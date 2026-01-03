# scripts/summarize_run.py

import os, json, argparse, csv, time
import numpy as np
import torch
from torch.nn.functional import softmax
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet
from utils.profiling import estimate_flops_tiny_audiocnn
import matplotlib.pyplot as plt


def load_json_safepath(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def append_csv_compat(path, row: dict):
    """
    Append a row to CSV without breaking existing header.
    - If CSV doesn't exist: create with row keys as header.
    - If CSV exists: reuse existing header; ignore extra keys; fill missing as blank.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        fieldnames = list(row.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
        return

    # Read existing header
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)

    if not header:
        header = list(row.keys())

    # Keep only known columns; fill missing
    filtered = {k: row.get(k, "") for k in header}

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow(filtered)


def ece_score(conf, corr, n_bins=15):
    """conf: (N,) confidences; corr: (N,) correctness (0/1)."""
    conf = np.asarray(conf)
    corr = np.asarray(corr).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_summ = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        sel = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(sel):
            bin_summ.append({"bin": [float(lo), float(hi)], "count": 0, "acc": None, "conf": None})
            continue
        p = sel.mean()
        acc = corr[sel].mean()
        cbar = conf[sel].mean()
        ece += p * abs(acc - cbar)
        bin_summ.append({"bin": [float(lo), float(hi)], "count": int(sel.sum()), "acc": float(acc), "conf": float(cbar)})
    return float(ece), bin_summ


def plot_hist_and_reliability(run_dir, key, conf, corr, n_bins=15):
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    # Histogram
    plt.figure()
    plt.hist(conf, bins=30, range=(0, 1))
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence Histogram – {key}")
    p_hist = os.path.join(run_dir, "plots", f"{key}_conf_hist.png")
    plt.savefig(p_hist, bbox_inches="tight")
    plt.close()

    # Reliability
    ece, bins = ece_score(conf, corr, n_bins=n_bins)
    accs, confs = [], []
    for b in bins:
        if b["acc"] is not None:
            accs.append(b["acc"])
            confs.append(b["conf"])
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence (bin mean)")
    plt.ylabel("Accuracy (bin)")
    plt.title(f"Reliability – {key} (ECE={ece:.3f})")
    p_rel = os.path.join(run_dir, "plots", f"{key}_reliability.png")
    plt.savefig(p_rel, bbox_inches="tight")
    plt.close()
    return {"ece": ece, "bins": bins, "hist_path": p_hist, "reliability_path": p_rel}


def plot_conf_vs_correct(run_dir, key, conf, corr, jitter=0.02):
    """Scatter: confidence (x) vs correctness (y∈{0,1})."""
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    y = np.asarray(corr).astype(float)
    x = np.asarray(conf).astype(float)
    yj = y + (np.random.rand(*y.shape) - 0.5) * jitter
    plt.figure()
    plt.scatter(x, yj, s=8, alpha=0.5)
    plt.yticks([0, 1], ["wrong", "correct"])
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Confidence")
    plt.ylabel("Correctness")
    plt.title(f"Confidence vs Correctness – {key}")
    p_sc = os.path.join(run_dir, "plots", f"{key}_conf_vs_correct.png")
    plt.savefig(p_sc, bbox_inches="tight")
    plt.close()
    return p_sc


@torch.no_grad()
def collect_exit_logits_on_split(model, dl, device):
    """Returns dict per-exit: probs (N,C), y (N,), conf (N,), corr (N,) for using that exit directly."""
    all_batches = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg = model(x)
        pr = [softmax(l, dim=1) for l in lg]
        all_batches.append(([p.cpu() for p in pr], y.cpu()))
    probs = [torch.cat([a[0][k] for a in all_batches], 0).numpy() for k in range(3)]
    ytrue = torch.cat([a[1] for a in all_batches], 0).numpy()
    out = {}
    for k in range(3):
        conf = probs[k].max(axis=1)
        pred = probs[k].argmax(axis=1)
        corr = (pred == ytrue).astype(np.float32)
        out[f"exit{k+1}"] = {"probs": probs[k], "y": ytrue, "conf": conf, "corr": corr}
    return out


@torch.no_grad()
def policy_eval(run_dir, segments_csv, features_root, save_plots=True):
    # Load tau and temps (tau may not exist for EA; handled below)
    th = load_json_safepath(os.path.join(run_dir, "thresholds.json"), {"tau": 0.95})
    tau = float(th.get("tau", 0.95))

    temps = load_json_safepath(
        os.path.join(run_dir, "temperature.json"),
        {"temperatures": [1.0, 1.0, 1.0]},
    ).get("temperatures", [1.0, 1.0, 1.0])

    temps = [max(float(t), 0.5) for t in temps]

    # If policy_test.py wrote policy_results.json, use it (EA/greedy)
    policy_results = load_json_safepath(os.path.join(run_dir, "policy_results.json"), None)

    # Default policy metadata
    policy_name = "greedy"
    avg_exit_depth = None
    flip_rate = None
    exit_consistency = None
    ea_threshold = None
    ea_mode = None
    ea_min_exit = None
    ea_stable_k = None
    ea_flip_penalty = None

    # If policy_results exists, read policy name early (affects plotting decisions)
    if isinstance(policy_results, dict):
        policy_name = policy_results.get("policy", policy_name)

    # IMPORTANT: if the actual policy is not greedy, do NOT generate policy calibration plots here
    # because we do not have EA per-sample confidences/correctness in this script.
    if policy_name != "greedy":
        save_plots = False

    # data & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, dl_te, label2id = make_loaders(segments_csv, features_root, batch_size=64, num_workers=0)
    num_classes = len(label2id)

    model = ExitNet(TinyAudioCNN(), (16, 32), 64, num_classes).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(run_dir, "ckpt", "best.pt"), map_location=device))

    # Greedy policy evaluation (used only if no policy_results override exists)
    n = 0
    correct = 0
    exits = []
    confs = []
    corrs = []

    def scale(lg, t):
        return lg / max(t, 1e-3)

    for x, y in dl_te:
        x, y = x.to(device), y.to(device)
        logits = [scale(l, temps[i]) for i, l in enumerate(model(x))]
        probs = [softmax(l, dim=1) for l in logits]
        for i in range(x.size(0)):
            taken = 2
            for k in (0, 1, 2):
                if float(probs[k][i].max()) >= tau:
                    taken = k
                    break
            p = probs[taken][i]
            pred = int(torch.argmax(p))
            conf = float(torch.max(p))
            corr = float(pred == int(y[i]))
            correct += int(corr)
            exits.append(taken + 1)
            n += 1
            confs.append(conf)
            corrs.append(corr)

    # Greedy-derived defaults (may be overridden by policy_results.json)
    policy_acc = correct / max(n, 1)
    p1 = exits.count(1) / max(n, 1)
    p2 = exits.count(2) / max(n, 1)
    p3 = exits.count(3) / max(n, 1)
    avg_exit_depth = float(np.mean(exits)) if exits else None

    # Override with policy_results.json (authoritative for v0.2 EA)
    if isinstance(policy_results, dict):
        if "accuracy" in policy_results:
            policy_acc = float(policy_results["accuracy"])
        if "avg_exit_depth" in policy_results:
            avg_exit_depth = float(policy_results["avg_exit_depth"])
        if "flip_rate" in policy_results:
            flip_rate = float(policy_results["flip_rate"])
        if "exit_consistency" in policy_results:
            exit_consistency = float(policy_results["exit_consistency"])

        if "exit_mix" in policy_results and isinstance(policy_results["exit_mix"], dict):
            mx = policy_results["exit_mix"]
            p1 = float(mx.get("e1", mx.get("exit1", p1)))
            p2 = float(mx.get("e2", mx.get("exit2", p2)))
            p3 = float(mx.get("e3", mx.get("exit3", p3)))

        if "ea" in policy_results and isinstance(policy_results["ea"], dict):
            ea = policy_results["ea"]
            ea_threshold = ea.get("ea_threshold", None)
            ea_mode = ea.get("ea_mode", None)
            ea_min_exit = ea.get("ea_min_exit", None)
            ea_stable_k = ea.get("ea_stable_k", None)
            ea_flip_penalty = ea.get("ea_flip_penalty", None)

    # For non-greedy policies, tau is not meaningful
    if policy_name != "greedy":
        tau = None

    # feature shape for FLOPs
    import pandas as pd
    seg = pd.read_csv(segments_csv)
    test_row = seg[seg["split"] == "test"].iloc[0]
    feat_path = os.path.join(features_root, test_row["feat_relpath"])
    n_mels, frames = np.load(feat_path).shape

    # FLOPs
    fl = estimate_flops_tiny_audiocnn(n_mels=int(n_mels), frames=int(frames), num_classes=num_classes)
    full_mflops = fl["exit3"] / 1e6
    expected_mflops = (p1 * fl["exit1"] + p2 * fl["exit2"] + p3 * fl["exit3"]) / 1e6
    saving_pct = 100.0 * (1.0 - expected_mflops / full_mflops)

    # Warn if we probably used greedy exit_mix for an EA run
    saving_note = None
    if policy_name != "greedy" and not (isinstance(policy_results, dict) and isinstance(policy_results.get("exit_mix", None), dict)):
        saving_note = "WARNING: compute_saving_pct likely uses greedy exit_mix; add exit_mix to policy_results.json for accurate EA compute."

    # Policy calibration plots (ONLY for greedy)
    policy_calib = {"ece": None, "bins": None, "hist_path": None, "reliability_path": None, "scatter_path": None}
    if save_plots:
        policy_calib = plot_hist_and_reliability(run_dir, "policy_test", np.array(confs), np.array(corrs), n_bins=15)
        policy_scatter = plot_conf_vs_correct(run_dir, "policy_test", np.array(confs), np.array(corrs))
        policy_calib["scatter_path"] = policy_scatter

    # Per-exit calibration on TEST (no early exit; just each head)
    _, _, dl_te2, _ = make_loaders(segments_csv, features_root, batch_size=128, num_workers=0)
    per_exit = collect_exit_logits_on_split(model, dl_te2, device)
    per_exit_calib = {}
    for k in (1, 2, 3):
        conf = per_exit[f"exit{k}"]["conf"]
        corr = per_exit[f"exit{k}"]["corr"]
        per_exit_calib[f"exit{k}"] = plot_hist_and_reliability(run_dir, f"exit{k}_test", conf, corr, n_bins=15)
        sc_path = plot_conf_vs_correct(run_dir, f"exit{k}_test", conf, corr)
        per_exit_calib[f"exit{k}"]["scatter_path"] = sc_path

    return {
        "tau": tau,
        "temperatures": temps,
        "exit_mix": {"e1": p1, "e2": p2, "e3": p3},
        "policy_test_acc": float(policy_acc),
        "avg_exit_depth": avg_exit_depth,
        "flip_rate": flip_rate,
        "exit_consistency": exit_consistency,
        "policy_name": policy_name,
        "ea_threshold": ea_threshold,
        "ea_mode": ea_mode,
        "ea_min_exit": ea_min_exit,
        "ea_stable_k": ea_stable_k,
        "ea_flip_penalty": ea_flip_penalty,
        "policy_results_found": bool(policy_results),
        "saving_note": saving_note,
        "n_mels": int(n_mels),
        "frames": int(frames),
        "num_classes": num_classes,
        "expected_mflops": float(expected_mflops),
        "full_mflops": float(full_mflops),
        "compute_saving_pct": float(saving_pct),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "policy_calibration": policy_calib,
        "per_exit_calibration": per_exit_calib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--report_name", default="summary.json")
    ap.add_argument("--experiments_csv", default="runs/experiments.csv")
    ap.add_argument("--no_plots", action="store_true")

    ap.add_argument(
        "--no_log",
        action="store_true",
        help="Do not append a row to runs/experiments.csv (useful when called from run_reports.ps1).",
    )

    args = ap.parse_args()

    metrics = load_json_safepath(os.path.join(args.run_dir, "metrics.json"), {})
    report = load_json_safepath(os.path.join(args.run_dir, "report.json"), {})
    calib = load_json_safepath(os.path.join(args.run_dir, "temperature.json"), {})
    thres = load_json_safepath(os.path.join(args.run_dir, "thresholds.json"), {})

    policy = policy_eval(args.run_dir, args.segments_csv, args.features_root, save_plots=not args.no_plots)

    run_id = os.path.basename(args.run_dir.rstrip("\\/"))
    summary = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics_tail": metrics.get("val", [-1])[-1] if metrics else None,
        "val_curve": metrics.get("val", []),
        "test_report": report,
        "temperature": calib,
        "thresholds": thres,
        "policy_summary": policy,
    }

    out_json = os.path.join(args.run_dir, args.report_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved", out_json)

    if args.no_log:
        print("[summarize_run] --no_log set: skipping append to", args.experiments_csv)
        return

    row = {
        "run_id": run_id,
        "policy_name": policy.get("policy_name", "greedy"),
        "tau": "" if policy["tau"] is None else policy["tau"],
        "temp_e1": policy["temperatures"][0],
        "temp_e2": policy["temperatures"][1],
        "temp_e3": policy["temperatures"][2],
        "test_acc_policy": policy["policy_test_acc"],
        "avg_exit_depth": policy.get("avg_exit_depth", ""),
        "exit_e1": policy["exit_mix"]["e1"],
        "exit_e2": policy["exit_mix"]["e2"],
        "exit_e3": policy["exit_mix"]["e3"],
        "expected_mflops": policy["expected_mflops"],
        "full_mflops": policy["full_mflops"],
        "compute_saving_pct": policy["compute_saving_pct"],
        "flip_rate": policy.get("flip_rate", ""),
        "exit_consistency": policy.get("exit_consistency", ""),
        "ea_threshold": policy.get("ea_threshold", ""),
        "ea_mode": policy.get("ea_mode", ""),
        "ea_min_exit": policy.get("ea_min_exit", ""),
        "ea_stable_k": policy.get("ea_stable_k", ""),
        "ea_flip_penalty": policy.get("ea_flip_penalty", ""),
        "ece_policy": policy["policy_calibration"]["ece"],
        "ece_exit1": policy["per_exit_calibration"]["exit1"]["ece"],
        "ece_exit2": policy["per_exit_calibration"]["exit2"]["ece"],
        "ece_exit3": policy["per_exit_calibration"]["exit3"]["ece"],
        "n_mels": policy["n_mels"],
        "frames": policy["frames"],
        "num_classes": policy["num_classes"],
        "torch_version": policy["torch_version"],
        "cuda": policy["cuda_available"],
        "saving_note": policy.get("saving_note", ""),
    }

    append_csv_compat(args.experiments_csv, row)
    print("Logged row to", args.experiments_csv)


if __name__ == "__main__":
    main()
