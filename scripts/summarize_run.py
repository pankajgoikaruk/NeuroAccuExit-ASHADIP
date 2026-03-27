# scripts/summarize_run.py

import os
import json
import argparse
import csv
import time

import numpy as np
import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net, load_run_model_cfg
from utils.profiling import conv2d_flops


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

    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)

    if not header:
        header = list(row.keys())

    filtered = {k: row.get(k, "") for k in header}
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow(filtered)


def ece_score(conf, corr, n_bins=15):
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

    plt.figure()
    plt.hist(conf, bins=30, range=(0, 1))
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence Histogram – {key}")
    p_hist = os.path.join(run_dir, "plots", f"{key}_conf_hist.png")
    plt.savefig(p_hist, bbox_inches="tight")
    plt.close()

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
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    x = np.asarray(conf)
    y = np.asarray(corr)
    yj = np.clip(y + np.random.uniform(-jitter, jitter, size=len(y)), -0.2, 1.2)

    plt.figure()
    plt.scatter(x, yj, s=6, alpha=0.4)
    plt.xlabel("Confidence")
    plt.ylabel("Correct (jittered)")
    plt.title(f"Confidence vs Correct – {key}")
    p_sc = os.path.join(run_dir, "plots", f"{key}_conf_scatter.png")
    plt.savefig(p_sc, bbox_inches="tight")
    plt.close()
    return p_sc


def infer_feature_shape(segments_csv, features_root):
    import pandas as pd
    seg = pd.read_csv(segments_csv)
    test_row = seg[seg["split"] == "test"].iloc[0]
    feat_rel = str(test_row["feat_relpath"]).replace("\\", "/")
    feat_path = os.path.join(features_root, feat_rel)
    n_mels, frames = np.load(feat_path).shape
    return int(n_mels), int(frames)


def estimate_flops_tiny_audiocnn_tapblocks(n_mels: int, frames: int, num_classes: int, tap_blocks):
    tap_blocks = sorted(set(int(b) for b in tap_blocks))
    if any(b < 1 or b > 4 for b in tap_blocks):
        raise ValueError(f"tap_blocks must be in [1..4]. Got: {tap_blocks}")

    ch = [16, 24, 32, 48, 64]
    H, W = int(n_mels), int(frames)

    flops = {}
    total = 0

    # block1 conv + pool
    f1, h1, w1 = conv2d_flops(H, W, cin=1, cout=ch[0], k=3, stride=1, padding=1)
    total += f1
    H1, W1 = h1 // 2, w1 // 2
    exit_idx = 1
    if 1 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[0] * num_classes)
        exit_idx += 1

    # block2 conv + pool
    f2, h2, w2 = conv2d_flops(H1, W1, cin=ch[0], cout=ch[1], k=3, stride=1, padding=1)
    total += f2
    H2, W2 = h2 // 2, w2 // 2
    if 2 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[1] * num_classes)
        exit_idx += 1

    # block3 conv
    f3, h3, w3 = conv2d_flops(H2, W2, cin=ch[1], cout=ch[2], k=3, stride=1, padding=1)
    total += f3
    if 3 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[2] * num_classes)
        exit_idx += 1

    # block4 conv
    f4, h4, w4 = conv2d_flops(h3, w3, cin=ch[2], cout=ch[3], k=3, stride=1, padding=1)
    total += f4
    if 4 in tap_blocks:
        flops[f"exit{exit_idx}"] = total + 2 * (ch[3] * num_classes)
        exit_idx += 1

    # block5 conv (final)
    f5, h5, w5 = conv2d_flops(h4, w4, cin=ch[3], cout=ch[4], k=3, stride=1, padding=1)
    total += f5
    flops[f"exit{exit_idx}"] = total + 2 * (ch[4] * num_classes)
    return flops


@torch.no_grad()
def collect_exit_logits_on_split(model, dl, device):
    """Returns dict per-exit: probs (N,C), y (N,), conf (N,), corr (N,) for using that exit directly."""
    model.eval()
    K = model.num_exits

    probs_chunks = [[] for _ in range(K)]
    y_chunks = []

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg_list = model(x)  # length K
        pr_list = [softmax(lg, dim=1).detach().cpu() for lg in lg_list]
        for k in range(K):
            probs_chunks[k].append(pr_list[k])
        y_chunks.append(y.detach().cpu())

    probs = [torch.cat(ch, 0).numpy() if len(ch) > 0 else None for ch in probs_chunks]
    ytrue = torch.cat(y_chunks, 0).numpy()

    out = {}
    for k in range(K):
        p = probs[k]
        conf = p.max(axis=1)
        pred = p.argmax(axis=1)
        corr = (pred == ytrue).astype(np.float32)
        out[f"exit{k+1}"] = {"probs": p, "y": ytrue, "conf": conf, "corr": corr}
    return out


@torch.no_grad()
def policy_eval(run_dir, segments_csv, features_root, tap_blocks, n_mels: int, save_plots=True):
    """
    Produces summary metrics for the run. K-exit compatible.

    If policy_results.json exists, it is treated as authoritative for:
      - policy_name, accuracy, avg_exit_depth, exit_mix, flip_rate, exit_consistency, EA knobs
    Otherwise we compute a greedy policy result using thresholds.json + temperature.json.
    """
    th = load_json_safepath(os.path.join(run_dir, "thresholds.json"), {"tau": 0.95})
    tau = float(th.get("tau", 0.95))

    # Data & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, dl_te, label2id = make_loaders(segments_csv, features_root, batch_size=64, num_workers=0)
    num_classes = len(label2id)

    model_cfg = load_run_model_cfg(run_dir)
    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device).eval()

    ckpt = os.path.join(run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    K = model.num_exits

    temps = load_json_safepath(os.path.join(run_dir, "temperature.json"), {"temperatures": None}).get("temperatures", None)
    if temps is None:
        temps = [1.0] * K
    temps = [max(float(t), 1e-3) for t in temps]
    if len(temps) < K:
        temps = temps + [temps[-1]] * (K - len(temps))
    elif len(temps) > K:
        temps = temps[:K]

    policy_results = load_json_safepath(os.path.join(run_dir, "policy_results.json"), None)

    policy_name = "greedy"
    avg_exit_depth = None
    flip_rate = None
    exit_consistency = None
    ea_threshold = None
    ea_mode = None
    ea_min_exit = None
    ea_stable_k = None
    ea_flip_penalty = None

    if isinstance(policy_results, dict):
        policy_name = policy_results.get("policy", policy_name)

    # If not greedy, we skip policy calibration plots (we don't have per-sample conf/corr for EA here)
    if policy_name != "greedy":
        save_plots = False

    # Greedy policy evaluation (fallback)
    n = 0
    correct = 0
    exits = []
    confs = []
    corrs = []

    def scale(lg, t):
        return lg / max(float(t), 1e-3)

    for x, y in dl_te:
        x, y = x.to(device), y.to(device)
        logits_list = model(x)
        logits_list = [scale(logits_list[i], temps[i]) for i in range(K)]
        probs_list = [softmax(lg, dim=1) for lg in logits_list]

        for i in range(x.size(0)):
            taken = K - 1
            for k in range(K):
                if float(probs_list[k][i].max()) >= tau:
                    taken = k
                    break
            p = probs_list[taken][i]
            pred = int(torch.argmax(p))
            conf = float(torch.max(p))
            corr = float(pred == int(y[i]))
            correct += int(corr)
            exits.append(taken + 1)
            n += 1
            confs.append(conf)
            corrs.append(corr)

    policy_acc = correct / max(n, 1)
    avg_exit_depth = float(np.mean(exits)) if exits else None

    exit_counts = {f"e{i}": exits.count(i) / max(n, 1) for i in range(1, K + 1)}

    # Override with policy_results.json if available
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
            # accept dynamic e1..eK
            for i in range(1, K + 1):
                key = f"e{i}"
                if key in mx:
                    exit_counts[key] = float(mx[key])

        if "ea" in policy_results and isinstance(policy_results["ea"], dict):
            ea = policy_results["ea"]
            ea_threshold = ea.get("ea_threshold", None)
            ea_mode = ea.get("ea_mode", None)
            ea_min_exit = ea.get("ea_min_exit", None)
            ea_stable_k = ea.get("ea_stable_k", None)
            ea_flip_penalty = ea.get("ea_flip_penalty", None)

    if policy_name != "greedy":
        tau = None

    # feature shape + FLOPs (K-generic)
    n_mels_inf, frames = infer_feature_shape(segments_csv, features_root)
    fl = estimate_flops_tiny_audiocnn_tapblocks(n_mels=int(n_mels_inf), frames=int(frames), num_classes=num_classes, tap_blocks=tap_blocks)
    full_mflops = float(fl[f"exit{K}"]) / 1e6
    expected_mflops = sum(float(exit_counts.get(f"e{i}", 0.0)) * float(fl[f"exit{i}"]) for i in range(1, K + 1)) / 1e6
    saving_pct = 100.0 * (1.0 - expected_mflops / max(full_mflops, 1e-12))

    saving_note = None
    if policy_name != "greedy" and not (isinstance(policy_results, dict) and isinstance(policy_results.get("exit_mix", None), dict)):
        saving_note = "WARNING: compute_saving_pct likely uses greedy exit_mix; add exit_mix to policy_results.json for accurate EA compute."

    # policy calibration plots (only greedy)
    policy_calib = {"ece": None, "bins": None, "hist_path": None, "reliability_path": None, "scatter_path": None}
    if save_plots:
        policy_calib = plot_hist_and_reliability(run_dir, "policy_test", np.array(confs), np.array(corrs), n_bins=15)
        policy_calib["scatter_path"] = plot_conf_vs_correct(run_dir, "policy_test", np.array(confs), np.array(corrs))

    # Per-exit calibration on TEST
    _, _, dl_te2, _ = make_loaders(segments_csv, features_root, batch_size=128, num_workers=0)
    per_exit = collect_exit_logits_on_split(model, dl_te2, device)

    per_exit_calib = {}
    for i in range(1, K + 1):
        conf = per_exit[f"exit{i}"]["conf"]
        corr = per_exit[f"exit{i}"]["corr"]
        per_exit_calib[f"exit{i}"] = plot_hist_and_reliability(run_dir, f"exit{i}_test", conf, corr, n_bins=15)
        per_exit_calib[f"exit{i}"]["scatter_path"] = plot_conf_vs_correct(run_dir, f"exit{i}_test", conf, corr)

    return {
        "policy_name": policy_name,
        "K": int(K),
        "tap_blocks": [int(x) for x in tap_blocks],
        "tau": tau,
        "temperatures": temps,
        "exit_mix": exit_counts,
        "policy_test_acc": float(policy_acc),
        "avg_exit_depth": avg_exit_depth,
        "flip_rate": flip_rate,
        "exit_consistency": exit_consistency,
        "ea_threshold": ea_threshold,
        "ea_mode": ea_mode,
        "ea_min_exit": ea_min_exit,
        "ea_stable_k": ea_stable_k,
        "ea_flip_penalty": ea_flip_penalty,
        "expected_mflops": float(expected_mflops),
        "full_mflops": float(full_mflops),
        "compute_saving_pct": float(saving_pct),
        "saving_note": saving_note,
        "policy_calibration": policy_calib,
        "per_exit_calibration": per_exit_calib,
        "n_mels": int(n_mels_inf),
        "frames": int(frames),
        "num_classes": int(num_classes),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "flops": {k: float(v) for k, v in fl.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--report_name", default="summary.json")
    ap.add_argument("--experiments_csv", default="runs/experiments.csv")
    ap.add_argument("--no_plots", action="store_true")

    ap.add_argument("--no_log", action="store_true", help="Do not append to experiments CSV.")

    # Step 0 (K-exit)
    ap.add_argument("--tap_blocks", default="1,3", help="Comma list like 1,2,3,4. Default 1,3 (=3 exits).")
    ap.add_argument("--n_mels", type=int, default=64)

    args = ap.parse_args()

    tap_blocks = tuple(int(x) for x in str(args.tap_blocks).split(",") if str(x).strip())

    metrics = load_json_safepath(os.path.join(args.run_dir, "metrics.json"), {})
    report = load_json_safepath(os.path.join(args.run_dir, "report.json"), {})
    calib = load_json_safepath(os.path.join(args.run_dir, "temperature.json"), {})
    thres = load_json_safepath(os.path.join(args.run_dir, "thresholds.json"), {})

    policy = policy_eval(
        args.run_dir,
        args.segments_csv,
        args.features_root,
        tap_blocks=tap_blocks,
        n_mels=int(args.n_mels),
        save_plots=not args.no_plots,
    )

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

    # CSV row (dynamic K; extra keys ignored if CSV already exists)
    row = {
        "run_id": run_id,
        "policy_name": policy.get("policy_name", "greedy"),
        "K": policy.get("K", ""),
        "tap_blocks": ",".join(str(x) for x in policy.get("tap_blocks", [])),
        "tau": "" if policy["tau"] is None else policy["tau"],
        "test_acc_policy": policy["policy_test_acc"],
        "avg_exit_depth": policy.get("avg_exit_depth", ""),
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
        "n_mels": policy["n_mels"],
        "frames": policy["frames"],
        "num_classes": policy["num_classes"],
        "torch_version": policy["torch_version"],
        "cuda": policy["cuda_available"],
        "saving_note": policy.get("saving_note", ""),
    }

    # temps + exit mix + per-exit ECE (dynamic K)
    temps = policy.get("temperatures", [])
    mx = policy.get("exit_mix", {})
    calib_per = policy.get("per_exit_calibration", {})

    for i in range(1, int(policy.get("K", 0)) + 1):
        row[f"temp_e{i}"] = temps[i - 1] if (i - 1) < len(temps) else ""
        row[f"exit_e{i}"] = mx.get(f"e{i}", "")
        row[f"ece_exit{i}"] = calib_per.get(f"exit{i}", {}).get("ece", "")

    append_csv_compat(args.experiments_csv, row)
    print("Logged row to", args.experiments_csv)


if __name__ == "__main__":
    main()