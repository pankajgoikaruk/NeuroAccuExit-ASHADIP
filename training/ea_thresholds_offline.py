# training/ea_thresholds_offline.py
import os
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import f1_score

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet
from policies.depth_ea import depth_ea_decide


def _load_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _pad_or_trim(temps, K: int):
    if temps is None:
        return [1.0] * K
    temps = [max(float(t), 1e-3) for t in temps]
    if len(temps) < K:
        temps = temps + [temps[-1]] * (K - len(temps))
    elif len(temps) > K:
        temps = temps[:K]
    return temps


@torch.no_grad()
def collect_val_logits(model, dl, device):
    """Collect logits for val split once so sweeps are fast."""
    model.eval()
    K = model.num_exits

    logits_all = [[] for _ in range(K)]
    ys = []

    for x, y in dl:
        x = x.to(device)
        logits = model(x)  # list length K
        for k in range(K):
            logits_all[k].append(logits[k].detach().cpu())
        ys.append(y.detach().cpu())

    logits_all = [torch.cat(v, dim=0) for v in logits_all]  # each (N,C)
    y = torch.cat(ys, dim=0).numpy()
    return logits_all, y


@torch.no_grad()
def greedy_decide_from_logits(logits_list_cpu, temps, tau: float):
    """
    Greedy per-sample decision using tau on max prob at each exit.
    logits_list_cpu: list of K tensors on CPU, each (N,C)
    temps: list of K floats
    returns: pred (N,), taken (N,) in {0..K-1}
    """
    K = len(logits_list_cpu)
    N, C = logits_list_cpu[0].shape

    probs = []
    for k in range(K):
        lg = logits_list_cpu[k].float() / max(float(temps[k]), 1e-3)
        probs.append(torch.softmax(lg, dim=1))

    taken = torch.full((N,), K - 1, dtype=torch.long)
    pred = torch.zeros((N,), dtype=torch.long)

    for i in range(N):
        tk = K - 1
        for k in range(K):
            if float(probs[k][i].max()) >= tau:
                tk = k
                break
        taken[i] = tk
        pred[i] = int(torch.argmax(probs[tk][i]))

    return pred.numpy(), taken.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", required=True)
    ap.add_argument("--features_root", required=True)

    # Model shape
    ap.add_argument("--tap_blocks", default="1,3", help="Comma list like 1,2,3,4 (controls K). Default 1,3.")
    ap.add_argument("--n_mels", type=int, default=64)

    # EA config
    ap.add_argument("--ea_mode", default="logprob", choices=["logprob", "logits"])
    ap.add_argument("--ea_min_exit", type=int, default=0)

    # EA grids
    ap.add_argument("--ea_grid", default="0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35")
    ap.add_argument("--stable_k_grid", default="1,2")
    ap.add_argument("--flip_penalty_grid", default="0.0,0.01,0.02,0.05")

    # Greedy baseline if thresholds.json missing
    ap.add_argument("--greedy_tau_grid", default="0.70,0.75,0.80,0.85,0.90,0.92,0.95")

    # Tradeoff + constraints
    ap.add_argument("--lambda_depth", type=float, default=0.08)
    ap.add_argument("--enforce_better_than_greedy_depth", action="store_true")
    ap.add_argument("--min_depth_improve", type=float, default=0.00)
    ap.add_argument("--max_f1_drop", type=float, default=1.00)

    # Exit1 override sweep (passed to depth_ea_decide)
    ap.add_argument("--exit1_conf_grid", default="0.90,0.95")
    ap.add_argument("--exit1_margin_mult_grid", default="1.5,2.0,2.5")
    ap.add_argument("--exit1_margin_min", type=float, default=0.0)

    args = ap.parse_args()

    tap_blocks = tuple(int(x) for x in args.tap_blocks.split(",") if x.strip())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validation loader
    _, dl_va, _, label2id = make_loaders(args.segments_csv, args.features_root, batch_size=64, num_workers=0)
    num_classes = len(label2id)

    # Build model
    backbone = TinyAudioCNN(n_mels=args.n_mels, tap_blocks=tap_blocks)
    model = ExitNet(
        backbone,
        num_classes=num_classes,
        tap_dims=backbone.tap_dims,
        final_dim=backbone.final_dim,
    ).to(device).eval()

    ckpt = os.path.join(args.run_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    K = model.num_exits

    # Temps (pad/trim to K)
    temps = [1.0] * K
    tpath = os.path.join(args.run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = _load_json(tpath, {}).get("temperatures", temps)
    temps = _pad_or_trim(temps, K)

    # Collect logits once
    logits_val_cpu, y_val = collect_val_logits(model, dl_va, device)

    # ---------------------------
    # 1) Greedy baseline on VAL
    # ---------------------------
    thresholds_json = _load_json(os.path.join(args.run_dir, "thresholds.json"), None)
    greedy_baseline = None

    if isinstance(thresholds_json, dict) and "tau" in thresholds_json:
        greedy_tau = float(thresholds_json["tau"])
        pred_g, taken_g = greedy_decide_from_logits(logits_val_cpu, temps, greedy_tau)
        greedy_f1 = float(f1_score(y_val, pred_g, average="macro"))
        greedy_acc = float((pred_g == y_val).mean())
        greedy_depth = float((taken_g + 1).mean())
        greedy_baseline = {
            "tau": greedy_tau,
            "f1": greedy_f1,
            "acc": greedy_acc,
            "avg_exit_depth": greedy_depth,
            "source": "thresholds.json",
        }
    else:
        tau_grid = _parse_float_list(args.greedy_tau_grid)
        best_g = None
        for tau in tau_grid:
            pred_g, taken_g = greedy_decide_from_logits(logits_val_cpu, temps, float(tau))
            f1 = float(f1_score(y_val, pred_g, average="macro"))
            acc = float((pred_g == y_val).mean())
            depth = float((taken_g + 1).mean())
            cand = {"tau": float(tau), "f1": f1, "acc": acc, "avg_exit_depth": depth}
            if best_g is None or (cand["f1"] > best_g["f1"]) or (cand["f1"] == best_g["f1"] and cand["acc"] > best_g["acc"]):
                best_g = cand
        best_g["source"] = "swept_tau_grid"
        greedy_baseline = best_g

    greedy_depth_ref = float(greedy_baseline["avg_exit_depth"])
    greedy_f1_ref = float(greedy_baseline["f1"])

    # ---------------------------
    # 2) Sweep EA configs
    # ---------------------------
    thr_grid = _parse_float_list(args.ea_grid)
    stable_grid = _parse_int_list(args.stable_k_grid)
    flip_grid = _parse_float_list(args.flip_penalty_grid)

    exit1_conf_grid = [None] + _parse_float_list(args.exit1_conf_grid)
    exit1_mult_grid = _parse_float_list(args.exit1_margin_mult_grid)

    all_rows = []
    best = None
    best_score = None

    logits_val_dev = [lg.to(device) for lg in logits_val_cpu]

    for stable_k in stable_grid:
        for flip_penalty in flip_grid:
            for exit1_conf_min in exit1_conf_grid:
                for exit1_mult in exit1_mult_grid:
                    for thr in thr_grid:
                        out = depth_ea_decide(
                            logits_list=logits_val_dev,
                            temps=temps,
                            ea_mode=args.ea_mode,
                            ea_threshold=float(thr),
                            ea_min_exit=int(args.ea_min_exit),
                            ea_stable_k=int(stable_k),
                            ea_flip_penalty=float(flip_penalty),
                            ea_exit1_conf_min=exit1_conf_min,
                            ea_exit1_margin_mult=float(exit1_mult),
                            ea_exit1_margin_min=float(args.exit1_margin_min),
                        )

                        pred = out["pred_taken"].detach().cpu().numpy()
                        taken = out["taken"].detach().cpu().numpy()

                        f1 = float(f1_score(y_val, pred, average="macro"))
                        acc = float((pred == y_val).mean())
                        avg_exit = float((taken + 1).mean())

                        score = f1 + 0.10 * acc - float(args.lambda_depth) * avg_exit

                        row = {
                            "ea_threshold": float(thr),
                            "ea_stable_k": int(stable_k),
                            "ea_flip_penalty": float(flip_penalty),
                            "ea_exit1_conf_min": exit1_conf_min,
                            "ea_exit1_margin_mult": float(exit1_mult),
                            "ea_exit1_margin_min": float(args.exit1_margin_min),
                            "f1": f1,
                            "acc": acc,
                            "avg_exit_depth": avg_exit,
                            "score": float(score),
                        }
                        all_rows.append(row)

                        # constraints
                        if args.enforce_better_than_greedy_depth:
                            depth_ok = (avg_exit <= (greedy_depth_ref - float(args.min_depth_improve)))
                        else:
                            depth_ok = True
                        f1_ok = (f1 >= (greedy_f1_ref - float(args.max_f1_drop)))

                        if not (depth_ok and f1_ok):
                            continue

                        if best is None or (score > best_score):
                            best = row
                            best_score = float(score)

    used_fallback = False
    if best is None:
        used_fallback = True
        best = max(all_rows, key=lambda r: r["score"])
        best_score = float(best["score"])

    # ---------------------------
    # 3) Save ea_thresholds.json
    # ---------------------------
    payload = {
        "K": int(K),
        "tap_blocks": list(tap_blocks),
        "n_mels": int(args.n_mels),
        "num_classes": int(num_classes),

        # Exit1 override knobs
        "ea_exit1_conf_min": best.get("ea_exit1_conf_min", None),
        "ea_exit1_margin_mult": float(best.get("ea_exit1_margin_mult", 2.0)),
        "ea_exit1_margin_min": float(best.get("ea_exit1_margin_min", float(args.exit1_margin_min))),

        # Standard EA knobs
        "ea_threshold": float(best["ea_threshold"]),
        "ea_stable_k": int(best["ea_stable_k"]),
        "ea_flip_penalty": float(best["ea_flip_penalty"]),
        "ea_mode": args.ea_mode,
        "ea_min_exit": int(args.ea_min_exit),

        # Metrics (VAL)
        "f1": float(best["f1"]),
        "acc": float(best["acc"]),
        "avg_exit_depth": float(best["avg_exit_depth"]),
        "score": float(best_score),

        "temperatures_used": temps,
        "baseline_greedy_val": greedy_baseline,
        "selection": {
            "lambda_depth": float(args.lambda_depth),
            "enforce_better_than_greedy_depth": bool(args.enforce_better_than_greedy_depth),
            "min_depth_improve": float(args.min_depth_improve),
            "max_f1_drop": float(args.max_f1_drop),
            "used_fallback_no_constraint_solution": bool(used_fallback),
        },
        "grid": {
            "ea_grid": thr_grid,
            "stable_k_grid": stable_grid,
            "flip_penalty_grid": flip_grid,
            "exit1_conf_grid": exit1_conf_grid,  # includes None -> null
            "exit1_margin_mult_grid": exit1_mult_grid,
            "exit1_margin_min": float(args.exit1_margin_min),
            "greedy_tau_grid_used_if_missing": _parse_float_list(args.greedy_tau_grid),
        },
        "selected_by": "score = f1 + 0.10*acc - lambda_depth*avg_exit_depth (with optional constraints)",
    }

    outpath = os.path.join(args.run_dir, "ea_thresholds.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    sweep_path = os.path.join(args.run_dir, "ea_sweep_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print("Saved ea_thresholds.json")
    print("Saved ea_sweep_results.json")


if __name__ == "__main__":
    main()