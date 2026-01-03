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


@torch.no_grad()
def collect_val_logits(model, dl, device):
    """Collect logits for val split once so sweeps are fast."""
    logits_all = [[], [], []]
    ys = []
    for x, y in dl:
        x = x.to(device)
        logits = model(x)  # list of 3 tensors (B,C)
        for k in range(3):
            logits_all[k].append(logits[k].detach().cpu())
        ys.append(y.detach().cpu())
    logits_all = [torch.cat(v, dim=0) for v in logits_all]  # each (N,C)
    y = torch.cat(ys, dim=0).numpy()
    return logits_all, y


@torch.no_grad()
def greedy_decide_from_logits(logits_list_cpu, temps, tau: float):
    """
    Greedy per-sample decision using tau on max prob at each exit.
    logits_list_cpu: list of 3 tensors on CPU, each (N,C)
    temps: list of 3 floats
    returns: pred (N,), taken (N,) in {0,1,2}
    """
    assert len(logits_list_cpu) == 3
    N, C = logits_list_cpu[0].shape

    # scale logits and convert to probs per exit
    probs = []
    for k in range(3):
        lg = logits_list_cpu[k].float()
        t = max(float(temps[k]), 1e-3)
        lg = lg / t
        p = torch.softmax(lg, dim=1)
        probs.append(p)

    taken = torch.full((N,), 2, dtype=torch.long)
    pred = torch.zeros((N,), dtype=torch.long)

    for i in range(N):
        tk = 2
        for k in (0, 1, 2):
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

    # EA config
    ap.add_argument("--ea_mode", default="logprob", choices=["logprob", "logits"])
    ap.add_argument("--ea_min_exit", type=int, default=0)

    # EA grids (defaults give exit1 a real chance)
    ap.add_argument(
        "--ea_grid",
        default="0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35",
        help="EA margin thresholds (lower values allow earlier exits, incl. exit1).",
    )
    ap.add_argument("--stable_k_grid", default="1,2")
    ap.add_argument("--flip_penalty_grid", default="0.0,0.01,0.02,0.05")

    # Greedy baseline (for the “must beat greedy depth” constraint)
    ap.add_argument(
        "--greedy_tau_grid",
        default="0.70,0.75,0.80,0.85,0.90,0.92,0.95",
        help="Used only if thresholds.json is missing; selects greedy baseline on VAL.",
    )

    # Tradeoff + constraints
    ap.add_argument(
        "--lambda_depth",
        type=float,
        default=0.08,
        help="Penalty weight for avg_exit_depth in the EA score. Higher => more efficiency pressure.",
    )
    ap.add_argument(
        "--enforce_better_than_greedy_depth",
        action="store_true",
        help="Hard constraint: only accept EA configs with avg_exit_depth < greedy_baseline_depth.",
    )
    ap.add_argument(
        "--min_depth_improve",
        type=float,
        default=0.00,
        help="Require EA avg_exit_depth <= greedy_depth - min_depth_improve (when enforcement is on).",
    )
    ap.add_argument(
        "--max_f1_drop",
        type=float,
        default=1.00,
        help="Optional safety constraint: EA must have f1 >= greedy_f1 - max_f1_drop (set small like 0.02).",
    )

    args = ap.parse_args()

    # Temps (safe default)
    temps = [1.0, 1.0, 1.0]
    tpath = os.path.join(args.run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = _load_json(tpath, {}).get("temperatures", temps)
    temps = [max(float(t), 0.5) for t in temps]  # your stability clamp

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validation loader
    _, dl_va, _, label2id = make_loaders(
        args.segments_csv, args.features_root, batch_size=64, num_workers=0
    )
    num_classes = len(label2id)

    # Model
    model = ExitNet(TinyAudioCNN(), (16, 32), 64, num_classes).to(device).eval()
    ckpt = os.path.join(args.run_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # Collect logits once
    logits_val_cpu, y_val = collect_val_logits(model, dl_va, device)

    # ---------------------------
    # 1) Greedy baseline on VAL
    # ---------------------------
    thresholds_json = _load_json(os.path.join(args.run_dir, "thresholds.json"), None)
    greedy_tau = None
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
        # fallback: sweep greedy tau quickly from logits
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
        greedy_tau = float(best_g["tau"])

    greedy_depth_ref = float(greedy_baseline["avg_exit_depth"])
    greedy_f1_ref = float(greedy_baseline["f1"])

    # ---------------------------
    # 2) Sweep EA configs
    # ---------------------------
    thr_grid = _parse_float_list(args.ea_grid)
    stable_grid = _parse_int_list(args.stable_k_grid)
    flip_grid = _parse_float_list(args.flip_penalty_grid)

    all_rows = []
    best = None
    best_score = None

    # Move logits to device once for EA runs (val N is small; OK)
    logits_val_dev = [lg.to(device) for lg in logits_val_cpu]

    for stable_k in stable_grid:
        for flip_penalty in flip_grid:
            for thr in thr_grid:
                out = depth_ea_decide(
                    logits_list=logits_val_dev,
                    temps=temps,
                    ea_mode=args.ea_mode,
                    ea_threshold=float(thr),
                    ea_min_exit=int(args.ea_min_exit),
                    ea_stable_k=int(stable_k),
                    ea_flip_penalty=float(flip_penalty),
                )
                pred = out["pred_taken"].detach().cpu().numpy()
                taken = out["taken"].detach().cpu().numpy()

                f1 = float(f1_score(y_val, pred, average="macro"))
                acc = float((pred == y_val).mean())
                avg_exit = float((taken + 1).mean())

                # Simple tradeoff score:
                # - primary: F1
                # - penalize depth (encourage exit1/exit2)
                # - small acc tie-break inside score
                score = f1 + 0.10 * acc - float(args.lambda_depth) * avg_exit

                row = {
                    "ea_threshold": float(thr),
                    "ea_stable_k": int(stable_k),
                    "ea_flip_penalty": float(flip_penalty),
                    "f1": f1,
                    "acc": acc,
                    "avg_exit_depth": avg_exit,
                    "score": float(score),
                }
                all_rows.append(row)

                # Optional hard constraints
                if args.enforce_better_than_greedy_depth:
                    depth_ok = (avg_exit <= (greedy_depth_ref - float(args.min_depth_improve)))
                else:
                    depth_ok = True

                f1_ok = (f1 >= (greedy_f1_ref - float(args.max_f1_drop)))

                if not (depth_ok and f1_ok):
                    continue

                if best is None or (score > best_score):
                    best = row
                    best_score = score

    # If constraint was too strict, fallback to best score overall (but mark it)
    used_fallback = False
    if best is None:
        used_fallback = True
        # pick best score overall, no constraints
        best = max(all_rows, key=lambda r: r["score"])
        best_score = best["score"]

    # ---------------------------
    # 3) Save ea_thresholds.json
    # ---------------------------
    payload = {
        # required by policy_test.py
        "ea_threshold": float(best["ea_threshold"]),
        "ea_stable_k": int(best["ea_stable_k"]),
        "ea_flip_penalty": float(best["ea_flip_penalty"]),
        "ea_mode": args.ea_mode,
        "ea_min_exit": int(args.ea_min_exit),

        # selected metrics
        "f1": float(best["f1"]),
        "acc": float(best["acc"]),
        "avg_exit_depth": float(best["avg_exit_depth"]),
        "score": float(best_score),

        # metadata
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

    print("Saved ea_thresholds.json:", payload)
    print("Saved ea_sweep_results.json (all combos):", sweep_path)


if __name__ == "__main__":
    main()
