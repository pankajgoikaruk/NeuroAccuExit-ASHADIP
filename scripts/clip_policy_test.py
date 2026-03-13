# scripts/clip_policy_test.py

r"""
Clip-level (TIME) early-exit on top of your segment-level (DEPTH) early-exit.

Adds:
- clip_preds.csv (per-clip predictions + compute)
- confusion matrix (clip-level)
- per-class precision/recall (clip-level)
- prints Windows Saved (%) + Compute Saved (%) (if full baseline JSON exists)
- segment accuracy over processed windows ("policy test accuracy" on used segments)
- Reviewer-proof diagnostics:
  (1) Fixed-position segment accuracy for every clip (independent of time-exit):
      Acc_firstK, Acc_midK, Acc_lastK (+ n)
  (2) Stop-speed group consistency check (Depth×Time only):
      stop==2 vs stop==3-4 vs stop>=5, report early-window accuracy per group
- Optional: per-clip window counts printout:
    clip1 -> windows_total=30  |  id=...
    clip1 -> windows_total=30, windows_used=2  |  id=...

Key idea:
- segments.csv contains:
  wav_relpath (clip id), start (time), feat_relpath (feature pointer), split
- Group by wav_relpath, sort by start, and process segments sequentially.
- For each segment:
  - run the model
  - depth_ea_decide() chooses exit depth + pred_taken (DEPTH exit)
  - accumulate evidence over time (log-probs)
- TIME exit stops remaining segments when prediction becomes stable/confident.

IMPORTANT FIX:
- TIME stop confidence uses normalized posterior:
    post_stop = softmax(accum_logp / used_windows)
  so confidence does NOT explode after 2 windows.

Reviewer-proofing tweaks:
A) If --disable_time_exit: report "Clip-length distribution" over windows_total (not a stopping distribution)
B) Diagnostics are apples-to-apples:
   Fixed-position evaluation does NOT affect time-exit stopping and is NOT counted toward compute/used windows.

Example (PowerShell):

(1) Baseline (FULL clip; no time early-exit)
python -m scripts.clip_policy_test `
  --run_dir "runs\v0_3\v0_3_027" `
  --segments_csv "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\features" `
  --disable_time_exit `
  --eval_fixed_k_windows 3 `
  --print_clip_windows

(2) Depth×Time (time early-exit enabled)
python -m scripts.clip_policy_test `
  --run_dir "runs\v0_3\v0_3_027" `
  --segments_csv "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\features" `
  --time_conf 0.95 `
  --time_stable_k 2 `
  --time_min_windows 2 `
  --eval_fixed_k_windows 3 `
  --print_clip_windows

(3) If you train with 5 exits (tap_blocks 1,2,3,4):
python -m scripts.clip_policy_test `
  --tap_blocks 1,2,3,4 `
  --run_dir ... `
  --segments_csv ... `
  --features_root ...
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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


def _save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _pad_or_trim_temps(temps, K: int):
    """Pad/trim temperature list to length K."""
    if temps is None:
        temps = [1.0] * K
    temps = [max(float(t), 1e-3) for t in temps]
    if len(temps) < K:
        temps = temps + [temps[-1]] * (K - len(temps))
    elif len(temps) > K:
        temps = temps[:K]
    return temps


def _stop_posterior_from_accum(accum_logp_cpu: torch.Tensor, used_windows: int) -> torch.Tensor:
    """
    Posterior used for TIME stopping.
    Use normalized log-probs to avoid confidence blowing up with sum(logp).
    Shape: (C,)
    """
    M = max(int(used_windows), 1)
    logp_for_stop = accum_logp_cpu / float(M)  # geometric mean posterior
    return torch.softmax(logp_for_stop, dim=-1)


def _print_confusion(cm: np.ndarray, labels: list[str]):
    """Pretty-print confusion matrix with labels. Rows=true, Cols=pred."""
    width = max(7, max(len(l) for l in labels) + 2)
    header = " " * width + "".join([f"{l:>{width}}" for l in labels])
    print(header)
    for i, l in enumerate(labels):
        row = "".join([f"{cm[i, j]:>{width}d}" for j in range(cm.shape[1])])
        print(f"{l:>{width}}{row}")


def _dist_stats(values: list[int]) -> dict:
    """
    Generic stats + histogram for integer distributions (windows_used or windows_total).
    Returns JSON-friendly dict with string keys for hist.
    """
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None, "hist": {}, "n_clips": 0}
    arr = sorted(int(x) for x in values)
    n = len(arr)
    # median (even/odd)
    if n % 2 == 1:
        median = float(arr[n // 2])
    else:
        median = 0.5 * (arr[n // 2 - 1] + arr[n // 2])
    mean = float(sum(arr) / n)
    hist = dict(Counter(arr))
    return {
        "min": int(arr[0]),
        "median": float(median),
        "max": int(arr[-1]),
        "mean": float(mean),
        "hist": {str(k): int(v) for k, v in sorted(hist.items())},
        "n_clips": int(n),
    }


def _valid_window_rows(g_sorted: pd.DataFrame, features_root: str) -> list[dict]:
    """
    Build a list of valid windows (with existing feature files), preserving time order.
    Each element: {"feat_path": ..., "feat_relpath": ..., "start": ...}
    """
    rows = []
    for _, row in g_sorted.iterrows():
        feat_rel = str(row["feat_relpath"])
        if not feat_rel or feat_rel.strip() == "":
            continue
        feat_rel = feat_rel.replace("\\", "/")
        feat_path = os.path.join(features_root, feat_rel)
        if not os.path.exists(feat_path):
            raise SystemExit(f"Missing feature file: {feat_path}")
        rows.append({"feat_path": feat_path, "feat_relpath": feat_rel, "start": float(row["start"])})
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", required=True)
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--num_workers", type=int, default=0)

    # ---- Model shape (Step 0: K-exit + C-class) ----
    ap.add_argument(
        "--tap_blocks",
        default="1,3",
        help="Comma list like 1,2,3,4 controlling number of exits (K=len(tap_blocks)+1). Default 1,3 (=3 exits).",
    )
    ap.add_argument("--n_mels", type=int, default=64, help="Number of mel bins (must match feature extraction).")

    # ---- TIME early-exit knobs (clip-level) ----
    ap.add_argument("--disable_time_exit", action="store_true", help="Process full clip (no time stopping).")
    ap.add_argument("--time_min_windows", type=int, default=2, help="Do not stop before this many segments.")
    ap.add_argument("--time_stable_k", type=int, default=2, help="Require same clip prediction for K steps.")
    ap.add_argument("--time_conf", type=float, default=0.95, help="Stop if clip posterior max >= this.")
    ap.add_argument(
        "--time_margin",
        type=float,
        default=0.0,
        help="Optional: stop only if (top1 - top2) >= this (prob space). 0 disables margin check.",
    )
    ap.add_argument("--time_max_windows", type=int, default=100000, help="Safety cap per clip.")

    # ---- Reviewer-proof fixed-position diagnostic ----
    ap.add_argument(
        "--eval_fixed_k_windows",
        type=int,
        default=0,
        help=(
            "If >0, evaluate segment accuracy on fixed positions for EVERY clip (independent of time-exit): "
            "first-K, middle-K, last-K. Does NOT count toward compute/used/stopping."
        ),
    )

    # ---- Optional: per-clip window counts ----
    ap.add_argument(
        "--print_clip_windows",
        action="store_true",
        help="Print per-clip window counts like: clip1 -> windows_total=30 | id=... (and windows_used if time-exit).",
    )

    # ---- For Compute Saved (%) ----
    ap.add_argument(
        "--full_baseline_json",
        default="",
        help=(
            "Path to full-clip baseline JSON to compute compute_saved%%. "
            "If empty, uses run_dir/clip_policy_results_full.json when time-exit is enabled."
        ),
    )

    args = ap.parse_args()

    tap_blocks = tuple(int(x) for x in str(args.tap_blocks).split(",") if str(x).strip())
    if len(tap_blocks) == 0:
        raise SystemExit("tap_blocks parsed empty. Provide e.g. --tap_blocks 1,3 or 1,2,3,4")

    time_exit_enabled = not args.disable_time_exit
    device = "cuda" if torch.cuda.is_available() else "cpu"

    temps = None  # loaded after model is built (needs K)

    # ---- Load EA knobs (same keys as policy_test.py) ----
    ea_path = os.path.join(args.run_dir, "ea_thresholds.json")
    if not os.path.exists(ea_path):
        raise FileNotFoundError(f"ea_thresholds.json not found in run_dir: {ea_path}")
    ea_file = _load_json(ea_path)
    ea_cfg = {
        "ea_threshold": float(ea_file["ea_threshold"]),
        "ea_mode": ea_file.get("ea_mode", "logprob"),
        "ea_min_exit": int(ea_file.get("ea_min_exit", 0)),
        "ea_stable_k": int(ea_file.get("ea_stable_k", 1)),
        "ea_flip_penalty": float(ea_file.get("ea_flip_penalty", 0.0)),
        "ea_exit1_conf_min": ea_file.get("ea_exit1_conf_min", None),
        "ea_exit1_margin_mult": float(ea_file.get("ea_exit1_margin_mult", 2.0)),
        "ea_exit1_margin_min": float(ea_file.get("ea_exit1_margin_min", 0.0)),
    }

    # ---- Get label2id mapping via your loader (guarantees consistency) ----
    _, _, _, label2id = make_loaders(
        args.segments_csv, args.features_root, batch_size=64, num_workers=args.num_workers
    )
    id2label = {v: k for k, v in label2id.items()}
    labels_sorted = [id2label[i] for i in range(len(id2label))]
    num_classes = len(label2id)

    # ---- Build model (Step 0 compatible) ----
    backbone = TinyAudioCNN(n_mels=int(args.n_mels), tap_blocks=tap_blocks)
    model = ExitNet(
        backbone,
        num_classes=num_classes,
        tap_dims=backbone.tap_dims,
        final_dim=backbone.final_dim,
    ).to(device)

    ckpt = os.path.join(args.run_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    K = int(model.num_exits)

    # ---- Load temps (pad/trim to K) ----
    tpath = os.path.join(args.run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = _load_json(tpath).get("temperatures", None)
    else:
        temps = None
    temps = _pad_or_trim_temps(temps, K)

    # ---- Load segments.csv and group by clip (wav_relpath) ----
    df = pd.read_csv(args.segments_csv)
    req = {"wav_relpath", "label", "start", "duration", "split", "feat_relpath"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"segments.csv missing required columns: {sorted(list(missing))}")

    df["wav_relpath"] = df["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
    df["feat_relpath"] = df["feat_relpath"].astype(str).str.replace("\\", "/", regex=False)

    df_te = df[df["split"] == "test"].copy()
    if len(df_te) == 0:
        raise SystemExit("No test rows found in segments.csv (split=='test').")

    # ---- Metrics (clip-level + used-windows segment-level) ----
    n_clips = 0
    clip_correct = 0

    exit_mix = Counter()  # used windows only: e1..eK
    seg_count = 0
    seg_taken_eq_final = 0
    flip_count_total = 0

    # Segment accuracy over USED windows (processed segments)
    seg_correct = 0

    # Fixed-position diagnostics (independent of time-exit)
    K_fix = int(args.eval_fixed_k_windows)
    fix_first_correct = 0
    fix_first_count = 0
    fix_mid_correct = 0
    fix_mid_count = 0
    fix_last_correct = 0
    fix_last_count = 0

    # Stop-speed group diagnostics (Depth×Time only): use first-K within each stop group
    group_firstk = {
        "stop_2": {"clips": 0, "correct": 0, "count": 0},
        "stop_3_4": {"clips": 0, "correct": 0, "count": 0},
        "stop_5_plus": {"clips": 0, "correct": 0, "count": 0},
    }

    windows_used_list = []
    windows_total_list = []
    stop_reasons = Counter()
    compute_units_list = []

    y_true_clips = []
    y_pred_clips = []
    clip_rows = []

    features_root = args.features_root

    # Output filenames
    out_json_full = os.path.join(args.run_dir, "clip_policy_results_full.json")
    out_json_time = os.path.join(args.run_dir, "clip_policy_results_time.json")

    # Legacy / histogram outputs
    out_hist_legacy = os.path.join(args.run_dir, "windows_used_hist.json")
    out_hist_full = os.path.join(args.run_dir, "clip_length_hist.json")
    out_hist_time = os.path.join(args.run_dir, "windows_used_hist.json")

    for clip_id, g in df_te.groupby("wav_relpath"):
        g_sorted = g.sort_values("start").reset_index(drop=True)

        windows_total = int(len(g_sorted))
        windows_total_list.append(windows_total)

        # true clip label (constant per wav file)
        lab = str(g_sorted["label"].iloc[0])
        if g_sorted["label"].nunique() != 1:
            raise SystemExit(f"Clip has mixed labels (should not happen): {clip_id}")
        y_true = int(label2id[lab])

        # Build valid window list (ordered)
        win_rows = _valid_window_rows(g_sorted, features_root)
        n_valid = len(win_rows)
        if n_valid == 0:
            raise SystemExit(f"No valid feature windows for clip: {clip_id}")

        # ------------------------------
        # Reviewer-proof fixed-position diagnostics (independent of time-exit)
        # ------------------------------
        if K_fix > 0:
            # positions computed on valid windows list
            def _eval_positions(idxs):
                nonlocal fix_first_correct, fix_first_count, fix_mid_correct, fix_mid_count, fix_last_correct, fix_last_count
                for pos in idxs:
                    w = win_rows[pos]
                    S = np.load(w["feat_path"])
                    x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0).to(device)
                    logits_list = model(x)
                    out = depth_ea_decide(
                        logits_list=logits_list,
                        temps=temps,
                        ea_mode=ea_cfg["ea_mode"],
                        ea_threshold=ea_cfg["ea_threshold"],
                        ea_min_exit=ea_cfg["ea_min_exit"],
                        ea_stable_k=ea_cfg["ea_stable_k"],
                        ea_flip_penalty=ea_cfg["ea_flip_penalty"],
                        ea_exit1_conf_min=ea_cfg["ea_exit1_conf_min"],
                        ea_exit1_margin_mult=ea_cfg["ea_exit1_margin_mult"],
                        ea_exit1_margin_min=ea_cfg["ea_exit1_margin_min"],
                    )
                    pred_taken = int(out["pred_taken"][0].item())
                    return pred_taken

            # First-K
            k = min(K_fix, n_valid)
            first_idxs = list(range(0, k))
            for pos in first_idxs:
                S = np.load(win_rows[pos]["feat_path"])
                x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0).to(device)
                logits_list = model(x)
                out = depth_ea_decide(
                    logits_list=logits_list,
                    temps=temps,
                    ea_mode=ea_cfg["ea_mode"],
                    ea_threshold=ea_cfg["ea_threshold"],
                    ea_min_exit=ea_cfg["ea_min_exit"],
                    ea_stable_k=ea_cfg["ea_stable_k"],
                    ea_flip_penalty=ea_cfg["ea_flip_penalty"],
                    ea_exit1_conf_min=ea_cfg["ea_exit1_conf_min"],
                    ea_exit1_margin_mult=ea_cfg["ea_exit1_margin_mult"],
                    ea_exit1_margin_min=ea_cfg["ea_exit1_margin_min"],
                )
                pred_taken = int(out["pred_taken"][0].item())
                fix_first_correct += int(pred_taken == y_true)
                fix_first_count += 1

            # Middle-K
            mid_start = max((n_valid // 2) - (k // 2), 0)
            mid_start = min(mid_start, max(n_valid - k, 0))
            mid_idxs = list(range(mid_start, mid_start + k))
            for pos in mid_idxs:
                S = np.load(win_rows[pos]["feat_path"])
                x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0).to(device)
                logits_list = model(x)
                out = depth_ea_decide(
                    logits_list=logits_list,
                    temps=temps,
                    ea_mode=ea_cfg["ea_mode"],
                    ea_threshold=ea_cfg["ea_threshold"],
                    ea_min_exit=ea_cfg["ea_min_exit"],
                    ea_stable_k=ea_cfg["ea_stable_k"],
                    ea_flip_penalty=ea_cfg["ea_flip_penalty"],
                    ea_exit1_conf_min=ea_cfg["ea_exit1_conf_min"],
                    ea_exit1_margin_mult=ea_cfg["ea_exit1_margin_mult"],
                    ea_exit1_margin_min=ea_cfg["ea_exit1_margin_min"],
                )
                pred_taken = int(out["pred_taken"][0].item())
                fix_mid_correct += int(pred_taken == y_true)
                fix_mid_count += 1

            # Last-K
            last_idxs = list(range(max(n_valid - k, 0), n_valid))
            for pos in last_idxs:
                S = np.load(win_rows[pos]["feat_path"])
                x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0).to(device)
                logits_list = model(x)
                out = depth_ea_decide(
                    logits_list=logits_list,
                    temps=temps,
                    ea_mode=ea_cfg["ea_mode"],
                    ea_threshold=ea_cfg["ea_threshold"],
                    ea_min_exit=ea_cfg["ea_min_exit"],
                    ea_stable_k=ea_cfg["ea_stable_k"],
                    ea_flip_penalty=ea_cfg["ea_flip_penalty"],
                    ea_exit1_conf_min=ea_cfg["ea_exit1_conf_min"],
                    ea_exit1_margin_mult=ea_cfg["ea_exit1_margin_mult"],
                    ea_exit1_margin_min=ea_cfg["ea_exit1_margin_min"],
                )
                pred_taken = int(out["pred_taken"][0].item())
                fix_last_correct += int(pred_taken == y_true)
                fix_last_count += 1

        # ------------------------------
        # Main clip processing (Depth×Time)
        # ------------------------------
        n_clips += 1
        accum_logp = torch.zeros((num_classes,), dtype=torch.float32)  # CPU accumulator
        used = 0

        # time stopping state
        last_pred = None
        stable_count = 0

        stopped = False
        stop_reason = "eof"
        stop_conf = None
        stop_margin = None
        compute_units = 0

        # Iterate over valid windows only (this is what "processed" means)
        for w in win_rows:
            if used >= int(args.time_max_windows):
                stop_reasons["max_windows_cap"] += 1
                stopped = True
                stop_reason = "max_windows_cap"
                break

            feat_path = w["feat_path"]

            S = np.load(feat_path)  # (M, T)
            x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,M,T)
            logits_list = model(x)

            out = depth_ea_decide(
                logits_list=logits_list,
                temps=temps,
                ea_mode=ea_cfg["ea_mode"],
                ea_threshold=ea_cfg["ea_threshold"],
                ea_min_exit=ea_cfg["ea_min_exit"],
                ea_stable_k=ea_cfg["ea_stable_k"],
                ea_flip_penalty=ea_cfg["ea_flip_penalty"],
                ea_exit1_conf_min=ea_cfg["ea_exit1_conf_min"],
                ea_exit1_margin_mult=ea_cfg["ea_exit1_margin_mult"],
                ea_exit1_margin_min=ea_cfg["ea_exit1_margin_min"],
            )

            taken = int(out["taken"][0].item())            # 0..K-1
            pred_taken = int(out["pred_taken"][0].item())  # class id
            pred_final = int(out["pred_final"][0].item())  # class id

            # segment-level stats (used windows only)
            exit_mix[f"e{taken+1}"] += 1
            seg_taken_eq_final += int(pred_taken == pred_final)
            flip_count_total += int(out["flip_count"][0].item() > 0)

            # segment accuracy over processed windows
            seg_correct += int(pred_taken == y_true)
            seg_count += 1

            # compute proxy (depth units)
            compute_units += (taken + 1)

            # TIME evidence update
            if "logp_taken" in out:
                accum_logp += out["logp_taken"][0].detach().cpu()
            else:
                lg = logits_list[taken][0].float()
                t = max(float(temps[taken]), 1e-3)
                accum_logp += F.log_softmax((lg / t).detach().cpu(), dim=-1)

            used += 1

            # stop posterior (normalized)
            post_stop = _stop_posterior_from_accum(accum_logp, used_windows=used)
            conf = float(post_stop.max().item())

            # margin (top1 - top2) if enabled
            if float(args.time_margin) > 0.0:
                top2 = torch.topk(post_stop, k=2).values
                margin = float((top2[0] - top2[1]).item())
            else:
                margin = None

            pred_clip = int(torch.argmax(post_stop).item())

            # update stability counter
            if last_pred is None or pred_clip != last_pred:
                stable_count = 1
            else:
                stable_count += 1
            last_pred = pred_clip

            # TIME stopping condition
            if time_exit_enabled:
                if used >= int(args.time_min_windows):
                    ok_stable = stable_count >= int(args.time_stable_k)
                    ok_conf = conf >= float(args.time_conf)
                    ok_margin = True
                    if float(args.time_margin) > 0.0:
                        ok_margin = (margin is not None) and (margin >= float(args.time_margin))

                    if ok_stable and ok_conf and ok_margin:
                        stop_reasons["time_conf_stable"] += 1
                        stopped = True
                        stop_reason = "time_conf_stable"
                        stop_conf = conf
                        stop_margin = margin
                        break

        # end window loop
        windows_used_list.append(int(used))
        compute_units_list.append(float(compute_units))

        # final clip prediction:
        # if time_exit disabled, still use normalized posterior at used==total (works fine)
        post_final = _stop_posterior_from_accum(accum_logp, used_windows=used)
        y_pred = int(torch.argmax(post_final).item())
        y_true_clips.append(y_true)
        y_pred_clips.append(y_pred)

        clip_correct += int(y_pred == y_true)

        # window distribution output depends on mode
        if args.print_clip_windows:
            if time_exit_enabled:
                print(f"clip{n_clips} -> windows_total={windows_total}, windows_used={used}  |  id={clip_id}")
            else:
                print(f"clip{n_clips} -> windows_total={windows_total}  |  id={clip_id}")

        clip_rows.append(
            {
                "clip_id": clip_id,
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "y_true_label": str(id2label[y_true]),
                "y_pred_label": str(id2label[y_pred]),
                "windows_total": int(windows_total),
                "windows_used": int(used),
                "compute_units": float(compute_units),
                "stop_reason": str(stop_reason),
                "stop_conf": None if stop_conf is None else float(stop_conf),
                "stop_margin": None if stop_margin is None else float(stop_margin),
            }
        )

        windows_total_list.append(int(windows_total))
        windows_used_list.append(int(used))

        # Stop-speed group diagnostics
        if time_exit_enabled and K_fix > 0:
            if used == 2:
                group = "stop_2"
            elif used in (3, 4):
                group = "stop_3_4"
            else:
                group = "stop_5_plus"

            group_firstk[group]["clips"] += 1

            # compute early-window accuracy on FIRST-K windows (not counted toward compute/stopping)
            k_eval = min(K_fix, n_valid)
            for pos in range(0, k_eval):
                S = np.load(win_rows[pos]["feat_path"])
                x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0).to(device)
                logits_list = model(x)
                out = depth_ea_decide(
                    logits_list=logits_list,
                    temps=temps,
                    ea_mode=ea_cfg["ea_mode"],
                    ea_threshold=ea_cfg["ea_threshold"],
                    ea_min_exit=ea_cfg["ea_min_exit"],
                    ea_stable_k=ea_cfg["ea_stable_k"],
                    ea_flip_penalty=ea_cfg["ea_flip_penalty"],
                    ea_exit1_conf_min=ea_cfg["ea_exit1_conf_min"],
                    ea_exit1_margin_mult=ea_cfg["ea_exit1_margin_mult"],
                    ea_exit1_margin_min=ea_cfg["ea_exit1_margin_min"],
                )
                pred_taken = int(out["pred_taken"][0].item())
                group_firstk[group]["correct"] += int(pred_taken == y_true)
                group_firstk[group]["count"] += 1

    # ---- Clip accuracy ----
    clip_acc = clip_correct / max(n_clips, 1)

    # ---- Segment metrics over USED windows ----
    seg_acc_used = seg_correct / max(seg_count, 1)
    flip_rate = flip_count_total / max(seg_count, 1)
    exit_consistency = seg_taken_eq_final / max(seg_count, 1)

    # ---- Fixed-position diagnostics summary ----
    def _safe_div(a, b):
        return None if b == 0 else (a / b)

    acc_firstK = _safe_div(fix_first_correct, fix_first_count)
    acc_midK = _safe_div(fix_mid_correct, fix_mid_count)
    acc_lastK = _safe_div(fix_last_correct, fix_last_count)

    # ---- Stop-speed groups summary ----
    stop_groups_summary = {}
    for gname, d in group_firstk.items():
        stop_groups_summary[gname] = {
            "clips": int(d["clips"]),
            "firstK_acc": None if d["count"] == 0 else float(d["correct"] / d["count"]),
            "firstK_n_segments": int(d["count"]),
        }

    # ---- Windows/Compute Saved (%) ----
    avg_used = float(np.mean(windows_used_list)) if windows_used_list else 0.0
    avg_total = float(np.mean(windows_total_list)) if windows_total_list else 0.0
    frac_used = (avg_used / avg_total) if avg_total > 0 else 0.0
    windows_saved_pct = 100.0 * (1.0 - frac_used) if avg_total > 0 else 0.0

    avg_compute_units = float(np.mean(compute_units_list)) if compute_units_list else 0.0
    avg_depth_per_used_window = (avg_compute_units / max(avg_used, 1e-9)) if avg_used > 0 else 0.0

    compute_full_ref = None
    compute_saved_pct = None

    # If time-exit enabled: compare against FULL baseline JSON (if exists)
    if time_exit_enabled:
        baseline_json = args.full_baseline_json.strip()
        if baseline_json == "":
            baseline_json = out_json_full
        if os.path.exists(baseline_json):
            base = _load_json(baseline_json, {})
            compute_full_ref = base.get("avg_compute_units_sum_depth_over_used_windows", None)
            if compute_full_ref is not None:
                compute_saved_pct = 100.0 * (1.0 - (avg_compute_units / max(float(compute_full_ref), 1e-9)))

    # ---- Confusion + per-class ----
    cm = confusion_matrix(y_true_clips, y_pred_clips, labels=list(range(num_classes)))
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true_clips, y_pred_clips, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {}
    for i in range(num_classes):
        per_class[str(labels_sorted[i])] = {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }

    # ---- Distribution JSONs ----
    # legacy (for compatibility)
    hist_path_legacy = out_hist_legacy

    # mode-specific: full vs time
    if args.disable_time_exit:
        hist_path_mode = out_hist_full
        dist_payload = _dist_stats(windows_total_list)
    else:
        hist_path_mode = out_hist_time
        dist_payload = _dist_stats(windows_used_list)

    _save_json(hist_path_legacy, dist_payload)
    _save_json(hist_path_mode, dist_payload)

    # ---- Exit mix normalize (used windows) ----
    total_used = sum(exit_mix.values()) if exit_mix else 0
    exit_mix_norm = {k: (v / max(total_used, 1)) for k, v in sorted(exit_mix.items())}

    # ---- Final JSON payload ----
    results = {
        "policy": "ea",

        # Step 0 metadata (K-exit + C-class)
        "K": int(K),
        "tap_blocks": [int(x) for x in tap_blocks],
        "n_mels": int(args.n_mels),
        "num_classes": int(num_classes),

        # distribution
        "window_distribution": dist_payload,
        "window_distribution_json_mode": os.path.basename(hist_path_mode),
        "window_distribution_json_legacy": os.path.basename(hist_path_legacy),

        # segment metrics
        "segment_accuracy_over_used_windows": float(seg_acc_used),
        "n_segments_used_windows": int(seg_count),

        # fixed-position diagnostics
        "diagnostic_fixed_k": int(K_fix),
        "diagnostic_acc_firstK": None if acc_firstK is None else float(acc_firstK),
        "diagnostic_firstK_n_segments": int(fix_first_count),
        "diagnostic_acc_midK": None if acc_midK is None else float(acc_midK),
        "diagnostic_midK_n_segments": int(fix_mid_count),
        "diagnostic_acc_lastK": None if acc_lastK is None else float(acc_lastK),
        "diagnostic_lastK_n_segments": int(fix_last_count),

        # stop-speed group diagnostics
        "diagnostic_stop_groups_firstK": stop_groups_summary,

        # clip metrics
        "clip_accuracy": float(clip_acc),
        "n_clips": int(n_clips),

        "time_exit_enabled": bool(time_exit_enabled),
        "time_params": {
            "time_min_windows": int(args.time_min_windows),
            "time_stable_k": int(args.time_stable_k),
            "time_conf": float(args.time_conf),
            "time_margin": float(args.time_margin),
            "time_max_windows": int(args.time_max_windows),
        },

        "avg_windows_used": float(avg_used),
        "avg_windows_total": float(avg_total),
        "avg_fraction_windows_used": float(frac_used),

        "avg_compute_units_sum_depth_over_used_windows": float(avg_compute_units),
        "avg_depth_per_used_window": float(avg_depth_per_used_window),

        "windows_saved_pct": float(windows_saved_pct),
        "compute_full_ref_avg_units": None if compute_full_ref is None else float(compute_full_ref),
        "compute_saved_pct": None if compute_saved_pct is None else float(compute_saved_pct),

        "exit_mix_over_used_windows": {k: float(v) for k, v in exit_mix_norm.items()},
        "flip_rate_over_used_windows": float(flip_rate),
        "exit_consistency_taken_vs_final_over_used_windows": float(exit_consistency),

        "clip_confusion_matrix": cm.tolist(),
        "per_class": per_class,

        "stop_reasons": dict(stop_reasons),
        "ea": ea_cfg,
        "temperatures_used": temps,
    }

    # ---- Write outputs ----
    out_path_legacy = os.path.join(args.run_dir, "clip_policy_results.json")
    _save_json(out_path_legacy, results)

    out_path_mode = out_json_full if args.disable_time_exit else out_json_time
    _save_json(out_path_mode, results)

    preds_df = pd.DataFrame(clip_rows)
    preds_csv_legacy = os.path.join(args.run_dir, "clip_preds.csv")
    preds_df.to_csv(preds_csv_legacy, index=False)

    preds_csv_mode = os.path.join(
        args.run_dir, "clip_preds_full.csv" if args.disable_time_exit else "clip_preds_time.csv"
    )
    preds_df.to_csv(preds_csv_mode, index=False)

    # ---- Print summary ----
    print("== Clip-level (Depth × Time) Policy Test ==")
    print("Policy: ea")
    print(f"Policy test accuracy (segments, processed windows): {seg_acc_used:.4f}  (n_segments={seg_count})")
    print(f"Clip accuracy: {clip_acc:.4f}  (n_clips={n_clips})")

    if args.disable_time_exit:
        print(f"Avg windows used: {avg_used:.3f} / {avg_total:.3f}  (frac=1.000)")
        print(f"Avg compute units (sum depth over used windows): {avg_compute_units:.3f}")
        print(f"Avg depth per used window: {avg_depth_per_used_window:.3f}")
        print("Window distribution mode: clip length (windows_total)")
    else:
        print(f"Avg windows used: {avg_used:.3f} / {avg_total:.3f}  (frac={frac_used:.4f})")
        print(f"Windows Saved (%): {windows_saved_pct:.2f}%")
        print(f"Avg compute units (sum depth over used windows): {avg_compute_units:.3f}")
        print(f"Avg depth per used window: {avg_depth_per_used_window:.3f}")
        if compute_saved_pct is not None:
            print(f"Compute Saved (%): {compute_saved_pct:.2f}%  (ref full avg units={compute_full_ref:.3f})")
        print("Window distribution mode: windows used (stopping distribution)")

    if exit_mix_norm:
        print(
            "Exit mix (used windows): "
            + ", ".join([f"{k}={exit_mix_norm[k]:.3f}" for k in sorted(exit_mix_norm.keys())])
        )

    print(f"Flip-rate (used windows): {flip_rate:.4f}")
    print(f"Exit-consistency taken==final (used windows): {exit_consistency:.4f}")

    if K_fix > 0:
        print("== Fixed-position diagnostic (independent of time-exit) ==")
        print(f"K={K_fix}  Acc_firstK={acc_firstK:.4f} (n={fix_first_count})")
        print(f"K={K_fix}  Acc_midK={acc_midK:.4f} (n={fix_mid_count})")
        print(f"K={K_fix}  Acc_lastK={acc_lastK:.4f} (n={fix_last_count})")

    if time_exit_enabled and K_fix > 0:
        print("== Stop-speed group diagnostic (Depth×Time only; first-K accuracy) ==")
        for gname in ["stop_2", "stop_3_4", "stop_5_plus"]:
            d = stop_groups_summary[gname]
            print(
                f"{gname}: clips={d['clips']}, firstK_acc={d['firstK_acc']}, firstK_n={d['firstK_n_segments']}"
            )

    print("== Confusion matrix (clip-level) ==")
    _print_confusion(cm, labels_sorted)

    print(f"Wrote: {out_path_mode}")
    print(f"Wrote: {preds_csv_mode}")
    print(f"Wrote: {hist_path_mode}")


if __name__ == "__main__":
    main()