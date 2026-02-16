# scripts/policy_test.py

import os
import json
import argparse
from statistics import mean
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet

from policies.depth_ea import depth_ea_decide


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_extract_clip_ids(meta, B: int):
    """
    Try to extract per-sample clip ids (wav_relpath) from a third dataloader item.
    Supports: dict, list[str], list[dict], tuple, etc.
    """
    if meta is None:
        return None

    # meta could be a dict of lists
    if isinstance(meta, dict):
        for k in ("wav_relpath", "clip_id", "wav", "file", "relpath"):
            if k in meta:
                v = meta[k]
                if isinstance(v, (list, tuple)):
                    if len(v) == B:
                        return [str(x) for x in v]
                if torch.is_tensor(v):
                    if v.numel() == B:
                        return [str(x) for x in v.detach().cpu().tolist()]
                if isinstance(v, str):
                    return [v] * B
        return None

    # meta could be list[str] OR list[dict]
    if isinstance(meta, (list, tuple)) and len(meta) == B:
        if all(isinstance(x, (str, bytes)) for x in meta):
            out = []
            for x in meta:
                out.append(x.decode() if isinstance(x, bytes) else x)
            return [str(x) for x in out]
        if all(isinstance(x, dict) for x in meta):
            out = []
            for d in meta:
                v = d.get("wav_relpath", d.get("clip_id", None))
                out.append("" if v is None else str(v))
            if all(len(x) > 0 for x in out):
                return out
        return None

    return None


def main(run_dir: str, segments_csv: str, features_root: str, policy: str, num_workers: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load temps ----
    tpath = os.path.join(run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = _load_json(tpath).get("temperatures", [1.0, 1.0, 1.0])
    else:
        temps = [1.0, 1.0, 1.0]
    temps = [max(float(t), 1e-3) for t in temps]  # stability clamp

    # ---- Load policy config ----
    tau = None
    ea_cfg = None

    if policy == "greedy":
        th_path = os.path.join(run_dir, "thresholds.json")
        if not os.path.exists(th_path):
            raise FileNotFoundError(f"thresholds.json not found in run_dir: {th_path}")
        tau = float(_load_json(th_path)["tau"])

    elif policy == "ea":
        ea_path = os.path.join(run_dir, "ea_thresholds.json")
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
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # ---- Data ----
    _, _, dl_te, label2id = make_loaders(
        segments_csv, features_root, batch_size=64, num_workers=num_workers, return_meta=True
    )
    num_classes = len(label2id)

    # ---- CSV-order fallback for clip ids (SAFE: verify labels match) ----
    df = pd.read_csv(segments_csv)
    if "split" in df.columns:
        df_te = df[df["split"] == "test"].copy()
    else:
        df_te = df.copy()

    clip_csv_ok = False
    clip_ids_csv = None
    y_csv = None
    if {"wav_relpath", "label"}.issubset(set(df_te.columns)):
        df_te["wav_relpath"] = df_te["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
        df_te["label"] = df_te["label"].astype(str)
        df_te["_y_csv"] = df_te["label"].map(label2id)
        if df_te["_y_csv"].isna().any():
            clip_csv_ok = False
        else:
            clip_ids_csv = df_te["wav_relpath"].tolist()
            y_csv = df_te["_y_csv"].astype(int).tolist()
            clip_csv_ok = True

    # ---- Build model ----
    model = ExitNet(TinyAudioCNN(), (16, 32), 64, num_classes).to(device)
    ckpt = os.path.join(run_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ---- Segment-level metrics (same as before) ----
    n = 0
    correct = 0
    exits_taken = []

    flip_count_total = 0
    exit_consistent_total = 0

    # ---- Clip-level metrics (new) ----
    clip_enabled = True  # will auto-disable if mapping is unsafe
    clip_logp = {}       # clip_id -> torch (C,)
    clip_true = {}       # clip_id -> int label
    clip_units = Counter()   # clip_id -> sum depth units
    clip_windows = Counter() # clip_id -> number of segments seen

    # global sample index for CSV-order fallback
    global_i = 0
    csv_mismatch = 0

    def _get_clip_id_for_sample(meta_clip_ids, b_idx, y_val_int):
        nonlocal global_i, clip_enabled, csv_mismatch

        # 1) Preferred: meta-provided clip ids
        if meta_clip_ids is not None:
            cid = str(meta_clip_ids[b_idx]).replace("\\", "/")
            return cid

        # 2) Fallback: CSV order (only if safe + labels match)
        if clip_csv_ok and clip_ids_csv is not None and y_csv is not None:
            if global_i >= len(clip_ids_csv):
                clip_enabled = False
                return None
            cid = str(clip_ids_csv[global_i]).replace("\\", "/")
            y_expected = int(y_csv[global_i])
            if int(y_val_int) != y_expected:
                csv_mismatch += 1
                # disable to avoid reporting wrong clip metrics
                clip_enabled = False
                return None
            return cid

        # no mapping possible
        clip_enabled = False
        return None

    with torch.no_grad():
        for batch in dl_te:
            # allow loader to yield (x,y) or (x,y,meta)
            meta = None
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch

            x = x.to(device)
            y = y.to(device)
            logits_list = model(x)  # list of 3 tensors (B,C)
            B = x.size(0)

            meta_clip_ids = _try_extract_clip_ids(meta, B) if clip_enabled else None

            if policy == "greedy":
                scaled = [lg / max(float(temps[i]), 1e-3) for i, lg in enumerate(logits_list)]
                probs = [torch.softmax(lg, dim=1) for lg in scaled]

                for i in range(B):
                    # decide taken exit
                    taken = 2
                    for k in (0, 1, 2):
                        if float(probs[k][i].max()) >= tau:
                            taken = k
                            break

                    pred = int(torch.argmax(probs[taken][i]))
                    correct += int(pred == int(y[i]))
                    exits_taken.append(taken + 1)
                    n += 1

                    # clip metrics update
                    if clip_enabled:
                        cid = _get_clip_id_for_sample(meta_clip_ids, i, int(y[i]))
                        if cid is not None:
                            if cid not in clip_logp:
                                clip_logp[cid] = torch.zeros((num_classes,), dtype=torch.float32)
                                clip_true[cid] = int(y[i])

                            lg = (logits_list[taken][i].float() / max(float(temps[taken]), 1e-3)).detach().cpu()
                            logp = F.log_softmax(lg, dim=-1)

                            clip_logp[cid] += logp
                            clip_units[cid] += (taken + 1)
                            clip_windows[cid] += 1

                    global_i += 1

            else:
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

                taken_t = out["taken"]            # (B,) 0/1/2
                pred_taken = out["pred_taken"]    # (B,)
                pred_final = out["pred_final"]    # (B,)

                correct += int((pred_taken == y).sum().item())
                exits_taken.extend((taken_t + 1).detach().cpu().tolist())
                n += B

                flip_count_total += int((out["flip_count"] > 0).sum().item())
                exit_consistent_total += int((pred_taken == pred_final).sum().item())

                # clip metrics update (per sample for mapping)
                for i in range(B):
                    taken = int(taken_t[i].item())

                    if clip_enabled:
                        cid = _get_clip_id_for_sample(meta_clip_ids, i, int(y[i]))
                        if cid is not None:
                            if cid not in clip_logp:
                                clip_logp[cid] = torch.zeros((num_classes,), dtype=torch.float32)
                                clip_true[cid] = int(y[i])

                            # Prefer logp_taken if your depth_ea returns it; else fallback to taken-exit logp
                            if isinstance(out, dict) and ("logp_taken" in out):
                                logp = out["logp_taken"][i].detach().cpu().float()
                            else:
                                lg = (logits_list[taken][i].float() / max(float(temps[taken]), 1e-3)).detach().cpu()
                                logp = F.log_softmax(lg, dim=-1)

                            clip_logp[cid] += logp
                            clip_units[cid] += (taken + 1)
                            clip_windows[cid] += 1

                    global_i += 1

    # ---- Segment summaries ----
    acc = correct / max(n, 1)
    avg_exit = mean(exits_taken) if exits_taken else 0.0

    e1 = exits_taken.count(1) / max(n, 1)
    e2 = exits_taken.count(2) / max(n, 1)
    e3 = exits_taken.count(3) / max(n, 1)
    exit_mix = {"e1": float(e1), "e2": float(e2), "e3": float(e3)}

    print(f"Policy: {policy}")
    print(f"Policy test accuracy: {acc:.4f}")
    print(f"Avg exit depth: {avg_exit:.3f}")
    print(f"Exit mix: e1={exit_mix['e1']:.3f}, e2={exit_mix['e2']:.3f}, e3={exit_mix['e3']:.3f}")

    results = {
        "policy": policy,
        "accuracy": float(acc),
        "avg_exit_depth": float(avg_exit),
        "n_samples": int(n),
        "exit_mix": exit_mix,
        "temperatures_used": temps,
    }

    if policy == "ea":
        flip_rate = flip_count_total / max(n, 1)
        exit_consistency = exit_consistent_total / max(n, 1)
        print(f"Flip-rate: {flip_rate:.4f}")
        print(f"Exit-consistency (taken==final): {exit_consistency:.4f}")

        results.update({
            "flip_rate": float(flip_rate),
            "exit_consistency": float(exit_consistency),
            "ea": ea_cfg,
        })

    # ---- Clip summaries (new) ----
    # Only report if mapping was safe and we actually got clips.
    clip_metrics_available = bool(clip_enabled and len(clip_logp) > 0 and len(clip_true) == len(clip_logp))
    results["clip_metrics_available"] = bool(clip_metrics_available)

    if clip_csv_ok and (not clip_enabled):
        # helpful debug
        results["clip_metrics_disabled_reason"] = "clip_id mapping unsafe (likely loader order != segments.csv order or missing meta)."
        results["clip_metrics_csv_label_mismatches"] = int(csv_mismatch)

    if clip_metrics_available:
        n_clips = len(clip_logp)
        clip_correct = 0
        compute_units = []
        windows = []

        for cid, logp_sum in clip_logp.items():
            pred_clip = int(torch.argmax(logp_sum).item())
            y_true = int(clip_true[cid])
            clip_correct += int(pred_clip == y_true)

            compute_units.append(float(clip_units[cid]))
            windows.append(float(clip_windows[cid]))

        clip_acc = clip_correct / max(n_clips, 1)
        avg_units = float(np.mean(compute_units)) if compute_units else 0.0
        avg_windows = float(np.mean(windows)) if windows else 0.0
        avg_depth_per_window = (avg_units / max(avg_windows, 1e-9)) if avg_windows > 0 else 0.0

        # These match your clip_policy_test metrics (disable_time_exit case)
        results.update({
            "clip_accuracy": float(clip_acc),
            "n_clips": int(n_clips),
            "avg_windows_used": float(avg_windows),
            "avg_windows_total": float(avg_windows),  # segment-policy uses all windows
            "avg_fraction_windows_used": 1.0,
            "avg_compute_units_sum_depth_over_used_windows": float(avg_units),
            "avg_depth_per_used_window": float(avg_depth_per_window),
            "clip_agg": "sum_logp_over_segments",
        })

        print("== Clip-metrics (segment-policy, for fair comparison) ==")
        print(f"Clip accuracy: {clip_acc:.4f}  (n_clips={n_clips})")
        print(f"Avg windows used: {avg_windows:.3f} / {avg_windows:.3f}  (frac=1.000)")
        print(f"Avg compute units (sum depth over used windows): {avg_units:.3f}")
        print(f"Avg depth per used window: {avg_depth_per_window:.3f}")

    # ---- Write policy_results.json ----
    out_path = os.path.join(run_dir, "policy_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--policy", default="greedy", choices=["greedy", "ea"])
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    main(args.run_dir, args.segments_csv, args.features_root, args.policy, args.num_workers)
