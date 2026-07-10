#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
V0.10 Frozen LATS Config Evaluator
==================================

Applies a frozen label-wise aggregation + threshold config, such as the
LATS-v2 config from v0.9_4, to any segment-level probability CSV that follows
this schema:

    parent_clip_id
    <label> ground-truth columns
    <prob_prefix><label> probability columns

This script does not retrain a model and does not search new thresholds. It is
for controlled evaluation of a new segment-probability source, e.g. a v0.10
hint-pass model, using a frozen parent-level inference policy.

Example:

python scripts\v0.10\evaluate_frozen_lats_config_v010.py `
  --segment-pred-csv "human_talk_workspace\tata_v0.10_hint_pass\no_hint\parent_eval_segment_probs.csv" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --config-json "docs\tables\agentic_data_preprocessing_v0.9\v0.9_lats_v2\lats_v2_final_frozen_config.json" `
  --out-dir "human_talk_workspace\tata_v0.10_hint_pass\no_hint_lats_v2_eval" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --model-name "v0.10_no_hint_3exit"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score, jaccard_score
except Exception:  # pragma: no cover
    f1_score = None
    jaccard_score = None


DEFAULT_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
    "other_speaker_present",
    "music_present",
    "audience_reaction_present",
    "silence_present",
]


AGG_METHODS = {
    "mean",
    "max",
    "min",
    "median",
    "top2mean",
    "top3mean",
    "top4mean",
    "top5mean",
    "p75",
    "p90",
    "p95",
    "noisy_or",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply a frozen LATS/LATS-v2 labelwise aggregation-threshold config."
    )
    p.add_argument("--segment-pred-csv", required=True, type=Path)
    p.add_argument("--labels-json", default=None, type=Path)
    p.add_argument("--config-json", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--parent-id-col", default="parent_clip_id")
    p.add_argument("--prob-prefix", default="exit3_prob_")
    p.add_argument("--model-name", default="v0.10_hint_pass_candidate")
    p.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold if a label is missing a threshold in config.",
    )
    return p.parse_args()


def load_labels(labels_json: Path | None) -> List[str]:
    if labels_json is None:
        return list(DEFAULT_LABELS)
    with labels_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]

    for key in ("labels", "classes", "label_names"):
        if key in data and isinstance(data[key], list):
            return [str(x) for x in data[key]]

    if "label2id" in data and isinstance(data["label2id"], dict):
        return [k for k, _ in sorted(data["label2id"].items(), key=lambda kv: int(kv[1]))]

    raise RuntimeError(f"Could not infer labels from {labels_json}")


def load_frozen_config(config_json: Path, labels: Sequence[str], default_threshold: float) -> Dict[str, Dict[str, float | str]]:
    with config_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Supported structures:
    # 1) {"config": {label: {"aggregation": ..., "threshold": ...}}}
    # 2) {label: {"aggregation": ..., "threshold": ...}}
    # 3) {"maps": {"some_map": {"aggregation": {...}, "thresholds": {...}}}}
    if "config" in data and isinstance(data["config"], dict):
        raw = data["config"]
        cfg = {}
        for label in labels:
            item = raw.get(label, {})
            cfg[label] = {
                "aggregation": str(item.get("aggregation", "mean")).lower().strip(),
                "threshold": float(item.get("threshold", default_threshold)),
            }
        return validate_config(cfg, labels)

    if all(label in data for label in labels):
        cfg = {}
        for label in labels:
            item = data[label]
            if isinstance(item, dict):
                cfg[label] = {
                    "aggregation": str(item.get("aggregation", "mean")).lower().strip(),
                    "threshold": float(item.get("threshold", default_threshold)),
                }
            else:
                raise RuntimeError(f"Label {label!r} must map to an object in {config_json}")
        return validate_config(cfg, labels)

    if "maps" in data and isinstance(data["maps"], dict):
        if len(data["maps"]) != 1:
            raise RuntimeError(
                "Config JSON contains multiple maps. Please pass a single frozen config JSON, "
                "or create a JSON with top-level {'config': ...}."
            )
        map_obj = next(iter(data["maps"].values()))
        aggregation = map_obj.get("aggregation", {})
        thresholds = map_obj.get("thresholds", {})
        cfg = {}
        for label in labels:
            cfg[label] = {
                "aggregation": str(aggregation.get(label, "mean")).lower().strip(),
                "threshold": float(thresholds.get(label, default_threshold)),
            }
        return validate_config(cfg, labels)

    raise RuntimeError(
        f"Unsupported frozen config structure in {config_json}. Expected top-level 'config'."
    )


def validate_config(cfg: Dict[str, Dict[str, float | str]], labels: Sequence[str]) -> Dict[str, Dict[str, float | str]]:
    missing = [label for label in labels if label not in cfg]
    if missing:
        raise RuntimeError(f"Frozen config is missing labels: {missing}")

    for label in labels:
        method = str(cfg[label]["aggregation"]).lower().strip()
        threshold = float(cfg[label]["threshold"])
        if method not in AGG_METHODS:
            raise RuntimeError(f"Unsupported aggregation {method!r} for label {label!r}")
        if not (0.0 <= threshold <= 1.0):
            raise RuntimeError(f"Threshold for label {label!r} must be in [0,1], got {threshold}")
        cfg[label]["aggregation"] = method
        cfg[label]["threshold"] = threshold
    return cfg


def topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, min(int(k), int(values.size)))
    return float(np.sort(values)[-k:].mean())


def aggregate_values(values: Iterable[float], method: str) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0

    method = method.lower().strip()
    if method == "mean":
        return float(arr.mean())
    if method == "max":
        return float(arr.max())
    if method == "min":
        return float(arr.min())
    if method == "median":
        return float(np.median(arr))
    if method.startswith("top") and method.endswith("mean"):
        m = re.match(r"top(\d+)mean", method)
        if not m:
            raise RuntimeError(f"Invalid top-k aggregation method: {method}")
        return topk_mean(arr, int(m.group(1)))
    if method == "p75":
        return float(np.percentile(arr, 75))
    if method == "p90":
        return float(np.percentile(arr, 90))
    if method == "p95":
        return float(np.percentile(arr, 95))
    if method == "noisy_or":
        arr = np.clip(arr, 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - arr))

    raise RuntimeError(f"Unknown aggregation method: {method}")


def validate_input_columns(df: pd.DataFrame, labels: Sequence[str], parent_id_col: str, prob_prefix: str) -> None:
    required = [parent_id_col]
    for label in labels:
        required.append(label)
        required.append(f"{prob_prefix}{label}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns in segment prediction CSV:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def make_parent_tables(
    df: pd.DataFrame,
    labels: Sequence[str],
    cfg: Dict[str, Dict[str, float | str]],
    parent_id_col: str,
    prob_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parent_ids = sorted(df[parent_id_col].astype(str).unique().tolist())
    true_rows = []
    score_rows = []
    pred_rows = []

    grouped = df.copy()
    grouped[parent_id_col] = grouped[parent_id_col].astype(str)

    for parent_id, g in grouped.groupby(parent_id_col, sort=True):
        true_row = {parent_id_col: parent_id}
        score_row = {parent_id_col: parent_id}
        pred_row = {parent_id_col: parent_id}

        for label in labels:
            y_true = int(pd.to_numeric(g[label], errors="coerce").fillna(0).max() >= 0.5)
            method = str(cfg[label]["aggregation"])
            threshold = float(cfg[label]["threshold"])
            prob_col = f"{prob_prefix}{label}"
            score = aggregate_values(pd.to_numeric(g[prob_col], errors="coerce").values, method)
            y_pred = int(score >= threshold)

            true_row[label] = y_true
            score_row[label] = score
            pred_row[label] = y_pred

        true_rows.append(true_row)
        score_rows.append(score_row)
        pred_rows.append(pred_row)

    return pd.DataFrame(true_rows), pd.DataFrame(score_rows), pd.DataFrame(pred_rows)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if f1_score is None:
        raise RuntimeError("scikit-learn is required for metric computation.")

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    samples_f1 = float(f1_score(y_true, y_pred, average="samples", zero_division=0))
    exact = float(np.mean(np.all(y_true == y_pred, axis=1))) if y_true.size else 0.0
    hamming = float(np.mean(y_true != y_pred)) if y_true.size else 0.0
    if jaccard_score is not None:
        jaccard = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    else:
        jaccard = float("nan")
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "samples_f1": samples_f1,
        "exact_match": exact,
        "hamming_loss": hamming,
        "jaccard": jaccard,
        "avg_true_labels": float(y_true.sum(axis=1).mean()) if y_true.size else 0.0,
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()) if y_pred.size else 0.0,
        "label_count_abs_error": float(abs(y_true.sum(axis=1).mean() - y_pred.sum(axis=1).mean())) if y_true.size else 0.0,
        "n_parent_clips": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]) if y_true.ndim == 2 else 0,
    }


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    cfg: Dict[str, Dict[str, float | str]],
) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(labels):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "label": label,
                "aggregation": cfg[label]["aggregation"],
                "threshold": cfg[label]["threshold"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(yt.sum()),
                "predicted_positive": int(yp.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels_json)
    cfg = load_frozen_config(args.config_json, labels, args.default_threshold)

    df = pd.read_csv(args.segment_pred_csv)
    validate_input_columns(df, labels, args.parent_id_col, args.prob_prefix)

    y_true_df, score_df, pred_df = make_parent_tables(
        df=df,
        labels=labels,
        cfg=cfg,
        parent_id_col=args.parent_id_col,
        prob_prefix=args.prob_prefix,
    )

    y_true = y_true_df[labels].to_numpy(dtype=int)
    y_pred = pred_df[labels].to_numpy(dtype=int)

    metrics = compute_metrics(y_true, y_pred)
    metrics["method"] = "frozen_lats_config_eval"
    metrics["model_name"] = args.model_name
    metrics["segment_pred_csv"] = str(args.segment_pred_csv)
    metrics["config_json"] = str(args.config_json)

    eval_df = pd.DataFrame([metrics])
    per_label_df = per_label_metrics(y_true, y_pred, labels, cfg)

    eval_path = args.out_dir / "v010_frozen_lats_eval.csv"
    per_label_path = args.out_dir / "v010_frozen_lats_per_label.csv"
    true_path = args.out_dir / "v010_parent_truth.csv"
    score_path = args.out_dir / "v010_parent_scores.csv"
    pred_path = args.out_dir / "v010_parent_predictions.csv"
    config_used_path = args.out_dir / "v010_frozen_lats_config_used.json"
    results_json_path = args.out_dir / "v010_frozen_lats_eval.json"

    eval_df.to_csv(eval_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)
    y_true_df.to_csv(true_path, index=False)
    score_df.to_csv(score_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    with config_used_path.open("w", encoding="utf-8") as f:
        json.dump({"labels": labels, "config": cfg}, f, indent=2)
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_label": per_label_df.to_dict(orient="records")}, f, indent=2)

    print("\nV0.10 frozen LATS evaluation complete")
    print("-" * 100)
    print(f"Segment CSV:  {args.segment_pred_csv}")
    print(f"Config JSON:  {args.config_json}")
    print(f"Parent clips: {metrics['n_parent_clips']}")
    print(f"Output dir:   {args.out_dir}")
    print("")
    print(eval_df[["method", "macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]].to_string(index=False))
    print("")
    print(f"Saved summary:   {eval_path}")
    print(f"Saved per-label: {per_label_path}")


if __name__ == "__main__":
    main()
