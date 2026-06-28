#!/usr/bin/env python
"""
Evaluate a bank of fixed label-wise parent aggregation maps on the same segment-level
prediction CSV.

Designed for agentic_data_preprocessing_v0.9.

Input segment CSV should contain:
  - parent_clip_id column by default
  - one true-label column per label, e.g. Brene_Brown
  - one probability column per label, e.g. exit3_prob_Brene_Brown

The script evaluates every map from a JSON config under the same conditions:
  - same corrected-holdout segment probabilities
  - same parent-level ground-truth rule: max(segment true labels)
  - same fixed threshold by default
  - no retraining
  - no calibration split
  - no threshold search

Supported aggregation methods:
  - mean
  - max
  - top2mean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

ALLOWED_METHODS = {"mean", "max", "top2mean"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def threshold_slug(threshold: float) -> str:
    if abs(threshold - 0.5) < 1e-12:
        return "fixed_0p5"
    return "fixed_" + str(threshold).replace("-", "m").replace(".", "p")


def sanitize_name(name: str) -> str:
    allowed = []
    for ch in name:
        if ch.isalnum() or ch in {"_", "-"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "map"


def top_k_mean(values: np.ndarray, k: int = 2) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    if values.size <= k:
        return float(np.mean(values))
    top_values = np.partition(values, -k)[-k:]
    return float(np.mean(top_values))


def aggregate_values(values: Iterable[float], method: str) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").astype(float).to_numpy()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0

    if method == "mean":
        return float(np.mean(arr))
    if method == "max":
        return float(np.max(arr))
    if method == "top2mean":
        return top_k_mean(arr, k=2)
    raise ValueError(f"Unsupported aggregation method: {method}")


def load_labels(config: Mapping[str, Any], labels_json: Path | None, df: pd.DataFrame, prob_prefix: str) -> List[str]:
    if labels_json is not None:
        payload = load_json(labels_json)
        if isinstance(payload, list):
            labels = [str(x) for x in payload]
        elif isinstance(payload, dict) and "labels" in payload:
            labels = [str(x) for x in payload["labels"]]
        else:
            raise RuntimeError(
                f"Could not read labels from {labels_json}. Expected JSON list or dict with key 'labels'."
            )
        return labels

    if "labels" in config:
        return [str(x) for x in config["labels"]]

    inferred = [c[len(prob_prefix):] for c in df.columns if c.startswith(prob_prefix)]
    if not inferred:
        raise RuntimeError(
            "Could not infer labels. Provide --labels-json or include a 'labels' list in the map JSON."
        )
    return inferred


def validate_config(config: Mapping[str, Any], labels: List[str], default_threshold: float) -> Dict[str, Dict[str, Any]]:
    maps = config.get("maps")
    if not isinstance(maps, dict) or not maps:
        raise RuntimeError("Mapping config must contain a non-empty object at key 'maps'.")

    label_set = set(labels)
    normalized: Dict[str, Dict[str, Any]] = {}

    for map_name, map_payload in maps.items():
        if not isinstance(map_payload, dict):
            raise RuntimeError(f"Map '{map_name}' must be an object.")
        aggregation = map_payload.get("aggregation")
        if not isinstance(aggregation, dict):
            raise RuntimeError(f"Map '{map_name}' must contain an 'aggregation' object.")

        missing = sorted(label_set - set(aggregation.keys()))
        extra = sorted(set(aggregation.keys()) - label_set)
        if missing:
            raise RuntimeError(f"Map '{map_name}' is missing labels: {missing}")
        if extra:
            raise RuntimeError(f"Map '{map_name}' contains unknown labels: {extra}")

        bad_methods = sorted({str(m) for m in aggregation.values()} - ALLOWED_METHODS)
        if bad_methods:
            raise RuntimeError(
                f"Map '{map_name}' has unsupported aggregation methods {bad_methods}. "
                f"Allowed: {sorted(ALLOWED_METHODS)}"
            )

        thresholds_obj = map_payload.get("thresholds", None)
        scalar_threshold = float(map_payload.get("threshold", default_threshold))
        if thresholds_obj is None:
            thresholds = {lab: scalar_threshold for lab in labels}
        elif isinstance(thresholds_obj, dict):
            missing_thr = sorted(label_set - set(thresholds_obj.keys()))
            extra_thr = sorted(set(thresholds_obj.keys()) - label_set)
            if missing_thr:
                raise RuntimeError(f"Map '{map_name}' thresholds missing labels: {missing_thr}")
            if extra_thr:
                raise RuntimeError(f"Map '{map_name}' thresholds contain unknown labels: {extra_thr}")
            thresholds = {lab: float(thresholds_obj[lab]) for lab in labels}
        else:
            raise RuntimeError(f"Map '{map_name}' thresholds must be an object if provided.")

        normalized[map_name] = {
            "description": str(map_payload.get("description", "")),
            "aggregation": {lab: str(aggregation[lab]) for lab in labels},
            "thresholds": thresholds,
        }

    return normalized


def validate_input_csv(df: pd.DataFrame, labels: List[str], parent_id_col: str, prob_prefix: str) -> None:
    if parent_id_col not in df.columns:
        raise RuntimeError(f"Input CSV missing parent ID column: {parent_id_col}")

    missing_true = [lab for lab in labels if lab not in df.columns]
    if missing_true:
        raise RuntimeError(f"Input CSV missing true-label columns: {missing_true}")

    missing_probs = [f"{prob_prefix}{lab}" for lab in labels if f"{prob_prefix}{lab}" not in df.columns]
    if missing_probs:
        raise RuntimeError(f"Input CSV missing probability columns: {missing_probs}")


def build_parent_truth(df: pd.DataFrame, labels: List[str], parent_id_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for parent_id, group in df.groupby(parent_id_col, sort=False):
        row: Dict[str, Any] = {parent_id_col: parent_id}
        for lab in labels:
            values = pd.to_numeric(group[lab], errors="coerce").fillna(0).astype(int)
            row[lab] = int(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_parent_probs_for_map(
    df: pd.DataFrame,
    labels: List[str],
    parent_id_col: str,
    prob_prefix: str,
    aggregation: Mapping[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for parent_id, group in df.groupby(parent_id_col, sort=False):
        row: Dict[str, Any] = {parent_id_col: parent_id}
        for lab in labels:
            prob_col = f"{prob_prefix}{lab}"
            row[f"prob_{lab}"] = aggregate_values(group[prob_col].values, aggregation[lab])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_score": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
    }

    per_label_rows: List[Dict[str, Any]] = []
    for idx, lab in enumerate(labels):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        per_label_rows.append(
            {
                "label": lab,
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "support": int(yt.sum()),
                "predicted_positive": int(yp.sum()),
            }
        )
    return summary, per_label_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple fixed label-wise aggregation maps on a full corrected holdout segment-probability CSV."
    )
    parser.add_argument("--segment-pred-csv", required=True, help="Segment-level prediction CSV.")
    parser.add_argument("--maps-json", required=True, help="JSON config containing candidate aggregation maps.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--labels-json", default=None, help="Optional labels JSON. Overrides labels in maps JSON.")
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--prob-prefix", default="exit3_prob_")
    parser.add_argument("--threshold", type=float, default=None, help="Global default threshold. If omitted, config default_threshold or 0.5 is used.")
    parser.add_argument("--model-name", default="main_v08_human_corrected_balanced_3exit_20260610_084027")
    parser.add_argument("--split", default="final_raw_holdout_parent_level")
    parser.add_argument("--save-parent-predictions", action="store_true", help="Save one parent prediction CSV per map.")
    args = parser.parse_args()

    segment_csv = Path(args.segment_pred_csv)
    maps_json = Path(args.maps_json)
    out_dir = Path(args.out_dir)
    labels_json = Path(args.labels_json) if args.labels_json else None
    out_dir.mkdir(parents=True, exist_ok=True)

    if not segment_csv.exists():
        raise FileNotFoundError(f"Segment prediction CSV not found: {segment_csv}")
    if not maps_json.exists():
        raise FileNotFoundError(f"Maps JSON not found: {maps_json}")

    config = load_json(maps_json)
    df = pd.read_csv(segment_csv, low_memory=False)

    labels = load_labels(config, labels_json, df, args.prob_prefix)
    validate_input_csv(df, labels, args.parent_id_col, args.prob_prefix)

    default_threshold = float(args.threshold if args.threshold is not None else config.get("default_threshold", 0.5))
    maps = validate_config(config, labels, default_threshold)
    threshold_name = threshold_slug(default_threshold)

    truth_df = build_parent_truth(df, labels, args.parent_id_col)
    y_true = truth_df[labels].astype(int).values

    summary_rows: List[Dict[str, Any]] = []
    per_label_all: List[Dict[str, Any]] = []
    parent_prediction_long_rows: List[Dict[str, Any]] = []

    for map_name, payload in maps.items():
        aggregation = payload["aggregation"]
        thresholds = payload["thresholds"]
        parent_probs = build_parent_probs_for_map(df, labels, args.parent_id_col, args.prob_prefix, aggregation)

        prob_cols = [f"prob_{lab}" for lab in labels]
        y_prob = parent_probs[prob_cols].astype(float).values
        thr = np.array([thresholds[lab] for lab in labels], dtype=float).reshape(1, -1)
        y_pred = (y_prob >= thr).astype(int)

        metrics, per_label_rows = compute_metrics(y_true, y_pred, labels)

        method_counts = pd.Series([aggregation[lab] for lab in labels]).value_counts().to_dict()
        threshold_values = np.array([thresholds[lab] for lab in labels], dtype=float)

        summary_row = {
            "model": args.model_name,
            "method": map_name,
            "description": payload.get("description", ""),
            "threshold_mode": threshold_name,
            "split": args.split,
            "parent_clips": int(len(truth_df)),
            "segments": int(len(df)),
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "samples_f1": metrics["samples_f1"],
            "exact_match": metrics["exact_match"],
            "hamming_loss": metrics["hamming_loss"],
            "jaccard_score": metrics["jaccard_score"],
            "avg_true_labels": metrics["avg_true_labels"],
            "avg_pred_labels": metrics["avg_pred_labels"],
            "mean_count": int(method_counts.get("mean", 0)),
            "max_count": int(method_counts.get("max", 0)),
            "top2mean_count": int(method_counts.get("top2mean", 0)),
            "threshold_min": float(threshold_values.min()),
            "threshold_max": float(threshold_values.max()),
            "threshold_mean": float(threshold_values.mean()),
        }
        summary_rows.append(summary_row)

        for row in per_label_rows:
            lab = row["label"]
            row.update(
                {
                    "model": args.model_name,
                    "method": map_name,
                    "aggregation": aggregation[lab],
                    "threshold": thresholds[lab],
                }
            )
            per_label_all.append(row)

        parent_output = truth_df[[args.parent_id_col] + labels].copy()
        for lab in labels:
            parent_output[f"prob_{lab}"] = parent_probs[f"prob_{lab}"].values
            parent_output[f"pred_{lab}"] = y_pred[:, labels.index(lab)]
            parent_output[f"aggregation_{lab}"] = aggregation[lab]
            parent_output[f"threshold_{lab}"] = thresholds[lab]

        parent_output.insert(0, "method", map_name)
        if args.save_parent_predictions:
            parent_path = out_dir / f"v09_mapping_bank_parent_predictions_{sanitize_name(map_name)}.csv"
            parent_output.to_csv(parent_path, index=False)

        # Long compact output for all methods/labels/parents.
        for parent_idx, parent_id in enumerate(parent_output[args.parent_id_col].values):
            for label_idx, lab in enumerate(labels):
                parent_prediction_long_rows.append(
                    {
                        "method": map_name,
                        args.parent_id_col: parent_id,
                        "label": lab,
                        "true": int(y_true[parent_idx, label_idx]),
                        "prob": float(y_prob[parent_idx, label_idx]),
                        "pred": int(y_pred[parent_idx, label_idx]),
                        "aggregation": aggregation[lab],
                        "threshold": thresholds[lab],
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["macro_f1", "micro_f1", "hamming_loss"], ascending=[False, False, True]
    )
    per_label_df = pd.DataFrame(per_label_all)
    parent_long_df = pd.DataFrame(parent_prediction_long_rows)

    summary_path = out_dir / "v09_mapping_bank_summary.csv"
    per_label_path = out_dir / "v09_mapping_bank_per_label.csv"
    parent_long_path = out_dir / "v09_mapping_bank_parent_predictions_long.csv"
    config_used_path = out_dir / "v09_mapping_bank_config_used.json"
    result_json_path = out_dir / "v09_mapping_bank_results.json"

    summary_df.to_csv(summary_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)
    parent_long_df.to_csv(parent_long_path, index=False)
    save_json(config, config_used_path)

    result_payload = {
        "segment_pred_csv": str(segment_csv),
        "maps_json": str(maps_json),
        "labels_json": str(labels_json) if labels_json else None,
        "parent_id_col": args.parent_id_col,
        "prob_prefix": args.prob_prefix,
        "default_threshold": default_threshold,
        "labels": labels,
        "parent_clips": int(len(truth_df)),
        "segments": int(len(df)),
        "summary": summary_df.to_dict(orient="records"),
        "outputs": {
            "summary_csv": str(summary_path),
            "per_label_csv": str(per_label_path),
            "parent_predictions_long_csv": str(parent_long_path),
            "config_used_json": str(config_used_path),
        },
    }
    save_json(result_payload, result_json_path)

    print("\nV0.9 labelwise mapping-bank evaluation complete")
    print("-" * 100)
    print(f"Segment CSV: {segment_csv}")
    print(f"Maps JSON:   {maps_json}")
    print(f"Parent clips: {len(truth_df)}")
    print(f"Segments:     {len(df)}")
    print(f"Output dir:   {out_dir}")
    print("")
    display_cols = [
        "method",
        "macro_f1",
        "micro_f1",
        "samples_f1",
        "exact_match",
        "hamming_loss",
        "avg_pred_labels",
        "mean_count",
        "max_count",
        "top2mean_count",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved summary:   {summary_path}")
    print(f"Saved per-label: {per_label_path}")
    print(f"Saved parent predictions long: {parent_long_path}")


if __name__ == "__main__":
    main()
