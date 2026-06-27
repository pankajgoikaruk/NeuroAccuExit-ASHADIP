r"""
scripts/v0.9/evaluate_frozen_labelwise_aggregation_v09.py

Final frozen v0.9 labelwise parent aggregation evaluation.

Purpose
-------
Apply a fixed/frozen labelwise parent aggregation map to the full corrected
holdout segment-probability CSV.

This script does NOT perform repeated calibration/evaluation splits.
This script does NOT search thresholds.
This script does NOT retrain the model.

It simply evaluates:
    segment probabilities -> parent-level aggregation -> fixed threshold -> metrics

Default frozen v0.9 map is based on the repeated v07-style calibration run:
    - mean for stable labels where mean was selected consistently
    - top2mean for labels where top2mean was selected most often
    - fixed threshold 0.5 for all labels

Expected input columns
----------------------
parent_clip_id
<label true columns>, e.g. Brene_Brown
probability columns with prefix, e.g. exit3_prob_Brene_Brown

Example
-------
python scripts\v0.9\evaluate_frozen_labelwise_aggregation_v09.py ^
  --segment-pred-csv "human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv" ^
  --out-dir "human_talk_workspace\tata_v0.9_labelwise_calibration\final_frozen_v06_labelwise_fixed_0p5" ^
  --labels-json "configs\human_talk_10label_schema.json" ^
  --parent-id-col "parent_clip_id" ^
  --prob-prefix "exit3_prob_" ^
  --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)


DEFAULT_FROZEN_METHOD_MAP: Dict[str, str] = {
    # Tie in repeated selection: mean=10, top2mean=10. Choose safer mean.
    "Brene_Brown": "mean",

    # Repeated selection preferred top2mean.
    "Eckhart_Tolle": "top2mean",

    # Repeated selection strongly preferred mean.
    "Eric_Thomas": "mean",

    # Repeated selection slightly preferred top2mean.
    "Gary_Vee": "top2mean",

    # Repeated selection strongly preferred mean.
    "Jay_Shetty": "mean",
    "Nick_Vujicic": "mean",
    "other_speaker_present": "mean",
    "music_present": "mean",

    # Transient/context labels selected top2mean most consistently.
    "audience_reaction_present": "top2mean",
    "silence_present": "top2mean",
}

VALID_METHODS = {"mean", "max", "top2mean"}


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


def threshold_slug(value: float) -> str:
    if abs(value - 0.5) < 1e-12:
        return "fixed_0p5"
    return "fixed_" + str(value).replace(".", "p").replace("-", "m")


def load_labels(labels_json: Path | None, df: pd.DataFrame, prob_prefix: str) -> List[str]:
    if labels_json is not None:
        payload = load_json(labels_json)
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict) and "labels" in payload:
            return [str(x) for x in payload["labels"]]
        raise RuntimeError(
            f"Could not read labels from {labels_json}. Expected JSON list or dict with key 'labels'."
        )

    labels = [col[len(prob_prefix):] for col in df.columns if col.startswith(prob_prefix)]
    if not labels:
        raise RuntimeError(
            f"Could not infer labels. No columns found with probability prefix '{prob_prefix}'. "
            "Pass --labels-json or check --prob-prefix."
        )
    return labels


def load_method_map(method_map_json: Path | None) -> Dict[str, str]:
    if method_map_json is None:
        return dict(DEFAULT_FROZEN_METHOD_MAP)

    payload = load_json(method_map_json)
    if not isinstance(payload, dict):
        raise RuntimeError("--method-map-json must contain a JSON object: {label: method}")
    return {str(k): str(v) for k, v in payload.items()}


def validate_inputs(
    df: pd.DataFrame,
    labels: List[str],
    method_map: Dict[str, str],
    parent_id_col: str,
    prob_prefix: str,
) -> None:
    if parent_id_col not in df.columns:
        raise RuntimeError(f"Missing parent id column: {parent_id_col}")

    missing_true = [lab for lab in labels if lab not in df.columns]
    if missing_true:
        raise RuntimeError(f"Missing true-label columns: {missing_true}")

    missing_prob = [f"{prob_prefix}{lab}" for lab in labels if f"{prob_prefix}{lab}" not in df.columns]
    if missing_prob:
        raise RuntimeError(f"Missing probability columns: {missing_prob}")

    missing_methods = [lab for lab in labels if lab not in method_map]
    if missing_methods:
        raise RuntimeError(f"Method map missing labels: {missing_methods}")

    extra_methods = [lab for lab in method_map if lab not in labels]
    if extra_methods:
        print(f"WARNING: method map has labels not present in current schema; ignored: {extra_methods}")

    invalid = {lab: method for lab, method in method_map.items() if method not in VALID_METHODS}
    if invalid:
        raise RuntimeError(f"Invalid aggregation methods found: {invalid}. Valid: {sorted(VALID_METHODS)}")


def aggregate_values(values: pd.Series, method: str) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if values.empty:
        return 0.0

    if method == "mean":
        return float(values.mean())
    if method == "max":
        return float(values.max())
    if method == "top2mean":
        top = values.sort_values(ascending=False).head(2)
        return float(top.mean())

    raise RuntimeError(f"Unsupported method: {method}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_score": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
        "per_label": {},
    }

    for i, lab in enumerate(labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics["per_label"][lab] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
        }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen v0.9 labelwise aggregation on full corrected holdout."
    )
    parser.add_argument("--segment-pred-csv", required=True, help="Segment-level probability CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--labels-json", default=None, help="Optional label schema JSON.")
    parser.add_argument("--method-map-json", default=None, help="Optional frozen method-map JSON.")
    parser.add_argument("--parent-id-col", default="parent_clip_id", help="Parent clip id column.")
    parser.add_argument("--prob-prefix", default="exit3_prob_", help="Probability column prefix.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold for all labels.")
    parser.add_argument("--model-name", default="main_v08_human_corrected_balanced_3exit_20260610_084027")
    parser.add_argument("--split", default="final_raw_holdout_parent_level")
    args = parser.parse_args()

    segment_csv = Path(args.segment_pred_csv)
    out_dir = Path(args.out_dir)
    labels_json = Path(args.labels_json) if args.labels_json else None
    method_map_json = Path(args.method_map_json) if args.method_map_json else None
    threshold = float(args.threshold)

    if not segment_csv.exists():
        raise FileNotFoundError(f"Segment prediction CSV not found: {segment_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(segment_csv, low_memory=False)
    labels = load_labels(labels_json, df, args.prob_prefix)
    method_map = load_method_map(method_map_json)
    validate_inputs(df, labels, method_map, args.parent_id_col, args.prob_prefix)

    parent_rows: List[Dict[str, Any]] = []
    for parent_id, group in df.groupby(args.parent_id_col, sort=False):
        row: Dict[str, Any] = {args.parent_id_col: parent_id}

        # Preserve useful metadata if present.
        for meta_col in ["source_file", "source_path", "parent_source_file", "parent_source_path"]:
            if meta_col in group.columns:
                row[meta_col] = group[meta_col].iloc[0]

        for lab in labels:
            true_values = pd.to_numeric(group[lab], errors="coerce").fillna(0).astype(int)
            row[lab] = int(true_values.max())

            prob_col = f"{args.prob_prefix}{lab}"
            method = method_map[lab]
            row[f"prob_{lab}"] = aggregate_values(group[prob_col], method)
            row[f"aggregation_{lab}"] = method

        parent_rows.append(row)

    parent_df = pd.DataFrame(parent_rows)

    y_true = parent_df[labels].astype(int).values
    prob_cols = [f"prob_{lab}" for lab in labels]
    y_prob = parent_df[prob_cols].astype(float).values
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, labels)
    for lab in labels:
        metrics["per_label"][lab]["aggregation"] = method_map[lab]
        metrics["per_label"][lab]["threshold"] = threshold

    threshold_name = threshold_slug(threshold)
    aggregation_name = "frozen_labelwise_v09_fixed_0p5"

    summary_row = {
        "model": args.model_name,
        "threshold_mode": threshold_name,
        "aggregation": aggregation_name,
        "split": args.split,
        "parent_clips": int(len(parent_df)),
        "segments": int(len(df)),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "samples_f1": metrics["samples_f1"],
        "exact_match": metrics["exact_match"],
        "hamming_loss": metrics["hamming_loss"],
        "jaccard_score": metrics["jaccard_score"],
        "avg_true_labels": metrics["avg_true_labels"],
        "avg_pred_labels": metrics["avg_pred_labels"],
    }

    per_label_rows = []
    for lab in labels:
        per_label_rows.append({"label": lab, **metrics["per_label"][lab]})

    method_rows = [{"label": lab, "aggregation": method_map[lab], "threshold": threshold} for lab in labels]

    parent_out = out_dir / "parent_frozen_labelwise_v09_probabilities.csv"
    summary_out = out_dir / "parent_holdout_static_frozen_labelwise_v09_fixed_0p5.csv"
    per_label_out = out_dir / "parent_holdout_per_label_frozen_labelwise_v09_fixed_0p5.csv"
    method_out = out_dir / "frozen_labelwise_v09_method_map.csv"
    json_out = out_dir / "parent_holdout_eval_frozen_labelwise_v09_fixed_0p5.json"

    parent_df.to_csv(parent_out, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_out, index=False)
    pd.DataFrame(per_label_rows).to_csv(per_label_out, index=False)
    pd.DataFrame(method_rows).to_csv(method_out, index=False)

    result = {
        "input": {
            "segment_pred_csv": str(segment_csv),
            "parent_id_col": args.parent_id_col,
            "prob_prefix": args.prob_prefix,
            "labels_json": str(labels_json) if labels_json else None,
            "method_map_json": str(method_map_json) if method_map_json else None,
        },
        "labels": labels,
        "frozen_method_map": {lab: method_map[lab] for lab in labels},
        "threshold": threshold,
        "summary": summary_row,
        "per_label": metrics["per_label"],
        "outputs": {
            "parent_probabilities_csv": str(parent_out),
            "summary_csv": str(summary_out),
            "per_label_csv": str(per_label_out),
            "method_map_csv": str(method_out),
        },
    }
    save_json(result, json_out)

    print("\nFrozen v0.9 labelwise aggregation evaluation complete")
    print("-" * 100)
    print(f"Segment CSV:      {segment_csv}")
    print(f"Parent clips:     {len(parent_df)}")
    print(f"Segments:         {len(df)}")
    print(f"Threshold:        {threshold}")
    print(f"Output dir:       {out_dir}")
    print("\nFrozen aggregation map:")
    print(pd.DataFrame(method_rows).to_string(index=False))
    print("\nSummary:")
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print("\nPer-label metrics:")
    print(pd.DataFrame(per_label_rows).to_string(index=False))
    print(f"\nOutput JSON: {json_out}")


if __name__ == "__main__":
    main()
