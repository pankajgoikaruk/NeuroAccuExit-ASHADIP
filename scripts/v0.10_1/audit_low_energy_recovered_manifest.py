#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit manually reviewed TATA-LAWYER low-energy rows before using them in
NeuroAccuExit v0.10_1.

This script is intentionally read-only for source manifests. It writes only to
the supplied output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

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

START_CANDIDATES = ["start_sec", "segment_start_sec", "window_start_sec", "start_time", "start"]
END_CANDIDATES = ["end_sec", "segment_end_sec", "window_end_sec", "end_time", "end"]
FEATURE_CANDIDATES = ["feature_path", "features_path", "npy_path", "feature_file"]


def _read_labels(labels_json: Optional[str]) -> List[str]:
    if not labels_json:
        return DEFAULT_LABELS
    path = Path(labels_json)
    if not path.exists():
        return DEFAULT_LABELS
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ["labels", "classes", "label_names"]:
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]
    return DEFAULT_LABELS


def _truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _select_reviewed_low_energy_rows(df: pd.DataFrame, review_flag_col: str, candidate_id_col: str) -> pd.DataFrame:
    masks = []
    if review_flag_col in df.columns:
        masks.append(_truthy_series(df[review_flag_col]))
    if candidate_id_col in df.columns:
        masks.append(df[candidate_id_col].notna() & (df[candidate_id_col].astype(str).str.strip() != ""))
    if "v09_evaluation_group" in df.columns:
        masks.append(df["v09_evaluation_group"].astype(str).str.lower().str.contains("recovered|low", regex=True, na=False))
    if not masks:
        raise ValueError(
            "Could not identify reviewed low-energy rows. Expected one of: "
            f"{review_flag_col}, {candidate_id_col}, or v09_evaluation_group."
        )
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return df.loc[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reviewed low-energy rows for v0.10_1 ablation.")
    parser.add_argument("--base-train-manifest", default="human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/final_expanded_training_dataset_balanced/metadata/multilabel_features_manifest_balanced.csv")
    parser.add_argument("--low-energy-masked-manifest", default="human_talk_workspace/tata_v0.9_pipeline/tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/feature_cache/metadata/multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv")
    parser.add_argument("--corrected-holdout-manifest", default="human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv")
    parser.add_argument("--labels-json", default="configs/human_talk_10label_schema.json")
    parser.add_argument("--out-dir", default="human_talk_workspace/tata_v0.10_1_low_energy_recovery_ablation/audit")
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--review-flag-col", default="v09_masked_review_applied")
    parser.add_argument("--candidate-id-col", default="v09_review_candidate_id")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_train_manifest)
    low_path = Path(args.low_energy_masked_manifest)
    holdout_path = Path(args.corrected_holdout_manifest)

    missing = [str(p) for p in [base_path, low_path, holdout_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required manifest(s):\n" + "\n".join(missing))

    labels = _read_labels(args.labels_json)
    base_df = pd.read_csv(base_path)
    low_df = pd.read_csv(low_path)
    holdout_df = pd.read_csv(holdout_path)
    reviewed_df = _select_reviewed_low_energy_rows(low_df, args.review_flag_col, args.candidate_id_col)

    required_label_cols = [c for c in labels if c in reviewed_df.columns]
    missing_labels = [c for c in labels if c not in reviewed_df.columns]
    mask_cols = [f"mask_{c}" for c in labels]
    present_mask_cols = [c for c in mask_cols if c in reviewed_df.columns]

    parent_col = args.parent_id_col
    if parent_col not in reviewed_df.columns:
        raise ValueError(f"Missing parent id column in low-energy manifest: {parent_col}")
    if parent_col not in holdout_df.columns:
        raise ValueError(f"Missing parent id column in corrected holdout manifest: {parent_col}")

    start_col = _first_present(reviewed_df.columns, START_CANDIDATES)
    end_col = _first_present(reviewed_df.columns, END_CANDIDATES)
    feature_col = _first_present(reviewed_df.columns, FEATURE_CANDIDATES)

    holdout_parents = set(holdout_df[parent_col].astype(str))
    reviewed_parents = set(reviewed_df[parent_col].dropna().astype(str))
    holdout_overlap = reviewed_df[reviewed_df[parent_col].astype(str).isin(holdout_parents)].copy()

    if present_mask_cols:
        mask_numeric = reviewed_df[present_mask_cols].apply(pd.to_numeric, errors="coerce")
        fully_known_mask = mask_numeric.eq(1).all(axis=1)
    else:
        fully_known_mask = pd.Series(False, index=reviewed_df.index)

    if required_label_cols:
        label_numeric = reviewed_df[required_label_cols].apply(pd.to_numeric, errors="coerce")
        label_value_counts = {
            label: {
                "positive": int((label_numeric[label] == 1).sum()),
                "negative": int((label_numeric[label] == 0).sum()),
                "unknown_or_missing": int(label_numeric[label].isna().sum() + (~label_numeric[label].isin([0, 1]) & label_numeric[label].notna()).sum()),
            }
            for label in required_label_cols
        }
    else:
        label_value_counts = {}

    report = {
        "base_train_manifest": str(base_path),
        "low_energy_masked_manifest": str(low_path),
        "corrected_holdout_manifest": str(holdout_path),
        "labels": labels,
        "base_rows": int(len(base_df)),
        "low_energy_source_rows": int(len(low_df)),
        "reviewed_low_energy_rows": int(len(reviewed_df)),
        "reviewed_parent_count": int(len(reviewed_parents)),
        "holdout_rows": int(len(holdout_df)),
        "holdout_parent_count": int(len(holdout_parents)),
        "holdout_parent_overlap_rows": int(len(holdout_overlap)),
        "holdout_parent_overlap_parent_count": int(holdout_overlap[parent_col].astype(str).nunique()) if len(holdout_overlap) else 0,
        "present_label_columns": required_label_cols,
        "missing_label_columns": missing_labels,
        "present_mask_columns": present_mask_cols,
        "fully_known_reviewed_rows": int(fully_known_mask.sum()),
        "partially_or_unmasked_reviewed_rows": int((~fully_known_mask).sum()),
        "start_column_detected": start_col,
        "end_column_detected": end_col,
        "feature_column_detected": feature_col,
        "split_column_present": bool(args.split_col in reviewed_df.columns),
        "split_counts_reviewed": reviewed_df[args.split_col].astype(str).value_counts().to_dict() if args.split_col in reviewed_df.columns else {},
        "label_value_counts_reviewed": label_value_counts,
        "safety_policy": "Original manifests are read-only. Build step must write a copied/augmented manifest in v0.10_1 workspace only.",
    }

    (out_dir / "low_energy_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    reviewed_df.head(50).to_csv(out_dir / "reviewed_low_energy_sample_head50.csv", index=False)
    if len(holdout_overlap):
        holdout_overlap.to_csv(out_dir / "holdout_parent_overlap_rows.csv", index=False)

    print("[OK] Low-energy audit complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
