#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a non-destructive v0.10_1 training manifest by copying the current
v0.10/v0.8-HCB training manifest and appending manually reviewed low-energy
1-second rows from TATA-LAWYER.

Safety rules:
  - source manifests are read-only;
  - copied inputs are saved under the v0.10_1 workspace;
  - low-energy rows are linked through parent_clip_id;
  - corrected-holdout parent overlap is excluded by default;
  - only fully-known reviewed rows are used by default, because the existing
    training.train_multilabel script does not apply per-label masks;
  - low-energy feature_path values are rewritten to absolute paths under the
    TATA-LAWYER feature cache so ASHADIP can read features without copying or
    modifying the source repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

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

FEATURE_PATH_COLUMNS = ["feature_path", "features_path", "npy_path"]


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


def _normalize_split_value(x: object) -> str:
    return str(x).strip().lower()


def _dedupe_key_columns(df: pd.DataFrame, parent_col: str) -> List[str]:
    candidates = [
        ["feature_path"],
        ["features_path"],
        ["npy_path"],
        [parent_col, "start_sec", "end_sec"],
        [parent_col, "segment_start_sec", "segment_end_sec"],
        [parent_col, "window_start_sec", "window_end_sec"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    return []


def _detect_feature_col(df: pd.DataFrame) -> Optional[str]:
    for col in FEATURE_PATH_COLUMNS:
        if col in df.columns:
            return col
    return None


def _auto_low_energy_features_root(low_manifest_path: Path) -> Path:
    # Expected TATA-LAWYER layout:
    #   .../feature_cache/metadata/manifest.csv
    #   .../feature_cache/features/<feature_path>
    if low_manifest_path.parent.name.lower() == "metadata":
        return low_manifest_path.parent.parent / "features"
    return low_manifest_path.parent / "features"


def _rewrite_low_energy_feature_paths(
    reviewed: pd.DataFrame,
    low_manifest_path: Path,
    low_energy_features_root: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    """Rewrite only selected low-energy rows to absolute feature paths.

    Base rows are left untouched. Low-energy rows come from a separate TATA-LAWYER
    workspace, so relative paths such as recovered_low_energy/train/... must be
    resolved against the TATA-LAWYER feature cache before ASHADIP training.
    """
    reviewed = reviewed.copy()
    feature_col = _detect_feature_col(reviewed)
    report = {
        "feature_column_detected": feature_col,
        "low_energy_features_root": None,
        "low_energy_feature_paths_rewritten": 0,
        "low_energy_feature_paths_already_absolute": 0,
        "low_energy_feature_paths_missing_after_rewrite": 0,
        "low_energy_feature_missing_examples": [],
    }

    if feature_col is None:
        return reviewed, report

    features_root = Path(low_energy_features_root) if low_energy_features_root else _auto_low_energy_features_root(low_manifest_path)
    features_root = features_root.resolve()
    report["low_energy_features_root"] = str(features_root)

    rewritten = []
    already_abs = 0
    missing_examples = []

    for raw_value in reviewed[feature_col].astype(str):
        raw_value = raw_value.strip()
        raw_path = Path(raw_value)

        if raw_path.is_absolute():
            candidate = raw_path
            already_abs += 1
        else:
            # Primary: feature_cache/features/<relative feature_path>
            candidate = features_root / raw_path

            # Fallbacks for manifests that already include features/ or feature_cache/
            if not candidate.exists():
                alt1 = low_manifest_path.parent.parent / raw_path
                alt2 = low_manifest_path.parent / raw_path
                if alt1.exists():
                    candidate = alt1
                elif alt2.exists():
                    candidate = alt2

        candidate = candidate.resolve()
        rewritten.append(str(candidate))
        if not candidate.exists() and len(missing_examples) < 10:
            missing_examples.append(str(candidate))

    reviewed.loc[:, feature_col] = rewritten
    report["low_energy_feature_paths_rewritten"] = int(len(rewritten) - already_abs)
    report["low_energy_feature_paths_already_absolute"] = int(already_abs)
    report["low_energy_feature_paths_missing_after_rewrite"] = int(sum(not Path(x).exists() for x in rewritten))
    report["low_energy_feature_missing_examples"] = missing_examples

    if report["low_energy_feature_paths_missing_after_rewrite"]:
        raise FileNotFoundError(
            "Some selected low-energy feature files were not found after path rewrite. "
            "Check --low-energy-features-root. Examples:\n"
            + "\n".join(missing_examples)
        )

    return reviewed, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v0.10_1 low-energy augmented manifest safely.")
    parser.add_argument("--base-train-manifest", default="human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/final_expanded_training_dataset_balanced/metadata/multilabel_features_manifest_balanced.csv")
    parser.add_argument("--low-energy-masked-manifest", default="human_talk_workspace/tata_v0.9_pipeline/tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/feature_cache/metadata/multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv")
    parser.add_argument("--low-energy-features-root", default="", help="Optional TATA-LAWYER feature_cache/features root. Auto-derived from manifest if omitted.")
    parser.add_argument("--corrected-holdout-manifest", default="human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv")
    parser.add_argument("--labels-json", default="configs/human_talk_10label_schema.json")
    parser.add_argument("--workspace-root", default="human_talk_workspace/tata_v0.10_1_low_energy_recovery_ablation")
    parser.add_argument("--out-name", default="multilabel_features_manifest_v010_1_LOW_ENERGY_AUGMENTED.csv")
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--include-splits", default="train", help="Comma-separated low-energy source splits to append. Default: train only.")
    parser.add_argument("--review-flag-col", default="v09_masked_review_applied")
    parser.add_argument("--candidate-id-col", default="v09_review_candidate_id")
    parser.add_argument("--allow-holdout-parent-overlap", action="store_true", help="Unsafe for final evaluation; default excludes overlap.")
    parser.add_argument("--allow-partial-mask", action="store_true", help="Use partially known rows. Not recommended with training.train_multilabel.")
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace_root)
    meta_dir = workspace / "metadata"
    copies_dir = workspace / "source_copies"
    meta_dir.mkdir(parents=True, exist_ok=True)
    copies_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_train_manifest)
    low_path = Path(args.low_energy_masked_manifest)
    holdout_path = Path(args.corrected_holdout_manifest)
    for path in [base_path, low_path, holdout_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required manifest: {path}")

    # Save immutable copies for reproducibility. These are copies only, never source edits.
    shutil.copy2(base_path, copies_dir / base_path.name)
    shutil.copy2(low_path, copies_dir / low_path.name)
    shutil.copy2(holdout_path, copies_dir / holdout_path.name)

    labels = _read_labels(args.labels_json)
    base_df = pd.read_csv(base_path, low_memory=False)
    low_df = pd.read_csv(low_path, low_memory=False)
    holdout_df = pd.read_csv(holdout_path, low_memory=False)

    parent_col = args.parent_id_col
    if parent_col not in base_df.columns:
        raise ValueError(f"Base manifest missing parent id column: {parent_col}")
    if parent_col not in low_df.columns:
        raise ValueError(f"Low-energy manifest missing parent id column: {parent_col}")
    if parent_col not in holdout_df.columns:
        raise ValueError(f"Holdout manifest missing parent id column: {parent_col}")

    reviewed = _select_reviewed_low_energy_rows(low_df, args.review_flag_col, args.candidate_id_col)
    reviewed["v010_1_selected_reason"] = "reviewed_low_energy_candidate"

    included_splits = {_normalize_split_value(x) for x in args.include_splits.split(",") if str(x).strip()}
    if args.split_col in reviewed.columns:
        before = len(reviewed)
        reviewed = reviewed[reviewed[args.split_col].map(_normalize_split_value).isin(included_splits)].copy()
        split_filtered = before - len(reviewed)
    else:
        split_filtered = 0

    # Existing train_multilabel does not apply mask columns, so default to fully-known rows only.
    mask_cols = [f"mask_{label}" for label in labels]
    present_mask_cols = [c for c in mask_cols if c in reviewed.columns]
    partial_mask_filtered = 0
    if present_mask_cols and not args.allow_partial_mask:
        mask_numeric = reviewed[present_mask_cols].apply(pd.to_numeric, errors="coerce")
        fully_known = mask_numeric.eq(1).all(axis=1)
        partial_mask_filtered = int((~fully_known).sum())
        reviewed = reviewed[fully_known].copy()

    # Keep only valid 0/1 label rows for standard BCE training.
    available_labels = [c for c in labels if c in reviewed.columns]
    label_value_filtered = 0
    if available_labels:
        label_numeric = reviewed[available_labels].apply(pd.to_numeric, errors="coerce")
        valid_labels = label_numeric.isin([0, 1]).all(axis=1)
        label_value_filtered = int((~valid_labels).sum())
        reviewed = reviewed[valid_labels].copy()
        reviewed.loc[:, available_labels] = label_numeric.loc[valid_labels, available_labels].astype(int)

    # Exclude corrected-holdout overlap by default to avoid leakage.
    holdout_parents = set(holdout_df[parent_col].astype(str))
    holdout_overlap_mask = reviewed[parent_col].astype(str).isin(holdout_parents)
    holdout_overlap_filtered = int(holdout_overlap_mask.sum())
    excluded_holdout_overlap = reviewed[holdout_overlap_mask].copy()
    if holdout_overlap_filtered and not args.allow_holdout_parent_overlap:
        reviewed = reviewed[~holdout_overlap_mask].copy()

    reviewed, feature_rewrite_report = _rewrite_low_energy_feature_paths(
        reviewed=reviewed,
        low_manifest_path=low_path,
        low_energy_features_root=args.low_energy_features_root or None,
    )

    base_df = base_df.copy()
    base_df["v010_1_source"] = "base_manifest_copy"
    base_df["v010_1_low_energy_added"] = False
    reviewed["v010_1_source"] = "tata_lawyer_human_reviewed_low_energy"
    reviewed["v010_1_low_energy_added"] = True

    combined = pd.concat([base_df, reviewed], ignore_index=True, sort=False)

    duplicate_removed = 0
    if not args.no_dedupe:
        key_cols = _dedupe_key_columns(combined, parent_col)
        if key_cols:
            before = len(combined)
            combined = combined.drop_duplicates(subset=key_cols, keep="first").copy()
            duplicate_removed = before - len(combined)

    out_manifest = meta_dir / args.out_name
    selected_low_energy_path = meta_dir / "selected_low_energy_rows_used.csv"
    excluded_overlap_path = meta_dir / "excluded_holdout_parent_overlap_rows.csv"
    report_path = meta_dir / "build_v010_1_low_energy_augmented_report.json"

    combined.to_csv(out_manifest, index=False)
    reviewed.to_csv(selected_low_energy_path, index=False)
    if len(excluded_holdout_overlap):
        excluded_holdout_overlap.to_csv(excluded_overlap_path, index=False)

    report = {
        "base_train_manifest": str(base_path),
        "low_energy_masked_manifest": str(low_path),
        "corrected_holdout_manifest": str(holdout_path),
        "output_manifest": str(out_manifest),
        "source_copies_dir": str(copies_dir),
        "base_rows": int(len(base_df)),
        "reviewed_low_energy_rows_initial": int(len(_select_reviewed_low_energy_rows(low_df, args.review_flag_col, args.candidate_id_col))),
        "split_filtered_rows": int(split_filtered),
        "partial_mask_filtered_rows": int(partial_mask_filtered),
        "invalid_label_value_filtered_rows": int(label_value_filtered),
        "holdout_parent_overlap_rows_detected": int(holdout_overlap_filtered),
        "holdout_parent_overlap_excluded": bool(holdout_overlap_filtered and not args.allow_holdout_parent_overlap),
        "selected_low_energy_rows_appended_before_dedupe": int(len(reviewed)),
        "duplicates_removed_after_concat": int(duplicate_removed),
        "final_rows": int(len(combined)),
        "final_low_energy_added_rows": int(combined["v010_1_low_energy_added"].fillna(False).astype(bool).sum()),
        "include_splits": sorted(included_splits),
        "allow_partial_mask": bool(args.allow_partial_mask),
        "allow_holdout_parent_overlap": bool(args.allow_holdout_parent_overlap),
        **feature_rewrite_report,
        "safety_policy": "No source manifest was modified. Only copied/source_copies and metadata output were written.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[OK] v0.10_1 low-energy augmented manifest created")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
