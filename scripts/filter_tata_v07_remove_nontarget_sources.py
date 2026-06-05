# scripts\filter_tata_v07_remove_nontarget_sources.py

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


EXCLUDE_CLASSES = {
    "Les_Brown",
    "Mel_Robbins",
    "Oprah_Winfrey",
    "Rabin_Sharma",
    "Simon_Sinek",
}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def is_excluded_row(row) -> bool:
    text = " ".join(
        str(row.get(c, ""))
        for c in ["source_class_dir", "source_file", "source_path", "source_rel_path", "parent_clip_id"]
    )
    return any(cls in text for cls in EXCLUDE_CLASSES)


def filter_csv(src: Path, dst: Path, name: str) -> dict:
    df = pd.read_csv(src)
    mask = df.apply(is_excluded_row, axis=1)

    kept = df[~mask].copy()
    removed = df[mask].copy()

    dst.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(dst, index=False)

    removed_path = dst.with_name(dst.stem + "_REMOVED_NON_TARGET_ROWS.csv")
    removed.to_csv(removed_path, index=False)

    return {
        "name": name,
        "src": str(src),
        "dst": str(dst),
        "removed_rows_csv": str(removed_path),
        "original_rows": int(len(df)),
        "kept_rows": int(len(kept)),
        "removed_rows": int(len(removed)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v06_root", default="human_talk_workspace/tata_v0.6_raw_pipeline")
    parser.add_argument("--v07_root", default="human_talk_workspace/tata_v0.7_raw_pipeline")
    args = parser.parse_args()

    v06 = Path(args.v06_root)
    v07 = Path(args.v07_root)

    summary = {
        "generated_at": now_iso(),
        "v06_root": str(v06),
        "v07_root": str(v07),
        "excluded_source_classes": sorted(EXCLUDE_CLASSES),
        "outputs": [],
        "note": "v0.7 filtered experiment removes Les/Mel/Oprah/Rabin/Simon rows from manifests only; original audio is untouched.",
    }

    # Create folder structure
    (v07 / "manual_review_queue").mkdir(parents=True, exist_ok=True)
    (v07 / "raw_tata_pseudo_routing" / "hybrid").mkdir(parents=True, exist_ok=True)
    (v07 / "metadata").mkdir(parents=True, exist_ok=True)

    files = [
        (
            v06 / "manual_review_queue" / "01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv",
            v07 / "manual_review_queue" / "01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed_FILTERED_TARGET_ONLY.csv",
            "final_holdout_ground_truth",
        ),
        (
            v06 / "manual_review_queue" / "02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv",
            v07 / "manual_review_queue" / "02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed_FILTERED_TARGET_ONLY.csv",
            "corrected_hybrid_needs_review",
        ),
        (
            v06 / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted.csv",
            v07 / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted_FILTERED_TARGET_ONLY.csv",
            "hybrid_accepted",
        ),
        (
            v06 / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted_with_warning.csv",
            v07 / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_accepted_with_warning_FILTERED_TARGET_ONLY.csv",
            "hybrid_accepted_with_warning",
        ),
        (
            v06 / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_needs_review.csv",
            v07 / "raw_tata_pseudo_routing" / "hybrid" / "hybrid_needs_review_FILTERED_TARGET_ONLY.csv",
            "original_hybrid_needs_review",
        ),
        (
            v06 / "metadata" / "raw_pseudo_pool_parent_manifest.csv",
            v07 / "metadata" / "raw_pseudo_pool_parent_manifest_FILTERED_TARGET_ONLY.csv",
            "raw_pseudo_pool_parent_manifest",
        ),
        (
            v06 / "metadata" / "raw_final_holdout_parent_manifest.csv",
            v07 / "metadata" / "raw_final_holdout_parent_manifest_FILTERED_TARGET_ONLY.csv",
            "raw_final_holdout_parent_manifest",
        ),
    ]

    for src, dst, name in files:
        if src.exists():
            summary["outputs"].append(filter_csv(src, dst, name))
        else:
            summary["outputs"].append({
                "name": name,
                "src": str(src),
                "status": "missing",
            })

    # Copy label json from v0.6 final dataset if exists
    src_labels = v06 / "final_expanded_training_dataset" / "metadata" / "tata_v06_labels.json"
    dst_labels = v07 / "metadata" / "tata_v07_labels.json"
    if src_labels.exists():
        shutil.copy2(src_labels, dst_labels)

    summary_path = v07 / "metadata" / "v07_filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# TATA v0.7 Filter Summary")
    md.append("")
    md.append(f"Generated: `{summary['generated_at']}`")
    md.append("")
    md.append("## Removed source classes")
    md.append("")
    for cls in sorted(EXCLUDE_CLASSES):
        md.append(f"- `{cls}`")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append("| File | Original | Kept | Removed |")
    md.append("|---|---:|---:|---:|")
    for item in summary["outputs"]:
        if item.get("status") == "missing":
            md.append(f"| `{item['name']}` | missing | missing | missing |")
        else:
            md.append(f"| `{item['name']}` | {item['original_rows']} | {item['kept_rows']} | {item['removed_rows']} |")
    md.append("")
    md.append("Original v0.6 files and audio were not deleted.")

    (v07 / "metadata" / "v07_filter_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("v0.7 filtered manifests created")
    print("-" * 90)
    print(f"Output root: {v07}")
    print(f"Summary:     {summary_path}")
    for item in summary["outputs"]:
        if item.get("status") == "missing":
            print(f"{item['name']}: missing")
        else:
            print(f"{item['name']}: original={item['original_rows']} kept={item['kept_rows']} removed={item['removed_rows']}")


if __name__ == "__main__":
    main()