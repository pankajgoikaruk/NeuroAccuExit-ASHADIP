# scripts/lawyer_refine_weak_labels_v08.py
#
# LAWYER: Label-Aware Weak-label Yield Estimation and Refinement
#
# Refines weak parent-level labels for:
#   other_speaker_present, audience_reaction_present, silence_present
# using label-specific aggregation and routing rules.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = [
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

TARGET_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
]

FOCUS_LABELS = [
    "other_speaker_present",
    "audience_reaction_present",
    "silence_present",
]

EVENT_LABELS = [
    "music_present",
    "audience_reaction_present",
    "silence_present",
]

NON_TARGET_SOURCE_CLASSES = {
    "Les_Brown",
    "Mel_Robbins",
    "Oprah_Winfrey",
    "Rabin_Sharma",
    "Simon_Sinek",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64, np.integer)):
            return int(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def safe_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).values


def topk_mean(values: np.ndarray, k: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    k = max(1, min(int(k), values.size))
    return float(np.mean(np.sort(values)[-k:]))


def clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def is_non_target_source(row: dict) -> bool:
    text = " ".join(
        str(row.get(c, ""))
        for c in ["source_class_dir", "source_file", "source_path", "source_rel_path", "parent_clip_id"]
    )
    return any(cls in text for cls in NON_TARGET_SOURCE_CLASSES)


def score_to_zone(score: float, low: float, high: float) -> str:
    score = float(score)
    if score < float(low):
        return "reject_zone"
    if score > float(high):
        return "accept_zone"
    return "uncertain_zone"


def active_label_text(row: dict) -> str:
    return "|".join([lab for lab in LABELS if int(row.get(lab, 0)) == 1])


def aggregate_segment_evidence(
    seg_df: pd.DataFrame,
    *,
    speaker_alpha: float,
    audience_top_k: int,
    silence_energy_col: str | None,
    silence_energy_threshold: float,
    silence_vad_col: str | None,
    silence_vad_threshold: float,
) -> pd.DataFrame:
    if "parent_clip_id" not in seg_df.columns:
        raise RuntimeError("segment_predictions CSV must contain parent_clip_id")

    missing_prob_cols = [f"segment_prob_{lab}" for lab in LABELS if f"segment_prob_{lab}" not in seg_df.columns]
    if missing_prob_cols:
        raise RuntimeError(
            "segment_predictions CSV is missing required segment probability columns:\n"
            f"{missing_prob_cols}"
        )

    rows: list[dict] = []
    meta_cols = ["parent_clip_id", "source_file", "source_path", "source_rel_path", "source_class_dir"]

    for parent_id, group in seg_df.groupby("parent_clip_id", dropna=False):
        row: dict = {"parent_clip_id": str(parent_id), "num_segments": int(len(group))}

        for col in meta_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]

        # Generic segment evidence for all labels.
        for lab in LABELS:
            p = safe_float_array(group[f"segment_prob_{lab}"])
            row[f"lawyer_seg_max_{lab}"] = float(np.max(p)) if len(p) else 0.0
            row[f"lawyer_seg_mean_{lab}"] = float(np.mean(p)) if len(p) else 0.0
            row[f"lawyer_seg_top2_{lab}"] = topk_mean(p, 2)

        # 1) Known speaker identity: stable parent-level evidence.
        for lab in TARGET_LABELS:
            mean_p = float(row[f"lawyer_seg_mean_{lab}"])
            max_p = float(row[f"lawyer_seg_max_{lab}"])
            row[f"lawyer_score_{lab}"] = clip01(speaker_alpha * mean_p + (1.0 - speaker_alpha) * max_p)

        target_scores = [float(row[f"lawyer_score_{lab}"]) for lab in TARGET_LABELS]
        target_scores_sorted = sorted(target_scores, reverse=True)
        row["lawyer_target_max_score"] = float(target_scores_sorted[0]) if target_scores_sorted else 0.0
        row["lawyer_target_second_score"] = float(target_scores_sorted[1]) if len(target_scores_sorted) > 1 else 0.0
        row["lawyer_target_margin"] = float(row["lawyer_target_max_score"] - row["lawyer_target_second_score"])

        # 2) Open-set other-speaker evidence.
        other_direct = float(row["lawyer_seg_max_other_speaker_present"])
        speech_like_score = max(other_direct, max(float(row[f"lawyer_seg_max_{lab}"]) for lab in TARGET_LABELS))
        known_max = float(row["lawyer_target_max_score"])
        open_set_score = clip01(speech_like_score * (1.0 - known_max))

        row["lawyer_score_other_speaker_direct"] = other_direct
        row["lawyer_score_speech_like"] = speech_like_score
        row["lawyer_score_other_speaker_open_set"] = open_set_score
        row["lawyer_is_non_target_source"] = int(is_non_target_source(row))
        source_bonus = 1.0 if row["lawyer_is_non_target_source"] == 1 else 0.0
        row["lawyer_score_other_speaker_present"] = max(other_direct, open_set_score, source_bonus)

        # 3) music: keep as max event evidence.
        row["lawyer_score_music_present"] = float(row["lawyer_seg_max_music_present"])

        # 4) audience reactions are short/bursty, so top-k mean is better than full mean.
        audience_p = safe_float_array(group["segment_prob_audience_reaction_present"])
        row["lawyer_score_audience_reaction_present"] = topk_mean(audience_p, audience_top_k)

        # 5) silence: neural max evidence plus optional signal evidence.
        silence_p = safe_float_array(group["segment_prob_silence_present"])
        tata_silence_score = float(np.max(silence_p)) if len(silence_p) else 0.0

        acoustic_silence_score = 0.0
        acoustic_reason = "not_used"

        if silence_energy_col and silence_energy_col in group.columns:
            energy = pd.to_numeric(group[silence_energy_col], errors="coerce")
            energy_silent = energy <= float(silence_energy_threshold)
            acoustic_silence_score = max(acoustic_silence_score, 1.0 if bool(energy_silent.any()) else 0.0)
            acoustic_reason = f"energy_col={silence_energy_col},threshold={silence_energy_threshold}"

        if silence_vad_col and silence_vad_col in group.columns:
            vad = pd.to_numeric(group[silence_vad_col], errors="coerce")
            vad_silent = vad <= float(silence_vad_threshold)
            acoustic_silence_score = max(acoustic_silence_score, 1.0 if bool(vad_silent.any()) else 0.0)
            if acoustic_reason == "not_used":
                acoustic_reason = f"vad_col={silence_vad_col},threshold={silence_vad_threshold}"
            else:
                acoustic_reason += f";vad_col={silence_vad_col},threshold={silence_vad_threshold}"

        row["lawyer_score_silence_tata"] = tata_silence_score
        row["lawyer_score_silence_acoustic"] = acoustic_silence_score
        row["lawyer_silence_acoustic_reason"] = acoustic_reason
        row["lawyer_score_silence_present"] = max(tata_silence_score, acoustic_silence_score)

        rows.append(row)

    return pd.DataFrame(rows)


def attach_parent_context(evidence_df: pd.DataFrame, parent_csv: Path | None) -> pd.DataFrame:
    if parent_csv is None:
        return evidence_df
    if not parent_csv.exists():
        raise FileNotFoundError(f"parent_csv not found: {parent_csv}")

    parent_df = pd.read_csv(parent_csv, low_memory=False)
    if "parent_clip_id" not in parent_df.columns:
        raise RuntimeError("parent_csv must contain parent_clip_id")

    parent_df = parent_df.copy()
    parent_df["parent_clip_id"] = parent_df["parent_clip_id"].astype(str)
    evidence_df = evidence_df.copy()
    evidence_df["parent_clip_id"] = evidence_df["parent_clip_id"].astype(str)

    keep_cols = ["parent_clip_id"] + [c for c in parent_df.columns if c != "parent_clip_id" and c not in evidence_df.columns]
    return evidence_df.merge(parent_df[keep_cols], on="parent_clip_id", how="left")


def apply_lawyer_decisions(
    evidence_df: pd.DataFrame,
    *,
    target_threshold: float,
    target_margin_threshold: float,
    other_direct_threshold: float,
    other_speech_threshold: float,
    other_known_max_threshold: float,
    music_threshold: float,
    audience_threshold: float,
    audience_low: float,
    audience_high: float,
    silence_threshold: float,
    silence_low: float,
    silence_high: float,
) -> pd.DataFrame:
    df = evidence_df.copy()
    rows = []

    for _, row in df.iterrows():
        out = row.to_dict()
        uncertain_reasons: list[str] = []
        routing_reasons: list[str] = []

        target_active = []
        target_scores = {}
        for lab in TARGET_LABELS:
            score = float(out.get(f"lawyer_score_{lab}", 0.0))
            target_scores[lab] = score
            out[lab] = int(score >= float(target_threshold))
            out[f"parent_pred_{lab}"] = int(out[lab])
            if out[lab] == 1:
                target_active.append(lab)

        best_target = max(target_scores, key=target_scores.get)
        best_target_score = float(target_scores[best_target])
        sorted_target_scores = sorted(target_scores.values(), reverse=True)
        second_target_score = float(sorted_target_scores[1]) if len(sorted_target_scores) > 1 else 0.0
        target_margin = best_target_score - second_target_score

        out["lawyer_best_target_label"] = best_target
        out["lawyer_best_target_score"] = best_target_score
        out["lawyer_second_target_score"] = second_target_score
        out["lawyer_target_margin_after_refine"] = target_margin

        if len(target_active) > 1 and target_margin < float(target_margin_threshold):
            uncertain_reasons.append("multi_target_low_margin")

        direct_other = float(out.get("lawyer_score_other_speaker_direct", 0.0))
        speech_like = float(out.get("lawyer_score_speech_like", 0.0))
        known_max = float(out.get("lawyer_target_max_score", best_target_score))
        is_non_target = int(out.get("lawyer_is_non_target_source", 0)) == 1

        other_open_condition = speech_like >= float(other_speech_threshold) and known_max <= float(other_known_max_threshold)
        other_positive = is_non_target or direct_other >= float(other_direct_threshold) or other_open_condition

        out["other_speaker_present"] = int(other_positive)
        out["parent_pred_other_speaker_present"] = int(other_positive)

        if is_non_target:
            routing_reasons.append("non_target_source_forced_other")
        elif other_open_condition:
            routing_reasons.append("open_set_other_rule")
        elif direct_other >= float(other_direct_threshold):
            routing_reasons.append("direct_other_probability_rule")

        music_score = float(out.get("lawyer_score_music_present", 0.0))
        out["music_present"] = int(music_score >= float(music_threshold))
        out["parent_pred_music_present"] = int(out["music_present"])

        audience_score = float(out.get("lawyer_score_audience_reaction_present", 0.0))
        out["audience_reaction_present"] = int(audience_score >= float(audience_threshold))
        out["parent_pred_audience_reaction_present"] = int(out["audience_reaction_present"])
        out["lawyer_zone_audience_reaction_present"] = score_to_zone(audience_score, audience_low, audience_high)
        if out["lawyer_zone_audience_reaction_present"] == "uncertain_zone":
            uncertain_reasons.append("audience_reaction_uncertain")

        silence_score = float(out.get("lawyer_score_silence_present", 0.0))
        out["silence_present"] = int(silence_score >= float(silence_threshold))
        out["parent_pred_silence_present"] = int(out["silence_present"])
        out["lawyer_zone_silence_present"] = score_to_zone(silence_score, silence_low, silence_high)
        if out["lawyer_zone_silence_present"] == "uncertain_zone":
            uncertain_reasons.append("silence_uncertain")

        other_score = float(out.get("lawyer_score_other_speaker_present", 0.0))
        out["lawyer_zone_other_speaker_present"] = score_to_zone(
            other_score,
            min(other_known_max_threshold, 0.35),
            max(other_direct_threshold, 0.60),
        )
        if out["lawyer_zone_other_speaker_present"] == "uncertain_zone" and not is_non_target and not other_open_condition:
            uncertain_reasons.append("other_speaker_uncertain")

        num_active = int(sum(int(out.get(lab, 0)) for lab in LABELS))
        target_count = int(sum(int(out.get(lab, 0)) for lab in TARGET_LABELS))
        event_count = int(sum(int(out.get(lab, 0)) for lab in EVENT_LABELS))
        other_active = int(out.get("other_speaker_present", 0)) == 1

        out["num_active_labels"] = num_active
        out["labels"] = active_label_text(out)
        out["lawyer_uncertain_reasons"] = "|".join(sorted(set(uncertain_reasons)))
        out["lawyer_routing_reason_detail"] = "|".join(sorted(set(routing_reasons)))

        # Important v0.8 change: other-only high-confidence rows are kept as warning examples,
        # not automatically rejected, to improve other_speaker_present learning.
        if uncertain_reasons:
            out["routing_decision"] = "needs_review"
            out["routing_reason"] = "lawyer_uncertain_" + "|".join(sorted(set(uncertain_reasons)))
        elif num_active == 0:
            out["routing_decision"] = "rejected"
            out["routing_reason"] = "lawyer_no_reliable_label"
        elif target_count == 1 and not other_active and event_count == 0:
            out["routing_decision"] = "accepted"
            out["routing_reason"] = "lawyer_single_target_clean"
        else:
            out["routing_decision"] = "accepted_with_warning"
            if other_active and target_count == 0:
                out["routing_reason"] = "lawyer_other_only_high_confidence"
            elif event_count > 0:
                out["routing_reason"] = "lawyer_target_or_other_with_context_event"
            else:
                out["routing_reason"] = "lawyer_multi_label_warning"

        out["routing_mode"] = "lawyer_v08"
        out["review_status"] = "lawyer_pseudo_routed"
        out["manual_labels"] = out["labels"]
        if "notes" not in out or pd.isna(out.get("notes")):
            out["notes"] = ""

        rows.append(out)

    return pd.DataFrame(rows)


def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [
        "parent_clip_id", "source_file", "source_path", "source_rel_path", "source_class_dir",
        "routing_mode", "routing_decision", "routing_reason", "labels",
        *LABELS,
        "manual_labels", "review_status", "notes", "num_segments", "num_active_labels",
        "lawyer_best_target_label", "lawyer_best_target_score", "lawyer_second_target_score",
        "lawyer_target_margin_after_refine", "lawyer_is_non_target_source",
        "lawyer_uncertain_reasons", "lawyer_routing_reason_detail",
    ]

    score_cols = []
    for lab in LABELS:
        score_cols.extend([
            f"lawyer_score_{lab}", f"lawyer_seg_max_{lab}", f"lawyer_seg_mean_{lab}",
            f"lawyer_seg_top2_{lab}", f"parent_pred_{lab}",
        ])

    zone_cols = [
        "lawyer_zone_other_speaker_present", "lawyer_zone_audience_reaction_present",
        "lawyer_zone_silence_present", "lawyer_score_other_speaker_direct",
        "lawyer_score_other_speaker_open_set", "lawyer_score_speech_like",
        "lawyer_score_silence_tata", "lawyer_score_silence_acoustic",
        "lawyer_silence_acoustic_reason",
    ]

    ordered = [c for c in front + score_cols + zone_cols if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def write_summary_md(path: Path, summary: dict) -> None:
    lines = [
        "# LAWYER v0.8 Weak-label Refinement Summary",
        "",
        "LAWYER = **Label-Aware Weak-label Yield Estimation and Refinement**.",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Core idea",
        "",
        "LAWYER applies different weak-label rules for different label types:",
        "",
        "| Label type | Rule |",
        "|---|---|",
        "| Known speaker identity | `score = alpha * mean(segment_prob) + (1-alpha) * max(segment_prob)` |",
        "| Open-set other speaker | direct other evidence OR speech-like evidence with low known-speaker confidence |",
        "| Audience reaction | top-k mean over segment probabilities for bursty/transient reactions |",
        "| Silence | max TATA silence score plus optional acoustic silence evidence |",
        "",
        "## Counts",
        "",
        f"- Segment rows: `{summary['segment_rows']}`",
        f"- Parent clips: `{summary['parent_rows']}`",
        "",
        "### Routing counts",
        "",
        "| Decision | Rows |",
        "|---|---:|",
    ]
    for key, value in summary["routing_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "### Label counts after LAWYER", "", "| Label | Positive parent clips |", "|---|---:|"])
    for lab, count in summary["label_counts"].items():
        lines.append(f"| `{lab}` | {count} |")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "", "## Paper interpretation", "",
        "Compare `v0.6` original full pipeline, `v0.7` filtered ablation, and `v0.8` LAWYER full weak-label repair.",
        "The main per-label focus should be `other_speaker_present`, `audience_reaction_present`, and `silence_present`.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(df: pd.DataFrame, out_dir: Path, mode_name: str, summary_extra: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = reorder_output_columns(df)

    paths = {
        "all_parent_labels": out_dir / f"{mode_name}_parent_labels_all.csv",
        "accepted": out_dir / f"{mode_name}_accepted.csv",
        "accepted_with_warning": out_dir / f"{mode_name}_accepted_with_warning.csv",
        "needs_review": out_dir / f"{mode_name}_needs_review.csv",
        "rejected": out_dir / f"{mode_name}_rejected.csv",
        "manual_review_prefill": out_dir / f"{mode_name}_manual_review_prefill.csv",
        "summary_json": out_dir / f"{mode_name}_summary.json",
        "summary_md": out_dir / f"{mode_name}_summary.md",
    }

    df.to_csv(paths["all_parent_labels"], index=False)
    df[df["routing_decision"] == "accepted"].to_csv(paths["accepted"], index=False)
    df[df["routing_decision"] == "accepted_with_warning"].to_csv(paths["accepted_with_warning"], index=False)
    df[df["routing_decision"] == "needs_review"].to_csv(paths["needs_review"], index=False)
    df[df["routing_decision"] == "rejected"].to_csv(paths["rejected"], index=False)
    df[df["routing_decision"] == "needs_review"].to_csv(paths["manual_review_prefill"], index=False)

    outputs = {key: str(value) for key, value in paths.items()}
    summary = {
        "generated_at": now_iso(),
        **summary_extra,
        "parent_rows": int(len(df)),
        "routing_counts": {k: int(v) for k, v in df["routing_decision"].value_counts().to_dict().items()},
        "label_counts": {lab: int(pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int).sum()) for lab in LABELS},
        "focus_label_counts": {lab: int(pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int).sum()) for lab in FOCUS_LABELS},
        "outputs": outputs,
    }

    save_json(summary, paths["summary_json"])
    write_summary_md(paths["summary_md"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LAWYER v0.8 label-aware weak-label refinement.")
    parser.add_argument("--segment_predictions_csv", required=True)
    parser.add_argument("--parent_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode_name", default="lawyer_v08")
    parser.add_argument("--speaker_alpha", type=float, default=0.70)
    parser.add_argument("--target_threshold", type=float, default=0.50)
    parser.add_argument("--target_margin_threshold", type=float, default=0.10)
    parser.add_argument("--other_direct_threshold", type=float, default=0.55)
    parser.add_argument("--other_speech_threshold", type=float, default=0.55)
    parser.add_argument("--other_known_max_threshold", type=float, default=0.35)
    parser.add_argument("--music_threshold", type=float, default=0.50)
    parser.add_argument("--audience_top_k", type=int, default=2)
    parser.add_argument("--audience_threshold", type=float, default=0.50)
    parser.add_argument("--audience_low", type=float, default=0.35)
    parser.add_argument("--audience_high", type=float, default=0.65)
    parser.add_argument("--silence_threshold", type=float, default=0.50)
    parser.add_argument("--silence_low", type=float, default=0.35)
    parser.add_argument("--silence_high", type=float, default=0.65)
    parser.add_argument("--silence_energy_col", default=None)
    parser.add_argument("--silence_energy_threshold", type=float, default=-45.0)
    parser.add_argument("--silence_vad_col", default=None)
    parser.add_argument("--silence_vad_threshold", type=float, default=0.15)
    args = parser.parse_args()

    segment_predictions_csv = Path(args.segment_predictions_csv)
    parent_csv = Path(args.parent_csv) if args.parent_csv else None
    out_dir = Path(args.out_dir)

    if not segment_predictions_csv.exists():
        raise FileNotFoundError(f"segment_predictions_csv not found: {segment_predictions_csv}")

    print("\nLAWYER v0.8 weak-label refinement")
    print("-" * 90)
    print(f"Segment predictions: {segment_predictions_csv}")
    print(f"Parent context CSV:   {parent_csv if parent_csv else '(not provided)'}")
    print(f"Output dir:           {out_dir}")
    print(f"Mode name:            {args.mode_name}")
    print("-" * 90)

    seg_df = pd.read_csv(segment_predictions_csv, low_memory=False)
    evidence = aggregate_segment_evidence(
        seg_df,
        speaker_alpha=float(args.speaker_alpha),
        audience_top_k=int(args.audience_top_k),
        silence_energy_col=args.silence_energy_col,
        silence_energy_threshold=float(args.silence_energy_threshold),
        silence_vad_col=args.silence_vad_col,
        silence_vad_threshold=float(args.silence_vad_threshold),
    )
    evidence = attach_parent_context(evidence, parent_csv)
    refined = apply_lawyer_decisions(
        evidence,
        target_threshold=float(args.target_threshold),
        target_margin_threshold=float(args.target_margin_threshold),
        other_direct_threshold=float(args.other_direct_threshold),
        other_speech_threshold=float(args.other_speech_threshold),
        other_known_max_threshold=float(args.other_known_max_threshold),
        music_threshold=float(args.music_threshold),
        audience_threshold=float(args.audience_threshold),
        audience_low=float(args.audience_low),
        audience_high=float(args.audience_high),
        silence_threshold=float(args.silence_threshold),
        silence_low=float(args.silence_low),
        silence_high=float(args.silence_high),
    )

    summary_extra = {
        "method": "LAWYER: Label-Aware Weak-label Yield Estimation and Refinement",
        "segment_predictions_csv": str(segment_predictions_csv),
        "parent_csv": str(parent_csv) if parent_csv else None,
        "out_dir": str(out_dir),
        "mode_name": str(args.mode_name),
        "segment_rows": int(len(seg_df)),
        "parameters": vars(args),
        "important_rule": "Use accepted + accepted_with_warning automatically. Manually correct needs_review before final manifest construction for strict HITL reporting.",
    }

    summary = write_outputs(refined, out_dir, str(args.mode_name), summary_extra)

    print("\nLAWYER refinement complete")
    print("-" * 90)
    print(f"Parent rows: {summary['parent_rows']}")
    print("\nRouting counts:")
    print(pd.Series(summary["routing_counts"]).to_string())
    print("\nFocus label counts:")
    print(pd.Series(summary["focus_label_counts"]).to_string())
    print(f"\nSummary: {summary['outputs']['summary_md']}")


if __name__ == "__main__":
    main()
