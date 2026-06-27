#!/usr/bin/env python
r"""
Generic repeated label-aware parent aggregation + threshold calibration.

Required CSV columns:
    parent_clip_id
    <label>
    prob_<label>

Example PowerShell:
    python run_labelwise_aggregation_threshold_calibration.py `
      --segment-pred-csv "C:\path\segment_predictions.csv" `
      --out-dir "C:\\path\\v07_labelwise_calibration_out" `
      --fixed-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, hamming_loss


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

METHODS = ["mean", "max", "top2mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-pred-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--prob-prefix", default="prob_")
    return parser.parse_args()


def load_thresholds(labels: List[str], thresholds_json: Path | None, fixed_threshold: float) -> Dict[str, float]:
    if thresholds_json is None:
        return {lab: float(fixed_threshold) for lab in labels}
    data = json.loads(thresholds_json.read_text(encoding="utf-8"))
    missing = [lab for lab in labels if lab not in data]
    if missing:
        raise RuntimeError(f"Threshold JSON is missing labels: {missing}")
    return {lab: float(data[lab]) for lab in labels}


def aggregate_values(vals: np.ndarray, method: str) -> float:
    vals = np.asarray(vals, dtype=float)
    if method == "mean":
        return float(np.mean(vals))
    if method == "max":
        return float(np.max(vals))
    if method == "top2mean":
        top = np.sort(vals)[-min(2, len(vals)):]
        return float(np.mean(top))
    raise ValueError(f"Unknown aggregation method: {method}")


def validate_input(df: pd.DataFrame, labels: List[str], parent_id_col: str, prob_prefix: str) -> None:
    required = [parent_id_col] + labels + [f"{prob_prefix}{lab}" for lab in labels]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Input CSV missing required columns: {missing}")


def clean_input(df: pd.DataFrame, labels: List[str], prob_prefix: str) -> pd.DataFrame:
    df = df.copy()
    for lab in labels:
        df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0).astype(int)
        df[f"{prob_prefix}{lab}"] = pd.to_numeric(df[f"{prob_prefix}{lab}"], errors="coerce").fillna(0.0).astype(float)
    return df


def build_parent_table(df: pd.DataFrame, labels: List[str], parent_id_col: str, prob_prefix: str) -> pd.DataFrame:
    parent_rows = []
    for parent_id, group in df.groupby(parent_id_col, sort=False):
        row = {"parent_clip_id": parent_id, "segments": int(len(group))}
        for lab in labels:
            row[lab] = int(group[lab].max())
            vals = group[f"{prob_prefix}{lab}"].to_numpy(dtype=float)
            for method in METHODS:
                row[f"{method}_prob_{lab}"] = aggregate_values(vals, method)
        parent_rows.append(row)
    return pd.DataFrame(parent_rows)


def select_aggregation_only(cal: pd.DataFrame, labels: List[str], fixed_thresholds: Dict[str, float]) -> Dict[str, str]:
    selected = {}
    for lab in labels:
        y = cal[lab].to_numpy(dtype=int)
        threshold = fixed_thresholds[lab]
        best_method, best_f1 = METHODS[0], -1.0
        for method in METHODS:
            probs = cal[f"{method}_prob_{lab}"].to_numpy(dtype=float)
            pred = (probs >= threshold).astype(int)
            score = f1_score(y, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_method = method
        selected[lab] = best_method
    return selected


def select_aggregation_and_threshold(
    cal: pd.DataFrame,
    labels: List[str],
    fixed_thresholds: Dict[str, float],
    threshold_grid: np.ndarray,
) -> Dict[str, Dict[str, float | str | int]]:
    selected = {}
    for lab in labels:
        y = cal[lab].to_numpy(dtype=int)
        if int(y.sum()) == 0:
            selected[lab] = {
                "method": "mean",
                "threshold": fixed_thresholds[lab],
                "cal_f1": 0.0,
                "cal_precision": 0.0,
                "cal_support": 0,
                "fallback_used": 1,
            }
            continue

        best = None
        for method in METHODS:
            probs = cal[f"{method}_prob_{lab}"].to_numpy(dtype=float)
            for threshold in threshold_grid:
                threshold = float(threshold)
                pred = (probs >= threshold).astype(int)
                precision = precision_score(y, pred, zero_division=0)
                f1 = f1_score(y, pred, zero_division=0)
                item = {
                    "method": method,
                    "threshold": threshold,
                    "cal_f1": float(f1),
                    "cal_precision": float(precision),
                    "cal_support": int(y.sum()),
                    "fallback_used": 0,
                }
                if best is None:
                    best = item
                    continue

                better = False
                if item["cal_f1"] > best["cal_f1"] + 1e-12:
                    better = True
                elif abs(item["cal_f1"] - best["cal_f1"]) <= 1e-12:
                    if item["cal_precision"] > best["cal_precision"] + 1e-12:
                        better = True
                    elif abs(item["cal_precision"] - best["cal_precision"]) <= 1e-12:
                        item_dist = abs(item["threshold"] - fixed_thresholds[lab])
                        best_dist = abs(best["threshold"] - fixed_thresholds[lab])
                        if item_dist < best_dist:
                            better = True
                if better:
                    best = item
        selected[lab] = best
    return selected


def build_probs_fixed_method(frame: pd.DataFrame, labels: List[str], method: str, fixed_thresholds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    thresholds = np.asarray([fixed_thresholds[lab] for lab in labels], dtype=float)
    for _, row in frame.iterrows():
        probs.append([float(row[f"{method}_prob_{lab}"]) for lab in labels])
    return np.asarray(probs, dtype=float), thresholds


def build_probs_from_method_map(frame: pd.DataFrame, labels: List[str], method_map: Dict[str, str], fixed_thresholds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    thresholds = np.asarray([fixed_thresholds[lab] for lab in labels], dtype=float)
    for _, row in frame.iterrows():
        probs.append([float(row[f"{method_map[lab]}_prob_{lab}"]) for lab in labels])
    return np.asarray(probs, dtype=float), thresholds


def build_probs_from_full_selection(frame: pd.DataFrame, labels: List[str], selection: Dict[str, Dict[str, float | str | int]]) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    thresholds = np.asarray([float(selection[lab]["threshold"]) for lab in labels], dtype=float)
    for _, row in frame.iterrows():
        probs.append([float(row[f"{selection[lab]['method']}_prob_{lab}"]) for lab in labels])
    return np.asarray(probs, dtype=float), thresholds


def evaluate(frame: pd.DataFrame, labels: List[str], probs: np.ndarray, thresholds: np.ndarray) -> Dict[str, float | int]:
    y_true = frame[labels].to_numpy(dtype=int)
    pred = (probs >= thresholds.reshape(1, -1)).astype(int)
    return {
        "rows": int(len(frame)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, pred)),
        "avg_predicted_labels": float(pred.sum(axis=1).mean()),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
    }


def add_result(rows: list, seed: int, method: str, metrics: Dict[str, float | int]) -> None:
    item = dict(metrics)
    item["seed"] = seed
    item["method"] = method
    rows.append(item)


def main() -> None:
    args = parse_args()
    labels = list(args.labels)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_thresholds = load_thresholds(labels, args.thresholds_json, args.fixed_threshold)

    df = pd.read_csv(args.segment_pred_csv)
    validate_input(df, labels, args.parent_id_col, args.prob_prefix)
    df = clean_input(df, labels, args.prob_prefix)

    parents = build_parent_table(df, labels, args.parent_id_col, args.prob_prefix)
    parents.to_csv(out_dir / "parent_level_aggregated_probabilities.csv", index=False, encoding="utf-8-sig")

    threshold_grid = np.round(np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step), 4)

    results = []
    v06_selection_rows = []
    v07_selection_rows = []

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(parents))
        rng.shuffle(idx)

        cal_size = int(len(idx) * args.cal_fraction)
        cal = parents.iloc[idx[:cal_size]].reset_index(drop=True)
        ev = parents.iloc[idx[cal_size:]].reset_index(drop=True)

        for method in METHODS:
            probs, thresholds = build_probs_fixed_method(ev, labels, method, fixed_thresholds)
            metrics = evaluate(ev, labels, probs, thresholds)
            add_result(results, seed, f"{method}_fixed_thresholds", metrics)

        v06_map = select_aggregation_only(cal, labels, fixed_thresholds)
        for lab, method in v06_map.items():
            v06_selection_rows.append({"seed": seed, "label": lab, "selected_aggregation": method})

        probs, thresholds = build_probs_from_method_map(ev, labels, v06_map, fixed_thresholds)
        metrics = evaluate(ev, labels, probs, thresholds)
        add_result(results, seed, "v06_calibration_selected_aggregation_fixed_thresholds", metrics)

        v07_selection = select_aggregation_and_threshold(cal, labels, fixed_thresholds, threshold_grid)
        for lab, selected in v07_selection.items():
            v07_selection_rows.append({
                "seed": seed,
                "label": lab,
                "selected_aggregation": selected["method"],
                "selected_threshold": selected["threshold"],
                "cal_f1": selected["cal_f1"],
                "cal_precision": selected["cal_precision"],
                "cal_support": selected["cal_support"],
                "fallback_used": selected["fallback_used"],
            })

        probs, thresholds = build_probs_from_full_selection(ev, labels, v07_selection)
        metrics = evaluate(ev, labels, probs, thresholds)
        add_result(results, seed, "v07_aggregation_threshold_calibrated", metrics)

    results_df = pd.DataFrame(results)
    v06_selections_df = pd.DataFrame(v06_selection_rows)
    v07_selections_df = pd.DataFrame(v07_selection_rows)

    summary_df = (
        results_df.groupby("method")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            micro_f1_mean=("micro_f1", "mean"),
            micro_f1_std=("micro_f1", "std"),
            samples_f1_mean=("samples_f1", "mean"),
            samples_f1_std=("samples_f1", "std"),
            exact_match_mean=("exact_match", "mean"),
            exact_match_std=("exact_match", "std"),
            hamming_loss_mean=("hamming_loss", "mean"),
            hamming_loss_std=("hamming_loss", "std"),
            avg_predicted_labels_mean=("avg_predicted_labels", "mean"),
            avg_true_labels_mean=("avg_true_labels", "mean"),
        )
        .reset_index()
    )

    v06_selection_frequency = (
        v06_selections_df.groupby(["label", "selected_aggregation"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "count"], ascending=[True, False])
    )

    v07_selection_frequency = (
        v07_selections_df.groupby(["label", "selected_aggregation"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "count"], ascending=[True, False])
    )

    threshold_summary = (
        v07_selections_df.groupby("label")
        .agg(
            threshold_mean=("selected_threshold", "mean"),
            threshold_std=("selected_threshold", "std"),
            threshold_min=("selected_threshold", "min"),
            threshold_max=("selected_threshold", "max"),
            support_mean=("cal_support", "mean"),
            fallback_count=("fallback_used", "sum"),
        )
        .reset_index()
    )

    results_df.to_csv(out_dir / "seed_level_eval_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "repeated_eval_summary.csv", index=False, encoding="utf-8-sig")
    v06_selections_df.to_csv(out_dir / "v06_selected_aggregation_by_seed.csv", index=False, encoding="utf-8-sig")
    v06_selection_frequency.to_csv(out_dir / "v06_selection_frequency.csv", index=False, encoding="utf-8-sig")
    v07_selections_df.to_csv(out_dir / "v07_selected_aggregation_threshold_by_seed.csv", index=False, encoding="utf-8-sig")
    v07_selection_frequency.to_csv(out_dir / "v07_selection_frequency.csv", index=False, encoding="utf-8-sig")
    threshold_summary.to_csv(out_dir / "v07_threshold_summary.csv", index=False, encoding="utf-8-sig")

    (out_dir / "experiment_settings.json").write_text(
        json.dumps(
            {
                "segment_pred_csv": str(args.segment_pred_csv),
                "labels": labels,
                "methods": METHODS,
                "fixed_thresholds": fixed_thresholds,
                "seeds": list(range(args.seeds)),
                "cal_fraction": args.cal_fraction,
                "threshold_grid": [float(x) for x in threshold_grid],
                "parent_id_col": args.parent_id_col,
                "prob_prefix": args.prob_prefix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRepeated label-aware aggregation + threshold calibration")
    print("-" * 100)
    print(summary_df.to_string(index=False))

    print("\nV0.6 aggregation selection frequency:")
    print(v06_selection_frequency.to_string(index=False))

    print("\nV0.7 aggregation selection frequency:")
    print(v07_selection_frequency.to_string(index=False))

    print("\nV0.7 threshold summary:")
    print(threshold_summary.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "repeated_eval_summary.csv")
    print(out_dir / "v06_selection_frequency.csv")
    print(out_dir / "v07_selection_frequency.csv")
    print(out_dir / "v07_threshold_summary.csv")


if __name__ == "__main__":
    main()
