#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
LATS-v0.9: Label-wise Aggregation and Threshold Search
=======================================================

Purpose
-------
Searches the best parent-level aggregation method and decision threshold for
EACH label using frozen segment-level model probabilities. This is an
inference-time/post-hoc optimisation script; it does NOT retrain the model.

Typical use in NeuroAccuExit-ASHADIP / LABLEX v0.9:
  - Input: segment-level probability CSV produced by the frozen v0.8-HCB model
  - Search: label-wise aggregation method + threshold
  - Validation: repeated calibration/evaluation splits
  - Output: stable frozen label-wise config + full-holdout evaluation

Recommended command
-------------------
python scripts\v0.9\run_lats_labelwise_aggregation_threshold_search_v09.py `
  --segment-pred-csv "human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --out-dir "human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v09_search" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --seeds 20 `
  --cal-fraction 0.5 `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.01 `
  --aggregation-methods "mean,max,top2mean,top3mean" `
  --model-name "main_v08_human_corrected_balanced_3exit_20260610_084027"

Outputs
-------
  lats_repeated_eval_summary.csv
  lats_seed_selected_configs_long.csv
  lats_candidate_search_log.csv              (optional; default enabled)
  lats_label_selection_frequency.csv
  lats_threshold_summary.csv
  lats_final_frozen_config.json
  lats_final_full_holdout_eval.csv
  lats_final_full_holdout_per_label.csv
  lats_final_full_holdout_parent_predictions.csv
  lats_run_config.json

Notes
-----
The repeated split metrics are the safer estimate of generalisation because
settings are selected on calibration parents and evaluated on held-out parents.
The final full-holdout evaluation is useful for comparison/reporting, but it is
still derived from settings learned from repeated splits over this holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score, jaccard_score
except Exception:  # pragma: no cover - fallback only
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

# Conservative tie-break preference. Lower rank is preferred when scores tie.
METHOD_PREFERENCE = {
    "mean": 0,
    "top2mean": 1,
    "top3mean": 2,
    "median": 3,
    "p75": 4,
    "p90": 5,
    "max": 6,
    "top2max": 7,
    "top3max": 8,
}


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    support: int
    predicted_positive: int
    hamming_errors: int


@dataclass(frozen=True)
class SelectedRule:
    label: str
    aggregation: str
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    support: int
    predicted_positive: int
    hamming_errors: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LATS-v0.9: label-wise aggregation and threshold search."
    )

    parser.add_argument("--segment-pred-csv", required=True, help="Segment-level probability CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--labels-json", default=None, help="JSON file containing label list.")
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional comma-separated label list. Overrides --labels-json if supplied.",
    )
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--prob-prefix", default="exit3_prob_")
    parser.add_argument("--model-name", default="")

    parser.add_argument("--seeds", type=int, default=20, help="Number of random splits/seeds.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed value.")
    parser.add_argument("--cal-fraction", type=float, default=0.5, help="Fraction of parents used for calibration.")

    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--aggregation-methods",
        default="mean,max,top2mean,top3mean",
        help="Comma-separated aggregation methods, e.g. mean,max,top2mean,top3mean,p75,p90.",
    )

    parser.add_argument(
        "--selection-metric",
        default="f1",
        choices=["f1"],
        help="Metric used for per-label calibration search. Currently only F1.",
    )
    parser.add_argument(
        "--save-candidate-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full threshold/method search log.",
    )
    parser.add_argument(
        "--round-thresholds-to-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round final median thresholds back to the nearest searched grid value.",
    )

    return parser.parse_args()


def load_labels(labels_json: Optional[str], labels_arg: Optional[str]) -> List[str]:
    if labels_arg:
        labels = [x.strip() for x in labels_arg.split(",") if x.strip()]
        if not labels:
            raise RuntimeError("--labels was provided but no labels were parsed.")
        return labels

    if labels_json:
        path = Path(labels_json)
        if not path.exists():
            raise FileNotFoundError(f"Labels JSON not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        labels = extract_labels_from_json(obj)
        if labels:
            return labels

        raise RuntimeError(
            f"Could not parse labels from {path}. Expected a list or a dict containing "
            "one of: labels, label_names, classes, class_names."
        )

    return DEFAULT_LABELS.copy()


def extract_labels_from_json(obj) -> List[str]:
    """Accept several common schema shapes."""
    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            return list(obj)
        if all(isinstance(x, dict) for x in obj):
            out = []
            for item in obj:
                for key in ("name", "label", "class_name", "id"):
                    if key in item and isinstance(item[key], str):
                        out.append(item[key])
                        break
            return out

    if isinstance(obj, dict):
        for key in ("labels", "label_names", "classes", "class_names"):
            if key in obj:
                return extract_labels_from_json(obj[key])

        # Some schemas are {"0": "label_a", "1": "label_b", ...}
        if obj and all(str(k).isdigit() for k in obj.keys()) and all(isinstance(v, str) for v in obj.values()):
            return [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x))]

    return []


def parse_methods(methods: str) -> List[str]:
    out = [x.strip() for x in methods.split(",") if x.strip()]
    if not out:
        raise RuntimeError("No aggregation methods were provided.")

    # Validate by calling parser on dummy values.
    dummy = np.array([0.1, 0.2, 0.3], dtype=float)
    for method in out:
        _ = aggregate_values(dummy, method)
    return out


def make_threshold_grid(t_min: float, t_max: float, step: float) -> np.ndarray:
    if not (0 <= t_min <= 1 and 0 <= t_max <= 1):
        raise RuntimeError("Threshold min/max must be between 0 and 1.")
    if t_min > t_max:
        raise RuntimeError("threshold-min must be <= threshold-max.")
    if step <= 0:
        raise RuntimeError("threshold-step must be positive.")

    decimals = max(0, int(math.ceil(-math.log10(step))) + 2) if step < 1 else 2
    vals = np.arange(t_min, t_max + (step / 2.0), step)
    vals = np.round(vals, decimals)
    vals = vals[(vals >= t_min - 1e-12) & (vals <= t_max + 1e-12)]
    return vals.astype(float)


def find_prob_col(df: pd.DataFrame, label: str, prob_prefix: str) -> str:
    candidates = [
        f"{prob_prefix}{label}",
        f"prob_{label}",
        f"p_{label}",
        f"score_{label}",
        f"y_score_{label}",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(
        f"Could not find probability column for label '{label}'. Tried: {candidates}"
    )


def find_truth_col(df: pd.DataFrame, label: str) -> str:
    candidates = [
        label,
        f"true_{label}",
        f"y_true_{label}",
        f"target_{label}",
        f"label_{label}",
        f"{label}_true",
        f"{label}_label",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(
        f"Could not find ground-truth column for label '{label}'. Tried: {candidates}"
    )


def validate_input_columns(
    df: pd.DataFrame, labels: Sequence[str], parent_id_col: str, prob_prefix: str
) -> Tuple[Dict[str, str], Dict[str, str]]:
    if parent_id_col not in df.columns:
        raise RuntimeError(f"Parent ID column not found: {parent_id_col}")

    prob_cols = {}
    truth_cols = {}
    for label in labels:
        prob_cols[label] = find_prob_col(df, label, prob_prefix)
        truth_cols[label] = find_truth_col(df, label)
    return prob_cols, truth_cols


def aggregate_values(values: np.ndarray, method: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")

    method = method.strip().lower()

    if method == "mean":
        return float(np.mean(values))
    if method == "max":
        return float(np.max(values))
    if method == "min":
        return float(np.min(values))
    if method == "median":
        return float(np.median(values))

    # topKmean, e.g. top2mean/top3mean
    m = re.fullmatch(r"top(\d+)mean", method)
    if m:
        k = max(1, int(m.group(1)))
        top = np.sort(values)[::-1][: min(k, values.size)]
        return float(np.mean(top))

    # topKmax is effectively max of top K; included only for experiment naming compatibility.
    m = re.fullmatch(r"top(\d+)max", method)
    if m:
        return float(np.max(values))

    # p75 / p90 percentile support
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)", method)
    if m:
        p = float(m.group(1))
        if not 0 <= p <= 100:
            raise RuntimeError(f"Invalid percentile aggregation method: {method}")
        return float(np.percentile(values, p))

    raise RuntimeError(
        f"Unsupported aggregation method '{method}'. Supported examples: "
        "mean, max, median, top2mean, top3mean, top2max, p75, p90."
    )


def build_parent_truth(
    df: pd.DataFrame,
    labels: Sequence[str],
    parent_id_col: str,
    truth_cols: Dict[str, str],
) -> pd.DataFrame:
    truth = pd.DataFrame(index=sorted(df[parent_id_col].dropna().unique()))
    truth.index.name = parent_id_col

    grouped = df.groupby(parent_id_col, sort=True)
    for label in labels:
        s = grouped[truth_cols[label]].max()
        truth[label] = s.astype(int)

    truth = truth.reset_index()
    return truth


def build_parent_scores(
    df: pd.DataFrame,
    labels: Sequence[str],
    parent_id_col: str,
    prob_cols: Dict[str, str],
    methods: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Returns:
        scores_by_method[method] = DataFrame indexed by parent id, columns labels.
    """
    parent_ids = sorted(df[parent_id_col].dropna().unique())
    scores_by_method: Dict[str, pd.DataFrame] = {}

    grouped = df.groupby(parent_id_col, sort=True)

    for method in methods:
        score_df = pd.DataFrame(index=parent_ids)
        score_df.index.name = parent_id_col
        for label in labels:
            prob_col = prob_cols[label]
            if method == "mean":
                s = grouped[prob_col].mean()
            elif method == "max":
                s = grouped[prob_col].max()
            elif method == "median":
                s = grouped[prob_col].median()
            else:
                s = grouped[prob_col].apply(lambda x, m=method: aggregate_values(x.to_numpy(), m))
            score_df[label] = s.astype(float)
        scores_by_method[method] = score_df

    return scores_by_method


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return BinaryMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        support=int(np.sum(y_true == 1)),
        predicted_positive=int(np.sum(y_pred == 1)),
        hamming_errors=int(fp + fn),
    )


def method_rank(method: str) -> int:
    return METHOD_PREFERENCE.get(method, 100)


def candidate_is_better(candidate: dict, best: Optional[dict]) -> bool:
    if best is None:
        return True

    # Higher F1 is primary.
    eps = 1e-12
    if candidate["f1"] > best["f1"] + eps:
        return True
    if candidate["f1"] < best["f1"] - eps:
        return False

    # Lower binary Hamming errors is safer.
    if candidate["hamming_errors"] < best["hamming_errors"]:
        return True
    if candidate["hamming_errors"] > best["hamming_errors"]:
        return False

    # Fewer false positives helps avoid over-prediction.
    if candidate["fp"] < best["fp"]:
        return True
    if candidate["fp"] > best["fp"]:
        return False

    # Prefer threshold close to 0.5 to avoid extreme/unstable calibration.
    cand_dist = abs(candidate["threshold"] - 0.5)
    best_dist = abs(best["threshold"] - 0.5)
    if cand_dist < best_dist - eps:
        return True
    if cand_dist > best_dist + eps:
        return False

    # Prefer simpler/conservative aggregation.
    if method_rank(candidate["aggregation"]) < method_rank(best["aggregation"]):
        return True

    return False


def search_best_rule_for_label(
    label: str,
    y_true_cal: np.ndarray,
    score_by_method_cal: Dict[str, np.ndarray],
    thresholds: np.ndarray,
    methods: Sequence[str],
    seed: int,
    save_candidates: bool,
) -> Tuple[SelectedRule, List[dict]]:
    best: Optional[dict] = None
    candidate_rows: List[dict] = []

    for method in methods:
        y_score = np.asarray(score_by_method_cal[method], dtype=float)
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            bm = binary_metrics(y_true_cal, y_pred)
            row = {
                "seed": seed,
                "label": label,
                "aggregation": method,
                "threshold": float(threshold),
                "precision": bm.precision,
                "recall": bm.recall,
                "f1": bm.f1,
                "tp": bm.tp,
                "fp": bm.fp,
                "fn": bm.fn,
                "tn": bm.tn,
                "support": bm.support,
                "predicted_positive": bm.predicted_positive,
                "hamming_errors": bm.hamming_errors,
            }
            if candidate_is_better(row, best):
                best = row
            if save_candidates:
                candidate_rows.append(row)

    assert best is not None

    selected = SelectedRule(
        label=label,
        aggregation=best["aggregation"],
        threshold=float(best["threshold"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        f1=float(best["f1"]),
        tp=int(best["tp"]),
        fp=int(best["fp"]),
        fn=int(best["fn"]),
        tn=int(best["tn"]),
        support=int(best["support"]),
        predicted_positive=int(best["predicted_positive"]),
        hamming_errors=int(best["hamming_errors"]),
    )

    if save_candidates:
        # Mark selected row(s). Only the first exact selected candidate becomes True.
        selected_marked = False
        for row in candidate_rows:
            is_sel = (
                not selected_marked
                and row["aggregation"] == selected.aggregation
                and abs(row["threshold"] - selected.threshold) < 1e-12
                and row["label"] == selected.label
            )
            row["is_selected"] = bool(is_sel)
            if is_sel:
                selected_marked = True

    return selected, candidate_rows


def compute_multilabel_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1))) if y_true.size else 0.0
    hamming = float(np.mean(y_true != y_pred)) if y_true.size else 0.0
    avg_true_labels = float(np.mean(np.sum(y_true, axis=1))) if y_true.size else 0.0
    avg_pred_labels = float(np.mean(np.sum(y_pred, axis=1))) if y_pred.size else 0.0

    if f1_score is not None:
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        samples = float(f1_score(y_true, y_pred, average="samples", zero_division=0))
    else:  # fallback
        per_f1 = []
        for j in range(y_true.shape[1]):
            per_f1.append(binary_metrics(y_true[:, j], y_pred[:, j]).f1)
        macro = float(np.mean(per_f1)) if per_f1 else 0.0
        bm_micro = binary_metrics(y_true.ravel(), y_pred.ravel())
        micro = bm_micro.f1
        samples_vals = []
        for i in range(y_true.shape[0]):
            inter = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
            denom = np.sum(y_true[i]) + np.sum(y_pred[i])
            samples_vals.append(2 * inter / denom if denom > 0 else 0.0)
        samples = float(np.mean(samples_vals)) if samples_vals else 0.0

    if jaccard_score is not None:
        jaccard = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    else:
        vals = []
        for i in range(y_true.shape[0]):
            inter = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
            union = np.sum((y_true[i] == 1) | (y_pred[i] == 1))
            vals.append(inter / union if union > 0 else 0.0)
        jaccard = float(np.mean(vals)) if vals else 0.0

    out = {
        "macro_f1": macro,
        "micro_f1": micro,
        "samples_f1": samples,
        "exact_match": exact_match,
        "hamming_loss": hamming,
        "jaccard": jaccard,
        "avg_true_labels": avg_true_labels,
        "avg_pred_labels": avg_pred_labels,
        "n_parent_clips": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]) if y_true.ndim == 2 else len(labels),
    }
    return out


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    aggregation_by_label: Optional[Dict[str, str]] = None,
    threshold_by_label: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(labels):
        bm = binary_metrics(y_true[:, j], y_pred[:, j])
        rows.append(
            {
                "label": label,
                "aggregation": aggregation_by_label.get(label, "") if aggregation_by_label else "",
                "threshold": threshold_by_label.get(label, np.nan) if threshold_by_label else np.nan,
                "precision": bm.precision,
                "recall": bm.recall,
                "f1": bm.f1,
                "support": bm.support,
                "predicted_positive": bm.predicted_positive,
                "tp": bm.tp,
                "fp": bm.fp,
                "fn": bm.fn,
                "tn": bm.tn,
                "hamming_errors": bm.hamming_errors,
            }
        )
    return pd.DataFrame(rows)


def split_parent_ids(parent_ids: Sequence, cal_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < cal_fraction < 1:
        raise RuntimeError("--cal-fraction must be > 0 and < 1.")
    parent_ids = np.asarray(list(parent_ids))
    rng = np.random.default_rng(seed)
    shuffled = parent_ids.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    cal_n = int(round(n * cal_fraction))
    cal_n = max(1, min(n - 1, cal_n))
    return shuffled[:cal_n], shuffled[cal_n:]


def apply_rules(
    parent_index: Sequence,
    labels: Sequence[str],
    scores_by_method: Dict[str, pd.DataFrame],
    rules: Dict[str, SelectedRule],
) -> Tuple[np.ndarray, pd.DataFrame]:
    parent_index = list(parent_index)
    y_pred = np.zeros((len(parent_index), len(labels)), dtype=int)
    prob_df = pd.DataFrame({"parent_clip_id": parent_index})

    for j, label in enumerate(labels):
        rule = rules[label]
        scores = scores_by_method[rule.aggregation].loc[parent_index, label].to_numpy(dtype=float)
        y_pred[:, j] = (scores >= rule.threshold).astype(int)
        prob_df[f"score_{label}"] = scores
        prob_df[f"pred_{label}"] = y_pred[:, j]
        prob_df[f"aggregation_{label}"] = rule.aggregation
        prob_df[f"threshold_{label}"] = rule.threshold

    return y_pred, prob_df


def nearest_grid_value(value: float, grid: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(grid - value)))
    return float(grid[idx])


def build_final_frozen_config(
    selected_df: pd.DataFrame,
    labels: Sequence[str],
    thresholds: np.ndarray,
    round_to_grid: bool,
) -> Dict[str, dict]:
    final = {}

    for label in labels:
        sub = selected_df[selected_df["label"] == label].copy()
        if sub.empty:
            raise RuntimeError(f"No selected rows found for label: {label}")

        counts = Counter(sub["aggregation"].astype(str).tolist())
        # Highest count, then conservative method preference.
        sorted_methods = sorted(
            counts.keys(),
            key=lambda m: (-counts[m], method_rank(m), m),
        )
        final_method = sorted_methods[0]

        threshold_pool = sub[sub["aggregation"] == final_method]["threshold"].astype(float).to_numpy()
        if threshold_pool.size == 0:
            threshold_pool = sub["threshold"].astype(float).to_numpy()
        median_t = float(np.median(threshold_pool))
        final_t = nearest_grid_value(median_t, thresholds) if round_to_grid else median_t

        final[label] = {
            "aggregation": final_method,
            "threshold": float(final_t),
            "selection_count": int(counts[final_method]),
            "selection_fraction": float(counts[final_method] / len(sub)),
            "threshold_median_selected_method": float(median_t),
            "threshold_mean_selected_method": float(np.mean(threshold_pool)),
            "threshold_std_selected_method": float(np.std(threshold_pool, ddof=1)) if threshold_pool.size > 1 else 0.0,
            "all_method_counts": dict(counts),
        }

    return final


def selected_rules_from_final_config(final_config: Dict[str, dict], labels: Sequence[str]) -> Dict[str, SelectedRule]:
    rules = {}
    for label in labels:
        cfg = final_config[label]
        rules[label] = SelectedRule(
            label=label,
            aggregation=str(cfg["aggregation"]),
            threshold=float(cfg["threshold"]),
            precision=float("nan"),
            recall=float("nan"),
            f1=float("nan"),
            tp=0,
            fp=0,
            fn=0,
            tn=0,
            support=0,
            predicted_positive=0,
            hamming_errors=0,
        )
    return rules


def main() -> None:
    args = parse_args()

    segment_csv = Path(args.segment_pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not segment_csv.exists():
        raise FileNotFoundError(f"Segment prediction CSV not found: {segment_csv}")

    labels = load_labels(args.labels_json, args.labels)
    methods = parse_methods(args.aggregation_methods)
    thresholds = make_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    df = pd.read_csv(segment_csv)
    prob_cols, truth_cols = validate_input_columns(df, labels, args.parent_id_col, args.prob_prefix)

    # Normalize truth/prob columns to numeric.
    for label in labels:
        df[truth_cols[label]] = pd.to_numeric(df[truth_cols[label]], errors="coerce").fillna(0).astype(int)
        df[prob_cols[label]] = pd.to_numeric(df[prob_cols[label]], errors="coerce").astype(float)

    parent_truth_df = build_parent_truth(df, labels, args.parent_id_col, truth_cols)
    parent_truth_df = parent_truth_df.set_index(args.parent_id_col).sort_index()

    scores_by_method = build_parent_scores(df, labels, args.parent_id_col, prob_cols, methods)
    for method in methods:
        scores_by_method[method] = scores_by_method[method].sort_index()

    parent_ids = parent_truth_df.index.to_numpy()
    y_true_all = parent_truth_df[labels].to_numpy(dtype=int)

    repeated_rows = []
    selected_rows = []
    candidate_rows_all = []
    eval_per_label_rows = []

    for seed in seeds:
        cal_ids, eval_ids = split_parent_ids(parent_ids, args.cal_fraction, seed)
        cal_ids = list(cal_ids)
        eval_ids = list(eval_ids)

        selected_rules: Dict[str, SelectedRule] = {}

        for label in labels:
            y_true_cal = parent_truth_df.loc[cal_ids, label].to_numpy(dtype=int)
            score_by_method_cal = {
                method: scores_by_method[method].loc[cal_ids, label].to_numpy(dtype=float)
                for method in methods
            }

            selected, candidate_rows = search_best_rule_for_label(
                label=label,
                y_true_cal=y_true_cal,
                score_by_method_cal=score_by_method_cal,
                thresholds=thresholds,
                methods=methods,
                seed=seed,
                save_candidates=args.save_candidate_log,
            )
            selected_rules[label] = selected

            selected_rows.append(
                {
                    "seed": seed,
                    "label": label,
                    "aggregation": selected.aggregation,
                    "threshold": selected.threshold,
                    "cal_precision": selected.precision,
                    "cal_recall": selected.recall,
                    "cal_f1": selected.f1,
                    "cal_tp": selected.tp,
                    "cal_fp": selected.fp,
                    "cal_fn": selected.fn,
                    "cal_tn": selected.tn,
                    "cal_support": selected.support,
                    "cal_predicted_positive": selected.predicted_positive,
                    "cal_hamming_errors": selected.hamming_errors,
                }
            )

            if args.save_candidate_log:
                candidate_rows_all.extend(candidate_rows)

        # Evaluate selected seed-specific config on held-out evaluation parents.
        y_true_eval = parent_truth_df.loc[eval_ids, labels].to_numpy(dtype=int)
        y_pred_eval, _ = apply_rules(eval_ids, labels, scores_by_method, selected_rules)
        metrics = compute_multilabel_metrics(y_true_eval, y_pred_eval, labels)
        metrics.update(
            {
                "seed": seed,
                "cal_fraction": args.cal_fraction,
                "n_calibration_parents": len(cal_ids),
                "n_evaluation_parents": len(eval_ids),
                "mean_count": sum(1 for r in selected_rules.values() if r.aggregation == "mean"),
                "max_count": sum(1 for r in selected_rules.values() if r.aggregation == "max"),
                "top2mean_count": sum(1 for r in selected_rules.values() if r.aggregation == "top2mean"),
                "top3mean_count": sum(1 for r in selected_rules.values() if r.aggregation == "top3mean"),
            }
        )
        repeated_rows.append(metrics)

        agg_by_label = {label: selected_rules[label].aggregation for label in labels}
        thr_by_label = {label: selected_rules[label].threshold for label in labels}
        per_label_df = compute_per_label_metrics(
            y_true_eval, y_pred_eval, labels, agg_by_label, thr_by_label
        )
        per_label_df.insert(0, "seed", seed)
        eval_per_label_rows.extend(per_label_df.to_dict(orient="records"))

    repeated_df = pd.DataFrame(repeated_rows)
    selected_df = pd.DataFrame(selected_rows)
    eval_per_label_df = pd.DataFrame(eval_per_label_rows)

    # Summary by label/method selection frequency.
    freq_rows = []
    for label in labels:
        sub = selected_df[selected_df["label"] == label]
        counts = Counter(sub["aggregation"].astype(str).tolist())
        total = len(sub)
        for method in methods:
            method_sub = sub[sub["aggregation"] == method]
            thresholds_selected = method_sub["threshold"].astype(float).to_numpy()
            freq_rows.append(
                {
                    "label": label,
                    "aggregation": method,
                    "selection_count": int(counts.get(method, 0)),
                    "selection_fraction": float(counts.get(method, 0) / total) if total else 0.0,
                    "threshold_mean_when_selected": float(np.mean(thresholds_selected)) if thresholds_selected.size else np.nan,
                    "threshold_median_when_selected": float(np.median(thresholds_selected)) if thresholds_selected.size else np.nan,
                    "threshold_std_when_selected": float(np.std(thresholds_selected, ddof=1)) if thresholds_selected.size > 1 else 0.0 if thresholds_selected.size == 1 else np.nan,
                    "cal_f1_mean_when_selected": float(method_sub["cal_f1"].mean()) if not method_sub.empty else np.nan,
                }
            )
    freq_df = pd.DataFrame(freq_rows)

    # Threshold summary for the finally selected aggregation per label.
    final_config = build_final_frozen_config(
        selected_df=selected_df,
        labels=labels,
        thresholds=thresholds,
        round_to_grid=args.round_thresholds_to_grid,
    )

    threshold_summary_rows = []
    for label in labels:
        cfg = final_config[label]
        sub = selected_df[selected_df["label"] == label]
        method_sub = sub[sub["aggregation"] == cfg["aggregation"]]
        threshold_summary_rows.append(
            {
                "label": label,
                "final_aggregation": cfg["aggregation"],
                "final_threshold": cfg["threshold"],
                "selection_count": cfg["selection_count"],
                "selection_fraction": cfg["selection_fraction"],
                "threshold_mean_selected_method": cfg["threshold_mean_selected_method"],
                "threshold_median_selected_method": cfg["threshold_median_selected_method"],
                "threshold_std_selected_method": cfg["threshold_std_selected_method"],
                "cal_f1_mean_selected_method": float(method_sub["cal_f1"].mean()) if not method_sub.empty else np.nan,
                "cal_f1_median_selected_method": float(method_sub["cal_f1"].median()) if not method_sub.empty else np.nan,
                "method_counts_json": json.dumps(cfg["all_method_counts"], sort_keys=True),
            }
        )
    threshold_summary_df = pd.DataFrame(threshold_summary_rows)

    # Apply final frozen config to full holdout.
    final_rules = selected_rules_from_final_config(final_config, labels)
    y_pred_full, parent_pred_df = apply_rules(parent_ids, labels, scores_by_method, final_rules)
    y_true_full = y_true_all
    full_metrics = compute_multilabel_metrics(y_true_full, y_pred_full, labels)
    full_metrics.update(
        {
            "method": "lats_final_frozen_config_full_holdout",
            "model_name": args.model_name,
            "segment_pred_csv": str(segment_csv),
            "selection_source": "median/mode from repeated calibration splits",
            "seeds": args.seeds,
            "cal_fraction": args.cal_fraction,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "aggregation_methods": ",".join(methods),
        }
    )
    full_eval_df = pd.DataFrame([full_metrics])

    final_agg_by_label = {label: final_config[label]["aggregation"] for label in labels}
    final_thr_by_label = {label: float(final_config[label]["threshold"]) for label in labels}
    full_per_label_df = compute_per_label_metrics(
        y_true_full, y_pred_full, labels, final_agg_by_label, final_thr_by_label
    )
    full_per_label_df.insert(0, "method", "lats_final_frozen_config_full_holdout")

    # Add truth columns and IDs to parent prediction output.
    parent_pred_df = parent_pred_df.rename(columns={"parent_clip_id": args.parent_id_col})
    for j, label in enumerate(labels):
        parent_pred_df[f"true_{label}"] = y_true_full[:, j]

    # Persist outputs.
    repeated_df.to_csv(out_dir / "lats_repeated_eval_summary.csv", index=False)
    selected_df.to_csv(out_dir / "lats_seed_selected_configs_long.csv", index=False)
    eval_per_label_df.to_csv(out_dir / "lats_seed_eval_per_label.csv", index=False)
    freq_df.to_csv(out_dir / "lats_label_selection_frequency.csv", index=False)
    threshold_summary_df.to_csv(out_dir / "lats_threshold_summary.csv", index=False)
    full_eval_df.to_csv(out_dir / "lats_final_full_holdout_eval.csv", index=False)
    full_per_label_df.to_csv(out_dir / "lats_final_full_holdout_per_label.csv", index=False)
    parent_pred_df.to_csv(out_dir / "lats_final_full_holdout_parent_predictions.csv", index=False)

    if args.save_candidate_log:
        pd.DataFrame(candidate_rows_all).to_csv(out_dir / "lats_candidate_search_log.csv", index=False)

    run_config = {
        "script": "scripts/v0.9/run_lats_labelwise_aggregation_threshold_search_v09.py",
        "model_name": args.model_name,
        "segment_pred_csv": str(segment_csv),
        "labels_json": args.labels_json,
        "labels": labels,
        "parent_id_col": args.parent_id_col,
        "prob_prefix": args.prob_prefix,
        "prob_cols": prob_cols,
        "truth_cols": truth_cols,
        "aggregation_methods": methods,
        "thresholds": thresholds.tolist(),
        "seeds": seeds,
        "cal_fraction": args.cal_fraction,
        "n_segments": int(len(df)),
        "n_parent_clips": int(len(parent_ids)),
        "save_candidate_log": args.save_candidate_log,
        "tie_breaking": [
            "higher F1",
            "lower hamming errors",
            "fewer false positives",
            "threshold closer to 0.5",
            "simpler aggregation preference: mean > top2mean > top3mean > max",
        ],
    }
    with (out_dir / "lats_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    final_config_json = {
        "name": "lats_final_frozen_config_v09",
        "description": "Final label-wise aggregation and threshold config selected from repeated LATS calibration splits.",
        "model_name": args.model_name,
        "segment_pred_csv": str(segment_csv),
        "selection_rule": "aggregation = most frequently selected method; threshold = median selected threshold for that method",
        "labels": labels,
        "config": final_config,
        "full_holdout_metrics": full_metrics,
    }
    with (out_dir / "lats_final_frozen_config.json").open("w", encoding="utf-8") as f:
        json.dump(final_config_json, f, indent=2)

    # Console summary.
    eval_mean = repeated_df[["macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]].mean()
    eval_std = repeated_df[["macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]].std(ddof=1)

    print("\nLATS-v0.9 search complete")
    print("-" * 100)
    print(f"Segment CSV: {segment_csv}")
    print(f"Parent clips: {len(parent_ids)}")
    print(f"Segments:     {len(df)}")
    print(f"Labels:       {len(labels)}")
    print(f"Methods:      {', '.join(methods)}")
    print(f"Thresholds:   {thresholds[0]:.4f} to {thresholds[-1]:.4f} step {args.threshold_step}")
    print(f"Seeds:        {args.seeds}")
    print(f"Output dir:   {out_dir}")

    print("\nRepeated split evaluation mean ± std:")
    for key in ["macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]:
        print(f"  {key:16s}: {eval_mean[key]:.6f} ± {eval_std[key]:.6f}")

    print("\nFinal frozen config full-holdout evaluation:")
    for key in ["macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]:
        print(f"  {key:16s}: {full_metrics[key]:.6f}")

    print("\nFinal label-wise config:")
    for label in labels:
        cfg = final_config[label]
        print(f"  {label:28s} -> {cfg['aggregation']:9s} threshold={cfg['threshold']:.4f} "
              f"selected={cfg['selection_count']}/{args.seeds}")

    print("\nSaved files:")
    for name in [
        "lats_repeated_eval_summary.csv",
        "lats_seed_selected_configs_long.csv",
        "lats_seed_eval_per_label.csv",
        "lats_label_selection_frequency.csv",
        "lats_threshold_summary.csv",
        "lats_final_frozen_config.json",
        "lats_final_full_holdout_eval.csv",
        "lats_final_full_holdout_per_label.csv",
        "lats_final_full_holdout_parent_predictions.csv",
        "lats_run_config.json",
    ]:
        print(f"  {out_dir / name}")
    if args.save_candidate_log:
        print(f"  {out_dir / 'lats_candidate_search_log.csv'}")


if __name__ == "__main__":
    main()
