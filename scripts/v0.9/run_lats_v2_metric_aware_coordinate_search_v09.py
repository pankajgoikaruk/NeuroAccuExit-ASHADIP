#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
LATS-v2: Metric-Aware Label-wise Aggregation and Threshold Coordinate Search
============================================================================

Purpose
-------
LATS-v1 selects an aggregation-threshold pair independently per label using
label-level F1. LATS-v2 improves this by optimising the FULL multi-label
objective. It starts from an existing frozen configuration, then changes one
label at a time only when the complete parent-level multi-label score improves.

This is an inference-time/post-hoc optimisation layer. It does NOT retrain the
base neural model.

Recommended command
-------------------
python scripts\v0.9\run_lats_v2_metric_aware_coordinate_search_v09.py `
  --segment-pred-csv "human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --start-config-json "human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v09_search\lats_final_frozen_config.json" `
  --out-dir "human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v2_metric_coordinate_search" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --seeds 20 `
  --cal-fraction 0.5 `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.01 `
  --aggregation-methods "mean,max,top2mean,top3mean,top4mean,top5mean,median,p75,p90,noisy_or" `
  --objective-weights "macro_f1=0.40,micro_f1=0.20,samples_f1=0.20,exact_match=0.15,hamming_loss=-0.05,label_count_abs_error=-0.05" `
  --max-iterations 5 `
  --model-name "main_v08_human_corrected_balanced_3exit_20260610_084027"

Main outputs
------------
lats_v2_init_full_holdout_eval.csv
lats_v2_repeated_eval_summary.csv
lats_v2_seed_final_configs_long.csv
lats_v2_coordinate_search_log.csv
lats_v2_label_selection_frequency.csv
lats_v2_threshold_summary.csv
lats_v2_final_frozen_config.json
lats_v2_final_full_holdout_eval.csv
lats_v2_final_full_holdout_per_label.csv
lats_v2_final_vs_init_comparison.csv

Important reporting note
------------------------
The repeated split results are the safer generalisation estimate. The final
full-holdout result is useful for comparison with previous v0.9 outputs, but it
is still derived from a configuration frozen from repeated splits on the same
corrected holdout. Do not describe it as an external unseen test result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

# Conservative tie-break preference. Lower is preferred.
METHOD_PREFERENCE = {
    "mean": 0,
    "top2mean": 1,
    "top3mean": 2,
    "top4mean": 3,
    "top5mean": 4,
    "median": 5,
    "p75": 6,
    "p90": 7,
    "noisy_or": 8,
    "max": 9,
}

DEFAULT_START_CONFIG = {
    "Brene_Brown": {"aggregation": "top3mean", "threshold": 0.50},
    "Eckhart_Tolle": {"aggregation": "top2mean", "threshold": 0.50},
    "Eric_Thomas": {"aggregation": "mean", "threshold": 0.53},
    "Gary_Vee": {"aggregation": "top3mean", "threshold": 0.50},
    "Jay_Shetty": {"aggregation": "mean", "threshold": 0.80},
    "Nick_Vujicic": {"aggregation": "mean", "threshold": 0.43},
    "other_speaker_present": {"aggregation": "top3mean", "threshold": 0.76},
    "music_present": {"aggregation": "mean", "threshold": 0.49},
    "audience_reaction_present": {"aggregation": "max", "threshold": 0.68},
    "silence_present": {"aggregation": "top2mean", "threshold": 0.38},
}


@dataclass(frozen=True)
class Rule:
    aggregation: str
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LATS-v2 metric-aware coordinate search for label-wise parent inference."
    )

    parser.add_argument("--segment-pred-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--labels-json", default=None)
    parser.add_argument("--labels", default=None, help="Optional comma-separated labels; overrides --labels-json.")
    parser.add_argument("--start-config-json", default=None, help="LATS/LATS-v2 config JSON used as starting point.")
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--prob-prefix", default="exit3_prob_")
    parser.add_argument("--model-name", default="")

    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-12, help="Minimum objective improvement required.")

    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--aggregation-methods",
        default="mean,max,top2mean,top3mean,top4mean,top5mean,median,p75,p90,noisy_or",
        help="Comma-separated methods.",
    )
    parser.add_argument(
        "--objective-weights",
        default="macro_f1=0.40,micro_f1=0.20,samples_f1=0.20,exact_match=0.15,hamming_loss=-0.05,label_count_abs_error=-0.05",
        help="Comma-separated metric weights. Use negative weight for metrics to minimise.",
    )
    parser.add_argument(
        "--round-thresholds-to-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round final median thresholds to nearest grid value.",
    )
    parser.add_argument(
        "--save-candidate-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save every candidate evaluation. This can be large.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print seed/iteration/label progress while coordinate search is running.",
    )

    return parser.parse_args()


def extract_labels_from_json(obj) -> List[str]:
    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            return list(obj)
        if all(isinstance(x, dict) for x in obj):
            labels = []
            for item in obj:
                for key in ("name", "label", "class_name", "id"):
                    if key in item and isinstance(item[key], str):
                        labels.append(item[key])
                        break
            return labels
    if isinstance(obj, dict):
        for key in ("labels", "label_names", "classes", "class_names"):
            if key in obj:
                return extract_labels_from_json(obj[key])
        if obj and all(str(k).isdigit() for k in obj.keys()) and all(isinstance(v, str) for v in obj.values()):
            return [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x))]
    return []


def load_labels(labels_json: Optional[str], labels_arg: Optional[str]) -> List[str]:
    if labels_arg:
        labels = [x.strip() for x in labels_arg.split(",") if x.strip()]
        if not labels:
            raise RuntimeError("--labels was supplied but no labels were parsed.")
        return labels
    if labels_json:
        path = Path(labels_json)
        if not path.exists():
            raise FileNotFoundError(f"Labels JSON not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            labels = extract_labels_from_json(json.load(f))
        if not labels:
            raise RuntimeError(f"Could not parse labels from {path}")
        return labels
    return DEFAULT_LABELS.copy()


def parse_methods(methods: str) -> List[str]:
    out = [x.strip().lower() for x in methods.split(",") if x.strip()]
    if not out:
        raise RuntimeError("No aggregation methods supplied.")
    dummy = np.array([0.1, 0.2, 0.3])
    for m in out:
        _ = aggregate_values(dummy, m)
    return out


def parse_objective_weights(text: str) -> Dict[str, float]:
    weights = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise RuntimeError(f"Invalid objective weight item: {part}")
        key, value = part.split("=", 1)
        weights[key.strip()] = float(value.strip())
    if not weights:
        raise RuntimeError("No objective weights parsed.")
    return weights


def make_threshold_grid(t_min: float, t_max: float, step: float) -> np.ndarray:
    if not (0 <= t_min <= 1 and 0 <= t_max <= 1):
        raise RuntimeError("Threshold bounds must be in [0, 1].")
    if t_min > t_max:
        raise RuntimeError("threshold-min must be <= threshold-max.")
    if step <= 0:
        raise RuntimeError("threshold-step must be positive.")
    decimals = max(0, int(math.ceil(-math.log10(step))) + 2) if step < 1 else 2
    values = np.arange(t_min, t_max + step / 2.0, step)
    values = np.round(values, decimals)
    values = values[(values >= t_min - 1e-12) & (values <= t_max + 1e-12)]
    return values.astype(float)


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
    raise RuntimeError(f"Probability column not found for label {label}. Tried: {candidates}")


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
    raise RuntimeError(f"Truth column not found for label {label}. Tried: {candidates}")


def validate_columns(df: pd.DataFrame, labels: Sequence[str], parent_id_col: str, prob_prefix: str):
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
    if method == "median":
        return float(np.median(values))
    if method == "noisy_or":
        # Smooth any-segment evidence aggregator.
        clipped = np.clip(values, 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - clipped))

    m = re.fullmatch(r"top(\d+)mean", method)
    if m:
        k = max(1, int(m.group(1)))
        top = np.sort(values)[::-1][: min(k, values.size)]
        return float(np.mean(top))

    m = re.fullmatch(r"p(\d+(?:\.\d+)?)", method)
    if m:
        p = float(m.group(1))
        if not 0 <= p <= 100:
            raise RuntimeError(f"Invalid percentile method: {method}")
        return float(np.percentile(values, p))

    raise RuntimeError(
        f"Unsupported aggregation method {method}. Examples: mean,max,top2mean,top3mean,median,p75,p90,noisy_or"
    )


def build_parent_truth(df, labels, parent_id_col, truth_cols) -> Tuple[np.ndarray, List, pd.DataFrame]:
    parent_ids = sorted(df[parent_id_col].dropna().unique())
    grouped = df.groupby(parent_id_col, sort=True)
    truth_df = pd.DataFrame(index=parent_ids)
    truth_df.index.name = parent_id_col
    for label in labels:
        truth_df[label] = grouped[truth_cols[label]].max().astype(int)
    return truth_df[labels].to_numpy(dtype=int), parent_ids, truth_df.reset_index()


def build_score_tensor(df, labels, parent_id_col, prob_cols, methods) -> Tuple[Dict[str, np.ndarray], List]:
    parent_ids = sorted(df[parent_id_col].dropna().unique())
    grouped = df.groupby(parent_id_col, sort=True)
    scores_by_method: Dict[str, np.ndarray] = {}
    for method in methods:
        arr = np.zeros((len(parent_ids), len(labels)), dtype=float)
        for j, label in enumerate(labels):
            col = prob_cols[label]
            if method == "mean":
                s = grouped[col].mean()
            elif method == "max":
                s = grouped[col].max()
            elif method == "median":
                s = grouped[col].median()
            else:
                s = grouped[col].apply(lambda x, m=method: aggregate_values(x.to_numpy(), m))
            arr[:, j] = s.loc[parent_ids].to_numpy(dtype=float)
        scores_by_method[method] = arr
    return scores_by_method, parent_ids


def load_start_config(path: Optional[str], labels: Sequence[str], methods: Sequence[str]) -> Dict[str, Rule]:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Start config JSON not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if "config" in obj and isinstance(obj["config"], dict):
            raw = obj["config"]
        elif "aggregation" in obj and isinstance(obj["aggregation"], dict):
            # Mapping-bank style with separate thresholds optional.
            raw = {}
            thresholds = obj.get("thresholds", {})
            for label, agg in obj["aggregation"].items():
                raw[label] = {"aggregation": agg, "threshold": thresholds.get(label, obj.get("threshold", 0.5))}
        else:
            raw = obj
    else:
        raw = DEFAULT_START_CONFIG

    rules: Dict[str, Rule] = {}
    for label in labels:
        if label not in raw:
            raise RuntimeError(f"Start config missing label: {label}")
        item = raw[label]
        if isinstance(item, dict):
            agg = str(item.get("aggregation", item.get("method", ""))).lower()
            threshold = float(item.get("threshold"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            agg = str(item[0]).lower()
            threshold = float(item[1])
        else:
            raise RuntimeError(f"Invalid start config item for {label}: {item}")
        if agg not in methods:
            raise RuntimeError(
                f"Start config for {label} uses aggregation '{agg}', which is not in --aggregation-methods: {methods}"
            )
        rules[label] = Rule(aggregation=agg, threshold=threshold)
    return rules


def f1_scores(y_true, y_pred) -> Tuple[float, float, float]:
    if f1_score is not None:
        return (
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        )

    # fallback
    per_label = []
    for j in range(y_true.shape[1]):
        per_label.append(binary_f1(y_true[:, j], y_pred[:, j]))
    macro = float(np.mean(per_label)) if per_label else 0.0
    micro = binary_f1(y_true.ravel(), y_pred.ravel())
    samples_vals = []
    for i in range(y_true.shape[0]):
        denom = np.sum(y_true[i]) + np.sum(y_pred[i])
        inter = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
        samples_vals.append(2 * inter / denom if denom else 0.0)
    return macro, float(micro), float(np.mean(samples_vals))


def binary_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, objective_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    macro, micro, samples = f1_scores(y_true, y_pred)
    exact = float(np.mean(np.all(y_true == y_pred, axis=1))) if y_true.size else 0.0
    hamming = float(np.mean(y_true != y_pred)) if y_true.size else 0.0
    avg_true = float(np.mean(np.sum(y_true, axis=1))) if y_true.size else 0.0
    avg_pred = float(np.mean(np.sum(y_pred, axis=1))) if y_true.size else 0.0
    count_err = abs(avg_pred - avg_true)
    if jaccard_score is not None:
        jaccard = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    else:
        vals = []
        for i in range(y_true.shape[0]):
            inter = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
            union = np.sum((y_true[i] == 1) | (y_pred[i] == 1))
            vals.append(inter / union if union else 0.0)
        jaccard = float(np.mean(vals)) if vals else 0.0
    out = {
        "macro_f1": macro,
        "micro_f1": micro,
        "samples_f1": samples,
        "exact_match": exact,
        "hamming_loss": hamming,
        "jaccard": jaccard,
        "avg_true_labels": avg_true,
        "avg_pred_labels": avg_pred,
        "label_count_abs_error": count_err,
        "n_parent_clips": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]) if y_true.ndim == 2 else 0,
    }
    if objective_weights is not None:
        out["objective_score"] = objective_score(out, objective_weights)
    return out


def objective_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        if key not in metrics:
            raise RuntimeError(f"Objective metric '{key}' not available. Available: {sorted(metrics.keys())}")
        score += float(weight) * float(metrics[key])
    return float(score)


def rules_to_pred_matrix(
    rules: Dict[str, Rule],
    labels: Sequence[str],
    scores_by_method: Dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    y_pred = np.zeros((len(indices), len(labels)), dtype=int)
    for j, label in enumerate(labels):
        rule = rules[label]
        y_pred[:, j] = (scores_by_method[rule.aggregation][indices, j] >= rule.threshold).astype(int)
    return y_pred


def method_rank(method: str) -> int:
    return METHOD_PREFERENCE.get(method, 999)


def candidate_better(cand: Dict[str, float], best: Optional[Dict[str, float]], min_delta: float) -> bool:
    if best is None:
        return True
    eps = 1e-12
    if cand["objective_score"] > best["objective_score"] + max(min_delta, eps):
        return True
    if cand["objective_score"] < best["objective_score"] - eps:
        return False
    # Tie breakers: macro, exact, hamming, threshold closeness, method simplicity.
    for key, higher_is_better in [
        ("macro_f1", True),
        ("micro_f1", True),
        ("exact_match", True),
        ("samples_f1", True),
        ("hamming_loss", False),
        ("label_count_abs_error", False),
    ]:
        if higher_is_better:
            if cand[key] > best[key] + eps:
                return True
            if cand[key] < best[key] - eps:
                return False
        else:
            if cand[key] < best[key] - eps:
                return True
            if cand[key] > best[key] + eps:
                return False
    if abs(cand["candidate_threshold"] - 0.5) < abs(best["candidate_threshold"] - 0.5) - eps:
        return True
    if method_rank(cand["candidate_aggregation"]) < method_rank(best["candidate_aggregation"]):
        return True
    return False


def split_ids(n: int, cal_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < cal_fraction < 1:
        raise RuntimeError("--cal-fraction must be > 0 and < 1")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cal_n = int(round(n * cal_fraction))
    cal_n = max(1, min(n - 1, cal_n))
    return idx[:cal_n], idx[cal_n:]


def coordinate_search_one_split(
    seed: int,
    y_true_cal: np.ndarray,
    cal_idx: np.ndarray,
    labels: Sequence[str],
    methods: Sequence[str],
    thresholds: np.ndarray,
    scores_by_method: Dict[str, np.ndarray],
    start_rules: Dict[str, Rule],
    objective_weights: Dict[str, float],
    max_iterations: int,
    min_delta: float,
    save_candidate_log: bool,
    progress: bool = True,
    seed_position: int = 1,
    total_seeds: int = 1,
) -> Tuple[Dict[str, Rule], List[dict], List[dict], Dict[str, float]]:
    rules = {label: Rule(r.aggregation, float(r.threshold)) for label, r in start_rules.items()}
    y_pred = rules_to_pred_matrix(rules, labels, scores_by_method, cal_idx)
    current = compute_metrics(y_true_cal, y_pred, objective_weights)
    search_log: List[dict] = []
    candidate_log: List[dict] = []

    total_candidate_checks_per_label = len(methods) * len(thresholds)

    for iteration in range(1, max_iterations + 1):
        any_improved = False
        accepted_this_iteration = 0
        if progress:
            print(
                f"\n[Seed {seed_position}/{total_seeds} | seed={seed}] "
                f"Iteration {iteration}/{max_iterations} | "
                f"current objective={current['objective_score']:.6f}",
                flush=True,
            )

        for j, label in enumerate(labels):
            before_rule = rules[label]
            before_score = current["objective_score"]
            best_cand: Optional[Dict[str, float]] = None
            best_pred_col: Optional[np.ndarray] = None

            if progress:
                pct = 100.0 * ((iteration - 1) * len(labels) + (j + 1)) / max(1, max_iterations * len(labels))
                print(
                    f"  [{pct:5.1f}% seed progress] "
                    f"Checking label {j + 1}/{len(labels)}: {label} | "
                    f"current={before_rule.aggregation}+{before_rule.threshold:.2f} | "
                    f"candidates≈{total_candidate_checks_per_label}",
                    flush=True,
                )

            # Current candidate baseline for this label.
            current_row = dict(current)
            current_row.update(
                {
                    "seed": seed,
                    "iteration": iteration,
                    "label": label,
                    "candidate_aggregation": before_rule.aggregation,
                    "candidate_threshold": before_rule.threshold,
                    "is_current_before_label_update": True,
                }
            )
            best_cand = current_row
            best_pred_col = y_pred[:, j].copy()

            for method in methods:
                scores = scores_by_method[method][cal_idx, j]
                for threshold in thresholds:
                    cand_col = (scores >= threshold).astype(int)
                    if method == before_rule.aggregation and abs(float(threshold) - before_rule.threshold) < 1e-12:
                        continue
                    cand_pred = y_pred.copy()
                    cand_pred[:, j] = cand_col
                    cand_metrics = compute_metrics(y_true_cal, cand_pred, objective_weights)
                    cand_metrics.update(
                        {
                            "seed": seed,
                            "iteration": iteration,
                            "label": label,
                            "candidate_aggregation": method,
                            "candidate_threshold": float(threshold),
                            "is_current_before_label_update": False,
                        }
                    )
                    if save_candidate_log:
                        candidate_log.append(cand_metrics.copy())
                    if candidate_better(cand_metrics, best_cand, min_delta=0.0):
                        best_cand = cand_metrics
                        best_pred_col = cand_col

            assert best_cand is not None
            assert best_pred_col is not None

            improvement = best_cand["objective_score"] - before_score
            accepted = improvement > min_delta

            search_log.append(
                {
                    "seed": seed,
                    "iteration": iteration,
                    "label": label,
                    "before_aggregation": before_rule.aggregation,
                    "before_threshold": before_rule.threshold,
                    "after_aggregation": best_cand["candidate_aggregation"] if accepted else before_rule.aggregation,
                    "after_threshold": best_cand["candidate_threshold"] if accepted else before_rule.threshold,
                    "before_objective_score": before_score,
                    "after_objective_score": best_cand["objective_score"] if accepted else before_score,
                    "objective_improvement": improvement if accepted else 0.0,
                    "accepted": bool(accepted),
                    "candidate_macro_f1": best_cand["macro_f1"],
                    "candidate_micro_f1": best_cand["micro_f1"],
                    "candidate_samples_f1": best_cand["samples_f1"],
                    "candidate_exact_match": best_cand["exact_match"],
                    "candidate_hamming_loss": best_cand["hamming_loss"],
                    "candidate_avg_pred_labels": best_cand["avg_pred_labels"],
                }
            )

            if accepted:
                rules[label] = Rule(
                    aggregation=str(best_cand["candidate_aggregation"]),
                    threshold=float(best_cand["candidate_threshold"]),
                )
                y_pred[:, j] = best_pred_col
                current = compute_metrics(y_true_cal, y_pred, objective_weights)
                any_improved = True
                accepted_this_iteration += 1

            if progress:
                if accepted:
                    print(
                        f"      ACCEPTED: {label} -> "
                        f"{rules[label].aggregation}+{rules[label].threshold:.2f} | "
                        f"objective {before_score:.6f} -> {current['objective_score']:.6f} "
                        f"(Δ={current['objective_score'] - before_score:+.6f})",
                        flush=True,
                    )
                else:
                    print(
                        f"      kept: {label} -> {before_rule.aggregation}+{before_rule.threshold:.2f} | "
                        f"best candidate did not pass min_delta",
                        flush=True,
                    )

        if progress:
            print(
                f"  End iteration {iteration}/{max_iterations}: "
                f"accepted_moves={accepted_this_iteration}, "
                f"objective={current['objective_score']:.6f}",
                flush=True,
            )

        if not any_improved:
            if progress:
                print(
                    f"  No accepted moves in iteration {iteration}; stopping this seed early.",
                    flush=True,
                )
            break

    final_cal_metrics = compute_metrics(y_true_cal, y_pred, objective_weights)
    return rules, search_log, candidate_log, final_cal_metrics


def config_to_rows(seed: int, rules: Dict[str, Rule], labels: Sequence[str]) -> List[dict]:
    return [
        {
            "seed": seed,
            "label": label,
            "aggregation": rules[label].aggregation,
            "threshold": rules[label].threshold,
        }
        for label in labels
    ]


def build_final_config_from_seed_rules(
    seed_config_df: pd.DataFrame,
    labels: Sequence[str],
    thresholds: np.ndarray,
    round_to_grid: bool,
) -> Dict[str, Rule]:
    final: Dict[str, Rule] = {}
    for label in labels:
        sub = seed_config_df[seed_config_df["label"] == label].copy()
        if sub.empty:
            raise RuntimeError(f"No seed configs for label: {label}")
        counts = Counter(sub["aggregation"].astype(str).tolist())
        final_method = sorted(counts.keys(), key=lambda m: (-counts[m], method_rank(m), m))[0]
        selected_thresholds = sub[sub["aggregation"] == final_method]["threshold"].astype(float).to_numpy()
        if selected_thresholds.size == 0:
            selected_thresholds = sub["threshold"].astype(float).to_numpy()
        median_t = float(np.median(selected_thresholds))
        if round_to_grid:
            median_t = float(thresholds[int(np.argmin(np.abs(thresholds - median_t)))])
        final[label] = Rule(final_method, median_t)
    return final


def build_threshold_summary(seed_config_df: pd.DataFrame, final_rules: Dict[str, Rule], labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        sub = seed_config_df[seed_config_df["label"] == label].copy()
        counts = Counter(sub["aggregation"].astype(str).tolist())
        final_method = final_rules[label].aggregation
        method_sub = sub[sub["aggregation"] == final_method]
        thresholds = method_sub["threshold"].astype(float).to_numpy()
        rows.append(
            {
                "label": label,
                "final_aggregation": final_method,
                "final_threshold": final_rules[label].threshold,
                "selection_count": int(counts.get(final_method, 0)),
                "selection_fraction": float(counts.get(final_method, 0) / len(sub)) if len(sub) else 0.0,
                "threshold_mean_selected_method": float(np.mean(thresholds)) if thresholds.size else np.nan,
                "threshold_median_selected_method": float(np.median(thresholds)) if thresholds.size else np.nan,
                "threshold_std_selected_method": float(np.std(thresholds, ddof=1)) if thresholds.size > 1 else 0.0 if thresholds.size == 1 else np.nan,
                "method_counts_json": json.dumps(dict(counts), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def label_selection_frequency(seed_config_df: pd.DataFrame, labels: Sequence[str], methods: Sequence[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        sub = seed_config_df[seed_config_df["label"] == label]
        total = len(sub)
        for method in methods:
            method_sub = sub[sub["aggregation"] == method]
            th = method_sub["threshold"].astype(float).to_numpy()
            rows.append(
                {
                    "label": label,
                    "aggregation": method,
                    "selection_count": int(len(method_sub)),
                    "selection_fraction": float(len(method_sub) / total) if total else 0.0,
                    "threshold_mean_when_selected": float(np.mean(th)) if th.size else np.nan,
                    "threshold_median_when_selected": float(np.median(th)) if th.size else np.nan,
                    "threshold_std_when_selected": float(np.std(th, ddof=1)) if th.size > 1 else 0.0 if th.size == 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], rules: Dict[str, Rule]) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(labels):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append(
            {
                "label": label,
                "aggregation": rules[label].aggregation,
                "threshold": rules[label].threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(np.sum(yt == 1)),
                "predicted_positive": int(np.sum(yp == 1)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "hamming_errors": fp + fn,
            }
        )
    return pd.DataFrame(rows)


def parent_predictions_df(parent_ids, y_true, y_pred, labels, rules, scores_by_method) -> pd.DataFrame:
    df = pd.DataFrame({"parent_clip_id": parent_ids})
    all_idx = np.arange(len(parent_ids))
    for j, label in enumerate(labels):
        rule = rules[label]
        df[f"true_{label}"] = y_true[:, j]
        df[f"score_{label}"] = scores_by_method[rule.aggregation][all_idx, j]
        df[f"pred_{label}"] = y_pred[:, j]
        df[f"aggregation_{label}"] = rule.aggregation
        df[f"threshold_{label}"] = rule.threshold
    return df


def rules_to_json_config(rules: Dict[str, Rule], labels: Sequence[str]) -> Dict[str, dict]:
    return {label: {"aggregation": rules[label].aggregation, "threshold": float(rules[label].threshold)} for label in labels}


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
    objective_weights = parse_objective_weights(args.objective_weights)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    df = pd.read_csv(segment_csv)
    prob_cols, truth_cols = validate_columns(df, labels, args.parent_id_col, args.prob_prefix)
    for label in labels:
        df[truth_cols[label]] = pd.to_numeric(df[truth_cols[label]], errors="coerce").fillna(0).astype(int)
        df[prob_cols[label]] = pd.to_numeric(df[prob_cols[label]], errors="coerce").astype(float)

    y_true_all, parent_ids, _ = build_parent_truth(df, labels, args.parent_id_col, truth_cols)
    scores_by_method, score_parent_ids = build_score_tensor(df, labels, args.parent_id_col, prob_cols, methods)
    if list(parent_ids) != list(score_parent_ids):
        raise RuntimeError("Parent ID mismatch between truth and scores.")

    start_rules = load_start_config(args.start_config_json, labels, methods)
    init_pred_all = rules_to_pred_matrix(start_rules, labels, scores_by_method, np.arange(len(parent_ids)))
    init_metrics = compute_metrics(y_true_all, init_pred_all, objective_weights)
    init_metrics.update({"method": "initial_start_config_full_holdout", "model_name": args.model_name})
    pd.DataFrame([init_metrics]).to_csv(out_dir / "lats_v2_init_full_holdout_eval.csv", index=False)
    per_label_metrics(y_true_all, init_pred_all, labels, start_rules).to_csv(
        out_dir / "lats_v2_init_full_holdout_per_label.csv", index=False
    )

    repeated_rows = []
    seed_config_rows = []
    search_log_rows = []
    candidate_log_rows = []

    if args.progress:
        total_label_passes = len(seeds) * args.max_iterations * len(labels)
        approx_candidate_checks = total_label_passes * len(methods) * len(thresholds)
        print("\nStarting LATS-v2 metric-aware coordinate search", flush=True)
        print("-" * 100, flush=True)
        print(f"Seeds: {len(seeds)} | Max iterations/seed: {args.max_iterations} | Labels: {len(labels)}", flush=True)
        print(f"Methods: {len(methods)} | Thresholds: {len(thresholds)}", flush=True)
        print(f"Approx candidate checks upper bound: {approx_candidate_checks:,}", flush=True)
        print("Note: seeds stop early when no label improves.\n", flush=True)

    for seed_pos, seed in enumerate(seeds, start=1):
        cal_idx, eval_idx = split_ids(len(parent_ids), args.cal_fraction, seed)
        y_true_cal = y_true_all[cal_idx]
        y_true_eval = y_true_all[eval_idx]

        if args.progress:
            print("=" * 100, flush=True)
            print(
                f"Seed {seed_pos}/{len(seeds)} (seed={seed}) | "
                f"calibration parents={len(cal_idx)} | evaluation parents={len(eval_idx)}",
                flush=True,
            )

        seed_rules, search_log, candidate_log, final_cal_metrics = coordinate_search_one_split(
            seed=seed,
            y_true_cal=y_true_cal,
            cal_idx=cal_idx,
            labels=labels,
            methods=methods,
            thresholds=thresholds,
            scores_by_method=scores_by_method,
            start_rules=start_rules,
            objective_weights=objective_weights,
            max_iterations=args.max_iterations,
            min_delta=args.min_delta,
            save_candidate_log=args.save_candidate_log,
            progress=args.progress,
            seed_position=seed_pos,
            total_seeds=len(seeds),
        )

        y_pred_eval = rules_to_pred_matrix(seed_rules, labels, scores_by_method, eval_idx)
        eval_metrics = compute_metrics(y_true_eval, y_pred_eval, objective_weights)
        accepted_moves = sum(1 for row in search_log if row["accepted"])
        iterations_run = max([row["iteration"] for row in search_log], default=0)
        eval_metrics.update(
            {
                "seed": seed,
                "n_calibration_parents": len(cal_idx),
                "n_evaluation_parents": len(eval_idx),
                "iterations_run": iterations_run,
                "accepted_moves": accepted_moves,
                "mean_count": sum(1 for r in seed_rules.values() if r.aggregation == "mean"),
                "max_count": sum(1 for r in seed_rules.values() if r.aggregation == "max"),
                "top2mean_count": sum(1 for r in seed_rules.values() if r.aggregation == "top2mean"),
                "top3mean_count": sum(1 for r in seed_rules.values() if r.aggregation == "top3mean"),
                "top4mean_count": sum(1 for r in seed_rules.values() if r.aggregation == "top4mean"),
                "top5mean_count": sum(1 for r in seed_rules.values() if r.aggregation == "top5mean"),
                "noisy_or_count": sum(1 for r in seed_rules.values() if r.aggregation == "noisy_or"),
            }
        )
        # Add final calibration metrics as separate columns.
        for key, value in final_cal_metrics.items():
            eval_metrics[f"cal_{key}"] = value

        if args.progress:
            print(
                f"Seed {seed_pos}/{len(seeds)} evaluation: "
                f"macro={eval_metrics['macro_f1']:.6f}, "
                f"micro={eval_metrics['micro_f1']:.6f}, "
                f"exact={eval_metrics['exact_match']:.6f}, "
                f"hamming={eval_metrics['hamming_loss']:.6f}, "
                f"accepted_moves={accepted_moves}",
                flush=True,
            )

        repeated_rows.append(eval_metrics)
        seed_config_rows.extend(config_to_rows(seed, seed_rules, labels))
        search_log_rows.extend(search_log)
        candidate_log_rows.extend(candidate_log)

    repeated_df = pd.DataFrame(repeated_rows)
    seed_config_df = pd.DataFrame(seed_config_rows)
    search_log_df = pd.DataFrame(search_log_rows)

    repeated_df.to_csv(out_dir / "lats_v2_repeated_eval_summary.csv", index=False)
    seed_config_df.to_csv(out_dir / "lats_v2_seed_final_configs_long.csv", index=False)
    search_log_df.to_csv(out_dir / "lats_v2_coordinate_search_log.csv", index=False)
    if args.save_candidate_log:
        pd.DataFrame(candidate_log_rows).to_csv(out_dir / "lats_v2_candidate_search_log.csv", index=False)

    freq_df = label_selection_frequency(seed_config_df, labels, methods)
    freq_df.to_csv(out_dir / "lats_v2_label_selection_frequency.csv", index=False)

    final_rules = build_final_config_from_seed_rules(seed_config_df, labels, thresholds, args.round_thresholds_to_grid)
    threshold_summary_df = build_threshold_summary(seed_config_df, final_rules, labels)
    threshold_summary_df.to_csv(out_dir / "lats_v2_threshold_summary.csv", index=False)

    final_pred_all = rules_to_pred_matrix(final_rules, labels, scores_by_method, np.arange(len(parent_ids)))
    final_metrics = compute_metrics(y_true_all, final_pred_all, objective_weights)
    final_metrics.update(
        {
            "method": "lats_v2_final_frozen_config_full_holdout",
            "model_name": args.model_name,
            "selection_source": "metric-aware coordinate search repeated splits; final method=mode, threshold=median",
            "segment_pred_csv": str(segment_csv),
            "start_config_json": args.start_config_json or "DEFAULT_START_CONFIG",
            "seeds": args.seeds,
            "cal_fraction": args.cal_fraction,
            "max_iterations": args.max_iterations,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "aggregation_methods": ",".join(methods),
            "objective_weights": args.objective_weights,
        }
    )
    pd.DataFrame([final_metrics]).to_csv(out_dir / "lats_v2_final_full_holdout_eval.csv", index=False)
    per_label_metrics(y_true_all, final_pred_all, labels, final_rules).to_csv(
        out_dir / "lats_v2_final_full_holdout_per_label.csv", index=False
    )
    parent_predictions_df(parent_ids, y_true_all, final_pred_all, labels, final_rules, scores_by_method).to_csv(
        out_dir / "lats_v2_final_full_holdout_parent_predictions.csv", index=False
    )

    comparison = []
    for metric in [
        "objective_score",
        "macro_f1",
        "micro_f1",
        "samples_f1",
        "exact_match",
        "hamming_loss",
        "jaccard",
        "avg_true_labels",
        "avg_pred_labels",
        "label_count_abs_error",
    ]:
        comparison.append(
            {
                "metric": metric,
                "initial_start_config": init_metrics[metric],
                "lats_v2_final_config": final_metrics[metric],
                "difference_final_minus_initial": final_metrics[metric] - init_metrics[metric],
            }
        )
    pd.DataFrame(comparison).to_csv(out_dir / "lats_v2_final_vs_init_comparison.csv", index=False)

    final_config_json = {
        "name": "lats_v2_final_frozen_config_v09",
        "description": "Metric-aware coordinate-search label-wise aggregation and threshold config.",
        "model_name": args.model_name,
        "segment_pred_csv": str(segment_csv),
        "start_config_json": args.start_config_json or "DEFAULT_START_CONFIG",
        "objective_weights": objective_weights,
        "selection_rule": "For each seed, coordinate search optimises full multi-label objective on calibration parents. Final config uses most frequent aggregation per label and median threshold for that aggregation.",
        "labels": labels,
        "config": rules_to_json_config(final_rules, labels),
        "initial_full_holdout_metrics": init_metrics,
        "final_full_holdout_metrics": final_metrics,
    }
    with (out_dir / "lats_v2_final_frozen_config.json").open("w", encoding="utf-8") as f:
        json.dump(final_config_json, f, indent=2)

    run_config = {
        "script": "scripts/v0.9/run_lats_v2_metric_aware_coordinate_search_v09.py",
        "model_name": args.model_name,
        "segment_pred_csv": str(segment_csv),
        "labels_json": args.labels_json,
        "labels": labels,
        "parent_id_col": args.parent_id_col,
        "prob_prefix": args.prob_prefix,
        "prob_cols": prob_cols,
        "truth_cols": truth_cols,
        "start_config_json": args.start_config_json,
        "aggregation_methods": methods,
        "thresholds": thresholds.tolist(),
        "objective_weights": objective_weights,
        "seeds": seeds,
        "cal_fraction": args.cal_fraction,
        "max_iterations": args.max_iterations,
        "min_delta": args.min_delta,
        "n_segments": int(len(df)),
        "n_parent_clips": int(len(parent_ids)),
        "save_candidate_log": args.save_candidate_log,
        "progress": args.progress,
    }
    with (out_dir / "lats_v2_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # Console summary.
    repeated_metrics = ["objective_score", "macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]
    rep_mean = repeated_df[repeated_metrics].mean()
    rep_std = repeated_df[repeated_metrics].std(ddof=1)

    print("\nLATS-v2 metric-aware coordinate search complete")
    print("-" * 108)
    print(f"Segment CSV: {segment_csv}")
    print(f"Parent clips: {len(parent_ids)}")
    print(f"Segments:     {len(df)}")
    print(f"Labels:       {len(labels)}")
    print(f"Methods:      {', '.join(methods)}")
    print(f"Thresholds:   {thresholds[0]:.4f} to {thresholds[-1]:.4f} step {args.threshold_step}")
    print(f"Seeds:        {args.seeds}")
    print(f"Output dir:   {out_dir}")

    print("\nInitial start-config full-holdout evaluation:")
    for key in ["objective_score", "macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]:
        print(f"  {key:18s}: {init_metrics[key]:.6f}")

    print("\nRepeated split evaluation mean ± std:")
    for key in repeated_metrics:
        print(f"  {key:18s}: {rep_mean[key]:.6f} ± {rep_std[key]:.6f}")

    print("\nLATS-v2 final frozen config full-holdout evaluation:")
    for key in ["objective_score", "macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]:
        print(f"  {key:18s}: {final_metrics[key]:.6f}")

    print("\nFinal label-wise config:")
    for label in labels:
        rule = final_rules[label]
        row = threshold_summary_df[threshold_summary_df["label"] == label].iloc[0]
        print(
            f"  {label:28s} -> {rule.aggregation:9s} threshold={rule.threshold:.4f} "
            f"selected={int(row['selection_count'])}/{args.seeds}"
        )

    print("\nSaved files:")
    for name in [
        "lats_v2_init_full_holdout_eval.csv",
        "lats_v2_init_full_holdout_per_label.csv",
        "lats_v2_repeated_eval_summary.csv",
        "lats_v2_seed_final_configs_long.csv",
        "lats_v2_coordinate_search_log.csv",
        "lats_v2_label_selection_frequency.csv",
        "lats_v2_threshold_summary.csv",
        "lats_v2_final_frozen_config.json",
        "lats_v2_final_full_holdout_eval.csv",
        "lats_v2_final_full_holdout_per_label.csv",
        "lats_v2_final_full_holdout_parent_predictions.csv",
        "lats_v2_final_vs_init_comparison.csv",
        "lats_v2_run_config.json",
    ]:
        print(f"  {out_dir / name}")
    if args.save_candidate_log:
        print(f"  {out_dir / 'lats_v2_candidate_search_log.csv'}")


if __name__ == "__main__":
    main()
