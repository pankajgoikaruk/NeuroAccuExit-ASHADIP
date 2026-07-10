#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LATS-v2 metric-aware coordinate re-optimization from segment probabilities.

Input is the parent/segment probability CSV created by
scripts/evaluate_tata_final_holdout_parent_level.py, e.g.
parent_eval_segment_probs_fixed_0p5_mean.csv.

This script is intended for controlled v0.10 diagnostics:
- Start with per-label best aggregation/threshold rules.
- Coordinate-search label rules to improve a global multi-label objective.
- Save summary, per-label metrics, selected rules, parent predictions, and history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)


def load_labels(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for key in ("labels", "label_names", "classes", "class_names"):
            if key in data and isinstance(data[key], list):
                return [str(x) for x in data[key]]
        if all(isinstance(v, int) for v in data.values()):
            return [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
    raise ValueError(f"Could not infer labels from {path}")


def topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = min(k, values.size)
    return float(np.mean(np.sort(values)[-k:]))


def aggregate_group(values: np.ndarray, method: str) -> float:
    values = np.asarray(values, dtype=float)
    if method == "mean":
        return float(np.mean(values))
    if method == "max":
        return float(np.max(values))
    if method == "top2mean":
        return topk_mean(values, 2)
    if method == "top3mean":
        return topk_mean(values, 3)
    if method == "p75":
        return float(np.percentile(values, 75))
    if method == "p90":
        return float(np.percentile(values, 90))
    raise ValueError(f"Unknown aggregation method: {method}")


def build_parent_tables(
    df: pd.DataFrame,
    labels: List[str],
    parent_col: str,
    prob_prefix: str,
    methods: List[str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    missing_labels = [x for x in labels if x not in df.columns]
    missing_probs = [prob_prefix + x for x in labels if prob_prefix + x not in df.columns]
    if missing_labels or missing_probs:
        raise ValueError(
            "Missing columns:\n"
            f"labels={missing_labels}\n"
            f"probability_columns={missing_probs}"
        )

    y_true = df.groupby(parent_col, sort=True)[labels].max().astype(int)

    agg_tables: Dict[str, pd.DataFrame] = {}
    for method in methods:
        out = {}
        for label in labels:
            col = prob_prefix + label
            out[label] = df.groupby(parent_col, sort=True)[col].apply(
                lambda s, m=method: aggregate_group(s.to_numpy(dtype=float), m)
            )
        agg_tables[method] = pd.DataFrame(out).loc[y_true.index]

    return y_true, agg_tables


def predict_from_rules(
    agg_tables: Dict[str, pd.DataFrame],
    labels: List[str],
    rules: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    pred = pd.DataFrame(index=next(iter(agg_tables.values())).index)
    for label in labels:
        method = rules[label]["aggregation"]
        threshold = float(rules[label]["threshold"])
        pred[label] = (agg_tables[method][label] >= threshold).astype(int)
    return pred[labels]


def metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
    yt = y_true.to_numpy(dtype=int)
    yp = y_pred.to_numpy(dtype=int)
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(yt, yp, average="samples", zero_division=0)),
        "exact_match": float(accuracy_score(yt, yp)),
        "hamming_loss": float(hamming_loss(yt, yp)),
        "avg_true_labels": float(yt.sum(axis=1).mean()),
        "avg_pred_labels": float(yp.sum(axis=1).mean()),
        "parent_clips": int(yt.shape[0]),
    }


def objective_score(m: Dict[str, float], objective: str) -> float:
    if objective == "macro_priority":
        # Pos-weight experiment primarily asks whether rare/weak labels improve.
        return (
            0.50 * m["macro_f1"]
            + 0.15 * m["micro_f1"]
            + 0.15 * m["samples_f1"]
            + 0.15 * m["exact_match"]
            - 0.05 * m["hamming_loss"]
        )
    if objective == "global_consistency":
        return (
            0.20 * m["macro_f1"]
            + 0.25 * m["micro_f1"]
            + 0.25 * m["samples_f1"]
            + 0.25 * m["exact_match"]
            - 0.05 * m["hamming_loss"]
        )
    if objective == "macro_only":
        return m["macro_f1"]
    raise ValueError(f"Unknown objective: {objective}")


def per_label_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    rows = []
    yt = y_true.to_numpy(dtype=int)
    yp = y_pred.to_numpy(dtype=int)
    p, r, f, support = precision_recall_fscore_support(
        yt, yp, average=None, zero_division=0
    )
    for i, label in enumerate(labels):
        rows.append(
            {
                "label": label,
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
                "predicted_positive": int(yp[:, i].sum()),
            }
        )
    return pd.DataFrame(rows)


def make_thresholds(tmin: float, tmax: float, step: float) -> List[float]:
    vals = []
    x = tmin
    while x <= tmax + 1e-12:
        vals.append(round(float(x), 6))
        x += step
    return vals


def initial_labelwise_rules(
    y_true: pd.DataFrame,
    agg_tables: Dict[str, pd.DataFrame],
    labels: List[str],
    methods: List[str],
    thresholds: List[float],
) -> Dict[str, Dict[str, float]]:
    rules = {}
    for label in labels:
        best = None
        y = y_true[label].to_numpy(dtype=int)
        for method in methods:
            scores = agg_tables[method][label].to_numpy(dtype=float)
            for threshold in thresholds:
                pred = (scores >= threshold).astype(int)
                f1 = f1_score(y, pred, zero_division=0)
                # tie-breaker: prefer fewer positives, then simpler lower threshold stability
                pred_count = int(pred.sum())
                key = (float(f1), -abs(pred_count - int(y.sum())), -threshold)
                if best is None or key > best[0]:
                    best = (key, method, threshold, f1, pred_count)
        assert best is not None
        rules[label] = {"aggregation": best[1], "threshold": float(best[2])}
    return rules


def coordinate_search(
    y_true: pd.DataFrame,
    agg_tables: Dict[str, pd.DataFrame],
    labels: List[str],
    methods: List[str],
    thresholds: List[float],
    objective: str,
    max_iter: int,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    rules = initial_labelwise_rules(y_true, agg_tables, labels, methods, thresholds)
    pred = predict_from_rules(agg_tables, labels, rules)
    cur_metrics = metrics(y_true, pred)
    cur_score = objective_score(cur_metrics, objective)

    history = [{"iteration": 0, "label": "INIT", "score": cur_score, **cur_metrics}]

    for iteration in range(1, max_iter + 1):
        changed = False
        for label in labels:
            best_rules = None
            best_metrics = None
            best_score = cur_score
            original = dict(rules[label])

            for method in methods:
                scores = agg_tables[method][label].to_numpy(dtype=float)
                for threshold in thresholds:
                    trial = dict(rules)
                    trial[label] = {"aggregation": method, "threshold": float(threshold)}
                    # Efficient enough for current holdout size.
                    trial_pred = predict_from_rules(agg_tables, labels, trial)
                    m = metrics(y_true, trial_pred)
                    s = objective_score(m, objective)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_rules = trial[label]
                        best_metrics = m

            if best_rules is not None:
                rules[label] = best_rules
                cur_score = best_score
                cur_metrics = best_metrics or cur_metrics
                changed = True
                history.append(
                    {
                        "iteration": iteration,
                        "label": label,
                        "aggregation": rules[label]["aggregation"],
                        "threshold": rules[label]["threshold"],
                        "score": cur_score,
                        **cur_metrics,
                    }
                )
            else:
                rules[label] = original

        if not changed:
            history.append({"iteration": iteration, "label": "NO_CHANGE", "score": cur_score, **cur_metrics})
            break

    return rules, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-pred-csv", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--prob-prefix", default="exit3_prob_")
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--aggregation-methods", default="mean,max,top2mean,top3mean,p75,p90")
    parser.add_argument("--objective", choices=["macro_priority", "global_consistency", "macro_only"], default="macro_priority")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--model-name", default="unknown")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(Path(args.labels_json))
    methods = [x.strip() for x in args.aggregation_methods.split(",") if x.strip()]
    thresholds = make_thresholds(args.threshold_min, args.threshold_max, args.threshold_step)

    df = pd.read_csv(args.segment_pred_csv)
    if args.parent_id_col not in df.columns:
        raise ValueError(f"Missing parent id column: {args.parent_id_col}")

    y_true, agg_tables = build_parent_tables(
        df=df,
        labels=labels,
        parent_col=args.parent_id_col,
        prob_prefix=args.prob_prefix,
        methods=methods,
    )

    rules, history = coordinate_search(
        y_true=y_true,
        agg_tables=agg_tables,
        labels=labels,
        methods=methods,
        thresholds=thresholds,
        objective=args.objective,
        max_iter=args.max_iter,
    )

    y_pred = predict_from_rules(agg_tables, labels, rules)
    summary = metrics(y_true, y_pred)
    summary["model_name"] = args.model_name
    summary["objective"] = args.objective
    summary["segment_pred_csv"] = args.segment_pred_csv

    pd.DataFrame([summary]).to_csv(out_dir / "lats_v2_coordinate_reoptimized_summary.csv", index=False)
    per_label_metrics(y_true, y_pred, labels).to_csv(
        out_dir / "lats_v2_coordinate_reoptimized_per_label.csv", index=False
    )
    pd.DataFrame(
        [{"label": k, **v} for k, v in rules.items()]
    ).to_csv(out_dir / "lats_v2_selected_rules.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "lats_v2_coordinate_history.csv", index=False)

    parent_pred = pd.concat(
        [
            y_true.add_prefix("true_"),
            y_pred.add_prefix("pred_"),
        ],
        axis=1,
    )
    parent_pred.insert(0, args.parent_id_col, y_true.index)
    parent_pred.to_csv(out_dir / "lats_v2_parent_predictions.csv", index=False)

    (out_dir / "lats_v2_selected_rules.json").write_text(
        json.dumps(rules, indent=2), encoding="utf-8"
    )
    (out_dir / "lats_v2_run_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    print("LATS-v2 coordinate re-optimization complete")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
