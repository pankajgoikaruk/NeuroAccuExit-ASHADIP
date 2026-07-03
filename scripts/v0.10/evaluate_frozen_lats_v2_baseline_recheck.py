from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_score, recall_score


LATS_V2_CONFIG = {
    "Brene_Brown":               {"aggregation": "top3mean", "threshold": 0.50},
    "Eckhart_Tolle":             {"aggregation": "top2mean", "threshold": 0.50},
    "Eric_Thomas":               {"aggregation": "mean",     "threshold": 0.54},
    "Gary_Vee":                  {"aggregation": "top3mean", "threshold": 0.50},
    "Jay_Shetty":                {"aggregation": "mean",     "threshold": 0.82},
    "Nick_Vujicic":              {"aggregation": "mean",     "threshold": 0.43},
    "other_speaker_present":     {"aggregation": "top3mean", "threshold": 0.76},
    "music_present":             {"aggregation": "mean",     "threshold": 0.49},
    "audience_reaction_present": {"aggregation": "max",      "threshold": 0.68},
    "silence_present":           {"aggregation": "p75",      "threshold": 0.34},
}


def load_labels(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["labels", "label_names", "classes", "class_names"]:
            if key in obj:
                val = obj[key]
                if isinstance(val, list):
                    return val

    raise RuntimeError(f"Could not parse labels from: {path}")


def aggregate(values, method: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return 0.0

    method = method.lower().strip()

    if method == "mean":
        return float(np.mean(values))

    if method == "max":
        return float(np.max(values))

    if method == "top2mean":
        top = np.sort(values)[::-1][: min(2, len(values))]
        return float(np.mean(top))

    if method == "top3mean":
        top = np.sort(values)[::-1][: min(3, len(values))]
        return float(np.mean(top))

    if method == "p75":
        return float(np.percentile(values, 75))

    raise RuntimeError(f"Unsupported aggregation method: {method}")


def compute_metrics(y_true, y_pred, labels):
    result = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_score": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
        "parent_clips": int(y_true.shape[0]),
        "num_labels": int(y_true.shape[1]),
    }

    per_label = []
    for i, label in enumerate(labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        per_label.append({
            "label": label,
            "aggregation": LATS_V2_CONFIG[label]["aggregation"],
            "threshold": LATS_V2_CONFIG[label]["threshold"],
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
            "tn": int(((yt == 0) & (yp == 0)).sum()),
        })

    return result, pd.DataFrame(per_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-pred-csv", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--parent-id-col", default="parent_clip_id")
    parser.add_argument("--prob-prefix", default="exit3_prob_")
    args = parser.parse_args()

    segment_csv = Path(args.segment_pred_csv)
    labels_json = Path(args.labels_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(segment_csv, low_memory=False)
    labels = load_labels(labels_json)

    missing = []
    for label in labels:
        if label not in df.columns:
            missing.append(label)
        if f"{args.prob_prefix}{label}" not in df.columns:
            missing.append(f"{args.prob_prefix}{label}")

    if missing:
        raise RuntimeError("Missing columns:\n" + "\n".join(missing))

    parent_ids = sorted(df[args.parent_id_col].dropna().unique())

    parent_truth = (
        df.groupby(args.parent_id_col, as_index=True)[labels]
        .max()
        .loc[parent_ids]
        .astype(int)
    )

    parent_pred = pd.DataFrame({args.parent_id_col: parent_ids})
    parent_score = pd.DataFrame({args.parent_id_col: parent_ids})

    y_pred_cols = []

    for label in labels:
        cfg = LATS_V2_CONFIG[label]
        prob_col = f"{args.prob_prefix}{label}"

        scores = (
            df.groupby(args.parent_id_col, as_index=True)[prob_col]
            .apply(lambda x: aggregate(x.to_numpy(), cfg["aggregation"]))
            .loc[parent_ids]
            .astype(float)
        )

        pred = (scores.values >= float(cfg["threshold"])).astype(int)

        parent_score[f"score_{label}"] = scores.values
        parent_pred[f"pred_{label}"] = pred
        y_pred_cols.append(f"pred_{label}")

    y_true = parent_truth[labels].values.astype(int)
    y_pred = parent_pred[y_pred_cols].values.astype(int)

    metrics, per_label_df = compute_metrics(y_true, y_pred, labels)

    metrics["method"] = "frozen_lats_v2_baseline_recheck"
    metrics["segment_pred_csv"] = str(segment_csv)
    metrics["labels_json"] = str(labels_json)

    summary_df = pd.DataFrame([metrics])

    summary_path = out_dir / "frozen_lats_v2_baseline_recheck_summary.csv"
    per_label_path = out_dir / "frozen_lats_v2_baseline_recheck_per_label.csv"
    parent_pred_path = out_dir / "frozen_lats_v2_baseline_recheck_parent_predictions.csv"
    config_path = out_dir / "frozen_lats_v2_config_used.json"

    summary_df.to_csv(summary_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)

    parent_out = pd.DataFrame({args.parent_id_col: parent_ids})
    for label in labels:
        parent_out[f"true_{label}"] = parent_truth[label].values
        parent_out[f"score_{label}"] = parent_score[f"score_{label}"].values
        parent_out[f"pred_{label}"] = parent_pred[f"pred_{label}"].values
    parent_out.to_csv(parent_pred_path, index=False)

    config_payload = {
        "name": "frozen_lats_v2_baseline_recheck",
        "config": LATS_V2_CONFIG,
        "metrics": metrics,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print("\nFrozen LATS-v2 baseline recheck complete")
    print("-" * 90)
    print(summary_df[[
        "macro_f1",
        "micro_f1",
        "samples_f1",
        "exact_match",
        "hamming_loss",
        "avg_true_labels",
        "avg_pred_labels",
        "parent_clips",
    ]].to_string(index=False))

    print("\nExpected previous baseline:")
    print("Macro-F1   = 0.8673")
    print("Micro-F1   = 0.9458")
    print("Samples-F1 = 0.9517")
    print("Exact      = 0.8604")
    print("Hamming    = 0.0158")

    print("\nSaved:")
    print(summary_path)
    print(per_label_path)
    print(parent_pred_path)
    print(config_path)


if __name__ == "__main__":
    main()
