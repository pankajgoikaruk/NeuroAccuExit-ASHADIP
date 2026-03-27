# scripts/analyse_run.py

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, roc_curve, auc

from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net, load_run_model_cfg


def load_json_safepath(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def plot_training_curves(metrics_path: Path, plots_dir: Path):
    """
    Plot training loss and validation accuracy per exit over epochs
    using metrics.json produced by training/train.py (K-exit compatible).
    """
    metrics = load_json_safepath(metrics_path, {})
    if not metrics:
        print(f"[analyse_run] No metrics.json found at {metrics_path}, skipping training curves.")
        return

    train_hist = metrics.get("train", [])
    val_hist = metrics.get("val", [])
    if not train_hist or not val_hist:
        print("[analyse_run] metrics.json has no train/val history, skipping training curves.")
        return

    # --- Train loss ---
    epochs = [e["epoch"] for e in train_hist]
    losses = [e["loss"] for e in train_hist]

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs epoch")
    plt.grid(True)
    out = plots_dir / "train_loss.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyse_run] Saved {out}")

    # --- Validation accuracy per exit (dynamic K) ---
    epochs_val = [e["epoch"] for e in val_hist]
    if not val_hist or "acc" not in val_hist[0]:
        print("[analyse_run] No per-exit val acc found, skipping val acc plot.")
        return

    K = len(val_hist[0]["acc"])
    plt.figure()
    for k in range(K):
        acc_k = [e["acc"][k] for e in val_hist]
        plt.plot(epochs_val, acc_k, marker="o", label=f"exit{k+1}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy per exit vs epoch")
    plt.legend()
    plt.grid(True)
    out = plots_dir / "val_acc_exits.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyse_run] Saved {out}")


@torch.no_grad()
def collect_test_predictions(run_dir: Path, segments_csv: Path, features_root: Path, tap_blocks, n_mels: int, device: str = None):
    """
    Reload trained ExitNet model from ckpt/best.pt and run on TEST set.

    Returns:
      y_true        (N,)
      y_pred_exits  list length K, each (N,)
      y_prob_exits  list length K, each (N,C)
      label2id
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, dl_te, label2id = make_loaders(str(segments_csv), str(features_root), batch_size=64, num_workers=4)
    num_classes = len(label2id)

    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_path}")

    model_cfg = load_run_model_cfg(str(run_dir))
    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    K = model.num_exits

    y_true_list = []
    y_pred_exits = [[] for _ in range(K)]
    y_prob_exits = [[] for _ in range(K)]

    for x, y in dl_te:
        x = x.to(device)
        y_true_list.extend(y.numpy().tolist())

        logits_list = model(x)  # length K
        probs_list = [softmax(lg, dim=1).cpu().numpy() for lg in logits_list]

        for k in range(K):
            preds = np.argmax(probs_list[k], axis=1)
            y_pred_exits[k].extend(preds.tolist())
            y_prob_exits[k].append(probs_list[k])

    y_true = np.array(y_true_list, dtype=np.int64)
    y_prob_exits = [np.concatenate(chunks, axis=0) if len(chunks) > 0 else None for chunks in y_prob_exits]
    y_pred_exits = [np.array(preds, dtype=np.int64) for preds in y_pred_exits]
    return y_true, y_pred_exits, y_prob_exits, label2id


def compute_and_plot_confusion_matrices(y_true, y_pred_exits, label2id, plots_dir: Path, out_json: Path):
    id2label = {v: k for k, v in label2id.items()}
    labels_sorted = [id2label[i] for i in range(len(id2label))]

    cm_info = {}
    for exit_idx, y_pred in enumerate(y_pred_exits, start=1):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(id2label))))
        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = cm.astype(float) / np.maximum(row_sums, 1e-12)

        plt.figure(figsize=(5, 4))
        plt.imshow(cm_norm, interpolation="nearest")
        plt.title(f"Confusion matrix – exit{exit_idx}")
        plt.colorbar()
        tick_marks = np.arange(len(labels_sorted))
        plt.xticks(tick_marks, labels_sorted, rotation=45, ha="right")
        plt.yticks(tick_marks, labels_sorted)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        out_png = plots_dir / f"cm_exit{exit_idx}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[analyse_run] Saved {out_png}")

        cm_info[f"exit{exit_idx}"] = {
            "labels": labels_sorted,
            "counts": cm.tolist(),
            "row_normalised": cm_norm.tolist(),
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cm_info, f, indent=2)
    print(f"[analyse_run] Saved confusion matrices JSON -> {out_json}")
    return cm_info


def compute_and_plot_roc(y_true, y_prob_exits, plots_dir: Path, out_json: Path):
    """
    ROC/AUC for binary only (num_classes == 2). For multiclass, returns {}.
    """
    if not y_prob_exits or y_prob_exits[0] is None:
        print("[analyse_run] No probabilities for exit1, skipping ROC/AUC.")
        return {}

    num_classes = y_prob_exits[0].shape[1]
    if num_classes != 2:
        print(f"[analyse_run] ROC/AUC only implemented for binary tasks, got num_classes={num_classes}. Skipping.")
        return {}

    roc_info = {}
    plt.figure()

    for exit_idx, probs in enumerate(y_prob_exits, start=1):
        if probs is None:
            continue
        y_score = probs[:, 1]  # class 1 as positive
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_info[f"exit{exit_idx}"] = {
            "auc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        plt.plot(fpr, tpr, label=f"exit{exit_idx} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves per exit")
    plt.legend()
    plt.grid(True)

    out_png = plots_dir / "roc_exits.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[analyse_run] Saved {out_png}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(roc_info, f, indent=2)
    print(f"[analyse_run] Saved ROC JSON -> {out_json}")
    return roc_info


def build_analysis_summary(run_dir: Path, cm_info: dict, roc_info: dict, label_names):
    report = load_json_safepath(run_dir / "report.json", {})
    summary = load_json_safepath(run_dir / "summary.json", {})

    out = {
        "run_id": summary.get("run_id", run_dir.name),
        "classification_per_exit": report,
        "policy_summary": summary.get("policy_summary", {}),
        "confusion_matrices": cm_info,
        "roc_auc": roc_info,
        "label_names": label_names,
    }

    out_path = run_dir / "analysis_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[analyse_run] Saved consolidated analysis JSON -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a run directory, e.g. runs/20251113_142831")
    ap.add_argument("--segments_csv", default="data_cache/segments.csv", help="Path to segments.csv used for this run")
    ap.add_argument("--features_root", default="data_cache/features", help="Root directory for .npy features used for this run")

    # Step 0 (K-exit)
    ap.add_argument("--tap_blocks", default="1,3", help="Comma list like 1,2,3,4. Default 1,3 (=3 exits).")
    ap.add_argument("--n_mels", type=int, default=64)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    tap_blocks = tuple(int(x) for x in str(args.tap_blocks).split(",") if str(x).strip())

    # 1) Training curves
    metrics_path = run_dir / "metrics.json"
    plot_training_curves(metrics_path, plots_dir)

    # 2) Test predictions + confusion + ROC
    y_true, y_pred_exits, y_prob_exits, label2id = collect_test_predictions(
        run_dir,
        Path(args.segments_csv),
        Path(args.features_root),
        tap_blocks=tap_blocks,
        n_mels=int(args.n_mels),
    )

    id2label = {v: k for k, v in label2id.items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    cm_json_path = run_dir / "confusion_matrices.json"
    cm_info = compute_and_plot_confusion_matrices(y_true, y_pred_exits, label2id, plots_dir, cm_json_path)

    roc_json_path = run_dir / "roc_curves.json"
    roc_info = compute_and_plot_roc(y_true, y_prob_exits, plots_dir, roc_json_path)

    # 3) consolidated JSON
    build_analysis_summary(run_dir, cm_info, roc_info, label_names)


if __name__ == "__main__":
    main()