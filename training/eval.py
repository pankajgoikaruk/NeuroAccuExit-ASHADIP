# training/eval.py

from __future__ import annotations

import argparse
import inspect
import json
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


def _parse_tap_blocks(value):
    """
    Parse tap blocks from:
      - None
      - "1,2,3,4"
      - [1,2,3,4]
      - (1,2,3,4)
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    value = str(value).strip()
    if value == "":
        return None

    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _build_backbone(n_mels: int, tap_blocks=None):
    """
    Build TinyAudioCNN in a way that is compatible with both:
    - old v0.1.6 backbone: TinyAudioCNN(n_mels=64)
    - new generic backbone: TinyAudioCNN(n_mels=64, tap_blocks=(1,2,3,4))
    """
    sig = inspect.signature(TinyAudioCNN.__init__)
    kwargs = {}

    if "n_mels" in sig.parameters:
        kwargs["n_mels"] = int(n_mels)

    if tap_blocks is not None and "tap_blocks" in sig.parameters:
        kwargs["tap_blocks"] = tap_blocks

    return TinyAudioCNN(**kwargs)


@torch.no_grad()
def main(
    run_dir,
    segments_csv,
    features_root,
    num_classes=2,
    n_mels=64,
    tap_blocks=None,
    batch_size=64,
    num_workers=4,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dl_tr, dl_va, dl_te, label2id = make_loaders(
        segments_csv, features_root, batch_size, num_workers
    )

    tap_blocks = _parse_tap_blocks(tap_blocks)
    backbone = _build_backbone(n_mels=n_mels, tap_blocks=tap_blocks)

    # Compatible with both old and new backbone code
    model_kwargs = {
        "backbone": backbone,
        "num_classes": int(num_classes),
    }

    if not hasattr(backbone, "tap_dims"):
        model_kwargs["tap_dims"] = (16, 32)
    if not hasattr(backbone, "final_dim"):
        model_kwargs["final_dim"] = 64

    model = ExitNet(**model_kwargs).to(device)

    ckpt_path = os.path.join(run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = None
    num_exits = None

    for x, y in dl_te:
        x = x.to(device)
        logits = model(x)

        if y_pred is None:
            num_exits = len(logits)
            y_pred = [[] for _ in range(num_exits)]

        for k, lg in enumerate(logits):
            preds = torch.argmax(lg, dim=1).cpu().numpy().tolist()
            y_pred[k].extend(preds)

        y_true.extend(y.numpy().tolist())

    if y_pred is None:
        raise RuntimeError("Test loader is empty. No predictions were generated.")

    reports = {}
    confusion_matrices = {}

    for k in range(num_exits):
        exit_name = f"exit{k + 1}"
        reports[exit_name] = classification_report(
            y_true,
            y_pred[k],
            output_dict=True,
            zero_division=0,
        )
        confusion_matrices[exit_name] = confusion_matrix(
            y_true,
            y_pred[k],
            labels=list(range(int(num_classes))),
        ).tolist()

    id2label = {int(v): str(k) for k, v in label2id.items()}

    out = {
        "num_exits": num_exits,
        "num_classes": int(num_classes),
        "tap_blocks": list(tap_blocks) if tap_blocks is not None else None,
        "label2id": label2id,
        "id2label": id2label,
        "reports": reports,
        "confusion_matrices": confusion_matrices,
    }

    out_path = os.path.join(run_dir, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {out_path}")
    for k in range(num_exits):
        exit_name = f"exit{k + 1}"
        acc = reports[exit_name]["accuracy"]
        print(f"{exit_name} accuracy: {acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument(
        "--tap_blocks",
        type=str,
        default=None,
        help='Example: "1,2,3,4" for 5 exits total, or "1,3" for 3 exits total.',
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    main(
        run_dir=args.run_dir,
        segments_csv=args.segments_csv,
        features_root=args.features_root,
        num_classes=args.num_classes,
        n_mels=args.n_mels,
        tap_blocks=args.tap_blocks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )