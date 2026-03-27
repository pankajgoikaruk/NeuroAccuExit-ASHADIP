# training/eval.py
from __future__ import annotations

import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net, load_run_model_cfg


@torch.no_grad()
def main(run_dir: str, segments_csv: str, features_root: str, tap_blocks: tuple[int, ...], n_mels: int, batch_size: int, num_workers: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loaders (we evaluate on TEST)
    _, _, dl_te, label2id = make_loaders(segments_csv, features_root, batch_size, num_workers)
    num_classes = len(label2id)

    model_cfg = load_run_model_cfg(run_dir)
    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    ckpt = os.path.join(run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    K = model.num_exits
    y_true: list[int] = []
    y_pred: list[list[int]] = [[] for _ in range(K)]

    for x, y in dl_te:
        x = x.to(device)
        logits_list = model(x)  # length K
        for k in range(K):
            y_pred[k].extend(torch.argmax(logits_list[k], dim=1).cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())

    # Reports per exit
    reports = {f"exit{k+1}": classification_report(y_true, y_pred[k], output_dict=True) for k in range(K)}
    cms = {f"exit{k+1}": confusion_matrix(y_true, y_pred[k]).tolist() for k in range(K)}

    out = {
        "K": K,
        "tap_blocks": list(tap_blocks),
        "n_mels": int(n_mels),
        "num_classes": int(num_classes),
        "label2id": {str(k): int(v) for k, v in label2id.items()},
        "reports": reports,
        "confusion_matrices": cms,
    }

    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--tap_blocks", default="1,3", help="Comma list like 1,2,3,4. Default 1,3 (=3 exits).")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    tap_blocks = tuple(int(x) for x in args.tap_blocks.split(",") if x.strip())
    main(args.run_dir, args.segments_csv, args.features_root, tap_blocks, args.n_mels, args.batch_size, args.num_workers)