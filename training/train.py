# training/train.py

from __future__ import annotations

import os
import sys
import argparse
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy

from utils.config import parse_args_with_config, ensure_dirs, save_config
from utils.logging import make_run_dir, save_json
from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net


def set_global_seed(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _parse_tap_blocks(value) -> Optional[tuple]:
    """
    Accept:
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


def _pad_or_trim(seq: Sequence[float], target_len: int, pad_value: Optional[float] = None) -> List[float]:
    """
    Pad/trim numeric sequence to target_len.
    If pad_value is None, pad with the last element (or 1.0 if empty).
    """
    vals = [float(x) for x in seq]
    if len(vals) >= target_len:
        return vals[:target_len]

    if pad_value is None:
        pad_value = vals[-1] if len(vals) > 0 else 1.0

    vals = vals + [float(pad_value)] * (target_len - len(vals))
    return vals


def train_one_epoch(model, dl, opt, device, loss_w):
    model.train()
    loss_sum, n = 0.0, 0
    correct = None

    for x, y in dl:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)

        if correct is None:
            correct = [0 for _ in range(len(logits))]

        losses = [cross_entropy(lg, y) for lg in logits]
        weights = _pad_or_trim(loss_w, len(losses))
        loss = sum(w * l for w, l in zip(weights, losses))

        loss.backward()
        opt.step()

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs

        for k, lg in enumerate(logits):
            pred = lg.argmax(1)
            correct[k] += int((pred == y).sum())

    if correct is None:
        correct = []

    acc = [c / max(n, 1) for c in correct]
    return loss_sum / max(n, 1), acc


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = None
    n = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if correct is None:
            correct = [0 for _ in range(len(logits))]

        n += x.size(0)
        for k, lg in enumerate(logits):
            pred = lg.argmax(1)
            correct[k] += int((pred == y).sum())

    if correct is None:
        correct = []

    acc = [c / max(n, 1) for c in correct]
    return acc


def _parse_extra_args():
    """
    Parse optional args without breaking parse_args_with_config().
    We remove our extra args from sys.argv first.
    """
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument("--run_dir", type=str, default=None,
                   help="Explicit run directory to write outputs.")
    p.add_argument("--device", type=str, default=None,
                   help="Force device: cpu | cuda (default: auto).")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Override cache directory containing segments.csv + features/.")
    p.add_argument("--segment_sec", type=float, default=None,
                   help="Optional: record segment_sec in effective config.")
    p.add_argument("--hop_sec", type=float, default=None,
                   help="Optional: record hop_sec in effective config.")
    p.add_argument("--variant", type=str, default=None,
                   help="Optional: record variant name in effective config.")

    # Generic K-exit control
    p.add_argument("--tap_blocks", type=str, default=None,
                   help='Comma-separated tap blocks. Example: "1,3" or "1,2,3,4".')

    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    extra = _parse_extra_args()
    cfg = parse_args_with_config()

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    # Device
    if extra.device is not None:
        device = extra.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    paths = (cfg.get("paths") or {})
    runs_root = paths.get("runs_root", "runs")
    cache_root = paths.get("cache_root", "data_cache")

    if extra.cache_dir:
        cache_root = extra.cache_dir

    ensure_dirs(runs_root)

    # Run dir
    if extra.run_dir:
        run_dir = extra.run_dir
        ensure_dirs(run_dir)
    else:
        run_dir = make_run_dir(runs_root)

    ckpt_dir = os.path.join(run_dir, "ckpt")
    ensure_dirs(ckpt_dir)

    # Save effective config
    cfg.setdefault("paths", {})
    cfg["paths"]["runs_root"] = runs_root
    cfg["paths"]["cache_root"] = cache_root

    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = device
    if extra.variant is not None:
        cfg["runtime"]["variant"] = extra.variant

    if extra.segment_sec is not None or extra.hop_sec is not None:
        cfg.setdefault("audio", {})
        if extra.segment_sec is not None:
            cfg["audio"]["segment_sec"] = float(extra.segment_sec)
        if extra.hop_sec is not None:
            cfg["audio"]["segment_hop"] = float(extra.hop_sec)

    # Read tap blocks from CLI first, then config, else default to backward-compatible 3-exit
    model_cfg = (cfg.get("model") or {})
    tap_blocks = (
        _parse_tap_blocks(extra.tap_blocks)
        or _parse_tap_blocks(model_cfg.get("tap_blocks"))
        or (1, 3)
    )
    cfg.setdefault("model", {})
    cfg["model"]["tap_blocks"] = list(tap_blocks)

    save_config(cfg, os.path.join(run_dir, "config_used.yaml"))

    # Data
    seg_csv = os.path.join(cache_root, "segments.csv")
    feat_root = os.path.join(cache_root, "features")

    tr = (cfg.get("train") or {})
    bs = int(tr.get("batch_size", 64))
    nw = int(tr.get("num_workers", 4))
    lr = float(tr.get("lr", 1e-3))
    wd = float(tr.get("weight_decay", 0.0))
    loss_w = tr.get("loss_weights", [0.3, 0.3, 1.0])
    epochs = int(tr.get("epochs", 40))

    dl_tr, dl_va, dl_te, label2id = make_loaders(
        seg_csv, feat_root, bs, nw, seed=seed
    )

    # Model
    n_mels = int((cfg.get("features") or {}).get("n_mels", 64))

    # Keep C-class generic: prefer dataset class count
    num_classes_data = len(label2id)
    num_classes_cfg = int((cfg.get("model") or {}).get("num_classes", num_classes_data))
    if num_classes_cfg != num_classes_data:
        print(
            f"[WARN] config num_classes={num_classes_cfg} but dataset has "
            f"{num_classes_data}. Using dataset value."
        )
    num_classes = num_classes_data

    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=(cfg.get("model") or {}),
    ).to(device)

    num_exits = model.num_exits
    loss_w = _pad_or_trim(loss_w, num_exits)

    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    metrics = {
        "meta": {
            "num_exits": num_exits,
            "tap_blocks": list(tap_blocks),
            "tap_dims": list(model.tap_dims) if hasattr(model, "tap_dims") else [],
            "final_dim": int(model.final_dim) if hasattr(model, "final_dim") else -1,
            "num_classes": num_classes,
            "loss_weights_used": loss_w,
            "exit_hint": (cfg.get("model") or {}).get("exit_hint", {}),
        },
        "train": [],
        "val": [],
    }

    best = -1.0

    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, device, loss_w)
        va_acc = evaluate(model, dl_va, device)

        metrics["train"].append({
            "epoch": ep + 1,
            "loss": tr_loss,
            "acc": tr_acc,
        })
        metrics["val"].append({
            "epoch": ep + 1,
            "acc": va_acc,
        })

        print(f"Epoch {ep + 1}: loss={tr_loss:.4f}, acc@exits={va_acc}")

        if len(va_acc) > 0 and va_acc[-1] > best:
            best = va_acc[-1]
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()