# training/train.py

import os
import sys
import argparse

import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy

from utils.config import parse_args_with_config, ensure_dirs, save_config
from utils.logging import make_run_dir, save_json
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


def train_one_epoch(model, dl, opt, device, loss_w):
    model.train()
    loss_sum, n = 0.0, 0
    correct = [0, 0, 0]
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        losses = [cross_entropy(lg, y) for lg in logits]
        loss = sum(w * l for w, l in zip(loss_w, losses))
        loss.backward()
        opt.step()

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs

        for k, lg in enumerate(logits):
            pred = lg.argmax(1)
            correct[k] += int((pred == y).sum())

    acc = [c / max(n, 1) for c in correct]
    return loss_sum / max(n, 1), acc


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = [0, 0, 0]
    n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        n += x.size(0)
        for k, lg in enumerate(logits):
            pred = lg.argmax(1)
            correct[k] += int((pred == y).sum())
    acc = [c / max(n, 1) for c in correct]
    return acc


def _parse_extra_args():
    """
    Parse our optional args without breaking parse_args_with_config().
    We remove our extra args from sys.argv, then parse_args_with_config() handles --config.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run_dir", type=str, default=None,
                   help="Explicit run directory to write outputs.")
    p.add_argument("--device", type=str, default=None,
                   help="Force device: cpu | cuda (default: auto).")
    p.add_argument("--segment_sec", type=float, default=None,
                   help="Optional: record segment_sec in effective config.")
    p.add_argument("--hop_sec", type=float, default=None,
                   help="Optional: record hop_sec in effective config.")
    p.add_argument("--variant", type=str, default=None,
                   help="Optional: record variant name in effective config.")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    extra = _parse_extra_args()
    cfg = parse_args_with_config()

    # Device selection
    if extra.device is not None:
        device = extra.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = (cfg.get("paths") or {})
    runs_root = paths.get("runs_root", "runs")
    cache_root = paths.get("cache_root", "data_cache")
    ensure_dirs(runs_root)

    # Choose run_dir
    if extra.run_dir:
        run_dir = extra.run_dir
        ensure_dirs(run_dir)
    else:
        run_dir = make_run_dir(runs_root)

    # Ensure ckpt directory exists
    ckpt_dir = os.path.join(run_dir, "ckpt")
    ensure_dirs(ckpt_dir)

    # --------- Build & save EFFECTIVE CONFIG used for this run ---------
    cfg.setdefault("paths", {})
    cfg["paths"]["runs_root"] = runs_root
    cfg["paths"]["cache_root"] = cache_root

    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = device
    if extra.variant is not None:
        cfg["runtime"]["variant"] = extra.variant

    # Record segment/hop if provided (helps reproducibility)
    if extra.segment_sec is not None or extra.hop_sec is not None:
        cfg.setdefault("audio", {})
        if extra.segment_sec is not None:
            cfg["audio"]["segment_sec"] = float(extra.segment_sec)
        if extra.hop_sec is not None:
            cfg["audio"]["segment_hop"] = float(extra.hop_sec)

    save_config(cfg, os.path.join(run_dir, "config_used.yaml"))
    # -------------------------------------------------------------------

    seg_csv = os.path.join(cache_root, "segments.csv")
    feat_root = os.path.join(cache_root, "features")

    tr = (cfg.get("train") or {})
    bs = int(tr.get("batch_size", 64))
    nw = int(tr.get("num_workers", 4))
    lr = float(tr.get("lr", 1e-3))
    wd = float(tr.get("weight_decay", 0.0))
    loss_w = tr.get("loss_weights", [0.3, 0.3, 1.0])
    epochs = int(tr.get("epochs", 40))

    dl_tr, dl_va, dl_te, label2id = make_loaders(seg_csv, feat_root, bs, nw)

    n_mels = int((cfg.get("features") or {}).get("n_mels", 64))
    num_classes = int((cfg.get("model") or {}).get("num_classes", 2))

    backbone = TinyAudioCNN(n_mels=n_mels)
    model = ExitNet(backbone, tap_dims=(16, 32), final_dim=64, num_classes=num_classes).to(device)

    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    metrics = {"train": [], "val": []}
    best = -1.0

    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, device, loss_w)
        va_acc = evaluate(model, dl_va, device)

        metrics["train"].append({"epoch": ep + 1, "loss": tr_loss, "acc": tr_acc})
        metrics["val"].append({"epoch": ep + 1, "acc": va_acc})

        print(f"Epoch {ep + 1}: loss={tr_loss:.4f}, acc@exits={va_acc}")

        if va_acc[-1] > best:
            best = va_acc[-1]
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
