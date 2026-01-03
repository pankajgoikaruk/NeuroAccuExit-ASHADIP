# training/train.py

import os
import sys
import argparse
import random
import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

from utils.config import parse_args_with_config, ensure_dirs, save_config
from utils.logging import make_run_dir, save_json
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


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


def _spec_augment(x, freq_mask=8, time_mask=12, num_masks=2):
    """
    Simple SpecAugment for log-mel features.
    x: (B, n_mels, frames) or (B, 1, n_mels, frames)
    Masks are applied with zeros.
    """
    if x.dim() == 4:
        # (B,1,M,T) -> (B,M,T)
        x2 = x[:, 0]
        add_channel_back = True
    else:
        x2 = x
        add_channel_back = False

    B, M, T = x2.shape
    out = x2.clone()

    for _ in range(num_masks):
        # Frequency mask
        if freq_mask > 0 and M > 1:
            f = random.randint(0, min(freq_mask, M - 1))
            f0 = random.randint(0, max(M - f, 1) - 1) if f > 0 else 0
            if f > 0:
                out[:, f0:f0 + f, :] = 0.0

        # Time mask
        if time_mask > 0 and T > 1:
            t = random.randint(0, min(time_mask, T - 1))
            t0 = random.randint(0, max(T - t, 1) - 1) if t > 0 else 0
            if t > 0:
                out[:, :, t0:t0 + t] = 0.0

    if add_channel_back:
        out = out.unsqueeze(1)
    return out


def _kd_kl(student_logits, teacher_logits, T=2.0):
    """
    KL( softmax(teacher/T) || softmax(student/T) ) * T^2
    Returns scalar.
    """
    T = float(T)
    # teacher detached
    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    # kl_div expects input=log_probs, target=probs
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
    return kl * (T * T)


def train_one_epoch(model, dl, opt, device, loss_w,
                    kd_enable=False, kd_alpha=0.5, kd_temp=2.0, kd_weights=(1.0, 1.0),
                    specaug_enable=False, specaug_cfg=None):
    model.train()
    loss_sum, n = 0.0, 0
    correct = [0, 0, 0]

    if specaug_cfg is None:
        specaug_cfg = {}

    for x, y in dl:
        x, y = x.to(device), y.to(device)

        # Optional SpecAugment (train only)
        if specaug_enable:
            fm = int(specaug_cfg.get("freq_mask", 8))
            tm = int(specaug_cfg.get("time_mask", 12))
            nm = int(specaug_cfg.get("num_masks", 2))
            x = _spec_augment(x, freq_mask=fm, time_mask=tm, num_masks=nm)

        opt.zero_grad()
        logits = model(x)  # list of 3 tensors (B,C)

        # CE losses for each exit
        ce_losses = [cross_entropy(lg, y) for lg in logits]
        loss = sum(float(w) * l for w, l in zip(loss_w, ce_losses))

        # Optional KD: teacher=exit3, students=exit1/exit2
        if kd_enable:
            teacher = logits[-1].detach()
            kd1 = _kd_kl(logits[0], teacher, T=kd_temp)
            kd2 = _kd_kl(logits[1], teacher, T=kd_temp)
            loss = loss + float(kd_alpha) * (float(kd_weights[0]) * kd1 + float(kd_weights[1]) * kd2)

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
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    extra = _parse_extra_args()
    cfg = parse_args_with_config()

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    if extra.device is not None:
        device = extra.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = (cfg.get("paths") or {})
    runs_root = paths.get("runs_root", "runs")
    cache_root = paths.get("cache_root", "data_cache")

    if extra.cache_dir:
        cache_root = extra.cache_dir

    ensure_dirs(runs_root)

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

    save_config(cfg, os.path.join(run_dir, "config_used.yaml"))

    seg_csv = os.path.join(cache_root, "segments.csv")
    feat_root = os.path.join(cache_root, "features")

    tr = (cfg.get("train") or {})
    bs = int(tr.get("batch_size", 64))
    nw = int(tr.get("num_workers", 4))
    lr = float(tr.get("lr", 1e-3))
    wd = float(tr.get("weight_decay", 0.0))
    epochs = int(tr.get("epochs", 40))

    # ✅ Loss weights (you already had this)
    # Recommend for stronger exit1: [1.0, 0.5, 0.2] (try it)
    loss_w = tr.get("loss_weights", [0.3, 0.3, 1.0])

    # ✅ KD config (optional)
    kd = tr.get("kd", {}) or {}
    kd_enable = bool(kd.get("enable", False))
    kd_alpha = float(kd.get("alpha", 0.5))      # strength of KD term
    kd_temp = float(kd.get("temp", 2.0))        # distillation temperature
    kd_weights = kd.get("weights", [1.0, 1.0])  # [exit1, exit2]
    if isinstance(kd_weights, (list, tuple)) and len(kd_weights) >= 2:
        kd_weights = (float(kd_weights[0]), float(kd_weights[1]))
    else:
        kd_weights = (1.0, 1.0)

    # SpecAugment config (optional)
    spec = tr.get("specaug", {}) or {}
    specaug_enable = bool(spec.get("enable", False))

    # deterministic loaders
    dl_tr, dl_va, dl_te, label2id = make_loaders(seg_csv, feat_root, bs, nw, seed=seed)

    n_mels = int((cfg.get("features") or {}).get("n_mels", 64))
    num_classes = int((cfg.get("model") or {}).get("num_classes", 2))

    backbone = TinyAudioCNN(n_mels=n_mels)
    model = ExitNet(backbone, tap_dims=(16, 32), final_dim=64, num_classes=num_classes).to(device)

    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    metrics = {"train": [], "val": []}
    best = -1.0

    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, dl_tr, opt, device, loss_w,
            kd_enable=kd_enable, kd_alpha=kd_alpha, kd_temp=kd_temp, kd_weights=kd_weights,
            specaug_enable=specaug_enable, specaug_cfg=spec
        )
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
