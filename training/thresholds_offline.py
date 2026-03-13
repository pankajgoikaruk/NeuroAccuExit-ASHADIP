# training/thresholds_offline.py
from __future__ import annotations

import os
import json
import argparse
import torch
from torch.nn.functional import softmax
from sklearn.metrics import f1_score

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet

try:
    import yaml
except ImportError:
    yaml = None


def _load_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def load_config_used(run_dir: str) -> dict:
    """Optional: read n_mels / tap_blocks from config_used.yaml (if available)."""
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not (yaml and os.path.exists(cfg_path)):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _pad_or_trim(temps, K: int):
    if temps is None:
        return [1.0] * K
    temps = [max(float(t), 1e-3) for t in temps]
    if len(temps) < K:
        temps = temps + [temps[-1]] * (K - len(temps))
    elif len(temps) > K:
        temps = temps[:K]
    return temps


@torch.no_grad()
def collect_val_logits(model, dl, device, max_samples=None, temps=None):
    """Collect logits once; optionally apply temperature scaling per-exit."""
    model.eval()
    K = model.num_exits

    all_logits = [[] for _ in range(K)]
    all_y = []
    seen = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg_list = model(x)  # length K

        if temps is not None:
            lg_list = [lg_list[k] / max(float(temps[k]), 1e-3) for k in range(K)]

        for k in range(K):
            all_logits[k].append(lg_list[k].detach().cpu())

        all_y.append(y.detach().cpu())
        seen += x.size(0)

        if max_samples is not None and seen >= max_samples:
            break

    all_logits = [torch.cat(L, 0) for L in all_logits]
    all_y = torch.cat(all_y, 0)
    return all_logits, all_y


def eval_policy_for_tau(logits_list, y_true, tau: float):
    """Greedy early-exit: stop at first exit where max prob >= tau; otherwise final exit."""
    K = len(logits_list)
    probs = [softmax(l, dim=1) for l in logits_list]

    y_hat = []
    for i in range(y_true.numel()):
        tk = K - 1
        for k in range(K):
            if float(probs[k][i].max()) >= tau:
                tk = k
                break
        y_hat.append(int(torch.argmax(probs[tk][i])))

    y_np = y_true.numpy()
    f1 = float(f1_score(y_np, y_hat, average="macro"))
    acc = float((y_np == y_hat).mean())
    return f1, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--tap_blocks", default="", help="Comma list like 1,3. If empty, try config_used.yaml, else 1,3.")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--tau", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95])
    ap.add_argument("--max_samples", type=int, default=0, help="limit val samples to speed up (0=all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # validation loader (optimize on VAL)
    _, dl_val, _, label2id = make_loaders(args.segments_csv, args.features_root, batch_size=128, num_workers=2)
    num_classes = len(label2id)

    # tap_blocks resolution: CLI > config_used.yaml > default
    cfg = load_config_used(args.run_dir)
    cfg_tap = (cfg.get("model") or {}).get("tap_blocks", None)
    if args.tap_blocks.strip():
        tap_blocks = tuple(int(x) for x in args.tap_blocks.split(",") if x.strip())
    elif isinstance(cfg_tap, (list, tuple)) and len(cfg_tap) > 0:
        tap_blocks = tuple(int(x) for x in cfg_tap)
    else:
        tap_blocks = (1, 3)

    # n_mels: CLI overrides config
    n_mels = int((cfg.get("features") or {}).get("n_mels", args.n_mels))

    # build model
    backbone = TinyAudioCNN(n_mels=n_mels, tap_blocks=tap_blocks)
    model = ExitNet(
        backbone,
        num_classes=num_classes,
        tap_dims=backbone.tap_dims,
        final_dim=backbone.final_dim,
    ).to(device)

    ckpt = os.path.join(args.run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    K = model.num_exits

    # temps (pad/trim to K)
    temps = None
    tpath = os.path.join(args.run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = _load_json(tpath, {}).get("temperatures", None)
        temps = _pad_or_trim(temps, K)
        print("Using temperatures:", temps)
    else:
        print("No temperature.json found; using raw logits.")

    max_samples = None if (args.max_samples or 0) <= 0 else int(args.max_samples)

    print("Collecting validation logits...")
    logits_val, y_val = collect_val_logits(model, dl_val, device, max_samples=max_samples, temps=temps)
    print(f"Collected {y_val.numel()} validation samples.")

    best = None
    print("Sweeping tau grid...")
    for tau in args.tau:
        f1, acc = eval_policy_for_tau(logits_val, y_val, float(tau))
        print(f" tau={tau:.3f} -> macroF1={f1:.4f}, acc={acc:.4f}")
        if best is None or (f1 > best["f1"]) or (f1 == best["f1"] and acc > best["acc"]):
            best = {"tau": float(tau), "f1": float(f1), "acc": float(acc)}

    payload = dict(best)
    payload.update({
        "K": int(K),
        "tap_blocks": list(tap_blocks),
        "n_mels": int(n_mels),
        "num_classes": int(num_classes),
        "temperatures_used": temps,
    })

    outpath = os.path.join(args.run_dir, "thresholds.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved thresholds.json:", payload)


if __name__ == "__main__":
    main()