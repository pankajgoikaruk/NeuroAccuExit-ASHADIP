# training/thresholds_offline.py

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


@torch.no_grad()
def collect_val_logits(model, dl, device, max_samples=None, temps=None, eps_temp=1e-3):
    """Collect logits once; optionally apply temperature scaling per-exit."""
    model.eval()
    all_logits = [[], [], []]
    all_y = []
    seen = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg = model(x)  # list of 3 tensors

        # Temperature scaling: logits / T
        if temps is not None:
            lg = [l / max(float(temps[i]), eps_temp) for i, l in enumerate(lg)]

        for k in range(3):
            all_logits[k].append(lg[k].cpu())
        all_y.append(y.cpu())

        seen += x.size(0)
        if max_samples is not None and seen >= max_samples:
            break

    all_logits = [torch.cat(L, 0) for L in all_logits]
    all_y = torch.cat(all_y, 0)
    return all_logits, all_y


def eval_policy_for_tau(logits_list, y_true, tau):
    """Greedy early-exit: stop at first exit where max prob >= tau."""
    probs = [softmax(l, dim=1) for l in logits_list]
    y_hat = []

    for i in range(y_true.numel()):
        pred = None
        for k in (0, 1, 2):
            if float(probs[k][i].max()) >= tau:
                pred = int(torch.argmax(probs[k][i]))
                break
        if pred is None:
            pred = int(torch.argmax(probs[-1][i]))
        y_hat.append(pred)

    y_np = y_true.numpy()
    f1 = f1_score(y_np, y_hat, average="macro")
    acc = (y_np == y_hat).mean()
    return f1, acc


def load_config_used(run_dir):
    """Optional: read n_mels / num_classes from config_used.yaml (if available)."""
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not (yaml and os.path.exists(cfg_path)):
        return {}

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--tau", nargs="+", type=float,
                    default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95])
    ap.add_argument("--max_samples", type=int, default=0,
                    help="limit val samples to speed up (0=all)")
    ap.add_argument("--eps_temp", type=float, default=1e-3,
                    help="minimum temperature to avoid divide-by-zero")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load validation loader (we optimize thresholds on VAL)
    _, dl_val, _, _ = make_loaders(
        args.segments_csv, args.features_root,
        batch_size=128, num_workers=2
    )

    # 2) Build model (prefer config_used.yaml if present)
    cfg = load_config_used(args.run_dir)
    n_mels = int((cfg.get("features") or {}).get("n_mels", 64))
    num_classes = int((cfg.get("model") or {}).get("num_classes", 2))

    model = ExitNet(
        TinyAudioCNN(n_mels=n_mels),
        (16, 32),
        64,
        num_classes
    ).to(device)

    ckpt = os.path.join(args.run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # 3) Load temperatures if available (use ALL 3; only tiny clamp)
    temps = None
    tpath = os.path.join(args.run_dir, "temperature.json")
    if os.path.exists(tpath):
        with open(tpath, "r", encoding="utf-8") as f:
            temps = json.load(f).get("temperatures", None)

        if temps is None or len(temps) != 3:
            print("temperature.json found but invalid; using raw logits.")
            temps = None
        else:
            temps = [float(t) for t in temps]
            temps = [max(t, args.eps_temp) for t in temps]  # DO NOT clamp to 0.5
            print("Using temperatures:", temps)
    else:
        print("No temperature.json found; using raw logits.")

    # 4) Precompute logits once
    max_samples = None if (args.max_samples or 0) <= 0 else int(args.max_samples)
    print("Collecting validation logits...")
    logits_val, y_val = collect_val_logits(
        model, dl_val, device,
        max_samples=max_samples,
        temps=temps,
        eps_temp=args.eps_temp
    )
    print(f"Collected {y_val.numel()} validation samples.")

    # 5) Sweep tau grid
    best = None
    print("Sweeping tau grid...")
    for tau in args.tau:
        f1, acc = eval_policy_for_tau(logits_val, y_val, tau)
        print(f"  tau={tau:.3f} -> macroF1={f1:.4f}, acc={acc:.4f}")
        if best is None or (f1 > best["f1"]) or (f1 == best["f1"] and acc > best["acc"]):
            best = {"tau": float(tau), "f1": float(f1), "acc": float(acc)}

    # 6) Save best thresholds (+ record temps used)
    outpath = os.path.join(args.run_dir, "thresholds.json")
    payload = dict(best)
    payload["temperatures_used"] = temps
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved thresholds.json:", payload)


if __name__ == "__main__":
    main()
