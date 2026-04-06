# training/calibrate.py

from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Optional

import torch
from torch.nn.functional import cross_entropy

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


class TempScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # T = exp(log_t) > 0
        self.log_t = torch.nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        t = torch.exp(self.log_t).clamp(min=1e-3)
        return logits / t


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


def _build_backbone(n_mels: int, tap_blocks=None):
    """
    Compatible with both:
    - old style: TinyAudioCNN(n_mels=64)
    - new style: TinyAudioCNN(n_mels=64, tap_blocks=(1,2,3,4))
    """
    sig = inspect.signature(TinyAudioCNN.__init__)
    kwargs = {}

    if "n_mels" in sig.parameters:
        kwargs["n_mels"] = int(n_mels)

    if tap_blocks is not None and "tap_blocks" in sig.parameters:
        kwargs["tap_blocks"] = tap_blocks

    return TinyAudioCNN(**kwargs)


@torch.no_grad()
def collect_val_logits(model, dl, device, max_samples=None):
    model.eval()

    all_logits = None
    all_y = []
    seen = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits_list = model(x)

        if all_logits is None:
            all_logits = [[] for _ in range(len(logits_list))]

        for k in range(len(logits_list)):
            all_logits[k].append(logits_list[k].detach().cpu())

        all_y.append(y.detach().cpu())
        seen += x.size(0)

        if max_samples and seen >= max_samples:
            break

    if all_logits is None or len(all_y) == 0:
        raise RuntimeError("Validation loader is empty. No logits collected.")

    all_logits = [torch.cat(L, dim=0) for L in all_logits]
    all_y = torch.cat(all_y, dim=0)
    return all_logits, all_y


def fit_temperature_for_exit(logits, y, max_iter=60, verbose=True):
    device = logits.device
    ts = TempScale().to(device)

    opt = torch.optim.LBFGS(
        ts.parameters(),
        lr=0.5,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = cross_entropy(ts(logits), y)
        loss.backward()
        return loss

    last = None
    for i in range(max_iter):
        loss = opt.step(closure)
        cur = float(loss.detach().cpu())
        t_val = float(torch.exp(ts.log_t).detach().cpu())

        if verbose and (last is None or abs(cur - last) > 1e-5):
            print(f"  iter {i + 1:02d}: loss={cur:.6f}, T={t_val:.4f}")

        last = cur

    return float(torch.exp(ts.log_t).detach().cpu())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all validation samples")
    ap.add_argument("--tap_blocks", type=str, default=None,
                    help='Example: "1,3" for 3 exits or "1,2,3,4" for 5 exits.')
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=None,
                    help="Optional override. If omitted, inferred from label2id.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    # Validation loader only
    _, dl_va, _, label2id = make_loaders(
        args.segments_csv,
        args.features_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = int(args.num_classes) if args.num_classes is not None else len(label2id)
    tap_blocks = _parse_tap_blocks(args.tap_blocks)

    # Build model
    backbone = _build_backbone(n_mels=args.n_mels, tap_blocks=tap_blocks)

    model_kwargs = {
        "backbone": backbone,
        "num_classes": num_classes,
    }

    # Backward compatibility if backbone has no metadata
    if not hasattr(backbone, "tap_dims"):
        model_kwargs["tap_dims"] = (16, 32)
    if not hasattr(backbone, "final_dim"):
        model_kwargs["final_dim"] = 64

    model = ExitNet(**model_kwargs).to(device)

    ckpt_path = os.path.join(args.run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("Collecting validation logits...")
    logits, y = collect_val_logits(
        model,
        dl_va,
        device,
        max_samples=(args.max_samples or None),
    )
    print(f"Collected {y.numel()} samples across {len(logits)} exits.")

    temps = []
    for k in range(len(logits)):
        print(f"Fitting temperature for exit {k + 1}/{len(logits)}...")
        t = fit_temperature_for_exit(
            logits[k].to(device),
            y.to(device),
            max_iter=40,
            verbose=True,
        )
        temps.append(t)

    out = {
        "temperatures": temps,
        "num_exits": len(logits),
        "tap_blocks": list(tap_blocks) if tap_blocks is not None else None,
        "num_classes": num_classes,
    }

    out_path = os.path.join(args.run_dir, "temperature.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved temperature.json:", out)