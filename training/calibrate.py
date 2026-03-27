# training/calibrate.py
import os
import json
import torch
from torch.nn.functional import cross_entropy

from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net, load_run_model_cfg


class TempScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = torch.nn.Parameter(torch.zeros(1))  # T = exp(log_t)

    def forward(self, logits):
        T = torch.exp(self.log_t).clamp(min=1e-3)
        return logits / T


@torch.no_grad()
def collect_val_logits(model, dl, device, max_samples=None):
    model.eval()
    all_logits = None
    all_y = []
    seen = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg_list = model(x)  # list length K

        if all_logits is None:
            all_logits = [[] for _ in range(len(lg_list))]

        for k in range(len(lg_list)):
            all_logits[k].append(lg_list[k].detach().cpu())

        all_y.append(y.detach().cpu())
        seen += x.size(0)

        if max_samples and seen >= max_samples:
            break

    all_logits = [torch.cat(L, 0) for L in all_logits]
    all_y = torch.cat(all_y, 0)
    return all_logits, all_y


def fit_temperature_for_exit(logits, y, max_iter=60):
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

    opt.step(closure)
    return float(torch.exp(ts.log_t).detach().cpu())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", required=True)
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--tap_blocks", default="", help="Comma list like 1,2,3,4 (optional).")
    ap.add_argument("--n_mels", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loaders
    _, dl_va, _, label2id = make_loaders(
        args.segments_csv, args.features_root, batch_size=128, num_workers=0
    )
    num_classes = len(label2id)

    # tap config (K-exit generic)
    tap_blocks = (1, 3) if not args.tap_blocks.strip() else tuple(int(x) for x in args.tap_blocks.split(","))

    model_cfg = load_run_model_cfg(args.run_dir)
    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=args.n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    ckpt_path = os.path.join(args.run_dir, "ckpt", "best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("Collecting validation logits...")
    logits, y = collect_val_logits(model, dl_va, device, max_samples=(args.max_samples or None))

    temps = []
    for k in range(len(logits)):
        print(f"Fitting temperature for exit {k+1}/{len(logits)}...")
        temps.append(fit_temperature_for_exit(logits[k].to(device), y.to(device), max_iter=40))

    out = {"temperatures": temps}
    with open(os.path.join(args.run_dir, "temperature.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved temperature.json:", out)