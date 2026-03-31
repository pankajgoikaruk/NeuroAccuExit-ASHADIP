import os
import json
import argparse
from statistics import mean

import torch
from torch.nn.functional import softmax

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


def main(run_dir, segments_csv, features_root, policy="greedy", num_workers=2):
    if policy != "greedy":
        raise ValueError(
            f"Current scripts.policy_test.py supports only greedy policy, got: {policy}"
        )

    # Load greedy threshold + temperatures
    tau = json.load(open(os.path.join(run_dir, "thresholds.json"), "r"))["tau"]
    temps = json.load(open(os.path.join(run_dir, "temperature.json"), "r"))["temperatures"]
    temps = [max(float(t), 1e-3) for t in temps]  # safer than clamping to 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ExitNet(TinyAudioCNN(), (16, 32), 64, 2).to(device)
    model.load_state_dict(
        torch.load(os.path.join(run_dir, "ckpt", "best.pt"), map_location=device)
    )
    model.eval()

    def scale(logits, t):
        return logits / max(float(t), 1e-3)

    _, _, dl_te, _ = make_loaders(
        segments_csv,
        features_root,
        batch_size=64,
        num_workers=num_workers
    )

    n = 0
    correct = 0
    exits = []
    flip_any_count = 0
    flip_count_sum = 0
    consistency_count = 0
    exit_counts = {"e1": 0, "e2": 0, "e3": 0}

    with torch.no_grad():
        for x, y in dl_te:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            logits = [scale(lg, temps[i]) for i, lg in enumerate(logits)]
            probs = [softmax(lg, dim=1) for lg in logits]

            for i in range(x.size(0)):
                # Predictions at all exits
                preds_all = [int(torch.argmax(p[i]).item()) for p in probs]

                # Greedy decision
                taken = len(probs) - 1
                for k in range(len(probs)):
                    if float(probs[k][i].max()) >= tau:
                        taken = k
                        break

                pred_taken = preds_all[taken]
                pred_final = preds_all[-1]

                correct += int(pred_taken == int(y[i]))
                exits.append(taken + 1)
                exit_counts[f"e{taken + 1}"] += 1

                # Flip metrics
                flip_any_count += int(len(set(preds_all)) > 1)
                flip_count_sum += sum(
                    1 for a, b in zip(preds_all[:-1], preds_all[1:]) if a != b
                )
                consistency_count += int(pred_taken == pred_final)

                n += 1

    acc = correct / n
    avg_exit_depth = mean(exits)
    exit_mix = {k: v / n for k, v in exit_counts.items()}
    flip_any_rate = flip_any_count / n
    avg_flip_count = flip_count_sum / n
    exit_consistency = consistency_count / n

    result = {
        "policy": "greedy",
        "accuracy": acc,
        "avg_exit_depth": avg_exit_depth,
        "n_samples": n,
        "n_segments": n,
        "exit_mix": exit_mix,
        "tau": float(tau),
        "temperatures_used": temps,
        "flip_any_rate": flip_any_rate,
        "avg_flip_count": avg_flip_count,
        "exit_consistency": exit_consistency,
    }

    out_path = os.path.join(run_dir, "policy_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    exit_mix_str = ", ".join(f"{k}={v:.4f}" for k, v in exit_mix.items())

    print(f"Policy test accuracy: {acc:.4f} (n_segments={n})")
    print(f"Avg exit depth: {avg_exit_depth:.3f}")
    print(f"Exit mix: {exit_mix_str}")
    print(f"Flip-any rate: {flip_any_rate:.4f}")
    print(f"Exit consistency: {exit_consistency:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_caches/segments.csv")
    ap.add_argument("--features_root", default="data_caches/features")
    ap.add_argument("--policy", default="greedy")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    main(
        run_dir=args.run_dir,
        segments_csv=args.segments_csv,
        features_root=args.features_root,
        policy=args.policy,
        num_workers=args.num_workers,
    )