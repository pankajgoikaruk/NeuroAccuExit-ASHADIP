# scripts/policy_test.py

import os
import json
import argparse
import torch
from statistics import mean

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet

# EA policy function (make sure file is: policies/depth_ea.py)
from policies.depth_ea import depth_ea_decide


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(run_dir: str, segments_csv: str, features_root: str, policy: str, num_workers: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load temps (required for both policies; fall back to 1.0s if missing) ----
    tpath = os.path.join(run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = _load_json(tpath).get("temperatures", [1.0, 1.0, 1.0])
    else:
        temps = [1.0, 1.0, 1.0]
    temps = [max(float(t), 0.5) for t in temps]  # stability clamp

    # ---- Load policy-specific thresholds ----
    tau = None
    ea_cfg = None

    if policy == "greedy":
        th_path = os.path.join(run_dir, "thresholds.json")
        if not os.path.exists(th_path):
            raise FileNotFoundError(f"thresholds.json not found in run_dir: {th_path}")
        tau = float(_load_json(th_path)["tau"])

    elif policy == "ea":
        ea_path = os.path.join(run_dir, "ea_thresholds.json")
        if not os.path.exists(ea_path):
            raise FileNotFoundError(f"ea_thresholds.json not found in run_dir: {ea_path}")
        ea_file = _load_json(ea_path)

        ea_threshold = float(ea_file["ea_threshold"])
        ea_mode = ea_file.get("ea_mode", "logprob")
        ea_min_exit = int(ea_file.get("ea_min_exit", 0))
        ea_stable_k = int(ea_file.get("ea_stable_k", 1))
        ea_flip_penalty = float(ea_file.get("ea_flip_penalty", 0.0))

        ea_cfg = {
            "ea_threshold": ea_threshold,
            "ea_mode": ea_mode,
            "ea_min_exit": ea_min_exit,
            "ea_stable_k": ea_stable_k,
            "ea_flip_penalty": ea_flip_penalty,
        }
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # ---- Data ----
    # (we read label2id to avoid hardcoding num_classes)
    _, _, dl_te, label2id = make_loaders(
        segments_csv, features_root, batch_size=64, num_workers=num_workers
    )
    num_classes = len(label2id)

    # ---- Build model ----
    model = ExitNet(TinyAudioCNN(), (16, 32), 64, num_classes).to(device)
    ckpt = os.path.join(run_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ---- Evaluate ----
    n = 0
    correct = 0
    exits_taken = []

    # EA-only diagnostics
    flip_count_total = 0
    exit_consistent_total = 0

    with torch.no_grad():
        for x, y in dl_te:
            x = x.to(device)
            y = y.to(device)
            logits_list = model(x)  # list of 3 tensors (B,C)

            if policy == "greedy":
                # scale logits per exit
                scaled = [lg / max(float(temps[i]), 1e-3) for i, lg in enumerate(logits_list)]
                probs = [torch.softmax(lg, dim=1) for lg in scaled]

                B = x.size(0)
                for i in range(B):
                    taken = 2
                    for k in (0, 1, 2):
                        if float(probs[k][i].max()) >= tau:
                            taken = k
                            break
                    pred = int(torch.argmax(probs[taken][i]))
                    correct += int(pred == int(y[i]))
                    exits_taken.append(taken + 1)  # store 1/2/3
                    n += 1

            else:
                # Depth-EA decision (batch-wise)
                out = depth_ea_decide(
                    logits_list=logits_list,
                    temps=temps,
                    ea_mode=ea_cfg["ea_mode"],
                    ea_threshold=ea_cfg["ea_threshold"],
                    ea_min_exit=ea_cfg["ea_min_exit"],
                    ea_stable_k=ea_cfg["ea_stable_k"],
                    ea_flip_penalty=ea_cfg["ea_flip_penalty"],
                )
                taken = out["taken"]            # (B,) values 0/1/2
                pred_taken = out["pred_taken"]  # (B,)
                pred_final = out["pred_final"]  # (B,)

                correct += int((pred_taken == y).sum().item())
                exits_taken.extend((taken + 1).detach().cpu().tolist())  # store 1/2/3
                n += x.size(0)

                flip_count_total += int((out["flip_count"] > 0).sum().item())
                exit_consistent_total += int((pred_taken == pred_final).sum().item())

    acc = correct / max(n, 1)
    avg_exit = mean(exits_taken) if exits_taken else 0.0

    # ---- Exit mix (needed for correct compute saving in summarize_run) ----
    e1 = exits_taken.count(1) / max(n, 1)
    e2 = exits_taken.count(2) / max(n, 1)
    e3 = exits_taken.count(3) / max(n, 1)
    exit_mix = {"e1": float(e1), "e2": float(e2), "e3": float(e3)}

    print(f"Policy: {policy}")
    print(f"Policy test accuracy: {acc:.4f}")
    print(f"Avg exit depth: {avg_exit:.3f}")
    print(f"Exit mix: e1={exit_mix['e1']:.3f}, e2={exit_mix['e2']:.3f}, e3={exit_mix['e3']:.3f}")

    results = {
        "policy": policy,
        "accuracy": float(acc),
        "avg_exit_depth": float(avg_exit),
        "n_samples": int(n),
        "exit_mix": exit_mix,
    }

    if policy == "ea":
        flip_rate = flip_count_total / max(n, 1)
        exit_consistency = exit_consistent_total / max(n, 1)
        print(f"Flip-rate: {flip_rate:.4f}")
        print(f"Exit-consistency (taken==final): {exit_consistency:.4f}")

        results.update({
            "flip_rate": float(flip_rate),
            "exit_consistency": float(exit_consistency),
            "ea": ea_cfg,
        })

    # write policy_results.json for summarize_run
    out_path = os.path.join(run_dir, "policy_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--policy", default="greedy", choices=["greedy", "ea"])
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    main(args.run_dir, args.segments_csv, args.features_root, args.policy, args.num_workers)
