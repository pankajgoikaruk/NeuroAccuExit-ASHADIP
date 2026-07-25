#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def norm_path(value: str) -> str:
    return str(Path(value).resolve()).replace("\\", "/").casefold()


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fair 3-exit/5-exit training configuration.")
    parser.add_argument("--run3", required=True, type=Path)
    parser.add_argument("--run5", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)
    args = parser.parse_args()

    cfg3 = load_json(args.run3.resolve() / "config_used.json")
    cfg5 = load_json(args.run5.resolve() / "config_used.json")
    weights3 = [float(value) for value in cfg3.get("loss_weights", [])]
    weights5 = [float(value) for value in cfg5.get("loss_weights", [])]

    fields = [
        "manifest", "features_root", "labels_json", "labels", "num_labels", "n_mels",
        "epochs", "batch_size", "num_workers", "lr", "weight_decay", "seed",
        "threshold", "use_pos_weight", "pos_weight_max", "label_balance_power",
        "synthetic_balance_power",
    ]
    checks: dict[str, bool] = {}
    for field in fields:
        left = cfg3.get(field)
        right = cfg5.get(field)
        if field in {"manifest", "features_root", "labels_json"}:
            checks[f"same_{field}"] = norm_path(str(left)) == norm_path(str(right))
        elif isinstance(left, float) or isinstance(right, float):
            checks[f"same_{field}"] = close(float(left), float(right))
        else:
            checks[f"same_{field}"] = left == right

    hint3 = cfg3.get("exit_hint", {}) or {}
    hint5 = cfg5.get("exit_hint", {}) or {}
    checks.update({
        "three_exit_taps_correct": list(cfg3.get("tap_blocks", [])) == [1, 3],
        "five_exit_taps_correct": list(cfg5.get("tap_blocks", [])) == [1, 2, 3, 4],
        "three_exit_count_correct": int(cfg3.get("num_exits", 0)) == 3,
        "five_exit_count_correct": int(cfg5.get("num_exits", 0)) == 5,
        "both_no_hint": not bool(hint3.get("enable", False)) and not bool(hint5.get("enable", False)),
        "final_loss_weight_matched": len(weights3) == 3 and len(weights5) == 5 and close(weights3[-1], weights5[-1]),
        "auxiliary_loss_budget_matched": len(weights3) == 3 and len(weights5) == 5 and close(sum(weights3[:-1]), sum(weights5[:-1])),
        "five_auxiliary_weights_equal": len(weights5) == 5 and max(weights5[:-1]) - min(weights5[:-1]) <= 1e-9,
    })

    passed = bool(all(checks.values()))
    payload = {
        "experiment": "v0.18_EE_fair_5exit_training_audit",
        "status": "PASS" if passed else "FAIL",
        "fair_training_valid": passed,
        "run3": str(args.run3.resolve()),
        "run5": str(args.run5.resolve()),
        "checks": checks,
        "three_exit_loss_weights": weights3,
        "five_exit_loss_weights": weights5,
        "three_exit_auxiliary_loss_budget": sum(weights3[:-1]) if len(weights3) >= 2 else None,
        "five_exit_auxiliary_loss_budget": sum(weights5[:-1]) if len(weights5) >= 2 else None,
        "interpretation": (
            "The five-exit model is directly comparable at training-data and optimisation level; "
            "only the exit topology and distribution of the matched auxiliary-loss budget differ."
            if passed else
            "The architecture comparison is not valid until every failed training-fairness check is resolved."
        ),
    }
    args.out_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.resolve().open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
