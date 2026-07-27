#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Create the final v0.19 EE decision report.")
    p.add_argument("--comparison_csv", required=True, type=Path)
    p.add_argument("--policy_json", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--min_flops_saved_pct", type=float, default=5.0)
    args = p.parse_args()

    frame = pd.read_csv(args.comparison_csv.resolve())
    base = frame.loc[frame["method"] == "always_final"].iloc[0]
    policy = frame.loc[frame["method"] == "full_strict"].iloc[0]
    with args.policy_json.resolve().open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)

    constraints = frozen["selection_constraints"]
    checks = {
        "macro_f1_drop": float(policy["parent_macro_f1_drop_vs_own_final"]) <= float(constraints["max_parent_macro_f1_drop"]),
        "micro_f1_drop": float(policy["parent_micro_f1_drop_vs_own_final"]) <= float(constraints["max_parent_micro_f1_drop"]),
        "exact_match_drop": float(policy["parent_exact_drop_vs_own_final"]) <= float(constraints["max_parent_exact_match_drop"]),
        "hamming_increase": float(policy["parent_hamming_increase_vs_own_final"]) <= float(constraints["max_parent_hamming_increase"]),
        "minimum_compute_saving": float(policy["estimated_flops_saved_pct"]) >= float(args.min_flops_saved_pct),
        "validation_eligible": bool(policy["validation_eligible"]),
    }
    finalise = bool(all(checks.values()))
    decision = "FINALISE_V019_EXIT3_TO_EXIT5" if finalise else "STOP_EE_AND_FINALISE_V013_PER_LABEL_MARGIN"

    summary = {
        "experiment": "v0.19_EE_final_targeted_exit3_to_exit5",
        "decision": decision,
        "all_requirements_met": finalise,
        "checks": checks,
        "always_exit5": {
            "macro_f1": float(base["parent_macro_f1"]),
            "micro_f1": float(base["parent_micro_f1"]),
            "samples_f1": float(base["parent_samples_f1"]),
            "exact_match": float(base["parent_exact_match"]),
            "hamming_loss": float(base["parent_hamming_loss"]),
        },
        "v019_exit3_to_exit5": {
            "exit3_fraction": float(policy.get("exit3_fraction", 0.0)),
            "exit5_fraction": float(policy.get("exit5_fraction", 0.0)),
            "estimated_flops_saved_pct": float(policy["estimated_flops_saved_pct"]),
            "measured_speedup": float(policy["measured_speedup_vs_always_final"]),
            "macro_f1": float(policy["parent_macro_f1"]),
            "micro_f1": float(policy["parent_micro_f1"]),
            "samples_f1": float(policy["parent_samples_f1"]),
            "exact_match": float(policy["parent_exact_match"]),
            "hamming_loss": float(policy["parent_hamming_loss"]),
            "macro_f1_drop": float(policy["parent_macro_f1_drop_vs_own_final"]),
            "micro_f1_drop": float(policy["parent_micro_f1_drop_vs_own_final"]),
            "exact_match_drop": float(policy["parent_exact_drop_vs_own_final"]),
            "hamming_increase": float(policy["parent_hamming_increase_vs_own_final"]),
        },
        "fallback_baseline": {
            "method": "v0.13 per-label margin",
            "flops_saved_pct": 1.44,
            "parent_macro_f1": 0.858748,
            "parent_micro_f1": 0.951556,
            "parent_samples_f1": 0.957198,
            "parent_exact_match": 0.874279,
            "parent_hamming_loss": 0.014187,
        },
    }
    args.out_dir.resolve().mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "v019_final_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    rows = pd.DataFrame([
        {
            "method": "Always Exit 5", "stop_unit": "None", "decision_route": "Exit 5 only",
            "flops_saved_pct": 0.0, "speedup": 1.0,
            "parent_macro_f1": base["parent_macro_f1"], "parent_micro_f1": base["parent_micro_f1"],
            "parent_samples_f1": base["parent_samples_f1"], "exact_match": base["parent_exact_match"],
            "hamming_loss": base["parent_hamming_loss"], "holdout_constraints_met": True,
        },
        {
            "method": "v0.19 Exit 3 to Exit 5", "stop_unit": "Segment", "decision_route": "Exit 3 -> Exit 5",
            "flops_saved_pct": policy["estimated_flops_saved_pct"], "speedup": policy["measured_speedup_vs_always_final"],
            "parent_macro_f1": policy["parent_macro_f1"], "parent_micro_f1": policy["parent_micro_f1"],
            "parent_samples_f1": policy["parent_samples_f1"], "exact_match": policy["parent_exact_match"],
            "hamming_loss": policy["parent_hamming_loss"], "holdout_constraints_met": finalise,
        },
    ])
    rows.to_csv(args.out_dir / "v019_final_comparison_table.csv", index=False)

    report = f"""# v0.19_EE final targeted decision

## Decision

**{decision}**

## Corrected-holdout result

| Metric | Always Exit 5 | v0.19 Exit 3 -> Exit 5 | Change / saving |
|---|---:|---:|---:|
| FLOPs saved | 0.00% | {policy['estimated_flops_saved_pct']:.2f}% | {policy['estimated_flops_saved_pct']:.2f}% |
| Measured speedup | 1.000x | {policy['measured_speedup_vs_always_final']:.3f}x | {(policy['measured_speedup_vs_always_final']-1)*100:.2f}% |
| Parent Macro-F1 | {base['parent_macro_f1']:.6f} | {policy['parent_macro_f1']:.6f} | {-policy['parent_macro_f1_drop_vs_own_final']:.6f} |
| Parent Micro-F1 | {base['parent_micro_f1']:.6f} | {policy['parent_micro_f1']:.6f} | {-policy['parent_micro_f1_drop_vs_own_final']:.6f} |
| Parent Samples-F1 | {base['parent_samples_f1']:.6f} | {policy['parent_samples_f1']:.6f} | {policy['parent_samples_f1']-base['parent_samples_f1']:.6f} |
| Exact Match | {base['parent_exact_match']:.6f} | {policy['parent_exact_match']:.6f} | {-policy['parent_exact_drop_vs_own_final']:.6f} |
| Hamming Loss | {base['parent_hamming_loss']:.6f} | {policy['parent_hamming_loss']:.6f} | +{policy['parent_hamming_increase_vs_own_final']:.6f} |
| Exit-3 fraction | 0.00% | {100*policy.get('exit3_fraction', 0.0):.2f}% | — |

## Constraint checks

""" + "\n".join(f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in checks.items()) + "\n"
    (args.out_dir / "V019_FINAL_DECISION.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nFINAL DECISION: {decision}")


if __name__ == "__main__":
    main()
