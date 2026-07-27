#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the v0.19 final decision and paper tables.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--previous_v018", type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(args.comparison.resolve())
    full = current[current["method"] == "full_targeted"].iloc[0]
    validation_eligible = as_bool(full["validation_eligible"])
    holdout_passed = as_bool(full["holdout_constraints_met"])
    passed = bool(validation_eligible and holdout_passed and float(full["estimated_flops_saved_pct"]) > 0.0)
    recommendation = "FINALISE_V019_EXIT3_TO_EXIT5" if passed else "STOP_EE_AND_RETAIN_PREVIOUS_SAFE_BASELINE"
    decision = {
        "experiment": "v0.19_EE_final_targeted_exit3_to_exit5",
        "validation_eligible": validation_eligible,
        "holdout_constraints_met": holdout_passed,
        "final_requirements_met": passed,
        "final_recommendation": recommendation,
        "selected_method": "full_targeted" if passed else None,
        "estimated_flops_saved_pct": float(full["estimated_flops_saved_pct"]),
        "measured_speedup": float(full["measured_speedup_vs_always_final"]),
        "parent_macro_f1": float(full["parent_macro_f1"]),
        "parent_micro_f1": float(full["parent_micro_f1"]),
        "parent_exact_match": float(full["parent_exact_match"]),
        "parent_hamming_loss": float(full["parent_hamming_loss"]),
        "macro_drop": float(full["parent_macro_f1_drop_vs_final"]),
        "micro_drop": float(full["parent_micro_f1_drop_vs_final"]),
        "exact_drop": float(full["parent_exact_drop_vs_final"]),
        "hamming_increase": float(full["parent_hamming_increase_vs_final"]),
        "rule": "v0.19 is the final targeted EE experiment; no further broad EE search is recommended.",
    }
    with (out / "v019_final_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    current.to_csv(out / "v019_final_targeted_table.csv", index=False)
    combined = current.copy()
    if args.previous_v018 and args.previous_v018.exists():
        previous = pd.read_csv(args.previous_v018.resolve())
        previous = previous[previous["method"].isin(["always_final", "no_exit1"])].copy()
        previous["experiment"] = "v0.18"
        combined["experiment"] = "v0.19"
        combined = pd.concat([previous, combined], ignore_index=True, sort=False)
    combined.to_csv(out / "v018_v019_targeted_comparison.csv", index=False)

    latex_rows = current[current["method"].isin(["always_final", "full_targeted"])].copy()
    with (out / "v019_final_targeted_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[H]\n\\centering\n")
        handle.write("\\caption{Final targeted Exit-3-to-Exit-5 corrected-holdout comparison.}\n")
        handle.write("\\label{tab:v019_targeted}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        handle.write("Method & FLOPs saved & Speedup & Macro-F1 & Micro-F1 & Samples-F1 & Exact & Hamming $\\downarrow$ \\\\\n")
        handle.write("\\midrule\n")
        for _, row in latex_rows.iterrows():
            method = "Always Exit 5" if row["method"] == "always_final" else "v0.19 Exit 3--5 targeted"
            handle.write(
                f"{method} & {row['estimated_flops_saved_pct']:.2f}\\% & "
                f"{row['measured_speedup_vs_always_final']:.3f}$\\times$ & "
                f"{row['parent_macro_f1']:.6f} & {row['parent_micro_f1']:.6f} & "
                f"{row['parent_samples_f1']:.6f} & {row['parent_exact_match']:.6f} & "
                f"{row['parent_hamming_loss']:.6f} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    print("\nV0.19 final decision")
    print("-" * 80)
    print(json.dumps(decision, indent=2))
    print(f"Saved outputs: {out}")


if __name__ == "__main__":
    main()
