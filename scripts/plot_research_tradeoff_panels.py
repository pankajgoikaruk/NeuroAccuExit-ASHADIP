#!/usr/bin/env python3
"""
Reusable plotting script for NeuroAccuExit trade-off figures.

Updated paper-friendly behavior
------------------------------
- Uses policy-style accuracy for all three panels:
    1) Segment policy accuracy               -> policy_results.json["accuracy"]
    2) Full-clip policy accuracy            -> prefers segment_accuracy_over_processed_windows
    3) Depth×Time policy accuracy           -> prefers segment_accuracy_over_processed_windows
- Avoids saturated clip-level accuracy as the default y metric for full/time panels.
- No subplot titles by default.
- Supports compact PDF-friendly figure sizes.
- Can save:
    * one combined 3-panel figure
    * separate figures
    * or both

Examples (PowerShell)
---------------------
A) No-hint progression, compact combined PDF + separate PDFs
python .\scripts\plot_research_tradeoff_panels.py `
  --runs_root .\runs `
  --mode progression `
  --runs kexit_dev\kexit_dev_001 kexit_dev\kexit_dev_002 kexit_greedy_no_hint\kexit_greedy_no_hint_001 `
  --labels "3-exit greedy" "5-exit greedy" "5-exit greedy no-hint" `
  --x_metric avg_windows_used `
  --out .\plots\no_hint_progression_avg_windows.pdf `
  --out_dir .\plots\individual_pdf `
  --save_individual `
  --csv_out .\plots\no_hint_progression_avg_windows.csv

B) Hint vs no-hint paired comparison, only separate PDFs
python .\scripts\plot_research_tradeoff_panels.py `
  --runs_root .\runs `
  --mode paired `
  --pairs "Greedy|kexit_greedy_no_hint\kexit_greedy_no_hint_001|kexit_greedy_hint\kexit_greedy_hint_001" `
  --connect_pairs `
  --x_metric expected_mflops `
  --save_individual `
  --skip_combined `
  --out_dir .\plots\hint_vs_nohint_pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ---------- JSON helpers ----------

def _load_json(path: Path) -> Optional[dict]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _first_existing_json(run_dir: Path, names: List[str]) -> Optional[dict]:
    for name in names:
        obj = _load_json(run_dir / name)
        if obj is not None:
            return obj
    return None


def _pick_first(*values):
    for v in values:
        if v is not None:
            return v
    return None


# ---------- Metric extraction ----------

def load_run_metrics(run_dir: Path) -> Dict[str, object]:
    """
    Load the core metrics needed for paper figures from one run directory.
    """
    policy = _load_json(run_dir / "policy_results.json") or {}
    full = _load_json(run_dir / "clip_policy_results_full.json") or {}
    time_res = _first_existing_json(run_dir, ["clip_policy_results_time.json", "clip_policy_results.json"]) or {}
    profiling = _load_json(run_dir / "profiling.json") or {}

    # Detect hint either from JSON or from path name.
    hint_flag = False
    if isinstance(time_res, dict):
        hint_flag = bool(time_res.get("exit_hint", False))
    if not hint_flag:
        hint_flag = "hint" in str(run_dir).lower()

    # Paper-facing accuracy fields:
    # Prefer policy-style window accuracy over saturated clip_accuracy.
    full_policy_test_accuracy = _pick_first(
        full.get("segment_accuracy_over_processed_windows"),
        full.get("segment_accuracy_over_used_windows"),
        full.get("clip_accuracy"),
    )
    time_policy_test_accuracy = _pick_first(
        time_res.get("segment_accuracy_over_processed_windows"),
        time_res.get("segment_accuracy_over_used_windows"),
        time_res.get("clip_accuracy"),
    )

    data = {
        "run_relpath": str(run_dir).replace("\\", "/"),
        "run_id": run_dir.name,
        "variant": run_dir.parent.name,
        "is_hint": hint_flag,

        # Segment policy
        "segment_policy_accuracy": policy.get("accuracy"),
        "avg_exit_depth": policy.get("avg_exit_depth"),
        "segment_flip_any_rate": policy.get("flip_any_rate"),
        "segment_exit_consistency": policy.get("exit_consistency"),

        # Full-clip baseline
        "full_clip_accuracy": full.get("clip_accuracy"),
        "full_segment_accuracy_over_processed_windows": full.get("segment_accuracy_over_processed_windows"),
        "full_segment_accuracy_over_used_windows": full.get("segment_accuracy_over_used_windows"),
        "full_policy_test_accuracy": full_policy_test_accuracy,
        "full_avg_windows_used": full.get("avg_windows_used"),
        "full_avg_windows_total": full.get("avg_windows_total"),
        "full_avg_compute_units": full.get("avg_compute_units_sum_depth_over_used_windows"),
        "full_windows_saved_pct": full.get("windows_saved_pct"),
        "full_compute_saved_pct": full.get("compute_saved_pct"),
        "full_avg_depth_per_used_window": full.get("avg_depth_per_used_window"),
        "full_flip_rate_over_used_windows": full.get("flip_rate_over_used_windows"),
        "full_exit_consistency": full.get("exit_consistency_taken_vs_final_over_used_windows"),

        # Depth×Time
        "time_clip_accuracy": time_res.get("clip_accuracy"),
        "time_segment_accuracy_over_processed_windows": time_res.get("segment_accuracy_over_processed_windows"),
        "time_segment_accuracy_over_used_windows": time_res.get("segment_accuracy_over_used_windows"),
        "time_policy_test_accuracy": time_policy_test_accuracy,
        "time_avg_windows_used": time_res.get("avg_windows_used"),
        "time_avg_windows_total": time_res.get("avg_windows_total"),
        "time_avg_compute_units": time_res.get("avg_compute_units_sum_depth_over_used_windows"),
        "time_windows_saved_pct": time_res.get("windows_saved_pct"),
        "time_compute_saved_pct": time_res.get("compute_saved_pct"),
        "time_avg_depth_per_used_window": time_res.get("avg_depth_per_used_window"),
        "time_flip_rate_over_used_windows": time_res.get("flip_rate_over_used_windows"),
        "time_exit_consistency": time_res.get("exit_consistency_taken_vs_final_over_used_windows"),

        # Profiling
        "expected_mflops": profiling.get("expected_mflops"),
        "full_mflops": profiling.get("full_mflops"),
        "compute_saving_pct_profile": profiling.get("compute_saving_pct"),
        "full_forward_latency_ms": profiling.get("full_forward_latency_ms"),
    }
    return data


def x_metric_value(metrics: Dict[str, object], x_metric: str):
    """
    Map user-facing x metric names to extracted metrics.
    """
    mapping = {
        "avg_windows_used": metrics.get("time_avg_windows_used"),
        "expected_mflops": metrics.get("expected_mflops"),
        "avg_compute_units": metrics.get("time_avg_compute_units"),
        "windows_saved_pct": metrics.get("time_windows_saved_pct"),
        "compute_saved_pct": metrics.get("time_compute_saved_pct"),
        "avg_exit_depth": metrics.get("avg_exit_depth"),
        "latency_ms_full_forward": metrics.get("full_forward_latency_ms"),
    }
    if x_metric not in mapping:
        raise KeyError(f"Unsupported x_metric: {x_metric}")
    return mapping[x_metric]


def x_metric_label(x_metric: str) -> str:
    labels = {
        "avg_windows_used": "Avg windows used/clip",
        "expected_mflops": "Expected MFLOPs",
        "avg_compute_units": "Avg compute units",
        "windows_saved_pct": "Windows saved (%)",
        "compute_saved_pct": "Compute saved (%)",
        "avg_exit_depth": "Avg exit depth",
        "latency_ms_full_forward": "Full forward latency (ms)",
    }
    return labels[x_metric]


# ---------- Input parsing ----------

def parse_pairs(pair_items: List[str]) -> List[Tuple[str, str, str]]:
    parsed: List[Tuple[str, str, str]] = []
    for item in pair_items:
        parts = item.split("|")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid pair spec: {item}\n"
                "Expected format: Label|no_hint_run_relpath|hint_run_relpath"
            )
        parsed.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return parsed


def build_progression_df(
    runs_root: Path,
    runs: List[str],
    labels: Optional[List[str]],
    x_metric: str,
) -> pd.DataFrame:
    if labels is not None and len(labels) != len(runs):
        raise ValueError("--labels must have the same length as --runs")

    rows = []
    for idx, run_rel in enumerate(runs):
        run_dir = runs_root / run_rel
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        m = load_run_metrics(run_dir)
        label = labels[idx] if labels is not None else run_dir.name
        rows.append({
            "label": label,
            "group": "progression",
            "pair_name": "",
            **m,
            "x_value": x_metric_value(m, x_metric),
        })

    return pd.DataFrame(rows)


def build_paired_df(
    runs_root: Path,
    pairs: List[Tuple[str, str, str]],
    x_metric: str,
) -> pd.DataFrame:
    rows = []
    for pair_name, no_hint_rel, hint_rel in pairs:
        no_hint_dir = runs_root / no_hint_rel
        hint_dir = runs_root / hint_rel

        if not no_hint_dir.exists():
            raise FileNotFoundError(f"No-hint run directory not found: {no_hint_dir}")
        if not hint_dir.exists():
            raise FileNotFoundError(f"Hint run directory not found: {hint_dir}")

        for group_label, run_dir in [("no-hint", no_hint_dir), ("hint", hint_dir)]:
            m = load_run_metrics(run_dir)
            label = f"{pair_name} ({group_label})"
            rows.append({
                "label": label,
                "group": group_label,
                "pair_name": pair_name,
                **m,
                "x_value": x_metric_value(m, x_metric),
            })

    return pd.DataFrame(rows)


# ---------- Plotting ----------

def _annotate_points(ax, df: pd.DataFrame, y_col: str):
    for _, row in df.iterrows():
        x = row["x_value"]
        y = row[y_col]
        if pd.isna(x) or pd.isna(y):
            continue
        ax.annotate(
            row["label"],
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )


def _plot_progression(ax, df: pd.DataFrame, y_col: str):
    sub = df.dropna(subset=["x_value", y_col]).copy()
    ax.scatter(sub["x_value"], sub[y_col], s=24)
    _annotate_points(ax, sub, y_col)


def _plot_paired(ax, df: pd.DataFrame, y_col: str, connect_pairs: bool):
    no_hint = df[df["group"] == "no-hint"].dropna(subset=["x_value", y_col]).copy()
    hint = df[df["group"] == "hint"].dropna(subset=["x_value", y_col]).copy()

    if not no_hint.empty:
        ax.scatter(no_hint["x_value"], no_hint[y_col], marker="o", s=24, label="No-hint")
    if not hint.empty:
        ax.scatter(hint["x_value"], hint[y_col], marker="s", s=24, label="Hint")

    _annotate_points(ax, no_hint, y_col)
    _annotate_points(ax, hint, y_col)

    if connect_pairs:
        for pair_name in sorted(set(df["pair_name"])):
            chunk = df[df["pair_name"] == pair_name].dropna(subset=["x_value", y_col])
            if len(chunk) == 2:
                chunk = chunk.sort_values("group")
                ax.plot(chunk["x_value"], chunk[y_col], linestyle="--", linewidth=0.8)


def plot_single_panel_scatter(
    df: pd.DataFrame,
    mode: str,
    x_metric: str,
    y_col: str,
    y_label: str,
    out_path: Path,
    connect_pairs: bool,
    width: float,
    height: float,
    show_legend: bool,
):
    fig, ax = plt.subplots(figsize=(width, height))

    if mode == "progression":
        _plot_progression(ax, df, y_col)
    elif mode == "paired":
        _plot_paired(ax, df, y_col, connect_pairs=connect_pairs)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    ax.set_xlabel(x_metric_label(x_metric))
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

    if mode == "paired" and show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, frameon=False)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_three_panel_scatter(
    df: pd.DataFrame,
    mode: str,
    x_metric: str,
    out_path: Path,
    connect_pairs: bool,
    width: float,
    height: float,
    show_legend: bool,
):
    fig, axes = plt.subplots(1, 3, figsize=(width, height))

    panels = [
        ("segment_policy_accuracy", "Segment policy acc."),
        ("full_policy_test_accuracy", "Full-clip policy acc."),
        ("time_policy_test_accuracy", "Depth×Time policy acc."),
    ]

    for ax, (y_col, y_label) in zip(axes, panels):
        if mode == "progression":
            _plot_progression(ax, df, y_col)
        elif mode == "paired":
            _plot_paired(ax, df, y_col, connect_pairs=connect_pairs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        ax.set_xlabel(x_metric_label(x_metric))
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    if mode == "paired" and show_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Plot reusable paper-friendly trade-off figures for NeuroAccuExit runs.")
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--mode", choices=["progression", "paired"], required=True)

    # progression mode
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Run paths relative to runs_root. Used in progression mode.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional labels aligned with --runs.")

    # paired mode
    parser.add_argument("--pairs", nargs="*", default=None,
                        help='Pair specs: "Label|no_hint_relpath|hint_relpath". Used in paired mode.')

    parser.add_argument("--x_metric", choices=[
        "avg_windows_used",
        "expected_mflops",
        "avg_compute_units",
        "windows_saved_pct",
        "compute_saved_pct",
        "avg_exit_depth",
        "latency_ms_full_forward",
    ], default="avg_windows_used")

    parser.add_argument("--out", type=Path, required=True,
                        help="Combined output path. Use .pdf for paper-ready vector output.")
    parser.add_argument("--csv_out", type=Path, default=None)
    parser.add_argument("--connect_pairs", action="store_true",
                        help="In paired mode, draw dashed lines between each no-hint / hint pair.")

    parser.add_argument("--save_individual", action="store_true",
                        help="Save each panel as a separate figure.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Directory for individual figures. Required if --save_individual is used.")
    parser.add_argument("--skip_combined", action="store_true",
                        help="Do not save the combined 3-panel figure.")

    # Paper-friendly sizing / formatting
    parser.add_argument("--combined_width", type=float, default=8.6,
                        help="Width of the combined figure in inches.")
    parser.add_argument("--combined_height", type=float, default=2.7,
                        help="Height of the combined figure in inches.")
    parser.add_argument("--single_width", type=float, default=3.0,
                        help="Width of each individual figure in inches.")
    parser.add_argument("--single_height", type=float, default=2.4,
                        help="Height of each individual figure in inches.")
    parser.add_argument("--no_legend", action="store_true",
                        help="Hide legend even in paired mode.")

    args = parser.parse_args()

    if args.mode == "progression":
        if not args.runs:
            raise ValueError("In progression mode, provide --runs.")
        df = build_progression_df(
            runs_root=args.runs_root,
            runs=args.runs,
            labels=args.labels,
            x_metric=args.x_metric,
        )
    else:
        if not args.pairs:
            raise ValueError("In paired mode, provide --pairs.")
        pair_specs = parse_pairs(args.pairs)
        df = build_paired_df(
            runs_root=args.runs_root,
            pairs=pair_specs,
            x_metric=args.x_metric,
        )

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_out, index=False)

    panels = [
        ("segment_policy_accuracy", "Segment policy acc.", "segment_policy_accuracy"),
        ("full_policy_test_accuracy", "Full-clip policy acc.", "full_policy_test_accuracy"),
        ("time_policy_test_accuracy", "Depth×Time policy acc.", "depth_time_policy_test_accuracy"),
    ]

    if args.save_individual:
        if args.out_dir is None:
            raise ValueError("Provide --out_dir when using --save_individual.")

        # Match extension to the combined output. Default to PDF if no suffix.
        ext = args.out.suffix if args.out.suffix else ".pdf"

        for y_col, y_label, stem in panels:
            single_out = args.out_dir / f"{stem}_vs_{args.x_metric}{ext}"
            plot_single_panel_scatter(
                df=df,
                mode=args.mode,
                x_metric=args.x_metric,
                y_col=y_col,
                y_label=y_label,
                out_path=single_out,
                connect_pairs=args.connect_pairs,
                width=args.single_width,
                height=args.single_height,
                show_legend=not args.no_legend,
            )
            print(f"Saved individual figure: {single_out}")

    if not args.skip_combined:
        plot_three_panel_scatter(
            df=df,
            mode=args.mode,
            x_metric=args.x_metric,
            out_path=args.out,
            connect_pairs=args.connect_pairs,
            width=args.combined_width,
            height=args.combined_height,
            show_legend=not args.no_legend,
        )
        print(f"Saved combined figure: {args.out}")

    if args.csv_out is not None:
        print(f"Saved source CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
