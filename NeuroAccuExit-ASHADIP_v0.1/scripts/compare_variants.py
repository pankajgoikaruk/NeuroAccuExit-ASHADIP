# scripts/compare_variants.py

import os
import json
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+_(\d+)$")


def find_new_run_dirs(root: Path):
    """
    NEW-ONLY layout (robust):
      runs/<VariantDir>/<RunDir>/
        must contain: meta.json and summary.json

    Example:
      runs/V0/V0_001/{meta.json, summary.json}
      runs/EA/EA_003/{meta.json, summary.json}

    We do NOT include legacy timestamp runs (no meta.json in your new scheme).
    """
    runs_root = root / "runs"
    if not runs_root.exists():
        return []

    run_dirs = []
    # Walk only under ./runs
    for dirpath, _, filenames in os.walk(runs_root):
        if "summary.json" not in filenames:
            continue
        if "meta.json" not in filenames:
            # New runs should always have meta.json -> ignore legacy
            continue

        run_dir = Path(dirpath)  # .../runs/<variant_dir>/<run_dir>
        variant_dir = run_dir.parent
        runs_dir = variant_dir.parent

        # enforce depth: runs/<variant_dir>/<run_dir>
        if runs_dir.name != "runs":
            continue

        run_dirs.append(run_dir)

    return run_dirs


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_variant_runid_from_meta(run_dir: Path):
    meta = load_json(run_dir / "meta.json")

    variant = meta.get("variant", None)
    run_id = meta.get("run_id", None)

    # Fallbacks
    if not run_id:
        run_id = run_dir.name
    if not variant:
        # Prefer meta.variant_safe if exists, else variant folder name
        variant = meta.get("variant_safe", run_dir.parent.name)

    return str(variant), str(run_id), meta


def load_summary_row(run_dir: Path):
    """
    Load meta.json + summary.json and extract key metrics for comparison.
    """
    variant, run_id, meta = parse_variant_runid_from_meta(run_dir)
    summ = load_json(run_dir / "summary.json")

    # Policy summary is the primary metrics location
    policy = summ.get("policy_summary", {})

    tau = policy.get("tau", None)
    temps = policy.get("temperatures", [None, None, None])
    exit_mix = policy.get("exit_mix", {})
    e1 = exit_mix.get("e1", None)
    e2 = exit_mix.get("e2", None)
    e3 = exit_mix.get("e3", None)

    # Some extra helpful columns (from meta), useful for debugging later
    device = meta.get("device", None)
    segment_sec = meta.get("segment_sec", None)
    hop_sec = meta.get("hop_sec", None)

    row = {
        "run_id": run_id,
        "variant": variant,

        "tau": tau,
        "temp_e1": temps[0] if len(temps) > 0 else None,
        "temp_e2": temps[1] if len(temps) > 1 else None,
        "temp_e3": temps[2] if len(temps) > 2 else None,

        "policy_test_acc": policy.get("policy_test_acc", None),

        "exit_e1": e1,
        "exit_e2": e2,
        "exit_e3": e3,

        "expected_mflops": policy.get("expected_mflops", None),
        "full_mflops": policy.get("full_mflops", None),
        "compute_saving_pct": policy.get("compute_saving_pct", None),

        "ece_policy": (policy.get("policy_calibration") or {}).get("ece", None),

        "n_mels": policy.get("n_mels", None),
        "frames": policy.get("frames", None),
        "num_classes": policy.get("num_classes", None),

        # meta fields (optional, but nice)
        "device": device,
        "segment_sec": segment_sec,
        "hop_sec": hop_sec,
    }

    return row


def make_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accuracy vs compute saving
    plt.figure()
    for variant, df_v in df.groupby("variant"):
        plt.scatter(
            df_v["compute_saving_pct"],
            df_v["policy_test_acc"],
            label=variant,
            s=40,
        )
        for _, row in df_v.iterrows():
            plt.text(
                row["compute_saving_pct"],
                row["policy_test_acc"],
                row["run_id"],
                fontsize=7,
                ha="left",
                va="bottom",
            )
    plt.xlabel("Compute saving (%)")
    plt.ylabel("Policy test accuracy")
    plt.title("Accuracy vs compute saving (new runs only)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_compute_saving.png", dpi=150)
    plt.close()

    # 2) Exit mix stacked bar per run
    plt.figure(figsize=(10, 4))
    df_plot = df.reset_index(drop=True)
    x = range(len(df_plot))
    e1 = df_plot["exit_e1"]
    e2 = df_plot["exit_e2"]
    e3 = df_plot["exit_e3"]

    plt.bar(x, e1, label="exit1")
    plt.bar(x, e2, bottom=e1, label="exit2")
    plt.bar(x, e3, bottom=e1 + e2, label="exit3")
    plt.xticks(x, df_plot["run_id"], rotation=45, ha="right")
    plt.ylabel("Fraction of samples")
    plt.title("Exit mix per run (new runs only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exit_mix_per_run.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Project root (default: current dir). Uses only ./runs/<variant>/<run>/ with meta.json+summary.json.",
    )
    ap.add_argument(
        "--out_csv",
        default="analysis/all_runs_summary.csv",
        help="Path to save aggregated CSV.",
    )
    ap.add_argument(
        "--out_dir",
        default="analysis/plots",
        help="Directory to save comparison plots.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    run_dirs = find_new_run_dirs(root)
    if not run_dirs:
        raise SystemExit(
            f"No NEW-style runs found under {root / 'runs'}.\n"
            f"Expected folders like runs/<Variant>/<RunId>/ containing meta.json and summary.json."
        )

    rows = []
    for rd in run_dirs:
        try:
            rows.append(load_summary_row(rd))
        except Exception as e:
            print(f"[warn] Skipping run at {rd} due to error: {e}")

    if not rows:
        raise SystemExit("Found run directories but could not load any rows (all failed).")

    df = pd.DataFrame(rows)

    # Optional: filter out weird/invalid run ids
    df = df[df["run_id"].astype(str).apply(lambda s: bool(RUN_ID_RE.match(s)))]
    df = df.sort_values(["variant", "run_id"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated CSV to {out_csv}")

    make_plots(df, Path(args.out_dir))
    print(f"Saved comparison plots under {args.out_dir}")


if __name__ == "__main__":
    main()




# # scripts/compare_variants.py
#
# import os
# import json
# import argparse
# import re
# from pathlib import Path
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+_(\d+)$")
#
#
# def find_summary_files_new_only(root: Path):
#     """
#     NEW-ONLY layout:
#       runs/<Variant>/<RunId>/summary.json
#         e.g. runs/V0/V0_001/summary.json
#              runs/EA/EA_003/summary.json
#
#     Ignores legacy:
#       runs/<timestamp>/summary.json
#       runs_v1/<timestamp>/summary.json
#       etc.
#     """
#     summaries = []
#     runs_root = root / "runs"
#     if not runs_root.exists():
#         return summaries
#
#     # Walk only under ./runs
#     for dirpath, _, filenames in os.walk(runs_root):
#         if "summary.json" not in filenames:
#             continue
#
#         run_dir = Path(dirpath)         # .../runs/<Variant>/<RunId>
#         variant_dir = run_dir.parent    # .../runs/<Variant>
#         runs_dir = variant_dir.parent   # .../runs
#
#         # Enforce exact new layout depth
#         if runs_dir.name != "runs":
#             continue
#
#         variant = variant_dir.name
#         run_id = run_dir.name
#
#         # Must match <Variant>_<digits>
#         if not run_id.startswith(variant + "_"):
#             continue
#         if not RUN_ID_RE.match(run_id):
#             continue
#
#         summaries.append(run_dir / "summary.json")
#
#     return summaries
#
#
# def parse_variant_from_path_new(summary_path: Path):
#     """
#     NEW layout:
#       .../runs/<Variant>/<RunId>/summary.json -> '<Variant>'
#     """
#     run_dir = summary_path.parent
#     variant_dir = run_dir.parent
#     runs_dir = variant_dir.parent
#     if runs_dir.name != "runs":
#         return None
#     return variant_dir.name
#
#
# def load_summary(summary_path: Path):
#     with open(summary_path, "r", encoding="utf-8") as f:
#         summ = json.load(f)
#
#     variant = parse_variant_from_path_new(summary_path)
#     run_id = summ.get("run_id", summary_path.parent.name)
#
#     # Ensure run_id is consistent with folder name (new scheme)
#     # Use folder name as canonical (keeps plots clean)
#     run_id = summary_path.parent.name
#
#     policy = summ.get("policy_summary", {})
#
#     tau = policy.get("tau", None)
#     temps = policy.get("temperatures", [None, None, None])
#     exit_mix = policy.get("exit_mix", {})
#     e1 = exit_mix.get("e1", None)
#     e2 = exit_mix.get("e2", None)
#     e3 = exit_mix.get("e3", None)
#
#     row = {
#         "run_id": run_id,
#         "variant": variant,
#         "tau": tau,
#         "temp_e1": temps[0] if len(temps) > 0 else None,
#         "temp_e2": temps[1] if len(temps) > 1 else None,
#         "temp_e3": temps[2] if len(temps) > 2 else None,
#         "policy_test_acc": policy.get("policy_test_acc", None),
#         "exit_e1": e1,
#         "exit_e2": e2,
#         "exit_e3": e3,
#         "expected_mflops": policy.get("expected_mflops", None),
#         "full_mflops": policy.get("full_mflops", None),
#         "compute_saving_pct": policy.get("compute_saving_pct", None),
#         "ece_policy": (policy.get("policy_calibration") or {}).get("ece", None),
#         "n_mels": policy.get("n_mels", None),
#         "frames": policy.get("frames", None),
#         "num_classes": policy.get("num_classes", None),
#     }
#     return row
#
#
# def make_plots(df: pd.DataFrame, out_dir: Path):
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     # 1) Accuracy vs compute saving
#     plt.figure()
#     for variant, df_v in df.groupby("variant"):
#         plt.scatter(
#             df_v["compute_saving_pct"],
#             df_v["policy_test_acc"],
#             label=variant,
#             s=40,
#         )
#         for _, row in df_v.iterrows():
#             plt.text(
#                 row["compute_saving_pct"],
#                 row["policy_test_acc"],
#                 row["run_id"],
#                 fontsize=7,
#                 ha="left",
#                 va="bottom",
#             )
#     plt.xlabel("Compute saving (%)")
#     plt.ylabel("Policy test accuracy")
#     plt.title("Accuracy vs compute saving (new runs only)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(out_dir / "acc_vs_compute_saving.png", dpi=150)
#     plt.close()
#
#     # 2) Exit mix stacked bar per run
#     plt.figure(figsize=(10, 4))
#     df_plot = df.reset_index(drop=True)
#     x = range(len(df_plot))
#     e1 = df_plot["exit_e1"]
#     e2 = df_plot["exit_e2"]
#     e3 = df_plot["exit_e3"]
#
#     plt.bar(x, e1, label="exit1")
#     plt.bar(x, e2, bottom=e1, label="exit2")
#     plt.bar(x, e3, bottom=e1 + e2, label="exit3")
#     plt.xticks(x, df_plot["run_id"], rotation=45, ha="right")
#     plt.ylabel("Fraction of samples")
#     plt.title("Exit mix per run (new runs only)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_dir / "exit_mix_per_run.png", dpi=150)
#     plt.close()
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--root",
#         default=".",
#         help="Project root (default: current dir). Only ./runs/<Variant>/<RunId>/summary.json is used.",
#     )
#     ap.add_argument(
#         "--out_csv",
#         default="analysis/all_runs_summary.csv",
#         help="Path to save aggregated CSV.",
#     )
#     ap.add_argument(
#         "--out_dir",
#         default="analysis/plots",
#         help="Directory to save comparison plots.",
#     )
#     args = ap.parse_args()
#
#     root = Path(args.root)
#     summaries = find_summary_files_new_only(root)
#     if not summaries:
#         raise SystemExit(
#             f"No NEW-style summary.json files found under {root / 'runs'}.\n"
#             f"Expected: runs/<Variant>/<RunId>/summary.json (e.g., runs/V0/V0_001/summary.json)"
#         )
#
#     rows = [load_summary(p) for p in summaries]
#     df = pd.DataFrame(rows).sort_values(["variant", "run_id"])
#
#     out_csv = Path(args.out_csv)
#     out_csv.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(out_csv, index=False)
#     print(f"Saved aggregated CSV to {out_csv}")
#
#     make_plots(df, Path(args.out_dir))
#     print(f"Saved comparison plots under {args.out_dir}")
#
#
# if __name__ == "__main__":
#     main()
