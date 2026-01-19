"""scripts/wandb_log_run.py

Post-hoc W&B logging for a completed run directory.

Design goal: *zero* impact on training / accuracy / efficiency.
This script only reads artifacts already produced by the pipeline:
  - meta.json (created by scripts/run_full.ps1)
  - metrics.json (training curves)
  - summary.json (from scripts/summarize_run.py)
  - profiling.json (from scripts/profile_latency.py)
  - plots/*.png (optional; produced by scripts/summarize_run.py)

Enable/disable via env var (recommended):
  ENABLE_WANDB=1
  WANDB_PROJECT=NeuroAccuExit-ASHADIP
  WANDB_MODE=offline   (or: online / disabled)
  WANDB_ENTITY=pankajgoikar-lancaster-university   (optional)

If ENABLE_WANDB is not truthy, this script exits cleanly as a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


TRUTHY = {"1", "true", "yes", "y", "on"}


def _enabled() -> bool:
    return os.getenv("ENABLE_WANDB", "").strip().lower() in TRUTHY


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON safely; supports UTF-8 BOM via utf-8-sig."""
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    """Flatten nested dicts into wandb-friendly key/value pairs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}{k}/", v, out)
    else:
        # keep scalars only (wandb config must be JSON-serializable)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            out[prefix[:-1]] = obj


def main() -> int:
    # --- prevent local ./wandb folder from shadowing the wandb package ---
    import sys

    cwd = Path.cwd()
    cwd_wandb = cwd / "wandb"

    # Remove common shadowing entries (cwd, "", and the scripts dir)
    if cwd_wandb.exists() and cwd_wandb.is_dir():
        if "" in sys.path:
            sys.path.remove("")
        cwd_str = str(cwd)
        if cwd_str in sys.path:
            sys.path.remove(cwd_str)

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir in sys.path:
        # Removing this is safe for this post-hoc script and avoids rare shadowing issues.
        sys.path.remove(scripts_dir)
    # --------------------------------------------------------------------

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)   # optional override (otherwise: run_dir.name)
    ap.add_argument("--group", default=None)  # optional override (otherwise: run_dir.parent.name)
    ap.add_argument("--tags", default="", help="Comma-separated tags")
    ap.add_argument("--log_plots", action="store_true", help="Upload plots/*.png if present")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    if not _enabled():
        print("[wandb_log_run] ENABLE_WANDB not set -> no-op")
        return 0

    # Import only when enabled.
    try:
        import wandb
    except Exception as e:
        print(f"[wandb_log_run] wandb import failed ({e}). Install with: pip install wandb")
        return 0

    # Helpful debug to confirm we're importing the *real* wandb package.
    print("[wandb_log_run] wandb.__file__ =", getattr(wandb, "__file__", None))

    project = args.project or os.getenv("WANDB_PROJECT", "NeuroAccuExit-ASHADIP")
    mode = os.getenv("WANDB_MODE", "offline")

    meta = _load_json(run_dir / "meta.json", {}) or {}
    summary = _load_json(run_dir / "summary.json", {}) or {}
    metrics = _load_json(run_dir / "metrics.json", {}) or {}
    profiling = _load_json(run_dir / "profiling.json", {}) or {}

    # --------- RUN NAME / GROUP (as you requested) ----------
    # Name shown on W&B site:
    #   v0_3_001, v0_3_002, v0_3_003, ...
    # If you pass --name, it overrides this.
    name = args.name or run_dir.name

    # Group runs by parent folder, e.g. runs/v0_3/...
    # If you pass --group, it overrides this.
    group = args.group or run_dir.parent.name

    # --------- Config (flattened) — safe, static metadata only ----------
    config: Dict[str, Any] = {}
    _flatten("meta/", meta, config)

    pol = (summary.get("policy_summary") or {})
    for k in ["policy_name", "tau", "ea_threshold", "ea_mode", "compute_saving_pct", "expected_mflops", "full_mflops"]:
        if k in pol:
            config[f"policy/{k}"] = pol.get(k)

    # --------- Tags ----------
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    variant = meta.get("variant") or (summary.get("policy_summary", {}) or {}).get("variant")
    policy_param = meta.get("policy")
    if variant:
        tags.append(str(variant))
    if policy_param:
        tags.append(f"policy_{policy_param}")
    if os.getenv("WANDB_TAGS"):
        tags.extend([t.strip() for t in os.getenv("WANDB_TAGS", "").split(",") if t.strip()])

    # Deduplicate while keeping order
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    # --------- Unique run ID per run_dir + no resume ----------
    # Stable across re-logs of the SAME run_dir, different across v0_3_001 vs v0_3_002 etc.
    run_uid = hashlib.md5(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:12]

    entity = os.getenv("WANDB_ENTITY") or None
    wb = wandb.init(
        entity=entity,
        project=project,
        name=name,          # display name: v0_3_003
        group=group,        # display grouping: v0_3
        config=config,
        mode=mode,
        tags=tags,
        id=run_uid,         # unique per run_dir
        resume="never",     # prevents overwriting/resuming older runs
    )

    try:
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
    except Exception:
        pass

    # -------------------- Log curves (from metrics.json) --------------------
    train_curve = metrics.get("train", []) if isinstance(metrics, dict) else []
    val_curve = metrics.get("val", []) if isinstance(metrics, dict) else []

    # Index by epoch to merge train + val.
    by_ep: Dict[int, Dict[str, Any]] = {}
    for r in train_curve:
        try:
            ep = int(r.get("epoch"))
        except Exception:
            continue
        by_ep.setdefault(ep, {})
        by_ep[ep]["train/loss"] = _safe_float(r.get("loss"))
        ta = r.get("acc")
        if isinstance(ta, (list, tuple)) and len(ta) == 3:
            by_ep[ep]["train/acc_exit1"] = _safe_float(ta[0])
            by_ep[ep]["train/acc_exit2"] = _safe_float(ta[1])
            by_ep[ep]["train/acc_exit3"] = _safe_float(ta[2])

    for r in val_curve:
        try:
            ep = int(r.get("epoch"))
        except Exception:
            continue
        by_ep.setdefault(ep, {})
        va = r.get("acc")
        if isinstance(va, (list, tuple)) and len(va) == 3:
            by_ep[ep]["val/acc_exit1"] = _safe_float(va[0])
            by_ep[ep]["val/acc_exit2"] = _safe_float(va[1])
            by_ep[ep]["val/acc_exit3"] = _safe_float(va[2])

    for ep in sorted(by_ep.keys()):
        payload = {k: v for k, v in by_ep[ep].items() if v is not None}
        payload["epoch"] = ep
        wandb.log(payload)

    # -------------------- Log final summary scalars (from summary.json) --------------------
    pol = summary.get("policy_summary") if isinstance(summary, dict) else {}
    if isinstance(pol, dict):
        scalars = {
            "test/policy_acc": _safe_float(pol.get("policy_test_acc")),
            "test/avg_exit_depth": _safe_float(pol.get("avg_exit_depth")),
            "test/compute_saving_pct": _safe_float(pol.get("compute_saving_pct")),
            "policy/tau": _safe_float(pol.get("tau")),
            "policy/expected_mflops": _safe_float(pol.get("expected_mflops")),
            "policy/full_mflops": _safe_float(pol.get("full_mflops")),
            "policy/temp_e1": _safe_float((pol.get("temperatures") or [None, None, None])[0] if isinstance(pol.get("temperatures"), list) else None),
            "policy/temp_e2": _safe_float((pol.get("temperatures") or [None, None, None])[1] if isinstance(pol.get("temperatures"), list) else None),
            "policy/temp_e3": _safe_float((pol.get("temperatures") or [None, None, None])[2] if isinstance(pol.get("temperatures"), list) else None),
            "policy/flip_rate": _safe_float(pol.get("flip_rate")),
            "policy/exit_consistency": _safe_float(pol.get("exit_consistency")),
            "policy/ea_threshold": _safe_float(pol.get("ea_threshold")),
        }

        mx = pol.get("exit_mix")
        if isinstance(mx, dict):
            scalars["policy/exit_mix_e1"] = _safe_float(mx.get("e1"))
            scalars["policy/exit_mix_e2"] = _safe_float(mx.get("e2"))
            scalars["policy/exit_mix_e3"] = _safe_float(mx.get("e3"))

        pc = pol.get("policy_calibration")
        if isinstance(pc, dict):
            scalars["calib/ece_policy"] = _safe_float(pc.get("ece"))

        pec = pol.get("per_exit_calibration")
        if isinstance(pec, dict):
            for ex in ("exit1", "exit2", "exit3"):
                if isinstance(pec.get(ex), dict):
                    scalars[f"calib/ece_{ex}"] = _safe_float(pec[ex].get("ece"))

        payload = {k: v for k, v in scalars.items() if v is not None}
        payload["epoch"] = 0
        wandb.log(payload)

    # -------------------- Log profiling (latency) --------------------
    if isinstance(profiling, dict):
        lat = profiling.get("latency_ms")
        if isinstance(lat, dict):
            wandb.log({
                "latency_ms/exit1": _safe_float(lat.get("exit1")),
                "latency_ms/exit2": _safe_float(lat.get("exit2")),
                "latency_ms/exit3": _safe_float(lat.get("exit3")),
                "epoch": 0,
            })

    # -------------------- Upload plots (optional) --------------------
    if args.log_plots:
        plots_dir = run_dir / "plots"
        if plots_dir.exists():
            for p in sorted(plots_dir.glob("*.png")):
                key = f"plots/{p.stem}"
                try:
                    wandb.log({key: wandb.Image(str(p)), "epoch": 0})
                except Exception:
                    pass

    wb.finish()
    print(f"[wandb_log_run] Logged run '{name}' to project '{project}' (mode={mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
