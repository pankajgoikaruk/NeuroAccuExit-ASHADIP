# scripts/v0.12_EE/label_aware_policy/common_v012.py

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, jaccard_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        raise TypeError(f"Cannot serialize {type(value)}")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=convert)


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result:
        raise ValueError("Expected at least one comma-separated float.")
    return sorted(set(result))


def parse_tap_blocks(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def load_run_config(run_dir: Path) -> dict[str, Any]:
    json_path = run_dir / "config_used.json"
    if json_path.exists():
        return load_json(json_path)

    yaml_path = run_dir / "config_used.yaml"
    if yaml_path.exists():
        from utils.config import load_config

        return load_config(yaml_path) or {}

    raise FileNotFoundError(
        f"No config_used.json or config_used.yaml found in run directory: {run_dir}"
    )


def load_labels(labels_json: Path, cfg: dict[str, Any]) -> list[str]:
    payload = load_json(labels_json)
    if isinstance(payload, list):
        labels = payload
    else:
        labels = payload.get("labels") or payload.get("classes") or cfg.get("labels")

    if not labels:
        raise RuntimeError(f"Could not resolve labels from {labels_json}")
    return [str(label) for label in labels]


def resolve_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(cfg.get("model") or {})
    if "exit_hint" not in model_cfg:
        exit_hint = cfg.get("exit_hint")
        model_cfg["exit_hint"] = (
            exit_hint if isinstance(exit_hint, dict) else {"enable": False}
        )
    return model_cfg


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def threshold_vector(mapping: dict[str, Any], labels: list[str]) -> np.ndarray:
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(f"Threshold mapping is missing labels: {missing}")
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def threshold_mapping(
    labels: list[str],
    thresholds: np.ndarray,
) -> dict[str, float]:
    return {
        label: float(thresholds[idx])
        for idx, label in enumerate(labels)
    }


def load_thresholds_by_exit(
    *,
    run_dir: Path,
    labels: list[str],
    num_exits: int,
    threshold_mode: str,
    fixed_threshold: float,
) -> list[np.ndarray]:
    if threshold_mode == "fixed_0p5":
        base = np.full(len(labels), float(fixed_threshold), dtype=np.float32)
        return [base.copy() for _ in range(num_exits)]

    comparison_path = run_dir / "threshold_tuning" / "threshold_comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError(
            "Per-exit threshold file not found. Expected:\n"
            f"  {comparison_path}\n"
            "Use --threshold_mode fixed_0p5 when the canonical run has no "
            "per-exit threshold artefact."
        )

    payload = load_json(comparison_path)
    exits = payload.get("exits", [])
    if len(exits) < num_exits:
        raise RuntimeError(
            f"Threshold file contains {len(exits)} exits; model has {num_exits}."
        )

    if threshold_mode == "tuned_per_exit":
        return [
            threshold_vector(exits[idx]["tuned_thresholds"], labels)
            for idx in range(num_exits)
        ]

    if threshold_mode == "final_exit_tuned":
        final_vector = threshold_vector(
            exits[num_exits - 1]["tuned_thresholds"],
            labels,
        )
        return [final_vector.copy() for _ in range(num_exits)]

    raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")


@torch.no_grad()
def collect_outputs(
    model: torch.nn.Module,
    loader,
    device: str,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    model.eval()
    target_parts: list[np.ndarray] = []
    probability_parts: list[list[np.ndarray]] | None = None
    frames: int | None = None

    for x, y in loader:
        if frames is None:
            frames = int(x.shape[-1])

        logits_by_exit = model(x.to(device))
        probabilities = [
            torch.sigmoid(logits).detach().cpu().numpy()
            for logits in logits_by_exit
        ]
        if probability_parts is None:
            probability_parts = [[] for _ in probabilities]

        target_parts.append(y.detach().cpu().numpy())
        for exit_idx, probs in enumerate(probabilities):
            probability_parts[exit_idx].append(probs)

    if not target_parts or probability_parts is None or frames is None:
        raise RuntimeError("Loader produced no samples.")

    return (
        np.concatenate(target_parts, axis=0).astype(np.int8),
        [
            np.concatenate(parts, axis=0).astype(np.float32)
            for parts in probability_parts
        ],
        frames,
    )


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "samples_f1": float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_score": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
    }


def load_lats_module():
    evaluator_path = (
        PROJECT_ROOT
        / "scripts"
        / "v0.10"
        / "evaluate_frozen_lats_config_v010.py"
    )
    if not evaluator_path.exists():
        raise FileNotFoundError(f"Frozen LATS evaluator not found: {evaluator_path}")

    spec = importlib.util.spec_from_file_location(
        "v010_frozen_lats_evaluator",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator module: {evaluator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_level_metrics(
    *,
    metadata_df: pd.DataFrame,
    labels: list[str],
    probabilities: np.ndarray,
    lats_config_json: Path,
    parent_id_col: str,
    lats_module,
) -> dict[str, float]:
    if parent_id_col not in metadata_df.columns:
        raise RuntimeError(
            f"Manifest does not contain parent ID column {parent_id_col!r}."
        )
    if len(metadata_df) != len(probabilities):
        raise RuntimeError(
            f"Metadata rows ({len(metadata_df)}) do not match probability rows "
            f"({len(probabilities)})."
        )

    frame = metadata_df[[parent_id_col, *labels]].copy()
    prefix = "dynamic_prob_"
    for label_idx, label in enumerate(labels):
        frame[f"{prefix}{label}"] = probabilities[:, label_idx]

    cfg = lats_module.load_frozen_config(
        lats_config_json,
        labels,
        0.5,
    )
    truth_df, _, pred_df = lats_module.make_parent_tables(
        df=frame,
        labels=labels,
        cfg=cfg,
        parent_id_col=parent_id_col,
        prob_prefix=prefix,
    )
    return lats_module.compute_metrics(
        truth_df[labels].to_numpy(dtype=int),
        pred_df[labels].to_numpy(dtype=int),
    )


def load_feature(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    array = np.load(path).astype(np.float32)
    if array.ndim != 2:
        raise RuntimeError(f"Expected [n_mels, T], got {array.shape}: {path}")
    return torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
