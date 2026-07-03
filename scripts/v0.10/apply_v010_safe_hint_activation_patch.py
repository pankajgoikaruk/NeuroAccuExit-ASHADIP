#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply v0.10 safe hint-pass activation patch.

Purpose
-------
This patch keeps old single-label / moth hint-pass behaviour safe by preserving
softmax as the default hint activation inside models/exit_net.py, while allowing
human-talk multi-label training/evaluation to explicitly use sigmoid hints.

Files patched:
  - models/exit_net.py
  - utils/model_factory.py
  - training/train_multilabel.py
  - scripts/run_tata_weakclip_experiment.ps1
  - scripts/evaluate_tata_final_holdout_parent_level.py

Run from repository root:
  python scripts/v0.10/apply_v010_safe_hint_activation_patch.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
BACKUP_SUFFIX = ".bak_v010_safe_hint_activation"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def backup(path: Path) -> None:
    bak = path.with_name(path.name + BACKUP_SUFFIX)
    if not bak.exists():
        bak.write_text(read(path), encoding="utf-8")


def replace_once(text: str, old: str, new: str, file_label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected block not found in {file_label}. Patch aborted.")
    return text.replace(old, new, 1)


def patch_exit_net() -> None:
    path = ROOT / "models" / "exit_net.py"
    text = read(path)
    if "hint_activation" in text and "def _hint_probs" in text:
        print(f"[SKIP] {path} already appears patched.")
        return

    backup(path)

    old_sig = '''        hint_dim: int = 0,\n        hint_source: str = "probs",\n        hint_detach: bool = True,\n        hint_use_stats: bool = True,\n    ):'''
    new_sig = '''        hint_dim: int = 0,\n        hint_source: str = "probs",\n        hint_detach: bool = True,\n        hint_use_stats: bool = True,\n        hint_activation: str = "softmax",\n    ):'''
    text = replace_once(text, old_sig, new_sig, str(path))

    old_cfg = '''        self.hint_dim = int(hint_dim)\n        self.hint_source = str(hint_source).lower().strip()\n        self.hint_detach = bool(hint_detach)\n        self.hint_use_stats = bool(hint_use_stats)\n        self.use_exit_hints = self.hint_dim > 0\n\n        if self.hint_source not in {"probs", "logits"}:\n            raise ValueError(\n                f"hint_source must be 'probs' or 'logits', got {self.hint_source}"\n            )'''
    new_cfg = '''        self.hint_dim = int(hint_dim)\n        self.hint_source = str(hint_source).lower().strip()\n        self.hint_detach = bool(hint_detach)\n        self.hint_use_stats = bool(hint_use_stats)\n        self.hint_activation = str(hint_activation).lower().strip()\n        self.use_exit_hints = self.hint_dim > 0\n\n        if self.hint_source not in {"probs", "logits"}:\n            raise ValueError(\n                f"hint_source must be 'probs' or 'logits', got {self.hint_source}"\n            )\n\n        if self.hint_activation not in {"softmax", "sigmoid"}:\n            raise ValueError(\n                "hint_activation must be 'softmax' or 'sigmoid', "\n                f"got {self.hint_activation}"\n            )'''
    text = replace_once(text, old_cfg, new_cfg, str(path))

    old_make_hint = '''    def _make_hint(self, logits: torch.Tensor, proj: nn.Module) -> torch.Tensor:\n        src = logits.detach() if self.hint_detach else logits\n\n        if self.hint_source == "probs":\n            base = F.softmax(src, dim=1)\n        else:\n            base = src\n\n        if self.hint_use_stats:\n            probs = F.softmax(src, dim=1)\n\n            conf = probs.max(dim=1, keepdim=True).values\n\n            top2 = probs.topk(k=min(2, probs.size(1)), dim=1).values\n            if top2.size(1) == 1:\n                margin = top2[:, :1]\n            else:\n                margin = top2[:, :1] - top2[:, 1:2]\n\n            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(\n                dim=1, keepdim=True\n            )\n\n            summary = torch.cat([base, conf, margin, entropy], dim=1)\n        else:\n            summary = base\n\n        return proj(summary)'''
    new_make_hint = '''    def _hint_probs(self, logits: torch.Tensor) -> torch.Tensor:\n        """Return probabilities for hint construction.\n\n        Default remains softmax for backward compatibility with old\n        single-label experiments. Human-talk multi-label runs should set\n        hint_activation='sigmoid'.\n        """\n        if self.hint_activation == "sigmoid":\n            return torch.sigmoid(logits)\n        return F.softmax(logits, dim=1)\n\n    def _hint_entropy(self, probs: torch.Tensor) -> torch.Tensor:\n        if self.hint_activation == "sigmoid":\n            p = probs.clamp_min(1e-8).clamp_max(1.0 - 1e-8)\n            return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).sum(\n                dim=1, keepdim=True\n            )\n\n        return -(probs * torch.log(probs.clamp_min(1e-8))).sum(\n            dim=1, keepdim=True\n        )\n\n    def _make_hint(self, logits: torch.Tensor, proj: nn.Module) -> torch.Tensor:\n        src = logits.detach() if self.hint_detach else logits\n\n        if self.hint_source == "probs":\n            base = self._hint_probs(src)\n        else:\n            base = src\n\n        if self.hint_use_stats:\n            probs = self._hint_probs(src)\n\n            conf = probs.max(dim=1, keepdim=True).values\n\n            top2 = probs.topk(k=min(2, probs.size(1)), dim=1).values\n            if top2.size(1) == 1:\n                margin = top2[:, :1]\n            else:\n                margin = top2[:, :1] - top2[:, 1:2]\n\n            entropy = self._hint_entropy(probs)\n\n            summary = torch.cat([base, conf, margin, entropy], dim=1)\n        else:\n            summary = base\n\n        return proj(summary)'''
    text = replace_once(text, old_make_hint, new_make_hint, str(path))

    write(path, text)
    print(f"[OK] Patched {path}")


def patch_model_factory() -> None:
    path = ROOT / "utils" / "model_factory.py"
    text = read(path)
    if '"hint_activation"' in text:
        print(f"[SKIP] {path} already appears patched.")
        return

    backup(path)

    old = '''    return {\n        "hint_dim": hint_dim,\n        "hint_source": str(hint_cfg.get("source", "probs")),\n        "hint_detach": bool(hint_cfg.get("detach", True)),\n        "hint_use_stats": bool(hint_cfg.get("use_stats", True)),\n    }'''
    new = '''    return {\n        "hint_dim": hint_dim,\n        "hint_source": str(hint_cfg.get("source", "probs")),\n        "hint_detach": bool(hint_cfg.get("detach", True)),\n        "hint_use_stats": bool(hint_cfg.get("use_stats", True)),\n        # Backward-compatible default is softmax. Multi-label human-talk\n        # experiments explicitly set activation='sigmoid'.\n        "hint_activation": str(\n            hint_cfg.get("activation", hint_cfg.get("hint_activation", "softmax"))\n        ),\n    }'''
    text = replace_once(text, old, new, str(path))
    write(path, text)
    print(f"[OK] Patched {path}")


def patch_train_multilabel() -> None:
    path = ROOT / "training" / "train_multilabel.py"
    text = read(path)
    if "--exit_hint" in text and "--hint_activation" in text:
        print(f"[SKIP] {path} already appears patched.")
        return

    backup(path)

    insert_after = '''    parser.add_argument(\n        "--synthetic_balance_power",\n        type=float,\n        default=0.0,\n        help="WeightedRandomSampler clean/synthetic balancing. 0 disables.",\n    )'''
    cli_block = '''    parser.add_argument(\n        "--synthetic_balance_power",\n        type=float,\n        default=0.0,\n        help="WeightedRandomSampler clean/synthetic balancing. 0 disables.",\n    )\n\n    parser.add_argument(\n        "--exit_hint",\n        action="store_true",\n        help="Enable local exit-to-exit hint passing for later exits.",\n    )\n    parser.add_argument("--hint_dim", type=int, default=8)\n    parser.add_argument(\n        "--hint_source",\n        default="probs",\n        choices=["probs", "logits"],\n        help="Use previous-exit probabilities or logits as the hint summary source.",\n    )\n    parser.add_argument(\n        "--hint_activation",\n        default="sigmoid",\n        choices=["softmax", "sigmoid"],\n        help="Probability activation for hint summaries. Use sigmoid for multi-label BCE tasks.",\n    )\n    parser.add_argument(\n        "--hint_detach",\n        dest="hint_detach",\n        action="store_true",\n        default=True,\n        help="Detach previous-exit logits before building hints.",\n    )\n    parser.add_argument(\n        "--no_hint_detach",\n        dest="hint_detach",\n        action="store_false",\n        help="Allow hint gradients to flow through previous-exit logits.",\n    )\n    parser.add_argument(\n        "--hint_use_stats",\n        dest="hint_use_stats",\n        action="store_true",\n        default=True,\n        help="Append confidence, margin, and entropy to hint summaries.",\n    )\n    parser.add_argument(\n        "--no_hint_use_stats",\n        dest="hint_use_stats",\n        action="store_false",\n        help="Use only the base previous-exit hint vector without confidence/margin/entropy stats.",\n    )'''
    text = replace_once(text, insert_after, cli_block, str(path))

    old_model_cfg = '''    # Important:\n    # For this first multi-label version, keep exit_hint disabled.\n    # Existing hint-passing was designed around softmax-style class probabilities.\n    model_cfg = {\n        "exit_hint": {\n            "enable": False,\n            "dim": 8,\n            "source": "probs",\n            "detach": True,\n            "use_stats": True,\n        }\n    }'''
    new_model_cfg = '''    model_cfg = {\n        "exit_hint": {\n            "enable": bool(args.exit_hint),\n            "dim": int(args.hint_dim),\n            "source": str(args.hint_source),\n            # Multi-label human-talk experiments should use sigmoid.\n            # The shared ExitNet default remains softmax for old single-label runs.\n            "activation": str(args.hint_activation),\n            "detach": bool(args.hint_detach),\n            "use_stats": bool(args.hint_use_stats),\n        }\n    }'''
    text = replace_once(text, old_model_cfg, new_model_cfg, str(path))

    old_print = '''    print(f"Exit hint:        disabled for first multi-label version")'''
    new_print = '''    print(f"Exit hint:        {model_cfg['exit_hint']}")'''
    text = replace_once(text, old_print, new_print, str(path))

    write(path, text)
    print(f"[OK] Patched {path}")


def patch_run_tata_ps1() -> None:
    path = ROOT / "scripts" / "run_tata_weakclip_experiment.ps1"
    text = read(path)
    if "HintActivation" in text and "--exit_hint" in text:
        print(f"[SKIP] {path} already appears patched.")
        return

    backup(path)

    old_param_tail = '''  [switch]$UsePosWeight,\n  [double]$PosWeightMax = 20.0,\n\n  [switch]$IncludeCheckpoint\n)'''
    new_param_tail = '''  [switch]$UsePosWeight,\n  [double]$PosWeightMax = 20.0,\n\n  [switch]$ExitHint,\n  [int]$HintDim = 8,\n  [ValidateSet("probs", "logits")]\n  [string]$HintSource = "probs",\n  [ValidateSet("softmax", "sigmoid")]\n  [string]$HintActivation = "sigmoid",\n  [bool]$HintDetach = $true,\n  [bool]$HintUseStats = $true,\n\n  [switch]$IncludeCheckpoint\n)'''
    text = replace_once(text, old_param_tail, new_param_tail, str(path))

    old_prints = '''    Write-Host "Device       = $Device"\n    Write-Host "UsePosWeight = $UsePosWeight"'''
    new_prints = '''    Write-Host "Device       = $Device"\n    Write-Host "UsePosWeight = $UsePosWeight"\n    Write-Host "ExitHint     = $ExitHint"\n    Write-Host "HintDim      = $HintDim"\n    Write-Host "HintSource   = $HintSource"\n    Write-Host "HintActivation = $HintActivation"\n    Write-Host "HintDetach   = $HintDetach"\n    Write-Host "HintUseStats = $HintUseStats"'''
    text = replace_once(text, old_prints, new_prints, str(path))

    old_after_pos = '''    if ($UsePosWeight) {\n        $TrainArgs += "--use_pos_weight"\n        $TrainArgs += "--pos_weight_max"\n        $TrainArgs += "$PosWeightMax"\n    }'''
    new_after_pos = '''    if ($UsePosWeight) {\n        $TrainArgs += "--use_pos_weight"\n        $TrainArgs += "--pos_weight_max"\n        $TrainArgs += "$PosWeightMax"\n    }\n\n    if ($ExitHint) {\n        $TrainArgs += "--exit_hint"\n        $TrainArgs += "--hint_dim"\n        $TrainArgs += "$HintDim"\n        $TrainArgs += "--hint_source"\n        $TrainArgs += "$HintSource"\n        $TrainArgs += "--hint_activation"\n        $TrainArgs += "$HintActivation"\n\n        if (-not $HintDetach) {\n            $TrainArgs += "--no_hint_detach"\n        }\n\n        if (-not $HintUseStats) {\n            $TrainArgs += "--no_hint_use_stats"\n        }\n    }'''
    text = replace_once(text, old_after_pos, new_after_pos, str(path))

    old_meta = '''        use_pos_weight = [bool]$UsePosWeight\n        pos_weight_max = $PosWeightMax\n        include_checkpoint_in_package = [bool]$IncludeCheckpoint'''
    new_meta = '''        use_pos_weight = [bool]$UsePosWeight\n        pos_weight_max = $PosWeightMax\n        exit_hint = [bool]$ExitHint\n        hint_dim = $HintDim\n        hint_source = $HintSource\n        hint_activation = $HintActivation\n        hint_detach = $HintDetach\n        hint_use_stats = $HintUseStats\n        include_checkpoint_in_package = [bool]$IncludeCheckpoint'''
    text = replace_once(text, old_meta, new_meta, str(path))

    write(path, text)
    print(f"[OK] Patched {path}")


def patch_evaluator() -> None:
    path = ROOT / "scripts" / "evaluate_tata_final_holdout_parent_level.py"
    text = read(path)
    if 'config.get("exit_hint"' in text:
        print(f"[SKIP] {path} already appears patched.")
        return

    backup(path)

    old = '''    model_cfg = {\n        "exit_hint": {\n            "enable": False,\n            "dim": 8,\n            "source": "probs",\n            "detach": True,\n            "use_stats": True,\n        }\n    }'''
    new = '''    # Rebuild with the same hint-pass architecture used during training.\n    # Old no-hint runs have no/disabled exit_hint in config_used.json, so they remain safe.\n    model_cfg = {\n        "exit_hint": config.get(\n            "exit_hint",\n            {\n                "enable": False,\n                "dim": 8,\n                "source": "probs",\n                "activation": "softmax",\n                "detach": True,\n                "use_stats": True,\n            },\n        )\n    }'''
    text = replace_once(text, old, new, str(path))
    write(path, text)
    print(f"[OK] Patched {path}")


def main() -> None:
    required = [
        ROOT / "models" / "exit_net.py",
        ROOT / "utils" / "model_factory.py",
        ROOT / "training" / "train_multilabel.py",
        ROOT / "scripts" / "run_tata_weakclip_experiment.ps1",
        ROOT / "scripts" / "evaluate_tata_final_holdout_parent_level.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    patch_exit_net()
    patch_model_factory()
    patch_train_multilabel()
    patch_run_tata_ps1()
    patch_evaluator()

    print("\nDone. Backups were created with suffix:", BACKUP_SUFFIX)
    print("Next: run py_compile and git diff before training.")


if __name__ == "__main__":
    main()
