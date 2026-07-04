# v0.10 Safe Hint Activation Patch

This patch keeps the shared `ExitNet` backward-compatible while allowing human-talk multi-label hint-pass to use sigmoid probabilities.

---

## Compatibility rule

| Experiment type | Hint activation |
|---|---|
| Old moth / single-label experiments | `softmax` default |
| Human-talk multi-label v0.10 | explicit `sigmoid` |

This avoids breaking old experiments while making the v0.10 multi-label hint vector semantically correct.

---

## Apply

```powershell
python scripts\v0.10\apply_v010_safe_hint_activation_patch.py

python -m py_compile `
  models\exit_net.py `
  utils\model_factory.py `
  training\train_multilabel.py `
  scripts\evaluate_tata_final_holdout_parent_level.py
```

---

## Check

```powershell
git diff -- `
  models\exit_net.py `
  utils\model_factory.py `
  training\train_multilabel.py `
  scripts\run_tata_weakclip_experiment.ps1 `
  scripts\evaluate_tata_final_holdout_parent_level.py
```

---

## Human-talk hint-pass run must use

```powershell
-ExitHint `
-HintActivation "sigmoid"
```

Old experiments that do not pass `hint_activation` keep the shared model default: `softmax`.

---

## Final empirical outcome

The patch was technically correct, but the current standard hint-pass method did not improve the human-talk multi-label results. The patch can remain because it is backward-compatible and useful for future gated/label-aware hinting experiments.
