# v0.10 Safe Hint Activation Patch

This patch keeps the shared `ExitNet` backward-compatible:

- Default hint activation remains `softmax` for old single-label / moth experiments.
- Human-talk multi-label v0.10 can explicitly use `sigmoid` by passing `--hint_activation sigmoid` or `-HintActivation "sigmoid"`.

## Apply

```powershell
python scripts\v0.10\apply_v010_safe_hint_activation_patch.py

python -m py_compile `
  models\exit_net.py `
  utils\model_factory.py `
  training\train_multilabel.py `
  scripts\evaluate_tata_final_holdout_parent_level.py
```

## Check

```powershell
git diff -- `
  models\exit_net.py `
  utils\model_factory.py `
  training\train_multilabel.py `
  scripts\run_tata_weakclip_experiment.ps1 `
  scripts\evaluate_tata_final_holdout_parent_level.py
```

## Human-talk hint-pass run must use

```powershell
-ExitHint `
-HintActivation "sigmoid"
```

Old experiments that do not pass `hint_activation` keep the shared model default: `softmax`.
