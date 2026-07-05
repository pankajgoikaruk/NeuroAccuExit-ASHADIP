# PowerShell Commands — v0.10 Documentation, Pos-Weight, and GitHub

## 1. Add latest docs and scripts to v0.10 branch

```powershell
git checkout agentic_data_preprocessing_v0.10
git status

git add README.md
git add DOC_STRUCTURE.md
git add docs/v0.10/
git add docs/tables/agentic_data_preprocessing_v0.10/
git add scripts/v0.10/run_v010_lats_v2_coordinate_reoptimize.py
git add scripts/v0.10/run_v010_no_hint_posweight_stability.ps1
git add scripts/v0.10/summarize_v010_posweight_results.ps1

git status
git commit -m "docs: update v0.10 pos-weight and LATS findings"
git push origin agentic_data_preprocessing_v0.10
```

---

## 2. Merge v0.10 updates into main

```powershell
git checkout main
git pull origin main
git merge agentic_data_preprocessing_v0.10
git push origin main
git status
```

If conflicts appear, stop and resolve them before pushing.

---

## 3. Pos-weight diagnostic command used

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v0.10\run_v010_no_hint_posweight_stability.ps1 `
  -Seeds 101,202,303 `
  -PosWeightMax 5.0 `
  -Device cpu `
  -Epochs 40 `
  -Objective macro_priority
```

Important note: the produced run was `seed_101202303`, so it must be treated as a single diagnostic run, not a valid 3-seed stability run.

---

## 4. Syntax check for LATS-v2 script

```powershell
python -m py_compile scripts\v0.10\run_v010_lats_v2_coordinate_reoptimize.py
```

---

## 5. Summarize pos-weight output

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v0.10\summarize_v010_posweight_results.ps1
```
