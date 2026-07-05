# GitHub Main-Branch Update Commands

## Step 1 — Commit on v0.10 branch

```powershell
git checkout agentic_data_preprocessing_v0.10
git status

git add README.md DOC_STRUCTURE.md docs/v0.10/ docs/tables/agentic_data_preprocessing_v0.10/
git add scripts/v0.10/run_v010_lats_v2_coordinate_reoptimize.py
git add scripts/v0.10/run_v010_no_hint_posweight_stability.ps1
git add scripts/v0.10/summarize_v010_posweight_results.ps1

git commit -m "docs: update v0.10 final experimental findings"
git push origin agentic_data_preprocessing_v0.10
```

## Step 2 — Merge into main

```powershell
git checkout main
git pull origin main
git merge agentic_data_preprocessing_v0.10
git push origin main
```

## Step 3 — Verify

```powershell
git status
git log --oneline -5
```

Expected status:

```text
nothing to commit, working tree clean
```
