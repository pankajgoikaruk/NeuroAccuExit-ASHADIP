# Apply v0.9_4 Documentation Updates

From the extracted documentation package root, copy the files into the repository root:

```powershell
$Repo = "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP"
$Pkg = "PATH_TO_EXTRACTED\LATS-v2"

Copy-Item -Recurse -Force "$Pkg\*" $Repo
```

Then inspect:

```powershell
cd $Repo
git status

git diff -- README.md DOC_STRUCTURE.md docs\INDEX.md docs\RESULTS.md docs\MULTILABEL_EXPERIMENT_LOG.md docs\APPENDIX.md docs\v0.9\v0.9_4 docs\results\v0.9 docs\reports\v0.9 docs\tables\agentic_data_preprocessing_v0.9
```

Commit:

```powershell
git add README.md DOC_STRUCTURE.md docs\INDEX.md docs\RESULTS.md docs\MULTILABEL_EXPERIMENT_LOG.md docs\APPENDIX.md docs\v0.9\v0.9_4 docs\results\v0.9 docs\reports\v0.9 docs\tables\agentic_data_preprocessing_v0.9

git commit -m "docs: document v0.9_4 LATS-v2 metric-aware results"

git push origin agentic_data_preprocessing_v0.9_4
```
