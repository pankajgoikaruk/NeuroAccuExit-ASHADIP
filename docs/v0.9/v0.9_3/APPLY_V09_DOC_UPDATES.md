# How to apply the v0.9_3 LATS documentation update

This package freezes the LATS-v0.9 novelty result.

Copy the package contents into your repository root:

```powershell
$Repo = "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP"
$Pkg = "PATH_TO_EXTRACTED\v09_LATS_freeze_package_UPDATED"

Copy-Item -Recurse -Force "$Pkg\*" $Repo
```

Then inspect:

```powershell
git diff -- README.md DOC_STRUCTURE.md docs\v0.9 docs\reports\v0.9 docs\results\v0.9 docs\tables\agentic_data_preprocessing_v0.9 configs\v0.9 scripts\v0.9
```

Recommended commit:

```powershell
git add README.md DOC_STRUCTURE.md docs\v0.9 docs\reports\v0.9 docs\results\v0.9 docs\tables\agentic_data_preprocessing_v0.9 configs\v0.9 scripts\v0.9
git commit -m "docs: freeze v0.9_3 LATS final result"
git push origin agentic_data_preprocessing_v0.9_3
```

Final frozen result:

```text
Method     = lats_final_frozen_config_v09
Macro-F1   = 0.8667
Micro-F1   = 0.9436
Samples-F1 = 0.9495
Exact      = 0.8524
Hamming    = 0.0165
```
