# How to apply the v0.9 documentation update

This package contains edited v0.9 documentation files plus copied evidence tables, config, and scripts.

Copy the package contents into your repository root:

```powershell
$Repo = "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP"
$Pkg = "PATH_TO_EXTRACTED\LABLEX_v0.9_documentation_update"

Copy-Item -Recurse -Force "$Pkg\*" $Repo
```

Then inspect the changes:

```powershell
git diff -- README.md DOC_STRUCTURE.md docs\v0.9 docs\reports\v0.9 docs\results\v0.9 docs\tables\agentic_data_preprocessing_v0.9 configs\v0.9 scripts\v0.9
```

Recommended commit:

```powershell
git add README.md DOC_STRUCTURE.md docs\v0.9 docs\reports\v0.9 docs\results\v0.9 docs\tables\agentic_data_preprocessing_v0.9 configs\v0.9 scripts\v0.9
git commit -m "docs: document v0.9 labelwise aggregation results"
git push origin agentic_data_preprocessing_v0.9
```

## Final reported method

```text
v09_frozen_frequency_plus_gary_mean
```

## Final result

```text
Macro-F1   = 0.8518
Micro-F1   = 0.9374
Samples-F1 = 0.9464
Exact      = 0.8431
Hamming    = 0.0183
```

## Important decision

Do not report old TATA-LAWYER thresholds as final for v0.9. They were tested and rejected for this model.
