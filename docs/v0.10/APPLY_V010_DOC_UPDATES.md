# Apply v0.10 Documentation Updates

Suggested copy plan from this package into the repository root:

```powershell
Copy-Item README.md .\README.md -Force
Copy-Item DOC_STRUCTURE.md .\DOC_STRUCTURE.md -Force
Copy-Item docs\v0.10 .\docs\v0.10 -Recurse -Force
Copy-Item docs\tables\agentic_data_preprocessing_v0.10 .\docs\tables\agentic_data_preprocessing_v0.10 -Recurse -Force
```

Then review:

```powershell
git diff -- README.md DOC_STRUCTURE.md docs\v0.10 docs\tables\agentic_data_preprocessing_v0.10
```

Recommended commit message:

```text
docs: add v0.10 hint-pass and LATS stability analysis
```
