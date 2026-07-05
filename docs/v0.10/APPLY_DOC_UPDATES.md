# Apply Documentation Updates

Copy these files into the repository root, preserving paths:

```text
README.md
DOC_STRUCTURE.md
docs/v0.10/
docs/tables/agentic_data_preprocessing_v0.10/
```

Then run:

```powershell
git diff -- README.md DOC_STRUCTURE.md docs/v0.10 docs/tables/agentic_data_preprocessing_v0.10
```

Recommended commit:

```powershell
git add README.md DOC_STRUCTURE.md docs/v0.10/ docs/tables/agentic_data_preprocessing_v0.10/
git commit -m "docs: add final v0.10 experimental analysis"
```
