# How to apply these v0.9 documentation updates

This package contains updated copies of the uploaded documentation files. They preserve the existing content and append v0.9 sections.

Copy them into your repository root:

```powershell
Copy-Item -Force README.md "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP\README.md"
Copy-Item -Force DOC_STRUCTURE.md "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP\DOC_STRUCTURE.md"
Copy-Item -Force APPENDIX.md "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP\docs\APPENDIX.md"
Copy-Item -Force COMMANDS_V08.md "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP\docs\COMMANDS_V08.md"
Copy-Item -Force MULTILABEL_EXPERIMENT_LOG.md "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP\docs\MULTILABEL_EXPERIMENT_LOG.md"
```

Then check:

```powershell
git diff -- README.md DOC_STRUCTURE.md docs\APPENDIX.md docs\COMMANDS_V08.md docs\MULTILABEL_EXPERIMENT_LOG.md
```

Commit on branch `agentic_data_preprocessing_v0.9`:

```powershell
git add README.md DOC_STRUCTURE.md docs\APPENDIX.md docs\COMMANDS_V08.md docs\MULTILABEL_EXPERIMENT_LOG.md
git commit -m "docs: add v0.9 frozen labelwise aggregation results"
git push origin agentic_data_preprocessing_v0.9
```
