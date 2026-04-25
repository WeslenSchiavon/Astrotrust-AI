$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "AstroTrust-AI MVP Pipeline"
Write-Host "========================================"

Write-Host "`n[1/7] Building ELAsTiCC2 light-curve dataset..."
python ..\src\data\build_elasticc2_lightcurve_dataset.py

Write-Host "`n[2/7] Extracting light-curve features..."
python ..\src\features\feature_extraction.py

Write-Host "`n[3/7] Running baseline Random Forest..."
python ..\experiments\run_baseline_random_forest.py

Write-Host "`n[4/7] Running early classification experiment..."
python ..\experiments\run_early_classification_random_forest.py

Write-Host "`n[5/7] Running uncertainty experiment..."
python ..\experiments\run_uncertainty_random_forest.py

Write-Host "`n[6/7] Analyzing uncertainty behavior..."
python ..\experiments\analyze_uncertainty_behavior.py

Write-Host "`n[7/7] Running follow-up prioritization..."
python ..\experiments\run_followup_prioritization.py

Write-Host "`nRunning follow-up enrichment analysis..."
python ..\experiments\analyze_followup_enrichment.py

Write-Host "`n========================================"
Write-Host "Pipeline finished successfully."
Write-Host "========================================"