$ErrorActionPreference = "Stop"

Write-Host "Starting AstroTrust-AI dashboard..."
python -m streamlit run ..\src\interface\app.py