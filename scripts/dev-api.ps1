# ConsentShield API (PowerShell)
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$root;$root\apps\api"
Set-Location "$root\apps\api"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
