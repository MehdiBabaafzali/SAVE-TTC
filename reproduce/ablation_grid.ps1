# Start / view ablation (Table V configurations) on the shared cache.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Run([string[]]$cmd) { & python @cmd; if ($LASTEXITCODE -ne 0) { throw "failed: python $cmd" } }
Run @("scripts/build_cache.py")
Run @("scripts/evaluate.py", "--config", "configs/ablation_grid.yaml")
Run @("scripts/make_tables.py")
