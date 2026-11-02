# Shared adversarial cache -> main comparison (3 defense seeds) + clean accuracy -> tables.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Run([string[]]$cmd) { & python @cmd; if ($LASTEXITCODE -ne 0) { throw "failed: python $cmd" } }
Run @("scripts/build_cache.py")
Run @("scripts/evaluate.py", "--config", "configs/main_table.yaml")
Run @("scripts/evaluate.py", "--config", "configs/clean.yaml")
Run @("scripts/make_tables.py")
Run @("scripts/make_tables.py", "--split", "clean")
