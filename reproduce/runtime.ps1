# Latency / memory benchmark (needs the Pets cache), then refresh the tables.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Run([string[]]$cmd) { & python @cmd; if ($LASTEXITCODE -ne 0) { throw "failed: python $cmd" } }
Run @("scripts/build_cache.py", "--datasets", "Pets")
Run @("scripts/benchmark_runtime.py", "--config", "configs/runtime.yaml")
Run @("scripts/make_tables.py")
