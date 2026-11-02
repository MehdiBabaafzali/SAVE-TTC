#!/usr/bin/env bash
# Latency / memory benchmark (needs the Pets cache), then refresh the tables.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_cache.py --datasets Pets
python scripts/benchmark_runtime.py --config configs/runtime.yaml
python scripts/make_tables.py
