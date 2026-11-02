#!/usr/bin/env bash
# Start / view ablation (Table V configurations) on the shared cache.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_cache.py
python scripts/evaluate.py --config configs/ablation_grid.yaml
python scripts/make_tables.py
