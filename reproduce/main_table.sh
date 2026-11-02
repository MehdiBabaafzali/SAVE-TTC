#!/usr/bin/env bash
# Shared adversarial cache -> main comparison (3 defense seeds) + clean accuracy -> tables.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_cache.py
python scripts/evaluate.py --config configs/main_table.yaml
python scripts/evaluate.py --config configs/clean.yaml
python scripts/make_tables.py
python scripts/make_tables.py --split clean
