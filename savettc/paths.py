"""Repository paths. Every location can be overridden by an environment variable or a CLI flag."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TTC_DIR = REPO_ROOT / "third_party" / "ttc"

DATA_ROOT = Path(os.environ.get("SAVETTC_DATA", REPO_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("SAVETTC_CACHE", REPO_ROOT / "caches"))
OUT_DIR = Path(os.environ.get("SAVETTC_OUT", REPO_ROOT / "outputs"))


def add_ttc_to_path():
    """Make the vendored TTC modules (replace, utils, attacks, func, models) importable."""
    p = str(TTC_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
