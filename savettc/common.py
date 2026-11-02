"""Small helpers shared by the scripts: seeding, YAML defaults, JSON output, run metadata."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_with_config(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Add --config; values in the YAML file become defaults, explicit CLI flags still win."""
    parser.add_argument("--config", type=str, default=None, help="YAML file with default arguments")
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        import yaml
        with open(pre.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            parser.error(f"unknown keys in {pre.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)


def write_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_info(extra: dict | None = None) -> dict:
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if extra:
        info.update(extra)
    return info
