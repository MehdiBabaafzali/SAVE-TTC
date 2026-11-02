"""Shared adversarial cache (PGD-10, eps=1/255 against undefended CLIP).

Stored as uint8 clean images plus an fp16 perturbation, so every defense configuration is evaluated on
exactly the same adversarial images.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import paths

DATASETS = ["Pets", "Flower102", "Aircraft", "DTD"]
LOADER_NAMES = {"Pets": "oxfordpet", "Flower102": "flowers102", "Aircraft": "fgvc_aircraft", "DTD": "dtd"}


def cache_path(dataset: str, cache_dir: Path = None, attack_seed: int = 1) -> Path:
    return Path(cache_dir or paths.CACHE_DIR) / f"{dataset}_pgd10_eps1_attackseed{attack_seed}.pt"


class AdvCache:
    def __init__(self, payload: dict):
        self.payload = payload
        self.dataset = payload["dataset"]
        self.texts = list(payload["texts"])
        self.labels = payload["labels"].long()
        self.n = int(self.labels.numel())
        self.eps_attack = float(payload["test_eps"])

    def clean(self, idx) -> torch.Tensor:
        return self.payload["clean_u8"][idx].float() / 255.0

    def adv(self, idx) -> torch.Tensor:
        x = self.clean(idx)
        d = self.payload["delta"][idx].float().clamp(-self.eps_attack, self.eps_attack)
        return (x + d).clamp(0, 1)


def load_cache(dataset: str, cache_dir: Path = None, attack_seed: int = 1) -> AdvCache:
    path = cache_path(dataset, cache_dir, attack_seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_cache.py first")
    return AdvCache(torch.load(path, map_location="cpu", weights_only=False))
