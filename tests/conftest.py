import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from savettc import paths  # noqa: E402

paths.add_ttc_to_path()

FALLBACK_CLASSES = ["abyssinian", "beagle", "bengal", "boxer", "chihuahua", "persian", "pug", "siamese"]


@pytest.fixture(scope="session")
def clip_model():
    from savettc.clip_model import load_clip
    from savettc.common import set_global_seed
    set_global_seed(0)
    return load_clip()


@pytest.fixture(scope="session")
def batch():
    """8 adversarial Pets images from the shared cache if it exists, else smooth synthetic images."""
    from savettc.clip_model import device
    from savettc.data import load_cache
    try:
        cache = load_cache("Pets")
        return cache.adv(list(range(8))).to(device), list(cache.texts), "pets_cache"
    except FileNotFoundError:
        g = torch.Generator().manual_seed(0)
        x = torch.rand(8, 3, 14, 14, generator=g)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic").clamp(0, 1)
        return x.to(device), [f"A photo of a {c}, a type of pet." for c in FALLBACK_CLASSES], "synthetic"


@pytest.fixture(scope="session")
def clf(clip_model, batch):
    from savettc.clip_model import make_classifier
    model, prompt_token = clip_model
    return make_classifier(model, prompt_token, batch[1])
