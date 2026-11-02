"""Frozen CLIP classifier used by every defense (ViT-B/32, fp32, zero-shot text heads)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

device = "cuda" if torch.cuda.is_available() else "cpu"


class ClipClassifier:
    """Zero-shot classifier on [0,1] images: bicubic resize to 224, CLIP normalisation,
    image encoder, cosine similarity to the class text embeddings."""

    def __init__(self, model, text_features: torch.Tensor, logit_scale: float,
                 prompt_token: Optional[torch.Tensor] = None):
        self.model = model
        self.module = model.module if hasattr(model, "module") else model
        self.text_features = text_features
        self.logit_scale = float(logit_scale)
        self.prompt_token = prompt_token
        dev = text_features.device
        self.mu = torch.tensor(CLIP_MEAN, device=dev).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=dev).view(3, 1, 1)

    def preprocess(self, x01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x01, size=(224, 224), mode="bicubic")
        return (x - self.mu) / self.std

    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        return self.module.encode_image(self.preprocess(x01), self.prompt_token)

    def logits(self, x01: torch.Tensor) -> torch.Tensor:
        f = self.encode(x01)
        f = f / f.norm(dim=-1, keepdim=True)
        return f @ self.text_features.t() * self.logit_scale


def load_clip(arch: str = "ViT-B/32"):
    """Load the prompt-capable CLIP from third_party/ttc (call savettc.paths.add_ttc_to_path() first).
    Returns (DataParallel model in eval mode, empty prompt token)."""
    from replace import clip
    from models.prompters import TokenPrompter
    from utils import convert_models_to_fp32

    model, _ = clip.load(arch, device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).to(device)
    with torch.no_grad():
        prompt_token = add_prompter().detach()
    return model, prompt_token


def text_features_for(model, texts):
    from replace import clip
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        f = model.module.encode_text(tokens)
        f = f / f.norm(dim=1, keepdim=True)
    return f, float(model.module.logit_scale.exp().item())


def make_classifier(model, prompt_token, texts) -> ClipClassifier:
    tf, ls = text_features_for(model, texts)
    return ClipClassifier(model, tf, ls, prompt_token)
