"""SAVE-TTC: start-and-view ensembling of test-time counterattacks.

    x --R independent TTC counterattacks (own uniform init + own tau gate)-->
    x_r --V fixed views of each x_r--> R*V softmax vectors --> mean --> argmax

The counterattack is the official TTC update (Xing et al., CVPR 2025): N signed steps that push the
CLIP image embedding away from that of the input, inside an l_inf ball of radius eps, followed by a
beta-weighted average of the pixel-space trajectory. A sample whose relative feature drift after the
first step is >= tau0 ("gate closed") keeps only its random start.

Random starts are drawn from per-image, per-restart seeded generators, so they do not depend on batch
size or ordering. tests/test_core.py checks every operation against the reference implementation.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

STREAM_DEFENSE = 11


# ----------------------------------------------------------------------------
# View families
# ----------------------------------------------------------------------------

def _crops(images: torch.Tensor):
    B, C, H, W = images.shape
    crop = int(H * 0.875)
    coords = [(0, 0), (0, W - crop), (H - crop, 0), (H - crop, W - crop)]
    out = []
    for top, left in coords:
        patch = images[:, :, top:top + crop, left:left + crop]
        out.append(F.interpolate(patch, size=(H, W), mode="bilinear", align_corners=False))
    return out


def views_code9(images: torch.Tensor) -> List[torch.Tensor]:
    """The 9 views used for all reported results:
    [identity, hflip, c_tl, hflip(c_tl), c_tr, hflip(c_tr), c_bl, hflip(c_bl), c_br]
    (87.5% corner crops resized back to the input size)."""
    views = [images, torch.flip(images, dims=[3])]
    for p in _crops(images):
        views.append(p)
        views.append(torch.flip(p, dims=[3]))
    return views[:9]


def views_paper9(images: torch.Tensor) -> List[torch.Tensor]:
    """Alternative 9-view family: hflip + 4 crops + 4 flipped crops (no identity)."""
    views = [torch.flip(images, dims=[3])]
    cs = _crops(images)
    views += cs
    views += [torch.flip(p, dims=[3]) for p in cs]
    return views


def views_id1(images: torch.Tensor) -> List[torch.Tensor]:
    return [images]


VIEW_FAMILIES: Dict[str, Callable] = {
    "id1": views_id1,
    "code9": views_code9,
    "paper9": views_paper9,
}

VIEW_NAMES = {
    "id1": ["identity"],
    "code9": ["identity", "hflip", "crop_tl", "hflip_crop_tl", "crop_tr", "hflip_crop_tr",
              "crop_bl", "hflip_crop_bl", "crop_br"],
    "paper9": ["hflip", "crop_tl", "crop_tr", "crop_bl", "crop_br",
               "hflip_crop_tl", "hflip_crop_tr", "hflip_crop_bl", "hflip_crop_br"],
}


# ----------------------------------------------------------------------------
# Config + seeded noise
# ----------------------------------------------------------------------------

@dataclass
class SaveConfig:
    """R starts, N counterattack steps, V views (from the view family)."""
    R: int = 3
    N: int = 4
    views: str = "code9"          # id1 => V=1 (plain TTC classification)
    eps: float = 4.0 / 255.0
    alpha: float = 1.0 / 255.0
    tau0: float = 0.2
    beta: float = 2.0

    @property
    def V(self) -> int:
        return len(VIEW_NAMES[self.views])

    def tag(self) -> str:
        base = f"R{self.R}_V{self.V}_N{self.N}"
        if self.views not in ("id1", "code9"):
            base += f"_{self.views}"
        return base

    @staticmethod
    def from_rvn(s: str, views9: str = "code9", **kw) -> "SaveConfig":
        """'3/9/4' -> R=3, V=9, N=4. '1/1/2' is the original TTC."""
        r, v, n = [int(t) for t in s.split("/")]
        if v == 1:
            fam = "id1"
        elif v == 9:
            fam = views9
        else:
            raise ValueError(f"V={v} has no declared view family")
        return SaveConfig(R=r, N=n, views=fam, **kw)


def image_seed(stream: int, base_seed: int, image_id: int, *extra: int) -> int:
    ss = np.random.SeedSequence([int(stream), int(base_seed), int(image_id)] + [int(e) for e in extra])
    return int(ss.generate_state(1, dtype=np.uint64)[0] & ((1 << 63) - 1))


def restart_seeds(stream: int, base_seed: int, image_ids: Sequence[int], r: int, *extra: int):
    return [image_seed(stream, base_seed, int(i), r, *extra) for i in image_ids]


def seeded_uniform(shape_per_image, eps: float, seeds: Sequence[int], device) -> torch.Tensor:
    out = []
    for s in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(int(s))
        out.append(torch.rand(shape_per_image, generator=g, device=device))
    u = torch.stack(out, 0)
    return (u * 2.0 - 1.0) * eps


def clamp_delta(delta: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Keep X + delta inside [0, 1]."""
    return torch.max(torch.min(delta, 1.0 - X), 0.0 - X)


# ----------------------------------------------------------------------------
# TTC counterattack (one start, batched)
# ----------------------------------------------------------------------------

@dataclass
class TTCStats:
    tau: torch.Tensor            # [B] relative feature drift after the first step
    gate_open: torch.Tensor      # [B] bool, tau < tau0
    grad_steps_executed: int     # backward passes actually run for the batch
    fwd_passes: int


def ttc_counterattack(clf, X: torch.Tensor, delta0: torch.Tensor, cfg: SaveConfig,
                      anchor: Optional[torch.Tensor] = None, skip_closed: bool = False):
    """Official TTC counterattack with an explicit random start `delta0`.

    Returns (Delta [B,3,H,W] detached, TTCStats); the counterattacked image is X + Delta.

    skip_closed=False runs all N steps for every sample (as the reference implementation does) and
    discards them for closed gates. skip_closed=True stops computing steps n>=1 for closed-gate
    samples. Closed-gate outputs are identical; open-gate outputs match up to floating-point rounding
    (steps run on a smaller sub-batch, and sign() can flip near-zero gradient entries).
    All reported results use skip_closed=False.
    """
    eps, alpha, N = cfg.eps, cfg.alpha, cfg.N
    delta0 = clamp_delta(delta0, X)
    B = X.size(0)
    if eps <= 0 or N == 0:
        return delta0.detach(), TTCStats(torch.zeros(B, device=X.device),
                                         torch.zeros(B, dtype=torch.bool, device=X.device), 0, 0)

    if anchor is None:
        with torch.no_grad():
            anchor = clf.encode(X)
    a_norm = anchor.norm(dim=-1)
    fwd = 1

    with torch.enable_grad():
        delta = delta0.clone().detach().requires_grad_(True)
        deltas = [delta0.detach().clone()]
        tau = None
        active_idx = torch.arange(B, device=X.device)
        grad_steps = 0
        for n in range(N):
            if skip_closed and n >= 1:
                idx = active_idx
                if idx.numel() == 0:
                    deltas.append(deltas[-1].clone())
                    continue
                d_sub = deltas[-1][idx].clone().requires_grad_(True)
                f = clf.encode(X[idx] + d_sub)
                loss = ((f - anchor[idx]) ** 2).sum(1).sum()
                g = torch.autograd.grad(loss, d_sub)[0]
                d_new = torch.clamp(d_sub + alpha * torch.sign(g), -eps, eps)
                d_new = clamp_delta(d_new, X[idx])
                full = deltas[-1].clone()
                full[idx] = d_new.detach()
                deltas.append(full)
                grad_steps += 1
                fwd += 1
                continue

            f = clf.encode(X + delta)
            fwd += 1
            if n == 0:
                tau = ((f - anchor).norm(dim=-1) / a_norm).detach()
                if skip_closed:
                    active_idx = torch.nonzero(tau < cfg.tau0).flatten()
            loss = ((f - anchor) ** 2).sum(1).sum()
            g = torch.autograd.grad(loss, delta)[0]
            grad_steps += 1
            d = torch.clamp(delta + alpha * torch.sign(g), min=-eps, max=eps)
            d = clamp_delta(d, X)
            delta.data[:, :, :, :] = d
            deltas.append(delta.data.clone())

    scheme_sign = (cfg.tau0 - tau).sign()
    Delta = torch.stack(deltas, dim=1)
    w = torch.arange(N + 1, device=X.device).unsqueeze(0).expand(B, -1).float()
    w = torch.exp(scheme_sign.view(-1, 1) * w * cfg.beta)
    w = w / w.sum(dim=1, keepdim=True)
    w_hard = torch.zeros_like(w)
    w_hard[:, 0] = 1.0
    w = torch.where(scheme_sign.unsqueeze(1) > 0, w, w_hard)
    Delta = (w.view(B, N + 1, 1, 1, 1) * Delta).sum(dim=1)
    return Delta.detach(), TTCStats(tau, tau < cfg.tau0, grad_steps, fwd)


# ----------------------------------------------------------------------------
# SAVE-TTC prediction
# ----------------------------------------------------------------------------

@dataclass
class SaveOutput:
    probs: torch.Tensor                      # [B,C] mean of the R*V softmax vectors
    gate_open: torch.Tensor                  # [B,R] bool
    tau: torch.Tensor                        # [B,R]
    d_view: Optional[torch.Tensor] = None    # [B] spread across views (diagnostics=True)
    d_start: Optional[torch.Tensor] = None   # [B] spread across starts (diagnostics=True)
    n_predictions: int = 0                   # R*V
    grad_steps_executed: int = 0             # summed over starts (batch-level)
    fwd_passes: int = 0


def save_predict(clf, X: torch.Tensor, cfg: SaveConfig, image_ids: Sequence[int], base_seed: int = 0,
                 stream: int = STREAM_DEFENSE, extra: Sequence[int] = (), skip_closed: bool = False,
                 diagnostics: bool = False, amp: bool = False) -> SaveOutput:
    """SAVE-TTC class probabilities for a batch X in [0,1] (B,3,224,224).

    image_ids identify the images (e.g. dataset indices) for the seeded random starts; base_seed
    selects the defense realisation.
    """
    view_fn = VIEW_FAMILIES[cfg.views]
    B = X.size(0)
    ctx = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
    with ctx:
        with torch.no_grad():
            anchor = clf.encode(X)
        probs_sum = None
        per_rv = [] if diagnostics else None
        gates, taus = [], []
        gsteps, fwd = 0, 1
        for r in range(cfg.R):
            d0 = seeded_uniform(X.shape[1:], cfg.eps,
                                restart_seeds(stream, base_seed, image_ids, r, *extra), X.device)
            Delta, st = ttc_counterattack(clf, X, d0, cfg, anchor=anchor, skip_closed=skip_closed)
            gsteps += st.grad_steps_executed
            fwd += st.fwd_passes - 1
            gates.append(st.gate_open)
            taus.append(st.tau)
            xr = X + Delta
            with torch.no_grad():
                for v in view_fn(xr):
                    p = clf.logits(v).float().softmax(dim=-1)
                    fwd += 1
                    probs_sum = p if probs_sum is None else probs_sum + p
                    if diagnostics:
                        per_rv.append(p)
    probs = probs_sum / (cfg.R * cfg.V)
    out = SaveOutput(probs=probs, gate_open=torch.stack(gates, 1), tau=torch.stack(taus, 1),
                     n_predictions=cfg.R * cfg.V, grad_steps_executed=gsteps, fwd_passes=fwd)
    if diagnostics:
        P = torch.stack(per_rv, 1).view(B, cfg.R, cfg.V, -1)
        m_r = P.mean(2, keepdim=True)
        pbar = P.mean((1, 2), keepdim=True)
        out.d_view = ((P - m_r) ** 2).sum(-1).mean((1, 2))
        out.d_start = ((m_r - pbar) ** 2).sum(-1).mean((1, 2))
    return out
