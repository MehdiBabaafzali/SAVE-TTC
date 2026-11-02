"""savettc.core against the reference implementation (tests/reference_impl.py).

    pytest -q tests
"""
import pytest
import torch

import reference_impl as ref
from savettc import core as sc


def ttc_ref(clip_model, X, cfg, seed):
    model, prompt_token = clip_model
    torch.manual_seed(seed)
    return ref.tau_thres_weighted_counterattacks(
        model, X, torch.nn.Identity(), lambda: prompt_token, alpha=cfg.alpha, attack_iters=cfg.N,
        epsilon=cfg.eps, tau_thres=cfg.tau0, beta=cfg.beta).detach()


def same_init(X, cfg, seed):
    torch.manual_seed(seed)
    return torch.zeros_like(X).uniform_(-cfg.eps, cfg.eps)


def test_preprocess_matches_ttc(clf, batch):
    from func import clip_img_preprocessing
    X = batch[0]
    assert (clf.preprocess(X) - clip_img_preprocessing(X)).abs().max().item() == 0.0


def test_views(batch):
    X = batch[0]
    v_ref, v_new = ref.make_tte_views(X), sc.views_code9(X)
    assert len(v_new) == 9 and all(torch.equal(a, b) for a, b in zip(v_ref, v_new))
    assert torch.equal(v_new[0], X)
    vp = sc.views_paper9(X)
    assert len(vp) == 9 and not any(torch.equal(v, X) for v in vp)
    assert sc.SaveConfig.from_rvn("3/9/4").V == 9 and sc.SaveConfig.from_rvn("1/1/2").V == 1


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("tau0", [0.2, 0.0, 10.0], ids=["tau0.2", "all_closed", "all_open"])
def test_ttc_parity(clip_model, clf, batch, N, tau0):
    X = batch[0]
    cfg = sc.SaveConfig(R=1, N=N, views="id1", tau0=tau0)
    expected = ttc_ref(clip_model, X, cfg, seed=123)
    Delta, st = sc.ttc_counterattack(clf, X, same_init(X, cfg, 123), cfg)
    assert (Delta - expected).abs().max().item() < 1e-6
    if tau0 == 0.0:
        assert not st.gate_open.any()
    if tau0 == 10.0:
        assert st.gate_open.all()


def test_counterattack_constraints(clf, batch):
    X = batch[0]
    cfg = sc.SaveConfig(R=1, N=4, views="id1")
    Delta, _ = sc.ttc_counterattack(clf, X, same_init(X, cfg, 1), cfg)
    xr = X + Delta
    assert xr.min().item() >= -1e-7 and xr.max().item() <= 1 + 1e-7
    assert Delta.abs().max().item() <= cfg.eps + 1e-7


def test_skip_closed_equivalent(clf, batch):
    X = batch[0]
    cfg = sc.SaveConfig(R=1, N=4, views="id1")
    d0 = same_init(X, cfg, 5)
    _, st = sc.ttc_counterattack(clf, X, d0, cfg)
    cfg.tau0 = float(st.tau.median())  # guarantees a mix of open and closed gates
    full, st_full = sc.ttc_counterattack(clf, X, d0, cfg)
    skip, st_skip = sc.ttc_counterattack(clf, X, d0, cfg, skip_closed=True)
    g = st_full.gate_open
    assert g.any() and not g.all()
    assert torch.equal(g, st_skip.gate_open)
    diff = (full - skip).abs()
    assert diff[~g].max().item() == 0.0
    # open gates: later steps run on a sub-batch, so sign() of near-zero gradients may flip
    assert diff[g].mean().item() < 1e-3
    with torch.no_grad():
        assert torch.equal(clf.logits(X + full).argmax(1), clf.logits(X + skip).argmax(1))
    assert st_skip.grad_steps_executed <= st_full.grad_steps_executed


@pytest.mark.parametrize("rvn", ["1/1/2", "1/9/4", "3/9/4"])
def test_save_parity(clip_model, clf, batch, rvn, monkeypatch):
    model, prompt_token = clip_model
    X = batch[0]
    c = sc.SaveConfig.from_rvn(rvn)
    ids = list(range(X.size(0)))
    queue = [sc.clamp_delta(sc.seeded_uniform(X.shape[1:], c.eps,
                                              sc.restart_seeds(sc.STREAM_DEFENSE, 5, ids, r), X.device), X)
             for r in range(c.R)]
    monkeypatch.setattr(ref, "init_delta", lambda X_, eps_, norm="l_inf": queue.pop(0).clone())
    expected = ref.save_logits(model, X, torch.nn.Identity(), lambda: prompt_token, clf.text_features,
                               clf.logit_scale, num_restarts=c.R, use_tte=(c.V == 9), ttc_stepsize=c.alpha,
                               ttc_numsteps=c.N, ttc_eps=c.eps, tau_thres=c.tau0, beta=c.beta).exp()
    out = sc.save_predict(clf, X, c, ids, base_seed=5, diagnostics=True)
    assert (out.probs - expected).abs().max().item() < 1e-5
    assert out.n_predictions == c.R * c.V
    assert tuple(out.gate_open.shape) == (X.size(0), c.R)
    assert out.grad_steps_executed <= c.R * c.N
    assert (out.probs.sum(1) - 1).abs().max().item() < 1e-4
    assert (out.d_view >= 0).all() and (out.d_start >= 0).all()


def test_mean_of_probabilities(clf, batch):
    X = batch[0]
    c = sc.SaveConfig.from_rvn("1/9/4")
    ids = list(range(X.size(0)))
    out = sc.save_predict(clf, X, c, ids, base_seed=9)
    d0 = sc.seeded_uniform(X.shape[1:], c.eps, sc.restart_seeds(sc.STREAM_DEFENSE, 9, ids, 0), X.device)
    Delta, _ = sc.ttc_counterattack(clf, X, d0, c)
    with torch.no_grad():
        L = torch.stack([clf.logits(v) for v in sc.views_code9(X + Delta)], 1)
    assert (out.probs - L.softmax(-1).mean(1)).abs().max().item() < 1e-5


def test_seeding_and_batching(clip_model, clf, batch):
    X = batch[0]
    c = sc.SaveConfig.from_rvn("3/9/4")
    ids = list(range(X.size(0)))
    o1 = sc.save_predict(clf, X, c, ids, base_seed=1)
    o2 = sc.save_predict(clf, X, c, ids, base_seed=1)
    o3 = sc.save_predict(clf, X, c, ids, base_seed=2)
    assert torch.equal(o1.probs, o2.probs)
    assert (o1.probs - o3.probs).abs().max().item() > 0
    # per-image generators give the same random starts for any batch split; the remaining differences
    # come from batch-size-dependent GPU kernels, amplified by sign() in the counterattack
    half = X.size(0) // 2
    pc = torch.cat([sc.save_predict(clf, X[:half], c, ids[:half], base_seed=1).probs,
                    sc.save_predict(clf, X[half:], c, ids[half:], base_seed=1).probs])
    assert (pc - o1.probs).abs().max().item() < 0.05
    assert torch.equal(pc.argmax(1), o1.probs.argmax(1))
    assert not o1.probs.requires_grad and torch.is_grad_enabled()
    model, _ = clip_model
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast path needs CUDA")
def test_amp_agrees_with_fp32(clf, batch):
    X = batch[0]
    c = sc.SaveConfig.from_rvn("3/9/4")
    ids = list(range(X.size(0)))
    p32 = sc.save_predict(clf, X, c, ids, base_seed=1).probs
    p16 = sc.save_predict(clf, X, c, ids, base_seed=1, amp=True).probs
    assert (p32.argmax(1) == p16.argmax(1)).float().mean().item() >= 0.75
