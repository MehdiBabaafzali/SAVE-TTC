"""Reference implementation the tests compare savettc.core against.

These are the original research-code functions (TTC counterattack from Xing et al., CVPR 2025, the
9-view family, and restart + view ensembling), kept as written apart from dropping the unused l_2 and
trajectory-logit branches. savettc.core must reproduce them exactly given the same random starts.
"""
import torch
import torch.nn.functional as F

from attacks import clamp, lower_limit, upper_limit
from func import clip_img_preprocessing


def init_delta(X, epsilon, norm="l_inf"):
    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    return delta


def tau_thres_weighted_counterattacks(model, X, prompter, add_prompter, alpha, attack_iters,
                                      norm="l_inf", epsilon=0, tau_thres: float = None, beta: float = None):
    """Original paper TTC: pixel-space weighted average of deltas."""
    delta = init_delta(X, epsilon, norm)
    if epsilon <= 0.:
        return delta
    delta.requires_grad = True
    if attack_iters == 0:
        return delta.data
    diff_ratio = None

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(prompter(clip_img_preprocessing(X)), prompt_token)
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = [delta.data.clone()]
    for _step_id in range(attack_iters):
        prompted_images = prompter(clip_img_preprocessing(X + delta))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        if _step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm
        scheme_sign = (tau_thres - diff_ratio).sign()

        l2_loss = ((((X_att_reps - X_ori_reps) ** 2).sum(1))).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())

    Delta = torch.stack(deltas_per_step, dim=1)
    weights = torch.arange(attack_iters + 1).unsqueeze(0).expand(X.size(0), -1).to(X.device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights /= weights.sum(dim=1, keepdim=True)
    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(X.size(0), attack_iters + 1, 1, 1, 1)
    return (weights * Delta).sum(dim=1)


def make_tte_views(images):
    B, C, H, W = images.shape
    views = [images, torch.flip(images, dims=[3])]
    crop = int(H * 0.875)
    coords = [(0, 0), (0, W - crop), (H - crop, 0), (H - crop, W - crop)]
    for top, left in coords:
        patch = images[:, :, top:top + crop, left:left + crop]
        patch = F.interpolate(patch, size=(H, W), mode='bilinear', align_corners=False)
        views.append(patch)
        views.append(torch.flip(patch, dims=[3]))
    return views[:9]


def classify_images(model, images, text_features, logit_scale, prompter, add_prompter, use_tte=False):
    """Return logits [bs, C]. If use_tte, average softmax over the views (returned as log-probs)."""
    prompt_token = add_prompter()
    if not use_tte:
        with torch.no_grad():
            feats = model.module.encode_image(prompter(clip_img_preprocessing(images)), prompt_token)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats @ text_features.t() * logit_scale
    probs_acc = None
    views = make_tte_views(images)
    with torch.no_grad():
        for v in views:
            feats = model.module.encode_image(prompter(clip_img_preprocessing(v)), prompt_token)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (feats @ text_features.t() * logit_scale).softmax(dim=-1)
            probs_acc = probs if probs_acc is None else probs_acc + probs
        return (probs_acc / len(views)).log()


def save_logits(model, images, prompter, add_prompter, text_features, logit_scale,
                num_restarts, use_tte, ttc_stepsize, ttc_numsteps, ttc_eps, tau_thres, beta):
    """R restarts of the counterattack, each classified (optionally over the 9 views); mean of softmax."""
    probs_acc = None
    for _ in range(max(1, num_restarts)):
        delta = tau_thres_weighted_counterattacks(model, images, prompter, add_prompter, alpha=ttc_stepsize,
                                                  attack_iters=ttc_numsteps, epsilon=ttc_eps,
                                                  tau_thres=tau_thres, beta=beta)
        logits = classify_images(model, images + delta, text_features, logit_scale, prompter, add_prompter,
                                 use_tte=use_tte)
        p = logits.softmax(dim=-1)
        probs_acc = p if probs_acc is None else probs_acc + p
    return (probs_acc / max(1, num_restarts)).log()
