"""Classify one image with CLIP, TTC and SAVE-TTC (optionally after a PGD attack on CLIP).

    python scripts/demo.py --image cat.jpg --classes "cat,dog,bird" --attack
    python scripts/demo.py --image leaf.png --classes_file my_classes.txt --save_config 3/9/4

Prompts are "This is a photo of a {class}". The attack is the evaluation attack: l_inf PGD on the
undefended CLIP (default eps = step = 1/255, 10 steps), aimed at the class CLIP predicts on the clean image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from savettc import core as sc  # noqa: E402
from savettc import paths  # noqa: E402
from savettc.clip_model import device, load_clip, make_classifier  # noqa: E402


def pgd_on_clip(clf, X, y, eps, step, steps, seed=0):
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    delta = (torch.rand(X.shape, generator=g, device=X.device) * 2 - 1) * eps
    delta = sc.clamp_delta(delta, X)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(clf.logits(X + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = torch.clamp(delta.detach() + step * grad.sign(), -eps, eps)
        delta = sc.clamp_delta(delta, X)
    return (X + delta).detach()


def report(name, probs, classes, k):
    p, i = probs[0].topk(min(k, len(classes)))
    top = ", ".join(f"{classes[j]} {100 * q:.1f}%" for q, j in zip(p.tolist(), i.tolist()))
    print(f"  {name:10s} -> {classes[i[0]]:20s} [{top}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--classes", default="", help="comma-separated class names")
    ap.add_argument("--classes_file", default="", help="one class name per line")
    ap.add_argument("--save_config", default="3/9/4", help="R/V/N for SAVE-TTC")
    ap.add_argument("--attack", action="store_true", help="also evaluate a PGD example crafted on CLIP")
    ap.add_argument("--eps", type=float, default=1.0, help="attack radius in 1/255 units")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="defense seed")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    if args.classes_file:
        classes = [c.strip() for c in Path(args.classes_file).read_text(encoding="utf-8").splitlines() if c.strip()]
    else:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(classes) < 2:
        sys.exit("give at least two classes via --classes or --classes_file")

    paths.add_ttc_to_path()
    model, prompt_token = load_clip()
    clf = make_classifier(model, prompt_token, [f"This is a photo of a {c}" for c in classes])
    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    X = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    ttc_cfg = sc.SaveConfig.from_rvn("1/1/2")
    save_cfg = sc.SaveConfig.from_rvn(args.save_config)
    inputs = {"clean": X}
    if args.attack:
        with torch.no_grad():
            y0 = clf.logits(X).argmax(1)
        e = args.eps / 255.0
        inputs[f"PGD-{args.steps} eps={args.eps:g}/255"] = pgd_on_clip(clf, X, y0, e, e, args.steps)

    for name, x in inputs.items():
        print(f"{name}:")
        with torch.no_grad():
            report("CLIP", clf.logits(x).softmax(-1), classes, args.topk)
        report("TTC", sc.save_predict(clf, x, ttc_cfg, [0], base_seed=args.seed).probs, classes, args.topk)
        out = sc.save_predict(clf, x, save_cfg, [0], base_seed=args.seed)
        report(f"SAVE {args.save_config}", out.probs, classes, args.topk)
        print(f"  gate open for {int(out.gate_open.sum())}/{save_cfg.R} starts, tau = "
              + ", ".join(f"{t:.3f}" for t in out.tau[0].tolist()))


if __name__ == "__main__":
    main()
