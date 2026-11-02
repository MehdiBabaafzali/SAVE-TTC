"""Build one shared adversarial cache per dataset (TTC evaluation protocol).

Untargeted l_inf PGD-10, eps=1/255, step 1/255, random start, against undefended CLIP ViT-B/32,
TTC loaders (Resize 256 / CenterCrop 224) and dataset-specific prompts, attack under autocast.
Every defense is then evaluated on exactly the same adversarial images.

    python scripts/build_cache.py --datasets Pets Flower102 Aircraft DTD
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from savettc import paths  # noqa: E402
from savettc.clip_model import device, load_clip, make_classifier  # noqa: E402
from savettc.common import run_info, set_global_seed, write_json  # noqa: E402
from savettc.data import DATASETS, LOADER_NAMES, cache_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--data_root", default=str(paths.DATA_ROOT))
    ap.add_argument("--cache_dir", default=str(paths.CACHE_DIR))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--attack_seed", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1.0, help="x/255")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--stepsize", type=float, default=1.0, help="x/255")
    ap.add_argument("--max_images", type=int, default=0,
                    help="smoke test: stop after ~n images (use a separate --cache_dir)")
    args = ap.parse_args()

    paths.add_ttc_to_path()
    from attacks import attack_pgd
    from replace import clip
    from utils import get_text_prompts_val, load_val_dataset

    model, prompt_token = load_clip()
    eps, alpha = args.eps / 255.0, args.stepsize / 255.0
    crit = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    summary = {}

    for name in args.datasets:
        out = cache_path(name, args.cache_dir, args.attack_seed)
        if out.exists():
            print(f"[skip] {out}")
            continue
        set_global_seed(args.attack_seed)
        la = argparse.Namespace(root=args.data_root, batch_size=args.batch_size, num_workers=0)
        ds, loader = load_val_dataset(la, LOADER_NAMES[name])
        texts = get_text_prompts_val([ds], [LOADER_NAMES[name]])[0]
        tokens = clip.tokenize(texts).to(device)
        clf = make_classifier(model, prompt_token, texts)
        cu8, dl, lab = [], [], []
        c_ok = a_ok = seen = 0
        max_dev = 0.0
        t0 = time.time()
        for x, y in tqdm(loader, desc=f"cache-{name}"):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                delta = attack_pgd(None, torch.nn.Identity(), model, None, None, lambda: prompt_token, crit,
                                   x, y, alpha, args.steps, "l_inf", text_tokens=tokens,
                                   epsilon=eps, dataset_name=None)
            adv = (x + delta.detach()).clamp(0, 1)
            max_dev = max(max_dev, (adv - x).abs().max().item())
            x_u8 = (x * 255.0).round().to(torch.uint8)
            with torch.no_grad():
                c_ok += (clf.logits(x).argmax(1) == y).sum().item()
                a_ok += (clf.logits(adv).argmax(1) == y).sum().item()
            cu8.append(x_u8.cpu())
            dl.append((adv - x).half().cpu())
            lab.append(y.cpu())
            seen += y.numel()
            if args.max_images and seen >= args.max_images:
                break
        n = sum(t.numel() for t in lab)
        payload = {
            "dataset": name, "loader_name": LOADER_NAMES[name],
            "clean_u8": torch.cat(cu8), "delta": torch.cat(dl), "labels": torch.cat(lab),
            "texts": texts, "test_eps": eps, "test_numsteps": args.steps, "test_stepsize": alpha,
            "attack_seed": args.attack_seed, "batch_size": args.batch_size, "n": n,
            "clip_clean_correct": c_ok, "clip_adv_correct": a_ok, "max_linf": max_dev,
            "truncated": bool(args.max_images),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        summary[name] = {"n": n, "clip_clean_acc": 100 * c_ok / n, "clip_adv_acc": 100 * a_ok / n,
                         "n_classes": len(texts), "prompt_example": texts[0],
                         "max_linf_x255": max_dev * 255, "seconds": time.time() - t0}
        print(name, summary[name], flush=True)

    if summary:
        write_json(Path(args.cache_dir) / f"summary_{int(time.time())}.json",
                   {**summary, "_run": run_info({"script": "build_cache.py", "args": vars(args)})})


if __name__ == "__main__":
    main()
