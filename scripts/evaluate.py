"""Evaluate CLIP / TTC / SAVE-TTC configurations on the shared adversarial cache.

Configurations are 'clip' (undefended) or R/V/N strings: 1/1/2 is the original TTC, 3/9/4 is SAVE-TTC.

    python scripts/evaluate.py --config configs/main_table.yaml
    python scripts/evaluate.py --configs clip 1/1/2 1/9/4 3/9/4 --seeds 0 1 2

Outputs (resumable: finished parts are skipped)
    <out_dir>/per_image/<dataset>/<cfg>_seed<s>_<split>.csv
    <out_dir>/summary/eval.jsonl        one line per (dataset, config, seed, split)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from savettc import core as sc  # noqa: E402
from savettc import paths  # noqa: E402
from savettc.clip_model import device, load_clip, make_classifier  # noqa: E402
from savettc.common import parse_with_config, run_info  # noqa: E402
from savettc.data import DATASETS, load_cache  # noqa: E402

CSV_FIELDS = ["image_idx", "label", "pred", "correct", "p_true", "p_max", "gate_open_frac", "tau_mean",
              "d_view", "d_start"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_part(clf, cache, cfg, cfg_tag, seed, split, idx, batch_size, amp, log_every=500):
    rows, n_correct, t_sync, gate_open, total_grad_steps, seen = [], 0, 0.0, [], 0, 0
    nan = float("nan")
    for b0 in range(0, len(idx), batch_size):
        bidx = idx[b0:b0 + batch_size]
        X = (cache.adv(bidx) if split == "adv" else cache.clean(bidx)).to(device)
        y = cache.labels[bidx].to(device)
        sync()
        t0 = time.perf_counter()
        if cfg is None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                probs = clf.logits(X).float().softmax(-1)
            gate = torch.zeros(len(bidx), 1, device=device)
            tau = torch.zeros(len(bidx), 1, device=device)
            dv = ds = torch.full((len(bidx),), nan, device=device)
        else:
            out = sc.save_predict(clf, X, cfg, [int(i) for i in bidx], base_seed=seed,
                                  diagnostics=(cfg.R * cfg.V > 1), amp=amp)
            probs, gate, tau = out.probs, out.gate_open.float(), out.tau
            dv = out.d_view if out.d_view is not None else torch.full((len(bidx),), nan, device=device)
            ds = out.d_start if out.d_start is not None else torch.full((len(bidx),), nan, device=device)
            total_grad_steps += out.grad_steps_executed * len(bidx)
        sync()
        t_sync += time.perf_counter() - t0
        pred = probs.argmax(1)
        ok = pred == y
        n_correct += int(ok.sum())
        py = probs.gather(1, y.view(-1, 1)).squeeze(1)
        gate_open.append(gate.mean(1).cpu())
        for j, i in enumerate(bidx):
            rows.append([int(i), int(y[j]), int(pred[j]), int(ok[j]), f"{py[j].item():.6g}",
                         f"{probs[j].max().item():.6g}", f"{gate[j].mean().item():.4g}",
                         f"{tau[j].mean().item():.5g}", f"{dv[j].item():.5g}", f"{ds[j].item():.5g}"])
        seen += len(bidx)
        if seen // log_every != (seen - len(bidx)) // log_every:
            print(f"    [{cache.dataset} {cfg_tag} s{seed} {split}] {seen}/{len(idx)} "
                  f"acc={100 * n_correct / seen:.2f}", flush=True)
    n = len(idx)
    return rows, {
        "correct": n_correct, "total": n, "acc": 100.0 * n_correct / n,
        "ms_per_image_amortised": 1000.0 * t_sync / n,
        "gate_open_frac_mean": float(torch.cat(gate_open).mean()) if cfg is not None else None,
        "inner_grad_steps_per_image_executed": (total_grad_steps / n) if cfg is not None else 0,
        "inner_steps_max_RN": (cfg.R * cfg.N) if cfg is not None else 0,
        "predictions_RV": (cfg.R * cfg.V) if cfg is not None else 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--configs", nargs="+", default=["clip", "1/1/2", "3/9/4"], help="'clip' or R/V/N")
    ap.add_argument("--views9", default="code9", choices=["code9", "paper9"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="defense seeds (random starts)")
    ap.add_argument("--splits", nargs="+", default=["adv"], choices=["adv", "clean"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--amp", choices=["on", "off"], default="on",
                    help="fp16 autocast, as in the TTC evaluation protocol")
    ap.add_argument("--cache_seed", type=int, default=1, help="attack seed of the cache")
    ap.add_argument("--cache_dir", default=str(paths.CACHE_DIR))
    ap.add_argument("--out_dir", default=str(paths.OUT_DIR))
    ap.add_argument("--max_images", type=int, default=0, help="only the first n images (smoke test)")
    args = parse_with_config(ap)

    paths.add_ttc_to_path()
    amp = args.amp == "on" and torch.cuda.is_available()
    out_dir = Path(args.out_dir)
    summ_path = out_dir / "summary" / "eval.jsonl"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    info = run_info({"script": "evaluate.py", "amp": amp})
    model, prompt_token = load_clip()

    for dname in args.datasets:
        cache = load_cache(dname, args.cache_dir, args.cache_seed)
        if cache.payload.get("truncated"):
            print(f"[warning] {dname}: truncated smoke-test cache (n={cache.n})")
        clf = make_classifier(model, prompt_token, cache.texts)
        idx = list(range(cache.n if args.max_images <= 0 else min(cache.n, args.max_images)))
        print(f"[{dname}] n={len(idx)} classes={len(cache.texts)} prompt='{cache.texts[0]}' amp={amp}", flush=True)
        for c in args.configs:
            cfg = None if c == "clip" else sc.SaveConfig.from_rvn(c, views9=args.views9)
            cfg_tag = "CLIP" if cfg is None else cfg.tag()
            suffix = "" if args.max_images <= 0 else f"_first{len(idx)}"
            for seed in ([0] if cfg is None else args.seeds):
                for split in args.splits:
                    pp = out_dir / "per_image" / dname / f"{cfg_tag}{suffix}_seed{seed}_{split}.csv"
                    if pp.exists():
                        print(f"  [skip] {pp.name}")
                        continue
                    t0 = time.time()
                    rows, res = run_part(clf, cache, cfg, cfg_tag, seed, split, idx, args.batch_size, amp)
                    pp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = pp.with_suffix(".tmp")
                    with open(tmp, "w", newline="", encoding="utf-8") as fh:
                        w = csv.writer(fh)
                        w.writerow(CSV_FIELDS)
                        w.writerows(rows)
                    os.replace(tmp, pp)
                    rec = {"dataset": dname,
                           "config": "clip" if cfg is None else f"{cfg.R}/{cfg.V}/{cfg.N}",
                           "cfg_tag": cfg_tag + suffix, "views": None if cfg is None else cfg.views,
                           "seed": seed, "split": split, "amp": amp, "batch_size": args.batch_size,
                           "cache_seed": args.cache_seed, "wall_seconds": time.time() - t0, **res,
                           "run": {k: info[k] for k in ("timestamp", "torch", "gpu")}}
                    with open(summ_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(f"  {dname} {cfg_tag} seed={seed} {split}: {res['correct']}/{res['total']} = "
                          f"{res['acc']:.2f}%", flush=True)
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
