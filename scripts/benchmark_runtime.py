"""Inference latency and GPU memory of TTC / SAVE-TTC configurations.

Images are pre-loaded to the GPU, so data loading and adversarial-example generation are excluded; the
timing covers the full defense call (counterattack forward/backward, tau gate, views, averaging).
CUDA events after synchronisation, warm-up batches, repeated passes, batch size 1 and 8. Memory is
torch.cuda.max_memory_allocated / max_memory_reserved after a reset (includes CLIP weights).

    python scripts/benchmark_runtime.py --config configs/runtime.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from savettc import bench, paths  # noqa: E402
from savettc import core as sc  # noqa: E402
from savettc.clip_model import device, load_clip, make_classifier  # noqa: E402
from savettc.common import parse_with_config, run_info, set_global_seed, write_json  # noqa: E402
from savettc.data import DATASETS, load_cache  # noqa: E402

DEFAULT_CONFIGS = ["1/1/2", "3/1/2", "1/9/2", "3/9/2", "1/9/4", "2/9/4", "3/9/4",
                   "1/9/6", "1/9/8", "1/9/12", "1/9/16", "1/9/20"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--dataset", default="Pets", choices=DATASETS)
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--n_images_bs8", type=int, default=256)
    ap.add_argument("--n_images_bs1", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--precisions", nargs="+", default=["fp32"], choices=["fp32", "amp"])
    ap.add_argument("--cache_dir", default=str(paths.CACHE_DIR))
    ap.add_argument("--out_dir", default=str(paths.OUT_DIR))
    args = parse_with_config(ap)
    if not torch.cuda.is_available():
        sys.exit("benchmark_runtime.py needs a CUDA GPU")

    paths.add_ttc_to_path()
    out_dir = Path(args.out_dir) / "runtime"
    raw_path, summ_path = out_dir / "runtime_raw.csv", out_dir / "runtime_summary.jsonl"
    summ_path.parent.mkdir(parents=True, exist_ok=True)

    set_global_seed(0)
    model, prompt_token = load_clip()
    cache = load_cache(args.dataset, args.cache_dir)
    clf = make_classifier(model, prompt_token, cache.texts)
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated() / 2 ** 20
    env = {"gpu_snapshot_before": bench.gpu_snapshot(), "base_allocated_mb_after_model_and_text": base_alloc,
           "run": run_info({"script": "benchmark_runtime.py", "args": vars(args)})}

    for bs in args.batch_sizes:
        n = args.n_images_bs8 if bs > 1 else args.n_images_bs1
        n = min((n // bs) * bs, (cache.n // bs) * bs)
        batches = [cache.adv(list(range(i, i + bs))).to(device) for i in range(0, n, bs)]
        for prec in args.precisions:
            amp = prec == "amp"
            for rvn in ["clip"] + list(args.configs):
                if rvn == "clip":
                    cfg = None

                    def fn(X, b):
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            return clf.logits(X).softmax(-1)
                else:
                    cfg = sc.SaveConfig.from_rvn(rvn)
                    stats = {"gate": [], "steps": 0}

                    def fn(X, b, cfg=cfg, stats=stats):
                        ids = list(range(b * X.size(0), (b + 1) * X.size(0)))
                        o = sc.save_predict(clf, X, cfg, ids, base_seed=0, amp=amp)
                        stats["gate"].append(o.gate_open.float().mean().item())
                        stats["steps"] += o.grad_steps_executed
                        return o.probs
                rows, peak_alloc, peak_res = bench.time_method(fn, batches, args.warmup, args.repeats)
                method = "CLIP" if cfg is None else ("TTC" if rvn == "1/1/2" else "SAVE-TTC")
                bench.append_raw(raw_path, "savettc", method, rvn, "faithful", prec, bs, rows)
                s = bench.summarise(rows, bs)
                rec = {"method": method, "rvn": rvn, "precision": prec, "batch_size": bs, "n_images": n,
                       "warmup_batches": args.warmup, "repeats": args.repeats, **s,
                       "peak_allocated_mb": peak_alloc, "peak_reserved_mb": peak_res,
                       "base_allocated_mb": base_alloc}
                if cfg is not None:
                    calls = len(stats["gate"])
                    rec.update({"max_inner_steps_RN": cfg.R * cfg.N, "predictions_RV": cfg.R * cfg.V,
                                "gate_open_frac": float(sum(stats["gate"]) / calls),
                                "batch_grad_steps_per_call": stats["steps"] / calls})
                with open(summ_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"bs={bs} {prec} {rvn:7s} {s['ms_per_image_mean']:8.2f} ms/img "
                      f"(med {s['ms_per_image_median']:.2f}, p90 {s['ms_per_image_p90']:.2f}) "
                      f"alloc {peak_alloc:.0f}MB res {peak_res:.0f}MB", flush=True)
        del batches
        torch.cuda.empty_cache()
    env["gpu_snapshot_after"] = bench.gpu_snapshot()
    write_json(out_dir / "equipment.json", env)


if __name__ == "__main__":
    main()
