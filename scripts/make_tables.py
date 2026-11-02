"""Aggregate evaluate.py / benchmark_runtime.py outputs into tables.

    python scripts/make_tables.py

Reads   <out_dir>/summary/eval.jsonl, <out_dir>/runtime/runtime_summary.jsonl (optional)
Writes  <out_dir>/tables/results_<split>.{md,csv}, per_dataset_<split>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from savettc import paths  # noqa: E402
from savettc.data import DATASETS  # noqa: E402


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def config_key(rec):
    return rec["config"] if rec.get("views") in (None, "id1", "code9") else f"{rec['config']} ({rec['views']})"


def rvn_sort_key(cfg):
    if cfg.startswith("clip"):
        return (-1, 0, 0, cfg)
    r, v, n = (int(t) for t in cfg.split(" ")[0].split("/"))
    return (v, r, n, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(paths.OUT_DIR))
    ap.add_argument("--split", default="adv", choices=["adv", "clean"])
    ap.add_argument("--include_partial", action="store_true",
                    help="use --max_images (smoke-test) results instead of full-dataset ones")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    recs = [r for r in read_jsonl(out_dir / "summary" / "eval.jsonl")
            if r["split"] == args.split and (("_first" in r["cfg_tag"]) == args.include_partial)]
    if not recs:
        sys.exit(f"no full-dataset '{args.split}' results in {out_dir / 'summary' / 'eval.jsonl'}")

    acc = {}  # (config, dataset, seed) -> acc ; later lines win
    for r in recs:
        acc[(config_key(r), r["dataset"], r["seed"])] = r["acc"]
    configs = sorted({k[0] for k in acc}, key=rvn_sort_key)
    datasets = [d for d in DATASETS if any(k[1] == d for k in acc)]

    lat = {}
    for r in read_jsonl(out_dir / "runtime" / "runtime_summary.jsonl"):
        if r.get("batch_size") == 8 and r.get("precision") == "fp32":
            lat[r["rvn"]] = r["ms_per_image_mean"]

    rows, per_ds = [], []
    for c in configs:
        seeds = sorted({k[2] for k in acc if k[0] == c and all((c, d, k[2]) in acc for d in datasets)})
        if not seeds:
            continue
        means = [np.mean([acc[(c, d, s)] for d in datasets]) for s in seeds]
        ds_means = {d: float(np.mean([acc[(c, d, s)] for s in seeds])) for d in datasets}
        rows.append({"config": c, "n_seeds": len(seeds), "mean": float(np.mean(means)),
                     "sd": float(np.std(means, ddof=1)) if len(seeds) > 1 else None,
                     "ms_per_image_bs8": lat.get(c.split(" ")[0]), **ds_means})
        for d in datasets:
            per_ds.append({"config": c, "dataset": d, "acc": ds_means[d], "n_seeds": len(seeds)})

    tdir = out_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    fields = ["config", "n_seeds", *datasets, "mean", "sd", "ms_per_image_bs8"]
    with open(tdir / f"results_{args.split}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(tdir / f"per_dataset_{args.split}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "dataset", "acc", "n_seeds"])
        w.writeheader()
        w.writerows(per_ds)

    head = "| R/V/N | " + " | ".join(datasets) + " | Avg. | ms/img (bs=8) |"
    lines = [f"{args.split} accuracy (%), mean over defense seeds; Avg. ± SD over seeds", "", head,
             "|" + "---|" * (len(datasets) + 3)]
    for r in rows:
        avg = f"{r['mean']:.2f}" + (f" ± {r['sd']:.2f}" if r["sd"] is not None else "")
        ms = f"{r['ms_per_image_bs8']:.1f}" if r["ms_per_image_bs8"] else "–"
        lines.append(f"| {r['config']} | " + " | ".join(f"{r[d]:.2f}" for d in datasets) + f" | {avg} | {ms} |")
    (tdir / f"results_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
