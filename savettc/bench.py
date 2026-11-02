"""Latency / memory measurement: CUDA events, warm-up, repeated passes over pre-loaded batches."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import torch

RAW_FIELDS = ["impl", "method", "rvn", "variant", "precision", "batch_size", "repeat", "batch_id",
              "n_images", "ms_batch", "ms_per_image"]


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,"
             "power.draw,power.limit,pstate,clocks.sm,clocks.max.sm", "--format=csv,noheader"],
            text=True, timeout=30).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def time_method(fn, batches, warmup: int, repeats: int):
    """fn(X, batch_id). Returns [(repeat, batch_id, ms)], peak allocated MB, peak reserved MB."""
    for i in range(min(warmup, len(batches))):
        fn(batches[i], i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for rep in range(repeats):
        for b, X in enumerate(batches):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            fn(X, b)
            e.record()
            torch.cuda.synchronize()
            rows.append((rep, b, s.elapsed_time(e)))
    return rows, torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20


def summarise(rows, bs):
    per_img = np.array([ms / bs for _, _, ms in rows])
    return {"ms_per_image_mean": float(per_img.mean()), "ms_per_image_median": float(np.median(per_img)),
            "ms_per_image_std": float(per_img.std(ddof=1)) if len(per_img) > 1 else 0.0,
            "ms_per_image_p90": float(np.percentile(per_img, 90)), "n_batches_timed": len(rows)}


def append_raw(path: Path, impl, method, rvn, variant, precision, bs, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(RAW_FIELDS)
        for rep, b, ms in rows:
            w.writerow([impl, method, rvn, variant, precision, bs, rep, b, bs, f"{ms:.4f}", f"{ms / bs:.4f}"])
