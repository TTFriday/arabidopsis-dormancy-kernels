"""Sensitivity of the block kernel to merging adjacent blocks (2x and 4x granularity)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import nested_cv_prediction as P12
from kernel_utils import feature_kernel_centered_scaled

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def merge_blocks(mean_dosage, chrom, size):
    merged = []
    for c in np.unique(chrom):
        idx = np.flatnonzero(chrom == c)
        for start in range(0, len(idx), size):
            merged.append(mean_dosage[:, idx[start:start + size]].mean(axis=1))
    return np.column_stack(merged)


def main():
    t0 = time.time()
    data = P12.load_data()
    y = data["y"]
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    chrom = blocks["chrom"].to_numpy()

    results = {}
    for label, feats in [
        ("1x (25-SNP blocks)", data["block_mean"]),
        ("2x merge", merge_blocks(data["block_mean"], chrom, 2)),
        ("4x merge", merge_blocks(data["block_mean"], chrom, 4)),
    ]:
        K, _, _, _ = feature_kernel_centered_scaled(feats)
        res = P12.nested_cv_kernel(K, y)
        key = label if label.startswith("1x") else f"{label} ({feats.shape[1]} windows)"
        results[key] = {
            "n_features": int(feats.shape[1]),
            "pearson_mean": float(res["pearson_mean"]),
            "pearson_sd": float(res["pearson_sd"]),
            "mae_mean": float(res["mae_mean"]),
        }
        print(f"[granularity] {key}: r={res['pearson_mean']:.4f}", flush=True)

    results["description"] = "corrected pipeline, 10 repeats x 5-fold nested CV"
    results["elapsed_seconds"] = time.time() - t0
    (TABLES / "block_granularity_sensitivity.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()