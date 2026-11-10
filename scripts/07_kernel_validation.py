"""Validation run for the centered and trace-normalized kernel implementation (short repeat count; the full nested cross-validation is reported by nested_cv_prediction.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the script directory to the path
sys.path.insert(0, str(Path(__file__).parent))
from kernel_utils import (
    feature_kernel_centered_scaled,
    solve_kernel_ridge_with_intercept,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261106)


def load_data():
    """Load genotype and phenotype data."""
    print("Loading data...")

    # Phenotype
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)
    ecotype_ids = pheno["ecotype_id"].to_numpy()

    # Genotypes
    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp["maf_83"].to_numpy()

    # Filter: MAF > 0.05 and non-monomorphic
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()[keep]

    # Build the SNP dosage matrix (accessions x SNPs)
    # Current implementation: 0/1 coding (haplotype)
    X_snp = (codes[keep, :] == minor[:, None]).astype(np.float32).T

    print(f"  Accessions: {len(y)}")
    print(f"  SNPs: {X_snp.shape[1]}")
    print(f"  Phenotype range: [{y.min():.1f}, {y.max():.1f}]")

    return y, X_snp, ecotype_ids, snp, keep


def build_block_kernel():
    """Build the block-mean kernel (centered and trace-normalized)."""
    print("Building block-mean kernel...")

    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    subset_path = PROCESSED / "block_features_83.npy"

    if subset_path.exists():
        block_mean = np.asarray(
            np.load(subset_path, mmap_mode="r")[:, :, 0], dtype=np.float64
        )
    else:
        block_feats = np.load(PROCESSED / "block_features.npy", mmap_mode="r")
        block_mean = np.asarray(block_feats[cols, :, 0], dtype=np.float64)

    print(f"  Blocks: {block_mean.shape[1]}")

    # Use the centered, trace-normalized feature kernel
    K_block, _, _, _ = feature_kernel_centered_scaled(block_mean)

    return K_block, block_mean


def simple_cv(K, y, n_repeats=3, n_folds=5, lambda_reg=1.0):
    """Simple cross-validation (implementation validation only)."""
    from sklearn.model_selection import KFold

    n = len(y)
    preds = np.zeros((n_repeats, n))

    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep)
        for tr_idx, te_idx in kf.split(K):
            K_train = K[np.ix_(tr_idx, tr_idx)]
            K_test = K[np.ix_(te_idx, tr_idx)]
            y_train = y[tr_idx]

            # Use the kernel ridge solver with an explicit intercept
            _, _, y_pred = solve_kernel_ridge_with_intercept(
                K_train, y_train, lambda_reg, K_test
            )
            preds[rep, te_idx] = y_pred

    # Compute metrics
    from scipy import stats

    r_per_rep = np.array([stats.pearsonr(preds[i], y)[0] for i in range(n_repeats)])
    mae_per_rep = np.array([np.mean(np.abs(preds[i] - y)) for i in range(n_repeats)])

    return {
        "pearson_mean": float(r_per_rep.mean()),
        "pearson_sd": float(r_per_rep.std()),
        "mae_mean": float(mae_per_rep.mean()),
        "preds": preds,
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Kernel implementation validation")
    print("=" * 60)

    # Load data
    y, X_snp, ecotype_ids, snp_meta, keep_idx = load_data()

    # Build the SNP kernel (centered and trace-normalized)
    print("\nBuilding SNP kernel (centered and trace-normalized)...")
    K_snp, X_snp_scaled, mean_snp, std_snp = feature_kernel_centered_scaled(X_snp)
    print(f"  SNP kernel trace/n: {np.trace(K_snp) / len(y):.4f}")

    # Build the block kernel (centered and trace-normalized)
    K_block, block_mean = build_block_kernel()
    print(f"  Block kernel trace/n: {np.trace(K_block) / len(y):.4f}")

    # Combined kernel: equal-weight dual kernel
    K_combined = 0.5 * K_snp + 0.5 * K_block
    print(f"  Combined kernel trace/n: {np.trace(K_combined) / len(y):.4f}")

    # Simple CV validation
    print("\nRunning simple cross-validation (3 repeats x 5 folds)...")

    # Select lambda first (within-training CV)
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    best_lambda = None
    best_mae = np.inf

    for lam in lambdas:
        result = simple_cv(K_combined, y, n_repeats=2, n_folds=3, lambda_reg=lam)
        if result["mae_mean"] < best_mae:
            best_mae = result["mae_mean"]
            best_lambda = lam

    print(f"  Best lambda: {best_lambda:.4f}")
    print(f"  Corresponding MAE: {best_mae:.2f} days")

    # Run the full CV with the best lambda
    result_snp = simple_cv(K_snp, y, n_repeats=3, n_folds=5, lambda_reg=best_lambda)
    result_block = simple_cv(K_block, y, n_repeats=3, n_folds=5, lambda_reg=best_lambda)
    result_combined = simple_cv(K_combined, y, n_repeats=3, n_folds=5, lambda_reg=best_lambda)

    print("\n" + "=" * 60)
    print("Prediction results (3 repeats x 5-fold CV)")
    print("=" * 60)
    print(f"{'Model':<20} {'Pearson r':<15} {'MAE (days)':<15}")
    print("-" * 60)
    print(f"{'SNP kernel':<20} {result_snp['pearson_mean']:<15.4f} {result_snp['mae_mean']:<15.2f}")
    print(f"{'Block-mean kernel':<20} {result_block['pearson_mean']:<15.4f} {result_block['mae_mean']:<15.2f}")
    print(f"{'Combined (0.5+0.5)':<20} {result_combined['pearson_mean']:<15.4f} {result_combined['mae_mean']:<15.2f}")

    # Save results
    output = {
        "description": "Validation of the centered and trace-normalized kernel implementation",
        "date": "2026-11-06",
        "n_repeats": 3,
        "n_folds": 5,
        "best_lambda": best_lambda,
        "results": {
            "snp_kernel": {
                "pearson_mean": result_snp["pearson_mean"],
                "pearson_sd": result_snp["pearson_sd"],
                "mae_mean": result_snp["mae_mean"],
            },
            "block_kernel": {
                "pearson_mean": result_block["pearson_mean"],
                "pearson_sd": result_block["pearson_sd"],
                "mae_mean": result_block["mae_mean"],
            },
            "combined_kernel": {
                "pearson_mean": result_combined["pearson_mean"],
                "pearson_sd": result_combined["pearson_sd"],
                "mae_mean": result_combined["mae_mean"],
            },
        },
        "kernel_properties": {
            "snp_trace_per_n": float(np.trace(K_snp) / len(y)),
            "block_trace_per_n": float(np.trace(K_block) / len(y)),
            "combined_trace_per_n": float(np.trace(K_combined) / len(y)),
        },
    }

    output_path = TABLES / "kernel_validation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("\nValidation complete.")


if __name__ == "__main__":
    main()