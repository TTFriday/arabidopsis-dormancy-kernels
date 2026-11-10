"""Nested cross-validation of SNP, block-mean, graph and combined kernels, with grouped leave-one-out evaluation across genetic subpopulations."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).parent))
from kernel_utils import (
    feature_kernel_centered_scaled,
    solve_kernel_ridge_with_intercept,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261106)

# Extended lambda grid (avoids the optimum sitting on a grid boundary)
LAMBDAS = np.logspace(-8, 2, 21)  # 1e-8 to 100

# CV settings
N_REPEATS = 10
N_FOLDS = 5
INNER_FOLDS = 5


def load_data():
    """Load all data."""
    print("=" * 70)
    print("Nested cross-validation of genomic kernels")
    print("=" * 70)
    print("\nLoading data...")

    # Phenotype
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)
    ecotype_ids = pheno["ecotype_id"].to_numpy()
    nativenames = pheno["nativename"].to_numpy()

    # SNP genotypes
    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp_meta = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp_meta["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp_meta["minor_code"].to_numpy() < 4))
    minor = snp_meta["minor_code"].to_numpy()[keep]
    X_snp = (codes[keep, :] == minor[:, None]).astype(np.float32).T

    # Block features
    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    subset_path = PROCESSED / "block_features_83.npy"
    if subset_path.exists():
        block_mean = np.asarray(
            np.load(subset_path, mmap_mode="r")[:, :, 0], dtype=np.float64
        )
    else:
        block_feats = np.load(PROCESSED / "block_features.npy", mmap_mode="r")
        block_mean = np.asarray(block_feats[cols, :, 0], dtype=np.float64)

    # Graph embeddings
    graph_path = PROCESSED / "graph_embeddings_83.npy"
    if graph_path.exists():
        graph_nodes = np.load(graph_path)
    else:
        graph_nodes = None

    # Kinship matrix (used for grouping)
    kinship = np.load(PROCESSED / "kinship_83.npy")

    print(f"  Accessions: {len(y)}")
    print(f"  SNPs: {X_snp.shape[1]:,}")
    print(f"  Blocks: {block_mean.shape[1]:,}")
    if graph_nodes is not None:
        print(f"  Graph embeddings: {graph_nodes.shape}")

    return {
        "y": y,
        "ecotype_ids": ecotype_ids,
        "nativenames": nativenames,
        "X_snp": X_snp,
        "block_mean": block_mean,
        "graph_nodes": graph_nodes,
        "kinship": kinship,
    }


def build_kernels(data):
    """Build all kernel matrices."""
    print("\nBuilding kernel matrices...")
    kernels = {}

    # SNP kernel
    K_snp, _, _, _ = feature_kernel_centered_scaled(data["X_snp"])
    kernels["snp"] = K_snp
    print(f"  SNP kernel: tr(K)/n = {np.trace(K_snp) / len(data['y']):.4f}")

    # Block-mean kernel
    K_block, _, _, _ = feature_kernel_centered_scaled(data["block_mean"])
    kernels["block"] = K_block
    print(f"  Block kernel: tr(K)/n = {np.trace(K_block) / len(data['y']):.4f}")

    # SSL graph kernel
    if data["graph_nodes"] is not None:
        n_blocks = data["graph_nodes"].shape[1]
        K_graph = np.zeros((len(data["y"]), len(data["y"])))
        for b in range(n_blocks):
            z = data["graph_nodes"][:, b, :].astype(np.float64)
            K_b, _, _, _ = feature_kernel_centered_scaled(z)
            K_graph += K_b
        K_graph /= n_blocks
        kernels["ssl_graph"] = K_graph
        print(f"  SSL graph kernel: tr(K)/n = {np.trace(K_graph) / len(data['y']):.4f}")

        # Random-initialization control (build the kernel after shuffling sample order)
        rng_rand = np.random.default_rng(42)
        perm = rng_rand.permutation(len(data["y"]))
        graph_nodes_perm = data["graph_nodes"][perm]
        K_rand_graph = np.zeros((len(data["y"]), len(data["y"])))
        for b in range(n_blocks):
            z = graph_nodes_perm[:, b, :].astype(np.float64)
            K_b, _, _, _ = feature_kernel_centered_scaled(z)
            K_rand_graph += K_b
        K_rand_graph /= n_blocks
        kernels["random_graph"] = K_rand_graph
        print(f"  Random graph kernel: tr(K)/n = {np.trace(K_rand_graph) / len(data['y']):.4f}")

    # Combined kernels (mixture weight fixed at w_SNP = 0.1)
    kernels["combined_w01"] = 0.1 * K_snp + 0.9 * K_block
    kernels["combined_w05"] = 0.5 * K_snp + 0.5 * K_block
    print(f"  Combined kernel (0.1+0.9): tr(K)/n = {np.trace(kernels['combined_w01']) / len(data['y']):.4f}")

    return kernels


def nested_cv_kernel(K, y, n_repeats=N_REPEATS, n_folds=N_FOLDS, inner_folds=INNER_FOLDS):
    """Nested cross-validation (outer evaluation, inner hyper-parameter selection)."""
    n = len(y)
    preds = np.zeros((n_repeats, n))
    chosen_lambdas = []

    for rep in range(n_repeats):
        kf_outer = KFold(n_splits=n_folds, shuffle=True, random_state=rep)
        for tr_idx, te_idx in kf_outer.split(K):
            K_train = K[np.ix_(tr_idx, tr_idx)]
            K_test = K[np.ix_(te_idx, tr_idx)]
            y_train = y[tr_idx]

            # Inner CV selects lambda
            best_mse = np.inf
            best_lam = LAMBDAS[len(LAMBDAS) // 2]
            kf_inner = KFold(n_splits=inner_folds, shuffle=True, random_state=rep * 100)

            for lam in LAMBDAS:
                inner_preds = np.zeros(len(y_train))
                for itr, iye in kf_inner.split(K_train):
                    K_itr = K_train[np.ix_(itr, itr)]
                    K_iye = K_train[np.ix_(iye, itr)]
                    y_itr = y_train[itr]
                    _, _, y_iye = solve_kernel_ridge_with_intercept(
                        K_itr, y_itr, lam, K_iye
                    )
                    inner_preds[iye] = y_iye
                mse = np.mean((inner_preds - y_train) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_lam = lam

            # Final fit
            _, _, y_pred = solve_kernel_ridge_with_intercept(
                K_train, y_train, best_lam, K_test
            )
            preds[rep, te_idx] = y_pred
            chosen_lambdas.append(best_lam)

    # Compute metrics
    r_per_rep = np.array([stats.pearsonr(preds[i], y)[0] for i in range(n_repeats)])
    rho_per_rep = np.array([stats.spearmanr(preds[i], y).statistic for i in range(n_repeats)])
    mae_per_rep = np.array([np.mean(np.abs(preds[i] - y)) for i in range(n_repeats)])
    rmse_per_rep = np.array([np.sqrt(np.mean((preds[i] - y) ** 2)) for i in range(n_repeats)])

    return {
        "pearson_mean": float(r_per_rep.mean()),
        "pearson_sd": float(r_per_rep.std()),
        "spearman_mean": float(rho_per_rep.mean()),
        "spearman_sd": float(rho_per_rep.std()),
        "mae_mean": float(mae_per_rep.mean()),
        "mae_sd": float(mae_per_rep.std()),
        "rmse_mean": float(rmse_per_rep.mean()),
        "rmse_sd": float(rmse_per_rep.std()),
        "lambda_mode": float(pd.Series(chosen_lambdas).mode()[0]),
        "lambda_median": float(np.median(chosen_lambdas)),
        "lambda_min": float(min(chosen_lambdas)),
        "lambda_max": float(max(chosen_lambdas)),
        "preds": preds,
    }


def baseline_pc_regression(X_snp, y, n_pcs=10):
    """PC-regression baseline."""
    # PCA
    mean = X_snp.mean(axis=0)
    X_centered = X_snp - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    PCs = U[:, :n_pcs] * S[:n_pcs]

    # Nested CV
    preds = np.zeros((N_REPEATS, len(y)))
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        for tr, te in kf.split(PCs):
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            model.fit(PCs[tr], y[tr])
            preds[rep, te] = model.predict(PCs[te])

    r_per_rep = np.array([stats.pearsonr(preds[i], y)[0] for i in range(N_REPEATS)])
    mae_per_rep = np.array([np.mean(np.abs(preds[i] - y)) for i in range(N_REPEATS)])

    return {
        "pearson_mean": float(r_per_rep.mean()),
        "pearson_sd": float(r_per_rep.std()),
        "mae_mean": float(mae_per_rep.mean()),
        "mae_sd": float(mae_per_rep.std()),
    }


def grouped_cv(K, y, kinship, n_groups=6):
    """Grouped (cross-subpopulation) extrapolation CV."""
    # Hierarchical clustering on the kinship matrix
    from scipy.cluster.hierarchy import linkage, fcluster

    # Convert to a distance matrix
    dist = 1 - kinship / kinship.max()
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering
    condensed_dist = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(condensed_dist, method="average")
    groups = fcluster(Z, t=n_groups, criterion="maxclust")

    # GroupKFold
    gkf = GroupKFold(n_splits=min(n_groups, 5))
    preds = np.zeros(len(y))
    counts = np.zeros(len(y))

    for tr_idx, te_idx in gkf.split(K, y, groups):
        K_train = K[np.ix_(tr_idx, tr_idx)]
        K_test = K[np.ix_(te_idx, tr_idx)]
        y_train = y[tr_idx]

        # Inner lambda selection (simplified: fixed lambda)
        best_lam = 1e-4

        _, _, y_pred = solve_kernel_ridge_with_intercept(
            K_train, y_train, best_lam, K_test
        )
        preds[te_idx] = y_pred
        counts[te_idx] += 1

    # Compute metrics
    r = stats.pearsonr(preds, y)[0]
    rho = stats.spearmanr(preds, y).statistic
    mae = np.mean(np.abs(preds - y))

    return {
        "pearson_r": float(r),
        "spearman_rho": float(rho),
        "mae": float(mae),
        "n_groups": n_groups,
        "group_sizes": [int((groups == g).sum()) for g in range(1, n_groups + 1)],
    }


def main():
    """Main entry point."""
    start_time = time.time()

    # Load data
    data = load_data()
    y = data["y"]

    # Build kernels
    kernels = build_kernels(data)

    # === Baseline models ===
    print("\n" + "=" * 70)
    print("Evaluating baseline models...")
    print("=" * 70)

    results = {}

    # Training mean
    print("\n[1/12] Training-mean baseline...")
    mean_pred = np.full(len(y), y.mean())
    results["mean_baseline"] = {
        "pearson_r": 0.0,
        "mae": float(np.mean(np.abs(mean_pred - y))),
        "description": "Predict the training mean for every sample",
    }
    print(f"  MAE: {results['mean_baseline']['mae']:.2f} days")

    # PC regression
    print("\n[2/12] PC-regression baseline...")
    results["pc_regression"] = baseline_pc_regression(data["X_snp"], y, n_pcs=10)
    print(f"  Pearson r: {results['pc_regression']['pearson_mean']:.4f} ± {results['pc_regression']['pearson_sd']:.4f}")
    print(f"  MAE: {results['pc_regression']['mae_mean']:.2f} ± {results['pc_regression']['mae_sd']:.2f} days")

    # === Kernel models ===
    print("\n" + "=" * 70)
    print("Evaluating kernel models (10 repeats x 5-fold nested CV)...")
    print("=" * 70)

    model_names = [
        ("snp", "SNP kernel"),
        ("block", "Block-mean kernel"),
        ("ssl_graph", "SSL graph kernel"),
        ("random_graph", "Random graph control"),
        ("combined_w01", "Combined kernel (0.1+0.9)"),
        ("combined_w05", "Combined kernel (0.5+0.5)"),
    ]

    for i, (key, name) in enumerate(model_names, start=3):
        if key not in kernels:
            print(f"\n[{i}/12] {name}: skipped (data not available)")
            continue

        print(f"\n[{i}/12] {name}...")
        K = kernels[key]
        result = nested_cv_kernel(K, y)
        results[key] = result

        print(f"  Pearson r: {result['pearson_mean']:.4f} ± {result['pearson_sd']:.4f}")
        print(f"  Spearman ρ: {result['spearman_mean']:.4f} ± {result['spearman_sd']:.4f}")
        print(f"  MAE: {result['mae_mean']:.2f} ± {result['mae_sd']:.2f} days")
        print(f"  RMSE: {result['rmse_mean']:.2f} ± {result['rmse_sd']:.2f} days")
        print(f"  λ: mode={result['lambda_mode']:.2e}, median={result['lambda_median']:.2e}")

    # === Grouped extrapolation ===
    print("\n" + "=" * 70)
    print("Grouped extrapolation evaluation...")
    print("=" * 70)

    grouped_results = {}
    for n_groups in [6, 8, 10]:
        print(f"\nLeave-one-group-out ({n_groups} groups)...")
        # Use the best single kernel (block kernel)
        grouped_results[f"{n_groups}_groups"] = grouped_cv(
            kernels["block"], y, data["kinship"], n_groups
        )
        print(f"  Pearson r: {grouped_results[f'{n_groups}_groups']['pearson_r']:.4f}")
        print(f"  MAE: {grouped_results[f'{n_groups}_groups']['mae']:.2f} days")
        print(f"  Group sizes: {grouped_results[f'{n_groups}_groups']['group_sizes']}")

    # === Save results ===
    elapsed = time.time() - start_time

    output = {
        "description": "Nested cross-validation of the genomic kernel comparison",
        "date": "2026-11-06",
        "cv_settings": {
            "n_repeats": N_REPEATS,
            "n_folds": N_FOLDS,
            "inner_folds": INNER_FOLDS,
            "lambda_grid": LAMBDAS.tolist(),
        },
        "results": {},
        "grouped_cv": grouped_results,
        "elapsed_seconds": elapsed,
    }

    # Remove the preds arrays (too large)
    for k, v in results.items():
        if isinstance(v, dict):
            output["results"][k] = {kk: vv for kk, vv in v.items() if kk != "preds"}
        else:
            output["results"][k] = v

    output_path = TABLES / "nested_cv_prediction.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("Result summary")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Pearson r':<15} {'MAE (days)':<15} {'λ (mode)':<12}")
    print("-" * 70)

    for name, result in results.items():
        if "pearson_mean" in result:
            r = f"{result['pearson_mean']:.4f} ± {result['pearson_sd']:.4f}"
            mae = f"{result['mae_mean']:.2f} ± {result['mae_sd']:.2f}"
            lam = f"{result.get('lambda_mode', 0):.2e}"
        else:
            r = f"{result.get('pearson_r', 0.0):.4f}"
            mae = f"{result['mae']:.2f}"
            lam = "N/A"
        print(f"{name:<25} {r:<15} {mae:<15} {lam:<12}")

    print(f"\nTotal elapsed: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print("\nNested cross-validation complete.")


if __name__ == "__main__":
    main()