"""Extended kernel experiments for the dormancy prediction analysis.

Read-only on the processed data; every output is written under
results/sensitivity/.

Experiments
-----------
1. Sanity reproduction of the published SNP / block-mean kernel results.
2. Nonlinear kernels: global RBF on block features, per-block RBF, degree-2
   polynomial, RBF on top-10 PCs, RBF on SSL sample embeddings.
3. Finer haplotype granularity: split each original block into <=10 / <=5 SNP
   sub-blocks and rebuild the mean-dosage kernel.
4. Phenotype transformations: log1p and Blom rank-normal targets, metrics
   reported back on the raw day scale.
5. Permutation significance of the block-kernel accuracy (500 draws).
6. Paired repeat-level comparisons between kernels.

Evaluation protocol is identical to the nested-CV prediction experiment:
10 repeats x 5-fold outer CV, 5-fold inner CV for hyper-parameter selection,
lambda grid logspace(-8, 2, 21). Kernels are built on the full 83-accession
feature matrix (same convention as the original experiment) and sliced per
fold; RBF bandwidth and lambda are selected jointly inside the inner loop."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from kernel_utils import (  # noqa: E402
    feature_kernel_centered_scaled,
    solve_kernel_ridge_with_intercept,
)

LAMBDAS = np.logspace(-8, 2, 21)
N_REPEATS = 10
N_FOLDS = 5
INNER_FOLDS = 5
GAMMAS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])


def load_data():
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)
    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()[keep]
    X_snp = (codes[keep, :] == minor[:, None]).astype(np.float32).T
    block_mean = np.asarray(
        np.load(PROCESSED / "block_features_83.npy", mmap_mode="r")[:, :, 0],
        dtype=np.float64,
    )
    graph_nodes = np.load(PROCESSED / "graph_embeddings_83.npy")
    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    sample_emb = np.load(PROCESSED / "sample_embeddings_1307.npy")[cols]
    return {
        "y": y,
        "codes": codes,
        "minor_all": snp["minor_code"].to_numpy(),
        "X_snp": X_snp,
        "block_mean": block_mean,
        "graph_nodes": graph_nodes,
        "sample_emb": sample_emb,
    }


def center_trace(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    tr = np.trace(Kc) / n
    return Kc / tr if tr > 1e-10 else Kc


def rbf_kernel(X, gamma):
    Xs = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    sq = np.maximum(
        (Xs ** 2).sum(1)[:, None] + (Xs ** 2).sum(1)[None, :] - 2 * Xs @ Xs.T, 0.0
    )
    return np.exp(-gamma * sq / Xs.shape[1])


def perblock_rbf(block_mean, gamma):
    """Average of per-block 1-D RBF kernels on mean dosage."""
    n, B = block_mean.shape
    K = np.zeros((n, n))
    for b in range(B):
        z = block_mean[:, b]
        sd = z.std()
        z = (z - z.mean()) / (sd if sd > 1e-8 else 1.0)
        d2 = (z[:, None] - z[None, :]) ** 2
        K += np.exp(-gamma * d2)
    return K / B


def graph_block_kernel(graph_nodes):
    n, B, _ = graph_nodes.shape
    K = np.zeros((n, n))
    for b in range(B):
        z = graph_nodes[:, b, :].astype(np.float64)
        Kb, _, _, _ = feature_kernel_centered_scaled(z)
        K += Kb
    return K / B


def panel_minor_codes():
    """Minor-allele definition used by the original block builder: argmin of
    allele counts over the full 1,307-accession panel; monomorphic -> 255."""
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    counts = snp[["n_A_all", "n_C_all", "n_G_all", "n_T_all"]].to_numpy()
    freq = np.where(
        counts > 0, counts / np.maximum(counts.sum(1, keepdims=True), 1), np.inf
    )
    minor = np.argmin(freq, axis=1).astype(np.uint8)
    minor[counts.max(1) == counts.sum(1)] = 255
    return minor


def finer_block_kernel(codes, minor_all, max_snps):
    """Split every original block into chunks of <= max_snps consecutive SNPs."""
    m = np.load(PROCESSED / "block_snp_map.npz")
    starts, ends = m["snp_start"], m["snp_end"]
    cols_out = []
    for s, e in zip(starts, ends):
        for a in range(s, e, max_snps):
            b = min(a + max_snps, e)
            sub = (codes[a:b, :] == minor_all[a:b, None]).astype(np.float64)
            cols_out.append(sub.mean(axis=0))
    X = np.stack(cols_out, axis=1)
    K, _, _, _ = feature_kernel_centered_scaled(X)
    return K, X.shape[1]


def metrics_from_preds(preds, y):
    r = np.array([stats.pearsonr(preds[i], y)[0] for i in range(len(preds))])
    rho = np.array([stats.spearmanr(preds[i], y).statistic for i in range(len(preds))])
    mae = np.array([np.mean(np.abs(preds[i] - y)) for i in range(len(preds))])
    rmse = np.array([np.sqrt(np.mean((preds[i] - y) ** 2)) for i in range(len(preds))])
    return {
        "pearson_mean": float(r.mean()),
        "pearson_sd": float(r.std()),
        "spearman_mean": float(rho.mean()),
        "spearman_sd": float(rho.std()),
        "mae_mean": float(mae.mean()),
        "mae_sd": float(mae.std()),
        "rmse_mean": float(rmse.mean()),
        "rmse_sd": float(rmse.std()),
        "r_per_repeat": r.tolist(),
    }


def nested_cv_single(K, y):
    """Original protocol: inner CV selects lambda only."""
    n = len(y)
    preds = np.zeros((N_REPEATS, n))
    lams = []
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        for tr, te in kf.split(K):
            Ktr, Kte = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
            ytr = y[tr]
            kfi = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=rep * 100)
            best = (np.inf, LAMBDAS[len(LAMBDAS) // 2])
            for lam in LAMBDAS:
                ip = np.zeros(len(ytr))
                for itr, iye in kfi.split(Ktr):
                    _, _, ip[iye] = solve_kernel_ridge_with_intercept(
                        Ktr[np.ix_(itr, itr)], ytr[itr], lam, Ktr[np.ix_(iye, itr)]
                    )
                mse = np.mean((ip - ytr) ** 2)
                if mse < best[0]:
                    best = (mse, lam)
            _, _, preds[rep, te] = solve_kernel_ridge_with_intercept(
                Ktr, ytr, best[1], Kte
            )
            lams.append(best[1])
    out = metrics_from_preds(preds, y)
    out["lambda_mode"] = float(pd.Series(lams).mode()[0])
    return out, preds


def nested_cv_multi(candidates, y):
    """candidates: list of (name, K). Inner CV selects (kernel, lambda) jointly."""
    n = len(y)
    preds = np.zeros((N_REPEATS, n))
    chosen = []
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        for tr, te in kf.split(y):
            ytr = y[tr]
            kfi = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=rep * 100)
            best = (np.inf, None, None)
            for name, K in candidates:
                Ktr = K[np.ix_(tr, tr)]
                for lam in LAMBDAS:
                    ip = np.zeros(len(ytr))
                    for itr, iye in kfi.split(Ktr):
                        _, _, ip[iye] = solve_kernel_ridge_with_intercept(
                            Ktr[np.ix_(itr, itr)], ytr[itr], lam, Ktr[np.ix_(iye, itr)]
                        )
                    mse = np.mean((ip - ytr) ** 2)
                    if mse < best[0]:
                        best = (mse, name, lam)
            name, lam = best[1], best[2]
            K = dict(candidates)[name]
            _, _, preds[rep, te] = solve_kernel_ridge_with_intercept(
                K[np.ix_(tr, tr)], ytr, lam, K[np.ix_(te, tr)]
            )
            chosen.append({"kernel": name, "lambda": float(lam)})
    out = metrics_from_preds(preds, y)
    counts = pd.Series([c["kernel"] for c in chosen]).value_counts()
    out["inner_selection_counts"] = {k: int(v) for k, v in counts.items()}
    return out, preds


def blom_transform(y):
    ranks = stats.rankdata(y, method="average")
    return stats.norm.ppf((ranks - 0.375) / (len(y) + 0.25))


def blom_inverse(z, y_ref):
    u = stats.norm.cdf(z)
    qs = (np.arange(len(y_ref)) + 0.5) / len(y_ref)
    return np.interp(u, qs, np.sort(y_ref))


def permutation_test(K, y, n_perm=500, lam=1e-8):
    """Fixed-lambda repeated 5-fold CV under phenotype permutation."""
    n = len(y)
    rng = np.random.default_rng(20261108)

    def cv_r(yy):
        rs = []
        for rep in range(N_REPEATS):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
            p = np.zeros(n)
            for tr, te in kf.split(K):
                _, _, p[te] = solve_kernel_ridge_with_intercept(
                    K[np.ix_(tr, tr)], yy[tr], lam, K[np.ix_(te, tr)]
                )
            rs.append(stats.pearsonr(p, yy)[0])
        return float(np.mean(rs))

    observed = cv_r(y)
    null = np.array([cv_r(rng.permutation(y)) for _ in range(n_perm)])
    pval = (1 + int((null >= observed).sum())) / (1 + n_perm)
    return {
        "observed_r": observed,
        "n_permutations": n_perm,
        "lambda_fixed": lam,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "null_max": float(null.max()),
        "p_permutation": float(pval),
    }


def main():
    t0 = time.time()
    data = load_data()
    y = data["y"]
    results = {"protocol": "10x5 outer, 5 inner, lambda logspace(-8,2,21)"}
    store = {"y": y}

    # ---- 1. sanity reproduction + base kernels ----
    K_snp, _, _, _ = feature_kernel_centered_scaled(data["X_snp"])
    K_block, _, _, _ = feature_kernel_centered_scaled(data["block_mean"])
    K_ssl = graph_block_kernel(data["graph_nodes"])
    print("[1] base kernels", flush=True)
    for name, K in [("snp", K_snp), ("block", K_block), ("ssl_graph", K_ssl)]:
        res, preds = nested_cv_single(K, y)
        results[name] = res
        store[name + "_preds"] = preds
        print(f"  {name}: r={res['pearson_mean']:.4f} mae={res['mae_mean']:.2f}", flush=True)

    # ---- 2. nonlinear kernels ----
    print("[2] nonlinear kernels", flush=True)
    rbf_block_cand = [(f"rbf_g{g:g}", center_trace(rbf_kernel(data["block_mean"], g))) for g in GAMMAS]
    res, preds = nested_cv_multi(rbf_block_cand, y)
    results["block_rbf_global"] = res
    store["block_rbf_global_preds"] = preds
    print(f"  block_rbf_global: r={res['pearson_mean']:.4f} sel={res['inner_selection_counts']}", flush=True)

    pb_cand = [(f"pbrbf_g{g:g}", center_trace(perblock_rbf(data["block_mean"], g))) for g in GAMMAS]
    res, preds = nested_cv_multi(pb_cand, y)
    results["block_rbf_perblock"] = res
    store["block_rbf_perblock_preds"] = preds
    print(f"  block_rbf_perblock: r={res['pearson_mean']:.4f} sel={res['inner_selection_counts']}", flush=True)

    Xs = (data["block_mean"] - data["block_mean"].mean(0)) / np.where(
        data["block_mean"].std(0) < 1e-8, 1.0, data["block_mean"].std(0)
    )
    K_poly = center_trace((1.0 + Xs @ Xs.T / Xs.shape[1]) ** 2)
    res, preds = nested_cv_single(K_poly, y)
    results["block_poly2"] = res
    store["block_poly2_preds"] = preds
    print(f"  block_poly2: r={res['pearson_mean']:.4f}", flush=True)

    Xc = data["X_snp"] - data["X_snp"].mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    PCs = U[:, :10] * S[:10]
    pc_cand = [(f"pcrbf_g{g:g}", center_trace(rbf_kernel(PCs, g))) for g in GAMMAS]
    res, preds = nested_cv_multi(pc_cand, y)
    results["pc_rbf"] = res
    store["pc_rbf_preds"] = preds
    print(f"  pc_rbf: r={res['pearson_mean']:.4f} sel={res['inner_selection_counts']}", flush=True)

    emb_cand = [(f"embrbf_g{g:g}", center_trace(rbf_kernel(data["sample_emb"], g))) for g in GAMMAS]
    res, preds = nested_cv_multi(emb_cand, y)
    results["ssl_sample_rbf"] = res
    store["ssl_sample_rbf_preds"] = preds
    print(f"  ssl_sample_rbf: r={res['pearson_mean']:.4f} sel={res['inner_selection_counts']}", flush=True)

    # ---- 3. finer granularity ----
    print("[3] finer blocks", flush=True)
    # sanity: reconstruct original 25-SNP block means with the panel minor codes
    m = np.load(PROCESSED / "block_snp_map.npz")
    minor_panel = panel_minor_codes()
    recon = np.stack(
        [
            (data["codes"][s:e, :] == minor_panel[s:e, None]).astype(np.float64).mean(0)
            for s, e in zip(m["snp_start"], m["snp_end"])
        ],
        axis=1,
    )
    sanity_err = float(np.abs(recon - data["block_mean"]).max())
    results["block_reconstruction_max_abs_err"] = sanity_err
    print(f"  reconstruction max err = {sanity_err:.2e}", flush=True)
    for chunk in (10, 5):
        K, nb = finer_block_kernel(data["codes"], minor_panel, chunk)
        res, preds = nested_cv_single(K, y)
        res["n_subblocks"] = int(nb)
        results[f"block_finer_{chunk}snp"] = res
        store[f"block_finer_{chunk}snp_preds"] = preds
        print(f"  finer {chunk}: B={nb} r={res['pearson_mean']:.4f}", flush=True)

    # ---- 4. phenotype transformations ----
    print("[4] phenotype transformations", flush=True)
    y_log = np.log1p(y)
    y_blom = blom_transform(y)
    for tname, yt, inv in [
        ("log1p", y_log, lambda z: np.expm1(z)),
        ("blom", y_blom, lambda z: blom_inverse(z, y)),
    ]:
        for kname, K in [("snp", K_snp), ("block", K_block)]:
            res, preds_t = nested_cv_single(K, yt)
            preds_day = np.stack([inv(preds_t[i]) for i in range(N_REPEATS)])
            day = metrics_from_preds(preds_day, y)
            results[f"{kname}_y_{tname}"] = {
                "model_scale": res,
                "day_scale": {k: v for k, v in day.items() if k != "r_per_repeat"},
            }
            store[f"{kname}_y_{tname}_preds_day"] = preds_day
            print(
                f"  {kname}+{tname}: r_model={res['pearson_mean']:.4f} "
                f"r_day={day['pearson_mean']:.4f} mae_day={day['mae_mean']:.2f}",
                flush=True,
            )

    # ---- 5. permutation significance ----
    print("[5] permutation test (500 draws)", flush=True)
    results["block_permutation"] = permutation_test(K_block, y, n_perm=500)
    print(f"  p={results['block_permutation']['p_permutation']:.4f}", flush=True)

    # ---- 6. paired comparisons ----
    print("[6] paired comparisons", flush=True)
    pairs = {}
    base_r = np.array(results["block"]["r_per_repeat"])
    for name in ["snp", "ssl_graph", "block_rbf_global", "block_rbf_perblock",
                 "block_poly2", "pc_rbf", "ssl_sample_rbf", "block_finer_10snp",
                 "block_finer_5snp"]:
        if name not in results:
            continue
        other = np.array(results[name]["r_per_repeat"])
        t = stats.ttest_rel(base_r, other)
        w = stats.wilcoxon(base_r, other)
        pairs[f"block_vs_{name}"] = {
            "mean_diff": float((base_r - other).mean()),
            "ttest_p": float(t.pvalue),
            "wilcoxon_p": float(w.pvalue),
        }
    results["paired_comparisons"] = pairs

    results["elapsed_seconds"] = time.time() - t0
    (OUT / "kernel_experiments.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(OUT / "kernel_experiments_preds.npz", **store)
    print(f"\nDone in {results['elapsed_seconds']:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()