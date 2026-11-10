"""Second public trait: does the haplotype-block design rule replicate on
flowering time?

The positive claims rest on a single phenotype (seed dormancy, DSDS50, 83
accessions). AtPolyDB's published table contains flowering time at 10/16/22
degrees C (traits 5/6/7) measured on 194 / 193 / 193 accessions that are all
present in the 1,307-accession genotype panel used here, and 83 of them are
the dormancy accessions. Flowering time is genetically distinct from
dormancy, far better powered, and has the same public genotype source, so it
is a genuine out-of-trait replication of the design rule rather than a
re-analysis.

Protocol: identical to the dormancy experiments (10 repeats x 5 outer folds,
5 inner folds, lambda grid 1e-8..1e2), but every quantity is computed *within*
the flowering-time cohort: block-mean features are taken from the published
1,307-accession block feature tensor at the cohort columns, SNP indicators use
the panel minor-allele definition with MAF > 0.05 computed inside the cohort,
and bandwidth selection is local to the cohort.

Reported:
  * block-mean vs SNP vs local RBF (selected and fixed gamma=0.1) vs global
    RBF vs poly2 vs PC-RBF vs SSL sample-embedding RBF,
  * granularity sweep in both directions (5-/10-SNP sub-blocks, 2x/4x merged
    windows),
  * paired naive and Nadeau-Bengio corrected tests against the block kernel,
  * accession-bootstrap CIs conditional on the fitted OOF predictions,
  * permutation significance of the best kernel (200 draws),
  * the 111-accession subset that excludes all dormancy-panel accessions, to
    show the replication is not carried by the original 83.

Result: results/sensitivity/second_trait.json and
results/sensitivity/second_trait_preds.npz"""

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
RAW_ATDB = ROOT / "data" / "raw" / "atpolydb"
OUT = ROOT / "results" / "sensitivity"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import kernel_experiments as ke  # noqa: E402
from kernel_utils import (  # noqa: E402
    feature_kernel_centered_scaled,
    solve_kernel_ridge_with_intercept,
)

RHO = 0.25
LAMBDAS = ke.LAMBDAS
GAMMAS = ke.GAMMAS
GAMMA_RULE = 0.1  # bandwidth preferred by the fixed-gamma ablation
TRAITS = {"FT10": "5_FT10", "FT16": "6_FT16", "FT22": "7_FT22"}
N_PERM_BEST = 200
PERM_SEED = 20261109
MAF_MIN = 0.05


def header_ids():
    with open(RAW_ATDB / "call_method_75" / "call_method_75_TAIR9.csv", "rb") as h:
        next(h)
        header = h.readline().decode().rstrip("\n").split(",")
    return np.array([int(x) for x in header[2:]])


def snp_design(codes, cols, chrom):
    """Minor-allele coding, MAF filter and SNP indicator matrix for a sample.

    Mirrors scripts/01_build_genotype_matrix.py: MAF = min(major frequency,
    1 - major frequency), the coded allele is the least frequent *observed*
    allele, monomorphic SNPs are dropped, and SNPs with MAF <= 0.05 are dropped.
    Both quantities are re-derived inside the cohort from genotypes only.
    """
    sub = codes[:, cols]
    counts = np.stack([(sub == k).sum(axis=1) for k in range(4)], axis=1)
    called = counts.sum(axis=1)
    freq = counts / np.maximum(called, 1)[:, None]
    max_freq = freq.max(axis=1)
    maf = np.minimum(max_freq, 1.0 - max_freq)
    minor = np.argmin(np.where(counts > 0, freq, np.inf), axis=1).astype(np.uint8)
    minor[(counts.max(axis=1) == called) | (called == 0)] = 255
    valid = (chrom > 0) & (called > 0.8 * len(cols))
    sel = np.flatnonzero(valid & (maf > MAF_MIN) & (minor < 4))
    if len(sel) < 1000:
        raise RuntimeError(f"SNP selection collapsed to {len(sel)} columns")
    X_snp = (sub[sel, :] == minor[sel, None]).astype(np.float64).T
    return X_snp, minor, maf, sel


def snp_definition_selfcheck():
    """Run the cohort code path on the 83 dormancy accessions and compare with
    the published matrices (genotypes_83.npy / data['X_snp']). A match proves
    the flowering-time SNP design is the same design, only re-fitted inside a
    different sample."""
    codes_all = np.load(PROCESSED / "genotypes_all_1307.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    X_cohort, minor, maf, sel = snp_design(codes_all, cols, snp["chrom"].to_numpy())
    codes83 = np.load(PROCESSED / "genotypes_83.npy")
    keep = np.flatnonzero((snp["maf_83"].to_numpy() > MAF_MIN)
                          & (snp["minor_code"].to_numpy() < 4))
    X_pub = (codes83[keep, :] == snp["minor_code"].to_numpy()[keep, None]).astype(np.float64).T
    return {
        "n_snps_cohort_path": int(X_cohort.shape[1]),
        "n_snps_published": int(X_pub.shape[1]),
        "minor_code_identical": bool(np.array_equal(minor, snp["minor_code"].to_numpy())),
        "maf_max_abs_diff": float(np.nanmax(np.abs(maf - snp["maf_83"].to_numpy()))),
        "design_matrix_identical": bool(
            X_cohort.shape == X_pub.shape and np.array_equal(X_cohort, X_pub)
        ),
    }


def cohort_data(trait_col):
    """Block-mean features, SNP indicators and phenotype for one trait cohort.

    Block features are read from the published 1,307-accession tensor at the
    cohort columns, so the feature definition (mean dosage plus sign-stable
    PCs per block) is identical to the dormancy experiments; only the sample
    changes. Genotypes are complete in the panel, so every SNP is fully called.
    """
    ids = header_ids()
    pos = {int(e): i for i, e in enumerate(ids)}
    raw = pd.read_csv(RAW_ATDB / "phenotype_published_raw.tsv", sep="\t",
                      encoding="latin-1").rename(columns={"ecotype id": "ecotype_id"})
    v = pd.to_numeric(raw[trait_col], errors="coerce")
    keep = raw.loc[v.notna(), ["ecotype_id"]].assign(y=v[v.notna()])
    keep = keep[keep.ecotype_id.isin(pos)]
    cols = np.array([pos[int(e)] for e in keep.ecotype_id.to_numpy()], dtype=np.int32)
    y = keep.y.to_numpy(float)

    bf = np.load(PROCESSED / "block_features.npy", mmap_mode="r")
    block_mean = np.asarray(bf[cols][:, :, 0], dtype=np.float64)

    codes_all = np.load(PROCESSED / "genotypes_all_1307.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    X_snp, minor, maf, sel = snp_design(codes_all, cols, snp["chrom"].to_numpy())

    dorm_cols = set(np.load(PROCESSED / "dormancy_cols_in_1307.npy").tolist())
    return {
        "y": y,
        "block_mean": block_mean,
        "X_snp": X_snp,
        "n_snps_used": int(len(sel)),
        "sel_snps": sel,
        "minor_cohort": minor,
        "cohort_cols": cols,
        "in_dormancy_panel": np.array([c in dorm_cols for c in cols.tolist()]),
        "maf": float(maf[sel].mean()),
    }


def merge_windows(block_mean, chrom, size):
    merged = []
    for c in np.unique(chrom):
        idx = np.flatnonzero(chrom == c)
        for start in range(0, len(idx), size):
            merged.append(block_mean[:, idx[start:start + size]].mean(axis=1))
    return np.column_stack(merged)


def sub_block_windows(codes, minor, snp_start, snp_end, chunk):
    cols_out = []
    for s, e in zip(snp_start, snp_end):
        for a in range(s, e, chunk):
            b = min(a + chunk, e)
            cols_out.append((codes[a:b, :] == minor[a:b, None]).astype(np.float64).mean(axis=0))
    return np.stack(cols_out, axis=1)


def nested_cv_single(K, y):
    n = len(y)
    preds = np.zeros((ke.N_REPEATS, n))
    lams = []
    for rep in range(ke.N_REPEATS):
        for tr, te in KFold(n_splits=ke.N_FOLDS, shuffle=True, random_state=rep).split(K):
            Ktr = K[np.ix_(tr, tr)]
            kfi = KFold(n_splits=ke.INNER_FOLDS, shuffle=True, random_state=rep * 100)
            best = (np.inf, LAMBDAS[len(LAMBDAS) // 2])
            for lam in LAMBDAS:
                ip = np.zeros(len(y[tr]))
                for itr, iye in kfi.split(Ktr):
                    _, _, ip[iye] = solve_kernel_ridge_with_intercept(
                        Ktr[np.ix_(itr, itr)], y[tr][itr], lam, Ktr[np.ix_(iye, itr)]
                    )
                mse = float(np.mean((ip - y[tr]) ** 2))
                if mse < best[0]:
                    best = (mse, lam)
            _, _, preds[rep, te] = solve_kernel_ridge_with_intercept(
                Ktr, y[tr], best[1], K[np.ix_(te, tr)]
            )
            lams.append(best[1])
    out = ke.metrics_from_preds(preds, y)
    out["lambda_mode"] = float(pd.Series(lams).mode()[0])
    return out, preds


def nested_cv_multi(candidates, y):
    n = len(y)
    preds = np.zeros((ke.N_REPEATS, n))
    chosen = []
    cand = dict(candidates)
    for rep in range(ke.N_REPEATS):
        for tr, te in KFold(n_splits=ke.N_FOLDS, shuffle=True, random_state=rep).split(y):
            ytr = y[tr]
            kfi = KFold(n_splits=ke.INNER_FOLDS, shuffle=True, random_state=rep * 100)
            best = (np.inf, None, None)
            for name, K in candidates:
                Ktr = K[np.ix_(tr, tr)]
                for lam in LAMBDAS:
                    ip = np.zeros(len(ytr))
                    for itr, iye in kfi.split(Ktr):
                        _, _, ip[iye] = solve_kernel_ridge_with_intercept(
                            Ktr[np.ix_(itr, itr)], ytr[itr], lam, Ktr[np.ix_(iye, itr)]
                        )
                    mse = float(np.mean((ip - ytr) ** 2))
                    if np.isfinite(mse) and mse < best[0]:
                        best = (mse, name, lam)
            name, lam = best[1], best[2]
            if name is None:
                raise RuntimeError("inner CV rejected every candidate (non-finite kernel)")
            _, _, preds[rep, te] = solve_kernel_ridge_with_intercept(
                cand[name][np.ix_(tr, tr)], ytr, lam, cand[name][np.ix_(te, tr)]
            )
            chosen.append(name)
    out = ke.metrics_from_preds(preds, y)
    out["inner_selection_counts"] = {k: int(v) for k, v in pd.Series(chosen).value_counts().items()}
    return out, preds


def paired(diff):
    d = np.asarray(diff, float)
    n = len(d)
    m, s2 = float(d.mean()), float(d.var(ddof=1))
    if s2 < 1e-24:
        return {"mean_diff": m, "p_naive": 1.0, "p_nadeau_bengio": 1.0}
    return {
        "mean_diff": m,
        "cohen_dz": float(m / np.sqrt(s2)),
        "p_naive": float(2 * stats.t.sf(abs(m / np.sqrt(s2 / n)), n - 1)),
        "p_nadeau_bengio": float(2 * stats.t.sf(abs(m / np.sqrt(s2 * (1 / n + RHO))), n - 1)),
    }


def boot_ci(preds, y, n_boot=5000, seed=1):
    p = np.asarray(preds, float).mean(axis=0)
    n = len(y)
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rs[b] = 0.0 if p[idx].std() < 1e-12 else stats.pearsonr(p[idx], y[idx])[0]
    return {"r_mean": float(stats.pearsonr(p, y)[0]),
            "ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]}


def permutation_r(K, y, n_perm, lam=1e-8, seed=PERM_SEED):
    n = len(y)
    rng = np.random.default_rng(seed)

    def one(yy):
        rs = []
        for rep in range(ke.N_REPEATS):
            p = np.zeros(n)
            for tr, te in KFold(n_splits=ke.N_FOLDS, shuffle=True, random_state=rep).split(K):
                _, _, p[te] = solve_kernel_ridge_with_intercept(
                    K[np.ix_(tr, tr)], yy[tr], lam, K[np.ix_(te, tr)]
                )
            rs.append(stats.pearsonr(p, yy)[0])
        return float(np.mean(rs))

    obs = one(y)
    null = np.array([one(rng.permutation(y)) for _ in range(n_perm)])
    return {
        "observed_r": obs, "n_perm": n_perm, "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)), "null_max": float(null.max()),
        "p_permutation": float((1 + int((null >= obs).sum())) / (1 + n_perm)),
    }


def evaluate_trait(name, col, blocks_df, snp_map, codes, minor, sample_emb_all, full=True):
    d = cohort_data(col)
    y, block_mean, X_snp = d["y"], d["block_mean"], d["X_snp"]
    n = len(y)
    print(f"\n=== {name}: n={n}, SNPs={X_snp.shape[1]} ===", flush=True)

    K_snp, _, _, _ = feature_kernel_centered_scaled(X_snp)
    K_block, _, _, _ = feature_kernel_centered_scaled(block_mean)
    K_local_fixed = ke.center_trace(ke.perblock_rbf(block_mean, GAMMA_RULE))
    local_cand = [(f"g{g:g}", ke.center_trace(ke.perblock_rbf(block_mean, g))) for g in GAMMAS]
    K_glob = ke.center_trace(ke.rbf_kernel(block_mean, 0.5))
    Xs = (block_mean - block_mean.mean(0)) / np.where(block_mean.std(0) < 1e-8, 1.0, block_mean.std(0))
    K_poly = ke.center_trace((1.0 + Xs @ Xs.T / Xs.shape[1]) ** 2)
    Xc = X_snp - X_snp.mean(0)
    # identical 10 leading principal-component scores as an SVD of Xc, from the
    # n x n Gram matrix (Xc has ~200,000 columns, so this is the cheap route)
    gram = Xc @ Xc.T
    ev, U = np.linalg.eigh(gram)
    order = np.argsort(ev)[::-1][:10]
    PCs = U[:, order] * np.sqrt(np.maximum(ev[order], 0.0))
    pc_cand = [(f"g{g:g}", ke.center_trace(ke.rbf_kernel(PCs, g))) for g in GAMMAS]

    out = {}
    preds_store = {}
    for label, K in [("snp", K_snp), ("block", K_block), ("local_rbf_fixed_0.1", K_local_fixed)]:
        res, preds = nested_cv_single(K, y)
        out[label] = res
        preds_store[label] = preds
        print(f"  {label:20s} r={res['pearson_mean']:.4f} mae={res['mae_mean']:.2f}", flush=True)
    for label, cand in [("local_rbf_selected", local_cand), ("pc_rbf_selected", pc_cand)]:
        res, preds = nested_cv_multi(cand, y)
        out[label] = res
        preds_store[label] = preds
        print(f"  {label:20s} r={res['pearson_mean']:.4f} sel={res['inner_selection_counts']}", flush=True)
    if full:
        for label, K in [("global_rbf", K_glob), ("poly2", K_poly)]:
            res, preds = nested_cv_single(K, y)
            out[label] = res
            preds_store[label] = preds
            print(f"  {label:20s} r={res['pearson_mean']:.4f}", flush=True)

        # granularity sweep
        chrom = blocks_df["chrom"].to_numpy()
        for size, lbl in [(2, "coarser_2x"), (4, "coarser_4x")]:
            X = merge_windows(block_mean, chrom, size)
            K, _, _, _ = feature_kernel_centered_scaled(X)
            res, preds = nested_cv_single(K, y)
            res["n_windows"] = int(X.shape[1])
            out[lbl] = res
            preds_store[lbl] = preds
            print(f"  {lbl:20s} r={res['pearson_mean']:.4f} (B={X.shape[1]})", flush=True)
        for chunk, lbl in [(10, "finer_10snp"), (5, "finer_5snp")]:
            X = sub_block_windows(codes[:, d["cohort_cols"]], d["minor_cohort"],
                                  snp_map["snp_start"], snp_map["snp_end"], chunk)
            K, _, _, _ = feature_kernel_centered_scaled(X)
            res, preds = nested_cv_single(K, y)
            res["n_windows"] = int(X.shape[1])
            out[lbl] = res
            preds_store[lbl] = preds
            print(f"  {lbl:20s} r={res['pearson_mean']:.4f} (B={X.shape[1]})", flush=True)

        # SSL sample-embedding RBF (embeddings are panel-wide, transductive)
        emb = sample_emb_all[d["cohort_cols"]]
        emb_cand = [(f"g{g:g}", ke.center_trace(ke.rbf_kernel(emb, g))) for g in GAMMAS]
        res, preds = nested_cv_multi(emb_cand, y)
        out["ssl_sample_rbf"] = res
        preds_store["ssl_sample_rbf"] = preds
        print(f"  {'ssl_sample_rbf':20s} r={res['pearson_mean']:.4f}", flush=True)

    # paired tests against the block-mean kernel
    r_block = np.array(out["block"]["r_per_repeat"])
    tests = {k: paired(np.array(v["r_per_repeat"]) - r_block)
             for k, v in out.items() if k != "block"}
    best_key = max(out, key=lambda k: out[k]["pearson_mean"])
    tests["local_rbf_vs_snp"] = paired(
        np.array(out["local_rbf_fixed_0.1"]["r_per_repeat"])
        - np.array(out["snp"]["r_per_repeat"])
    )

    trait_res = {
        "n_accessions": int(n),
        "n_in_dormancy_panel": int(d["in_dormancy_panel"].sum()),
        "phenotype_mean_days": float(y.mean()),
        "phenotype_sd_days": float(y.std(ddof=1)),
        "n_snps_used": int(d["n_snps_used"]),
        "mean_cohort_maf_used": float(d["maf"]),
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "r_per_repeat"} for k, v in out.items()},
        "paired_vs_block_mean": tests,
        "bootstrap_ci": {k: boot_ci(preds_store[k], y) for k in ["snp", "block", "local_rbf_fixed_0.1"]},
        "best_model": best_key,
    }
    if full:
        trait_res["permutation_best"] = permutation_r(
            ke.center_trace(ke.perblock_rbf(block_mean, GAMMA_RULE)), y, N_PERM_BEST
        )
        print(f"  permutation (local RBF): p={trait_res['permutation_best']['p_permutation']:.4f} "
              f"null_max={trait_res['permutation_best']['null_max']:.3f}", flush=True)

    # dormancy-panel-excluded subset for the three core kernels
    mask = ~d["in_dormancy_panel"]
    if mask.sum() >= 60:
        y_sub = y[mask]
        sub_full = {}
        for label, K in [("snp", K_snp), ("block", K_block),
                         ("local_rbf_fixed_0.1", K_local_fixed)]:
            Ks = K[np.ix_(mask, mask)]
            res, _ = nested_cv_single(Ks, y_sub)
            sub_full[label] = res
        r_block_sub = np.array(sub_full["block"]["r_per_repeat"])
        trait_res["subset_without_dormancy_panel"] = {
            "n": int(mask.sum()),
            "models": {k: {kk: vv for kk, vv in v.items() if kk != "r_per_repeat"}
                       for k, v in sub_full.items()},
            "paired_vs_block_mean": {
                k: paired(np.array(v["r_per_repeat"]) - r_block_sub)
                for k, v in sub_full.items() if k != "block"
            },
        }
        print(f"  subset without 83: n={mask.sum()} "
              f"block r={sub_full['block']['pearson_mean']:.4f} "
              f"snp r={sub_full['snp']['pearson_mean']:.4f} "
              f"local r={sub_full['local_rbf_fixed_0.1']['pearson_mean']:.4f}", flush=True)
    return trait_res, preds_store, d["cohort_cols"]


def main():
    t0 = time.time()
    blocks_df = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    snp_map = np.load(PROCESSED / "block_snp_map.npz")
    codes = np.load(PROCESSED / "genotypes_all_1307.npy")
    minor = pd.read_csv(PROCESSED / "snp_metadata.csv")["minor_code"].to_numpy()
    sample_emb_all = np.load(PROCESSED / "sample_embeddings_1307.npy")

    results = {
        "protocol": "identical to dormancy experiments: 10x5 nested CV, 5 inner, lambda 1e-8..1e2; all quantities computed within cohort",
        "gamma_rule": GAMMA_RULE,
        "snp_design_selfcheck": snp_definition_selfcheck(),
        "traits": {},
    }
    print("SNP design self-check: "
          + json.dumps(results["snp_design_selfcheck"]), flush=True)
    all_preds = {}
    for i, (name, col) in enumerate(TRAITS.items()):
        tr_res, preds, cols = evaluate_trait(
            name, col, blocks_df, snp_map, codes, minor, sample_emb_all, full=(i == 0)
        )
        results["traits"][name] = tr_res
        for k, v in preds.items():
            all_preds[f"{name}_{k}"] = v

    # cross-trait consistency of the ranking block >= snp and local >= block
    ranks = {}
    for name, tr in results["traits"].items():
        ranks[name] = {
            "block_minus_snp": tr["models"]["block"]["pearson_mean"] - tr["models"]["snp"]["pearson_mean"],
            "local_minus_block": tr["models"]["local_rbf_fixed_0.1"]["pearson_mean"] - tr["models"]["block"]["pearson_mean"],
        }
    results["cross_trait_deltas"] = ranks

    results["elapsed_seconds"] = time.time() - t0
    (OUT / "second_trait.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(OUT / "second_trait_preds.npz", **all_preds)
    print(f"\nDone in {results['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()