"""Inference reanalysis of the repeat-level kernel comparisons.

No new cross-validation runs except the two coarser-granularity kernels whose
per-repeat scores were not saved by the aggregation-sensitivity experiment;
those are rebuilt with the identical protocol so that paired comparisons
against the block-mean kernel use the same folds.

Contents
--------
* CV-dependence-aware paired inference for every repeat-level comparison:
    naive paired t, Nadeau-Bengio corrected resampled t (repeat-level
    adaptation, variance factor 1/R + n_test/n_train with R=10 and
    n_test/n_train=0.25), Bayesian correlated t (Benavoli et al. 2017,
    rho=0.25) reported as posterior P(mean difference > 0), Cohen's dz, and
    Holm adjustment across the nine primary kernel comparisons (naive and
    corrected families adjusted separately).
* Paired block-mean vs PC-regression comparison (per-repeat r from the
    prediction figure data; identical outer folds).
* Fisher-z confidence intervals for all fixed-prediction correlations
    (external cohorts, grouped CV, repeatability) and accession-bootstrap
    percentile CIs for the cross-validated kernels (conditional on the fitted
    out-of-fold predictions).

Outputs: results/sensitivity/inference_reanalysis.json and
results/sensitivity/coarser_granularity_preds.npz"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "sensitivity"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import kernel_experiments as ke  # noqa: E402
from kernel_utils import feature_kernel_centered_scaled  # noqa: E402

R_REPEATS = 10
RHO = 0.25  # n_test / n_train for 5-fold CV
N_BOOT = 10000
BOOT_SEED = 20261109


def repeat_level_tests(diff):
    """diff: per-repeat paired differences (model A - model B)."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    m = float(d.mean())
    s2 = float(d.var(ddof=1))
    df = n - 1
    se_naive = np.sqrt(s2 / n)
    t_naive = m / se_naive
    p_naive = float(2 * stats.t.sf(abs(t_naive), df))

    # Nadeau-Bengio corrected resampled t (repeat-level adaptation)
    nb_var = s2 * (1.0 / n + RHO)
    t_nb = m / np.sqrt(nb_var)
    p_nb = float(2 * stats.t.sf(abs(t_nb), df))

    # Bayesian correlated t (Benavoli et al. 2017)
    bayes_scale = np.sqrt(s2 * (1.0 / n + RHO / (1.0 - RHO)))
    prob_pos = float(stats.t.cdf(m / bayes_scale, df))

    return {
        "mean_diff": m,
        "sd_diff": float(np.sqrt(s2)),
        "cohen_dz": float(m / np.sqrt(s2)),
        "t_naive": float(t_naive),
        "p_naive": p_naive,
        "t_nadeau_bengio": float(t_nb),
        "p_nadeau_bengio": p_nb,
        "bayes_posterior_prob_positive": prob_pos,
    }


def holm_adjust(pvals):
    """Holm step-down; returns dict of adjusted p-values (same keys)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adj[k] = running
    return adj


def fisher_z(r, n):
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.959963985 * se), np.tanh(z + 1.959963985 * se)
    p = float(2 * stats.norm.sf(abs(z) / se))
    return {
        "r": float(r),
        "n": int(n),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_fisher_z": p,
        "includes_zero": bool(lo <= 0 <= hi),
    }


def bootstrap_ci(preds, y):
    """Accession bootstrap, conditional on the fitted OOF predictions."""
    p_mean = np.asarray(preds, dtype=float).mean(axis=0)
    n = len(y)
    rng = np.random.default_rng(BOOT_SEED)
    rs = np.empty(N_BOOT)
    maes = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        pb, yb = p_mean[idx], y[idx]
        if pb.std() < 1e-12 or yb.std() < 1e-12:
            rs[b] = 0.0
        else:
            rs[b] = stats.pearsonr(pb, yb)[0]
        maes[b] = np.mean(np.abs(pb - yb))
    return {
        "r_mean_oof_avg": float(stats.pearsonr(p_mean, y)[0]),
        "r_mean_of_repeats": float(np.mean([stats.pearsonr(p, y)[0] for p in preds])),
        "r_ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
        "mae_mean": float(np.mean(np.abs(p_mean - y))),
        "mae_ci95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "n_bootstrap": N_BOOT,
    }


def r_per_repeat(preds, y):
    return np.array([stats.pearsonr(preds[i], y)[0] for i in range(len(preds))])


def merge_blocks(mean_dosage, chrom, size):
    merged = []
    for c in np.unique(chrom):
        idx = np.flatnonzero(chrom == c)
        for start in range(0, len(idx), size):
            merged.append(mean_dosage[:, idx[start:start + size]].mean(axis=1))
    return np.column_stack(merged)


def main():
    t0 = time.time()
    res = {
        "description": (
            "Inference reanalysis. Repeat-level paired tests with "
            "Nadeau-Bengio correction (factor 1/10+0.25) and Bayesian "
            "correlated t (rho=0.25); Holm over the nine primary kernel "
            "comparisons; Fisher-z CIs for fixed-prediction correlations; "
            "accession-bootstrap CIs conditional on fitted OOF predictions."
        ),
        "rho": RHO,
        "n_repeats": R_REPEATS,
    }

    published = json.loads((OUT / "kernel_experiments.json").read_text(encoding="utf-8"))
    store = np.load(OUT / "kernel_experiments_preds.npz")
    y = store["y"]
    old = np.load(ROOT / "results" / "tables" / "prediction_figure_data.npz")

    # ---- coarser granularity: rebuild + rerun (per-repeat r not saved) ----
    print("[coarser] rebuilding merged-block kernels", flush=True)
    data = ke.load_data()
    blocks = pd.read_csv(ROOT / "data" / "processed" / "haplotype_blocks.csv")
    chrom = blocks["chrom"].to_numpy()
    coarse_store = {}
    coarse_r = {}
    for size, label in [(2, "block_coarser_2x"), (4, "block_coarser_4x")]:
        feats = merge_blocks(data["block_mean"], chrom, size)
        K, _, _, _ = feature_kernel_centered_scaled(feats)
        out, preds = ke.nested_cv_single(K, y)
        out["n_features"] = int(feats.shape[1])
        res[label] = out
        coarse_store[label + "_preds"] = preds
        coarse_r[label] = np.array(out["r_per_repeat"])
        print(f"  {label}: B={feats.shape[1]} r={out['pearson_mean']:.4f}", flush=True)
    np.savez_compressed(OUT / "coarser_granularity_preds.npz", y=y, **coarse_store)

    # ---- primary nine comparisons, three variance treatments ----
    print("[paired] primary comparisons", flush=True)
    base_r = r_per_repeat(store["block_preds"], y)
    primary_names = [
        "snp", "ssl_graph", "block_rbf_global", "block_rbf_perblock",
        "block_poly2", "pc_rbf", "ssl_sample_rbf",
        "block_finer_10snp", "block_finer_5snp",
    ]
    tests = {}
    for name in primary_names:
        other = r_per_repeat(store[name + "_preds"], y)
        tests[f"block_vs_{name}"] = repeat_level_tests(base_r - other)
    adj_naive = holm_adjust({k: v["p_naive"] for k, v in tests.items()})
    adj_nb = holm_adjust({k: v["p_nadeau_bengio"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_naive_holm"] = adj_naive[k]
        tests[k]["p_nadeau_bengio_holm"] = adj_nb[k]
    res["primary_comparisons"] = tests

    # ---- extended exploratory comparisons ----
    print("[extended] comparisons", flush=True)
    ext = {}
    ext["block_vs_pc_regression"] = repeat_level_tests(base_r - old["pc_r"])
    for label in coarse_r:
        ext[f"block_vs_{label}"] = repeat_level_tests(base_r - coarse_r[label])
    for tname in ["log1p", "blom"]:
        for kname in ["snp", "block"]:
            key = f"{kname}_y_{tname}_preds_day"
            rt = r_per_repeat(store[key], y)
            ref = r_per_repeat(store[kname + "_preds"], y)
            ext[f"{kname}_{tname}_vs_{kname}_raw"] = repeat_level_tests(rt - ref)
    # local RBF vs SNP (headline model vs standard baseline)
    ext["local_rbf_vs_snp"] = repeat_level_tests(
        r_per_repeat(store["block_rbf_perblock_preds"], y)
        - r_per_repeat(store["snp_preds"], y)
    )
    res["extended_comparisons_exploratory"] = ext

    # ---- bootstrap CIs for CV kernels ----
    print("[bootstrap] CIs", flush=True)
    boot = {}
    for name in ["snp", "block", "ssl_graph", "block_rbf_global",
                 "block_rbf_perblock", "block_poly2", "pc_rbf",
                 "ssl_sample_rbf", "block_finer_10snp", "block_finer_5snp"]:
        boot[name] = bootstrap_ci(store[name + "_preds"], y)
    for label in coarse_r:
        boot[label] = bootstrap_ci(coarse_store[label + "_preds"], y)
    res["bootstrap_ci_conditional_on_oof"] = boot

    # ---- Fisher-z CIs for fixed-prediction correlations ----
    print("[Fisher] CIs", flush=True)
    dryad = json.loads(
        (ROOT / "results" / "tables" / "external_projection.json").read_text(encoding="utf-8")
    )
    formal = json.loads(
        (ROOT / "results" / "tables" / "nested_cv_prediction.json").read_text(encoding="utf-8")
    )
    extval = json.loads(
        (ROOT / "results" / "tables" / "external_validation.json").read_text(encoding="utf-8")
    )
    fisher = {}
    ind = dryad["results"]["independent_transductive"]
    fisher["dryad_independent_block"] = fisher_z(ind["block"]["pearson_r"], ind["block"]["n"])
    fisher["dryad_independent_snp"] = fisher_z(ind["snp"]["pearson_r"], ind["snp"]["n"])
    same = dryad["results"]["same83_new_environment"]
    fisher["dryad_same83_block"] = fisher_z(same["block"]["pearson_r"], same["block"]["n"])
    rep = dryad["measurement_repeatability_same38"]
    fisher["repeatability_same38"] = fisher_z(rep["pearson_r"], rep["n"])
    for gk, gv in formal["grouped_cv"].items():
        fisher[f"grouped_{gk}"] = fisher_z(gv["pearson_r"], 83)
    for mk, mv in extval["models"].items():
        fisher[f"old_external29_{mk}"] = fisher_z(mv["pearson"], mv["n"])
    res["fisher_z_fixed_predictions"] = fisher

    res["elapsed_seconds"] = time.time() - t0
    (OUT / "inference_reanalysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- console summary ----
    print("\n=== primary comparisons (block - other) ===")
    for k, v in tests.items():
        print(
            f"{k:34s} d={v['mean_diff']:+.4f} dz={v['cohen_dz']:+.2f} "
            f"p_naive={v['p_naive']:.2e} holm={v['p_naive_holm']:.2e} "
            f"p_NB={v['p_nadeau_bengio']:.3f} holmNB={v['p_nadeau_bengio_holm']:.3f} "
            f"Pbayes+={v['bayes_posterior_prob_positive']:.3f}"
        )
    print("\n=== extended ===")
    for k, v in ext.items():
        print(
            f"{k:34s} d={v['mean_diff']:+.4f} p_naive={v['p_naive']:.2e} "
            f"p_NB={v['p_nadeau_bengio']:.3f} Pbayes+={v['bayes_posterior_prob_positive']:.3f}"
        )
    print("\n=== Fisher CIs ===")
    for k, v in fisher.items():
        print(f"{k:34s} r={v['r']:+.3f} CI=[{v['ci95_low']:+.3f},{v['ci95_high']:+.3f}] incl0={v['includes_zero']}")
    print(f"\nDone in {res['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()