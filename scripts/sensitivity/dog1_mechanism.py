"""DOG1 mechanism checks for the block architecture.

DOG1 (AT5G45830) is the best-known seed-dormancy locus in Arabidopsis, so two
questions arise about a haplotype-block model: whether the block partition
happens to cut the functional DOG1 variation in half, and how much of the
prediction accuracy is actually carried by that one region. The published
association scan gives no special status to DOG1's block (raw-block
p_imhof = 0.307, percentile 32.3), which invites exactly this question.

Analyses
--------
1. Containment. Gene body, promoter windows (2 kb / 5 kb upstream and
   downstream, both strands reported) and the whole locus are mapped onto the
   block partition; the number of analysis SNPs per window and the blocks they
   fall in are listed, so "split by block boundaries" can be checked rather
   than assumed.
2. Necessity. The dormancy kernels are re-evaluated with (a) the DOG1 block
   column removed from the block-mean and local-RBF kernels, (b) the DOG1-window
   SNPs removed from the SNP kernel, and (c) all chr5 blocks removed as a
   chromosome-scale reference. Paired tests on per-repeat r with the
   Nadeau-Bengio correction.
3. Sufficiency. Every one of the 8,589 blocks is also turned into a kernel on
   its own and scored by the same CV protocol (rank-1 case evaluated in closed
   form), giving a genome-wide null distribution of single-block predictive
   value. DOG1's percentile inside that distribution is the direct answer to
   "is DOG1 doing the work?".
4. Local resolution. The same single-window scoring is repeated on 10-SNP and
   5-SNP sub-windows to test whether finer granularity isolates the DOG1
   region better, which links the granularity result to a mechanism.
5. Haplotype contrast. Accessions are clustered on the DOG1-window SNP dosages
   (k = 2, 3) and dormancy is compared between clusters (Kruskal-Wallis,
   difference between extreme clusters in days, and the position of the
   reference accessions Col-0, Ler, Cvi, Sd).

Result: results/sensitivity/dog1_mechanism.json"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "sensitivity"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import kernel_experiments as ke  # noqa: E402
from kernel_utils import feature_kernel_centered_scaled  # noqa: E402

RHO = 0.25

# Reference / well-characterised accessions whose DOG1 haplotype cluster is
# reported for orientation (names as spelled in dormancy_phenotype.csv).
REFERENCE_ACCESSIONS = ("Col-0", "Ler-1", "Ws-0", "Est-1", "Shahdara", "Nok-3")
GAMMA_RULE = 0.1
DOG1 = {"gene": "AT5G45830", "name": "DOG1", "chr": 5, "start": 18589418, "end": 18591506}
BLOCK_OF_DOG1 = 7960
LOCUS_FLANK = 3000  # window used for "the DOG1 region" in the ablation
N_CLUSTER_SEEDS = 5


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


def block_window_indices(blocks_df, chrom, lo, hi):
    sel = blocks_df[(blocks_df.chrom == chrom) & (blocks_df.end > lo) & (blocks_df.start < hi)]
    return sel.block_id.to_numpy()


def single_kernel_cv_r(Z, y, lam=1e-8, n_repeats=10, n_folds=5):
    """CV r for every column of Z treated as its own rank-1 kernel.

    A single standardised feature with a linear kernel and an explicit
    intercept is exactly ridge regression on that feature, so the leave-fold-out
    prediction is computed in closed form column by column: with intercept
    b = mean(y_train), alpha_j = <z_j, y - b> / (||z_j||^2 + lam) and
    z_pred_i,j = b + z_ij * alpha_j (elementwise across columns).
    """
    n = len(y)
    acc = np.zeros(Z.shape[1])
    for rep in range(n_repeats):
        pred = np.zeros((n, Z.shape[1]))
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=rep).split(Z):
            b = y[tr].mean()
            num = Z[tr].T @ (y[tr] - b)
            den = (Z[tr] ** 2).sum(axis=0) + lam
            alpha = num / den
            pred[te] = b + Z[te] * alpha[None, :]
        rs = np.array([stats.pearsonr(pred[:, j], y)[0] for j in range(Z.shape[1])])
        acc += rs
    return acc / n_repeats


def rank1_reference_r(z, y, lam=1e-8):
    """Same CV score obtained through the published kernel-ridge code path on
    the rank-1 kernel of one feature, for cross-checking the closed form."""
    from kernel_utils import solve_kernel_ridge_with_intercept

    K, _, _, _ = feature_kernel_centered_scaled(z[:, None])
    n = len(y)
    rs = []
    for rep in range(10):
        pred = np.zeros(n)
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=rep).split(K):
            _, _, pred[te] = solve_kernel_ridge_with_intercept(
                K[np.ix_(tr, tr)], y[tr], lam, K[np.ix_(te, tr)]
            )
        rs.append(stats.pearsonr(pred, y)[0])
    return float(np.mean(rs))


def main():
    t0 = time.time()
    data = ke.load_data()
    y = data["y"]
    blocks_df = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    m = np.load(PROCESSED / "block_snp_map.npz")

    res = {"dog1": DOG1, "block_of_dog1": BLOCK_OF_DOG1,
           "published_association": {
               "p_imhof": 0.3068525636480295, "percentile_raw": 32.26219583187798,
               "q_bh": 0.9467111873203956}}

    # ---------------- 1. containment ----------------
    print("[1] containment", flush=True)
    row = blocks_df[blocks_df.block_id == BLOCK_OF_DOG1].iloc[0]
    snp_rows = np.flatnonzero((snp.chrom.to_numpy() == DOG1["chr"]) &
                             (snp.pos.to_numpy() >= DOG1["start"] - LOCUS_FLANK) &
                             (snp.pos.to_numpy() <= DOG1["end"] + LOCUS_FLANK))
    minor_all = ke.panel_minor_codes()
    windows = {
        "gene_body": (DOG1["start"], DOG1["end"]),
        "gene_plus_2kb": (DOG1["start"] - 2000, DOG1["end"] + 2000),
        "gene_plus_5kb": (DOG1["start"] - 5000, DOG1["end"] + 5000),
        "gene_plus_10kb": (DOG1["start"] - 10000, DOG1["end"] + 10000),
    }
    containment = {}
    for label, (lo, hi) in windows.items():
        ids = block_window_indices(blocks_df, DOG1["chr"], lo, hi)
        n_snp = int(((snp.chrom.to_numpy() == DOG1["chr"]) &
                     (snp.pos.to_numpy() >= lo) & (snp.pos.to_numpy() <= hi)).sum())
        frac_inside = float(len(set(ids) & {BLOCK_OF_DOG1}))
        containment[label] = {
            "start": int(lo), "end": int(hi), "n_blocks_touched": int(len(ids)),
            "blocks": [int(i) for i in ids], "n_snps": n_snp,
            "dog1_block_in_window": bool(BLOCK_OF_DOG1 in set(ids)),
        }
        print(f"  {label:16s} blocks={list(map(int, ids))} n_snps={n_snp}", flush=True)
    res["containment"] = containment
    res["dog1_block_definition"] = {
        "start": int(row.start), "end": int(row.end), "n_snps": int(row.n_snps),
        "gene_count": int(row.gene_count), "nearest_gene": row.nearest_gene,
        "fraction_of_gene_body_covered": float(
            min(row.end, DOG1["end"]) - max(row.start, DOG1["start"])
        ) / (DOG1["end"] - DOG1["start"]),
    }

    # ---------------- 2. necessity (ablation) ----------------
    print("[2] ablation", flush=True)
    published = json.loads((OUT / "kernel_experiments.json").read_text(encoding="utf-8"))
    r_block_pub = np.array(published["block"]["r_per_repeat"])
    bm = data["block_mean"]
    keep_all = np.ones(bm.shape[1], dtype=bool)
    keep_all[BLOCK_OF_DOG1] = False
    chr5 = blocks_df.chrom.to_numpy() == 5
    keep_no_chr5 = ~chr5

    abl = {}
    specs = [
        ("block_mean_drop_dog1_block", bm[:, keep_all]),
        ("block_mean_drop_chr5", bm[:, keep_no_chr5]),
    ]
    for label, feats in specs:
        K, _, _, _ = feature_kernel_centered_scaled(feats)
        out, preds = ke.nested_cv_single(K, y)
        abl[label] = {
            "pearson_mean": out["pearson_mean"], "pearson_sd": out["pearson_sd"],
            "mae_mean": out["mae_mean"], "n_blocks": int(feats.shape[1]),
            "vs_full_block_kernel": paired(np.array(out["r_per_repeat"]) - r_block_pub),
        }
        print(f"  {label:32s} r={out['pearson_mean']:.4f} "
              f"pNB={abl[label]['vs_full_block_kernel']['p_nadeau_bengio']:.3f}", flush=True)

    # local RBF without the DOG1 block (fixed rule bandwidth)
    bm_sub = bm[:, keep_all]
    K_local_full = ke.center_trace(ke.perblock_rbf(bm, GAMMA_RULE))
    K_local_sub = ke.center_trace(ke.perblock_rbf(bm_sub, GAMMA_RULE))
    r_local_full = np.array(published["block_rbf_perblock"]["r_per_repeat"])
    out_l, _ = ke.nested_cv_single(K_local_sub, y)
    # reference run of the fixed rule bandwidth from the gamma ablation
    gl = json.loads((OUT / "gamma_lambda_sensitivity.json").read_text(encoding="utf-8"))
    r_rule = np.array(gl["fixed_gamma_ablation"]["gamma_0.1"]["r_per_repeat"])
    abl["local_rbf_drop_dog1_block"] = {
        "pearson_mean": out_l["pearson_mean"], "pearson_sd": out_l["pearson_sd"],
        "mae_mean": out_l["mae_mean"], "n_blocks": int(bm_sub.shape[1]),
        "vs_rule_gamma_local_rbf": paired(np.array(out_l["r_per_repeat"]) - r_rule),
        "note": "reference = fixed gamma=0.1 local RBF on all blocks",
    }
    print(f"  {'local_rbf_drop_dog1_block':32s} r={out_l['pearson_mean']:.4f} "
          f"(reference {r_rule.mean():.4f})", flush=True)

    # SNP kernel without the DOG1-window SNPs
    codes, minor_all_local = data["codes"], ke.panel_minor_codes()
    snp_idx_in_panel = np.flatnonzero(
        (snp.chrom.to_numpy() == DOG1["chr"]) &
        (snp.pos.to_numpy() >= DOG1["start"] - LOCUS_FLANK) &
        (snp.pos.to_numpy() <= DOG1["end"] + LOCUS_FLANK) &
        (snp.maf_83.to_numpy() > 0.05) & (snp.minor_code.to_numpy() < 4)
    )
    # straightforward: rebuild the SNP indicator matrix, then drop the window
    maf = snp["maf_83"].to_numpy()
    minor_code = snp["minor_code"].to_numpy()
    sel = np.flatnonzero((maf > 0.05) & (minor_code < 4))
    in_window = (snp.chrom.to_numpy()[sel] == DOG1["chr"]) & \
                (snp.pos.to_numpy()[sel] >= DOG1["start"] - LOCUS_FLANK) & \
                (snp.pos.to_numpy()[sel] <= DOG1["end"] + LOCUS_FLANK)
    X_snp = (data["codes"][sel, :] == minor_code[sel, None]).astype(np.float64).T
    X_snp_drop = X_snp[:, ~in_window]
    K_snp_full, _, _, _ = feature_kernel_centered_scaled(X_snp)
    K_snp_drop, _, _, _ = feature_kernel_centered_scaled(X_snp_drop)
    r_snp_pub = np.array(published["snp"]["r_per_repeat"])
    out_s, _ = ke.nested_cv_single(K_snp_drop, y)
    abl["snp_drop_dog1_window"] = {
        "pearson_mean": out_s["pearson_mean"], "pearson_sd": out_s["pearson_sd"],
        "mae_mean": out_s["mae_mean"], "n_snps_dropped": int(in_window.sum()),
        "n_snps_total": int(X_snp.shape[1]),
        "vs_full_snp_kernel": paired(np.array(out_s["r_per_repeat"]) - r_snp_pub),
    }
    print(f"  {'snp_drop_dog1_window':32s} r={out_s['pearson_mean']:.4f} "
          f"dropped={int(in_window.sum())}", flush=True)
    res["ablation"] = abl

    # ---------------- 3. sufficiency (single-block null) ----------------
    print("[3] single-block sufficiency", flush=True)
    Z = (bm - bm.mean(0)) / np.where(bm.std(0) < 1e-8, 1.0, bm.std(0))
    r_single = single_kernel_cv_r(Z, y)
    pct = float((r_single < r_single[BLOCK_OF_DOG1]).mean() * 100)
    top = int(np.argmax(r_single))
    check = {
        "dog1_closed_form": float(r_single[BLOCK_OF_DOG1]),
        "dog1_kernel_ridge_reference": rank1_reference_r(Z[:, BLOCK_OF_DOG1], y),
        "argmax_closed_form": float(r_single[top]),
        "argmax_kernel_ridge_reference": rank1_reference_r(Z[:, top], y),
    }
    check["max_abs_difference"] = float(max(
        abs(check["dog1_closed_form"] - check["dog1_kernel_ridge_reference"]),
        abs(check["argmax_closed_form"] - check["argmax_kernel_ridge_reference"]),
    ))
    res["single_block_sufficiency"] = {
        "protocol": "each block as its own rank-1 kernel, 10 repeats x 5-fold CV, lambda=1e-8",
        "implementation_check_vs_published_kernel_ridge": check,
        "dog1_block_r": float(r_single[BLOCK_OF_DOG1]),
        "genome_mean": float(r_single.mean()), "genome_sd": float(r_single.std()),
        "genome_max": float(r_single.max()),
        "argmax_block": int(np.argmax(r_single)),
        "dog1_percentile_within_single_block_distribution": pct,
        "n_blocks": int(len(r_single)),
    }
    print(f"  DOG1 single-block r={r_single[BLOCK_OF_DOG1]:.4f} percentile={pct:.1f} "
          f"max={r_single.max():.4f} (block {int(np.argmax(r_single))})", flush=True)

    # ---------------- 4. local resolution at finer granularity ----------
    print("[4] sub-window resolution", flush=True)
    fine = {}
    for chunk in (10, 5):
        cols_out, owner = [], []
        for bi, (s, e) in enumerate(zip(m["snp_start"], m["snp_end"])):
            for a in range(s, e, chunk):
                b = min(a + chunk, e)
                cols_out.append((data["codes"][a:b, :] == minor_all[a:b, None]).astype(np.float64).mean(0))
                owner.append(bi)
        Xf = np.stack(cols_out, axis=1)
        Zf = (Xf - Xf.mean(0)) / np.where(Xf.std(0) < 1e-8, 1.0, Xf.std(0))
        rf = single_kernel_cv_r(Zf, y)
        owner = np.array(owner)
        in_dog1 = owner == BLOCK_OF_DOG1
        rank_all = stats.rankdata(-rf)
        best_sub = int(np.argmax(rf))
        fine[f"subwindow_{chunk}snp"] = {
            "n_subwindows": int(Xf.shape[1]),
            "subwindows_from_dog1_block": int(in_dog1.sum()),
            "max_abs_r_dog1_subwindows": float(np.abs(rf[in_dog1]).max()),
            "best_subwindow_rank_of_dog1_block": int(rank_all[in_dog1].min()),
            "best_subwindow_overall_r": float(rf[best_sub]),
            "best_subwindow_parent_block": int(owner[best_sub]),
            "genome_mean": float(rf.mean()), "genome_sd": float(rf.std()),
        }
        print(f"  {chunk}-SNP: best r={rf[best_sub]:.4f} (block {owner[best_sub]}); "
              f"DOG1 sub-windows max |r|={np.abs(rf[in_dog1]).max():.4f} "
              f"best rank {int(rank_all[in_dog1].min())}/{Xf.shape[1]}", flush=True)
    res["subwindow_resolution"] = fine

    # ---------------- 5. DOG1 haplotype contrast ----------------
    print("[5] haplotype contrast", flush=True)
    win_snps = snp_idx_in_panel
    D = (data["codes"][win_snps, :] == minor_all[win_snps, None]).astype(np.float64).T
    Ds = (D - D.mean(0)) / np.where(D.std(0) < 1e-8, 1.0, D.std(0))
    names = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")["nativename"].to_numpy()
    hap = {}
    for k in (2, 3):
        km = KMeans(n_clusters=k, n_init=N_CLUSTER_SEEDS, random_state=20261024).fit_predict(Ds)
        groups = [y[km == g] for g in range(k)]
        kw = stats.kruskal(*groups)
        cl = {"k": k, "sizes": [int((km == g).sum()) for g in range(k)],
              "group_means_days": [float(g.mean()) for g in groups],
              "group_sds_days": [float(g.std(ddof=1)) if len(g) > 1 else None for g in groups],
              "kruskal_p": float(kw.pvalue),
              "extreme_difference_days": float(max(g.mean() for g in groups) - min(g.mean() for g in groups)),
              "reference_accession_clusters": {
                  nm: (int(km[int(np.flatnonzero(names == nm)[0])]) if (names == nm).any() else None)
                  for nm in REFERENCE_ACCESSIONS}}
        if k == 2:
            cl["pairwise_t_p_between_extremes"] = float(stats.ttest_ind(groups[0], groups[1]).pvalue)
        hap[f"k{k}"] = cl
        print(f"  k={k}: sizes={cl['sizes']} means={[round(x,1) for x in cl['group_means_days']]} "
              f"KW p={cl['kruskal_p']:.3g}", flush=True)
    # permutation reference for the k=2 extreme difference
    rng = np.random.default_rng(20261109)
    km2 = KMeans(n_clusters=2, n_init=N_CLUSTER_SEEDS, random_state=20261024).fit_predict(Ds)
    obs_diff = abs(y[km2 == 0].mean() - y[km2 == 1].mean())
    null_diff = np.empty(2000)
    for i in range(2000):
        yp = rng.permutation(y)
        null_diff[i] = abs(yp[km2 == 0].mean() - yp[km2 == 1].mean())
    hap["k2_permutation_p_of_extreme_difference"] = float(
        (1 + int((null_diff >= obs_diff).sum())) / (1 + len(null_diff)))
    hap["k2_observed_extreme_difference_days"] = float(obs_diff)
    hap["n_snps_in_haplotype_window"] = int(len(win_snps))
    res["haplotype_contrast"] = hap
    print(f"  k=2 permutation p={hap['k2_permutation_p_of_extreme_difference']:.3f} "
          f"(observed difference {obs_diff:.1f} days)", flush=True)

    res["elapsed_seconds"] = time.time() - t0
    (OUT / "dog1_mechanism.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone in {res['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
