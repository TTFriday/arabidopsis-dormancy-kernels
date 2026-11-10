"""QC and exploratory analysis for the 83-accession dormancy panel.

Outputs go to results/tables and results/figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261023)


def load() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    return codes, snp, pheno


def dosage(codes: np.ndarray, snp: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """Minor-allele dosage 0/1/2 for the selected SNPs (SNPs x accessions); monomorphic -> NaN."""
    minor = snp["minor_code"].to_numpy()[idx]
    sub = codes[idx, :]
    out = np.full(sub.shape, np.nan, dtype=np.float32)
    ok = minor < 4
    out[ok, :] = (sub[ok, :] == minor[ok][:, None]).astype(np.float32)
    return out


def ld_prune(geno: np.ndarray, snp: pd.DataFrame, idx: np.ndarray, window_bp: int = 100_000, r2_thr: float = 0.2) -> np.ndarray:
    chrom = snp["chrom"].to_numpy()[idx]
    pos = snp["pos"].to_numpy()[idx]
    keep: list[int] = []
    for c in np.unique(chrom):
        local = np.flatnonzero(chrom == c)
        local = local[np.argsort(pos[local])]
        selected: list[int] = []
        for j in local:
            if not selected:
                selected.append(j)
                continue
            prev = np.array(selected)
            close = prev[pos[prev] >= pos[j] - window_bp]
            if close.size == 0:
                selected.append(j)
                continue
            a = geno[j, :]
            b = geno[close, :].T
            r = np.corrcoef(np.column_stack([a, b]), rowvar=False)[0, 1:]
            r = r[np.isfinite(r)]
            if r.size == 0 or np.max(np.abs(r)) < r2_thr:
                selected.append(j)
        keep.extend(selected)
    return idx[np.array(keep)]


def main() -> None:
    codes, snp, pheno = load()
    y = pheno["dsds50"].to_numpy()
    maf = snp["maf_83"].to_numpy()
    common = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    geno = dosage(codes, snp, common)
    geno = np.nan_to_num(geno, nan=0.0)

    # --- LD pruning and PCA ---
    pruned = ld_prune(geno, snp, common)
    print(f"common SNPs: {len(common):,}; LD-pruned: {len(pruned):,}")
    sel = np.flatnonzero(np.isin(common, pruned))
    raw = geno[sel, :]                      # SNPs x accessions
    g = raw.T                               # accessions x SNPs
    g = (g - g.mean(axis=0)) / np.where(g.std(axis=0) > 0, g.std(axis=0), 1.0)
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    pcs = u[:, :10] * s[:10]
    var_explained = s**2 / (s**2).sum()

    pd.DataFrame(
        {
            "accession": pheno["nativename"],
            "ecotype_id": pheno["ecotype_id"],
            "dsds50": y,
            **{f"PC{i + 1}": pcs[:, i] for i in range(10)},
        }
    ).to_csv(TABLES / "accession_pcs.csv", index=False)

    # --- kinship (VanRaden) on pruned SNPs ---
    af = raw.mean(axis=1) / 2
    z = raw - 2 * af[:, None]
    kin = z.T @ z / (2 * np.sum(af * (1 - af)))
    np.save(PROCESSED / "kinship_83.npy", kin)
    pd.DataFrame(kin, index=pheno["nativename"], columns=pheno["nativename"]).to_csv(
        TABLES / "kinship_83.csv"
    )

    # --- simple GWAS with PC covariates (sanity check, not the final model) ---
    gwas_idx = common[maf[common] > 0.05]
    X0 = np.column_stack([np.ones(len(y)), pcs[:, :3]])
    beta0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    resid0 = y - X0 @ beta0
    dof = len(y) - X0.shape[1]
    sigma2 = resid0 @ resid0 / dof

    rows = []
    for j in gwas_idx:
        x = codes[j, :].astype(np.float32)
        m = snp["minor_code"].iloc[j]
        d = (x == m).astype(np.float32)
        if d.std() == 0:
            continue
        X = np.column_stack([X0, d])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        s2 = r @ r / (len(y) - X.shape[1])
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1])
        t = beta[-1] / se
        rows.append((snp["chrom"].iloc[j], snp["pos"].iloc[j], beta[-1], t, 2 * stats.t.sf(abs(t), len(y) - X.shape[1])))
    gw = pd.DataFrame(rows, columns=["chrom", "pos", "beta", "t", "p"])
    gw["q"] = multipletests(gw["p"], method="fdr_bh")[1]
    gw = gw.sort_values("p").reset_index(drop=True)
    gw.to_csv(TABLES / "sanity_gwas_pc3.csv", index=False)
    lambda_gc = float(np.median(stats.chi2.isf(gw["p"].clip(lower=1e-300), 1)) / 0.4549)
    print(f"PC3-LM lambda_GC = {lambda_gc:.2f}; min p = {gw['p'].min():.2e}")
    print(gw.head(10).to_string(index=False))

    # --- EMMAX mixed-model sanity GWAS ---
    d_k, u_k = np.linalg.eigh(kin)
    d_k = np.clip(d_k, 0, None)
    yt = u_k.T @ y
    Xt = u_k.T @ X0
    n = len(y)
    p0 = X0.shape[1]

    def reml(delta: float) -> float:
        w = 1.0 / (delta * d_k + 1.0)
        xtwx = Xt.T @ (w[:, None] * Xt)
        xtwy = Xt.T @ (w * yt)
        beta = np.linalg.solve(xtwx, xtwy)
        r = yt - Xt @ beta
        rss = np.sum(w * r * r)
        s2 = rss / (n - p0)
        return -0.5 * ((n - p0) * np.log(s2) + np.sum(np.log(delta * d_k + 1.0)) + np.log(np.linalg.det(xtwx)))

    grid = np.logspace(-4, 3, 71)
    ll = np.array([reml(g) for g in grid])
    delta = float(grid[np.argmax(ll)])
    h2 = delta / (1 + delta)

    w = 1.0 / (delta * d_k + 1.0)
    sw = np.sqrt(w)
    Xw = Xt * sw[:, None]
    yw = yt * sw
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    beta0w = XtX_inv @ (Xw.T @ yw)
    resid_y = yw - Xw @ beta0w
    df = n - p0 - 1
    s2_reml = (resid_y @ resid_y) / (n - p0)

    row_sel = np.searchsorted(common, gwas_idx)
    Gt = u_k.T @ geno[row_sel, :].T          # transformed genotype, n x m
    Gw = Gt * sw[:, None]
    beta_g = Xw.T @ Gw                         # p x m
    proj = XtX_inv @ beta_g
    Gw_res = Gw - Xw @ proj
    numer = Gw_res.T @ resid_y
    denom = np.sum(Gw_res * Gw_res, axis=0)
    b_snp = numer / denom
    se_snp = np.sqrt(s2_reml / denom)
    t_snp = b_snp / se_snp
    p_snp = 2 * stats.t.sf(np.abs(t_snp), df)

    lmm = pd.DataFrame(
        {
            "chrom": snp["chrom"].to_numpy()[gwas_idx],
            "pos": snp["pos"].to_numpy()[gwas_idx],
            "beta": b_snp,
            "t": t_snp,
            "p": p_snp,
        }
    )
    lmm["q"] = multipletests(lmm["p"], method="fdr_bh")[1]
    lmm = lmm.sort_values("p").reset_index(drop=True)
    lmm.to_csv(TABLES / "sanity_gwas_emmax.csv", index=False)
    lambda_lmm = float(np.median(stats.chi2.isf(lmm["p"].clip(lower=1e-300), 1)) / 0.4549)
    print(f"EMMAX delta={delta:.3g} h2={h2:.2f}; lambda_GC={lambda_lmm:.2f}; min p={lmm['p'].min():.2e}")
    print(lmm.head(10).to_string(index=False))

    # --- LOCO EMMAX (kinship excludes the tested chromosome) ---
    pruned_chrom = snp["chrom"].to_numpy()[common[sel]]
    loco_rows = []
    loco_delta = {}
    for c in np.unique(snp["chrom"].to_numpy()):
        train = pruned_chrom != c
        raw_c = raw[train, :]
        af_c = raw_c.mean(axis=1) / 2
        z_c = raw_c - 2 * af_c[:, None]
        k_c = z_c.T @ z_c / (2 * np.sum(af_c * (1 - af_c)))
        dc, uc = np.linalg.eigh(k_c)
        dc = np.clip(dc, 0, None)
        yt_c = uc.T @ y
        xt_c = uc.T @ X0

        def reml_c(dlt, dc=dc, yt_c=yt_c, xt_c=xt_c):
            w_c = 1.0 / (dlt * dc + 1.0)
            xtwx = xt_c.T @ (w_c[:, None] * xt_c)
            b = np.linalg.solve(xtwx, xt_c.T @ (w_c * yt_c))
            r = yt_c - xt_c @ b
            s2 = np.sum(w_c * r * r) / (n - p0)
            return -0.5 * ((n - p0) * np.log(s2) + np.sum(np.log(dlt * dc + 1.0)) + np.log(np.linalg.det(xtwx)))

        ll_c = np.array([reml_c(g) for g in grid])
        dlt = float(grid[np.argmax(ll_c)])
        loco_delta[int(c)] = dlt

        w_c = 1.0 / (dlt * dc + 1.0)
        sw_c = np.sqrt(w_c)
        xw_c = xt_c * sw_c[:, None]
        yw_c = yt_c * sw_c
        inv_c = np.linalg.inv(xw_c.T @ xw_c)
        res_y = yw_c - xw_c @ (inv_c @ (xw_c.T @ yw_c))
        s2_c = (res_y @ res_y) / (n - p0)

        idx_c = np.flatnonzero(snp["chrom"].to_numpy()[gwas_idx] == c)
        gtc = uc.T @ geno[row_sel[idx_c], :].T
        gwc = gtc * sw_c[:, None]
        gw_res = gwc - xw_c @ (inv_c @ (xw_c.T @ gwc))
        denom_c = np.sum(gw_res * gw_res, axis=0)
        b_c = (gw_res.T @ res_y) / denom_c
        se_c = np.sqrt(s2_c / denom_c)
        t_c = b_c / se_c
        for k, j in enumerate(idx_c):
            loco_rows.append(
                (
                    int(c),
                    int(snp["pos"].iloc[gwas_idx[j]]),
                    float(b_c[k]),
                    float(t_c[k]),
                    float(2 * stats.t.sf(abs(t_c[k]), n - p0 - 1)),
                )
            )
        print(f"  LOCO chr{c}: delta={dlt:.3g}, h2={dlt / (1 + dlt):.2f}")

    loco = pd.DataFrame(loco_rows, columns=["chrom", "pos", "beta", "t", "p"])
    loco["q"] = multipletests(loco["p"], method="fdr_bh")[1]
    loco = loco.sort_values("p").reset_index(drop=True)
    loco.to_csv(TABLES / "sanity_gwas_loco.csv", index=False)
    lambda_loco = float(np.median(stats.chi2.isf(loco["p"].clip(lower=1e-300), 1)) / 0.4549)
    print(f"LOCO lambda_GC={lambda_loco:.2f}; min p={loco['p'].min():.2e}")
    print(loco.head(10).to_string(index=False))

    # --- LD decay (r2 vs distance) on a random subset ---
    chrom_common = snp["chrom"].to_numpy()[common]
    pos_common = snp["pos"].to_numpy()[common]
    pairs = []
    for c in np.unique(chrom_common):
        local = np.flatnonzero(chrom_common == c)
        pos = pos_common[local]
        sub = geno[local, :].T
        for _ in range(3000):
            i, k = sorted(RNG.choice(len(local), 2, replace=False))
            dist = pos[k] - pos[i]
            if dist <= 0 or dist > 500_000:
                continue
            r = np.corrcoef(sub[:, i], sub[:, k])[0, 1]
            if np.isfinite(r):
                pairs.append((c, int(dist), r * r))
    ld = pd.DataFrame(pairs, columns=["chrom", "distance_bp", "r2"])
    ld.to_csv(TABLES / "ld_decay_pairs.csv", index=False)

    # --- figures ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].hist(y, bins=20, color="#4c72b0", edgecolor="white")
    axes[0, 0].set_xlabel("DSDS50 (days)")
    axes[0, 0].set_ylabel("Accessions")
    axes[0, 0].set_title("Dormancy phenotype (n=83)")

    axes[0, 1].scatter(pcs[:, 0], pcs[:, 1], c=y, cmap="viridis", s=45, edgecolor="k", linewidth=0.3)
    axes[0, 1].set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}%)")
    axes[0, 1].set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}%)")
    axes[0, 1].set_title("Population structure")

    bins = np.arange(0, 500_001, 10_000)
    ld["bin"] = pd.cut(ld["distance_bp"], bins)
    ld_mean = ld.groupby("bin", observed=True)["r2"].mean()
    centers = [b.mid for b in ld_mean.index]
    axes[1, 0].plot(np.array(centers) / 1000, ld_mean.values, marker="o", ms=3)
    axes[1, 0].set_xlabel("Distance (kb)")
    axes[1, 0].set_ylabel("Mean r²")
    axes[1, 0].set_title("LD decay")

    axes[1, 1].scatter(-np.log10(loco["p"]), loco["chrom"], c=loco["chrom"], cmap="tab10", s=6, alpha=0.6)
    axes[1, 1].set_xlabel("-log10 p (LOCO EMMAX)")
    axes[1, 1].set_ylabel("Chromosome")
    axes[1, 1].set_title("Mixed-model GWAS")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_data_qc.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(kin, cmap="RdBu_r", vmin=-0.2, vmax=0.6)
    ax.set_title("Kinship (VanRaden)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_kinship.png", dpi=300)
    plt.close(fig)

    summary = {
        "common_snps_maf05": int(len(common)),
        "ld_pruned_snps": int(len(pruned)),
        "pc_variance_explained_top5": [float(v) for v in var_explained[:5]],
        "pc3_lm_lambda_gc": lambda_gc,
        "pc3_lm_min_p": float(gw["p"].min()),
        "emmax": {
            "delta": delta,
            "snp_heritability": h2,
            "lambda_gc": lambda_lmm,
            "min_p": float(lmm["p"].min()),
            "n_p_lt_1e-5": int((lmm["p"] < 1e-5).sum()),
            "top_hits": lmm.head(10).to_dict(orient="records"),
        },
        "loco_emmax": {
            "delta_per_chromosome": {str(k): v for k, v in loco_delta.items()},
            "lambda_gc": lambda_loco,
            "min_p": float(loco["p"].min()),
            "n_p_lt_1e-5": int((loco["p"] < 1e-5).sum()),
            "top_hits": loco.head(10).to_dict(orient="records"),
        },
        "ld_mean_r2_first_bins": {str(b): float(v) for b, v in ld_mean.head(8).items()},
        "kinship_mean_offdiag": float((kin.sum() - np.trace(kin)) / (len(y) * (len(y) - 1))),
    }
    (TABLES / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
