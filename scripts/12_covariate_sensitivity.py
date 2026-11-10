"""Covariate sensitivity of the block scan (PC1-5 versus the frozen PC1-3 null)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
import block_association_scan as P13

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def main():
    t0 = time.time()
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)
    P13.y_global = y
    n = len(y)
    pcs = pd.read_csv(TABLES / "accession_pcs.csv")
    X = np.column_stack([np.ones(n), pcs[["PC1", "PC2", "PC3", "PC4", "PC5"]].to_numpy()])

    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()
    G = (codes[keep, :] == minor[keep, None]).astype(np.float64)
    chrom = snp["chrom"].to_numpy()[keep]
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    row_map = np.full(len(snp), -1, dtype=int)
    row_map[keep] = np.arange(len(keep))

    nulls = {}
    for c in range(1, 6):
        m = chrom != c
        Gm = G[m, :]
        freq = Gm.mean(axis=1)
        ok = (freq > 0) & (freq < 1)
        Zg = (Gm[ok] - freq[ok, None]) / np.sqrt(freq[ok] * (1 - freq[ok]))[:, None]
        K = Zg.T @ Zg / Zg.shape[0]
        nulls[c] = P13.fit_null(0.5 * (K + K.T), X, y)

    rows = []
    for b in range(len(blocks)):
        c = int(blocks.loc[b, "chrom"])
        s = int(blocks.loc[b, "snp_start_row"])
        e = int(blocks.loc[b, "snp_end_row"])
        rr = row_map[s:e]
        rr = rr[rr >= 0]
        if len(rr) == 0:
            rows.append((b, np.nan))
            continue
        Z = G[rr, :].T
        Z = Z - Z.mean(axis=0)
        _, _, _, _, p_satt, _ = P13.block_test(Z, nulls[c])
        rows.append((b, p_satt))
    df = pd.DataFrame(rows, columns=["block_id", "p_pc5"])
    ok = df["p_pc5"].notna()
    df.loc[ok, "q_pc5"] = multipletests(df.loc[ok, "p_pc5"], method="fdr_bh")[1]
    df.to_csv(TABLES / "covariate_sensitivity.csv", index=False)

    primary = pd.read_csv(TABLES / "block_scan_imhof.csv")
    merged = primary[["block_id", "p_satt", "p_imhof"]].merge(df, on="block_id")
    ok = merged["p_pc5"].notna() & merged["p_satt"].notna()
    rho = stats.spearmanr(merged.loc[ok, "p_satt"], merged.loc[ok, "p_pc5"]).statistic
    r_log = stats.pearsonr(-np.log10(merged.loc[ok, "p_satt"]),
                           -np.log10(merged.loc[ok, "p_pc5"]))[0]
    summary = {
        "description": "PC1-5 covariate sensitivity (secondary)",
        "date": "2026-11-07",
        "n_blocks": int(ok.sum()),
        "min_p_pc5": float(df["p_pc5"].min()),
        "n_bh005_pc5": int((df["q_pc5"] < 0.05).sum()),
        "spearman_rank_corr_pc3_pc5": float(rho),
        "pearson_corr_log10p": float(r_log),
        "elapsed_seconds": time.time() - t0,
    }
    (TABLES / "covariate_sensitivity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()