"""Phenotype-transformation sensitivity (Blom rank-normal transform) of the block scan."""

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
    y_raw = pheno["dsds50"].to_numpy(float)
    n = len(y_raw)
    ranks = stats.rankdata(y_raw, method="average")
    y = stats.norm.ppf((ranks - 0.375) / (n + 0.25))   # Blom transform
    P13.y_global = y

    pcs = pd.read_csv(TABLES / "accession_pcs.csv")
    X = np.column_stack([np.ones(n), pcs[["PC1", "PC2", "PC3"]].to_numpy()])

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
        print(f"[transform] chr{c} null: delta={nulls[c]['delta']:.3g}", flush=True)

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
    df = pd.DataFrame(rows, columns=["block_id", "p_rank"])
    ok = df["p_rank"].notna()
    df.loc[ok, "q_rank"] = multipletests(df.loc[ok, "p_rank"], method="fdr_bh")[1]
    df.to_csv(TABLES / "phenotype_transform_sensitivity.csv", index=False)

    primary = pd.read_csv(TABLES / "block_scan_imhof.csv")
    merged = primary[["block_id", "chrom", "start", "end", "p_satt", "p_imhof"]].merge(
        df, on="block_id")
    ok = merged["p_rank"].notna() & merged["p_satt"].notna()
    rho = stats.spearmanr(merged.loc[ok, "p_satt"], merged.loc[ok, "p_rank"]).statistic

    n_tested = int(ok.sum())
    rank_pct = 100.0 * merged["p_rank"].rank(method="min") / n_tested
    dog = merged[(merged["chrom"] == 5) & (merged["start"] <= 18_591_506)
                 & (merged["end"] >= 18_589_418)]
    top = merged.loc[merged["p_rank"].idxmin()]
    summary = {
        "description": "rank-normal (Blom) phenotype sensitivity, secondary",
        "date": "2026-11-07",
        "n_blocks": n_tested,
        "min_p_rank": float(merged["p_rank"].min()),
        "top_block_rank": int(top["block_id"]),
        "top_block_region": f"{int(top['chrom'])}:{int(top['start'])}-{int(top['end'])}",
        "n_bh005_rank": int((merged["q_rank"] < 0.05).sum()),
        "spearman_rank_corr_raw_rank": float(rho),
        "dog1_percentile_rank": [float(rank_pct.loc[i]) for i in dog.index],
        "primary_top_block": int(primary.loc[primary["p_imhof"].idxmin(), "block_id"]),
        "elapsed_seconds": time.time() - t0,
    }
    (TABLES / "phenotype_transform_sensitivity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()