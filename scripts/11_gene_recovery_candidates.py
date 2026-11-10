"""Gene-block mapping, known dormancy gene recovery and candidate evidence grading."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "TABLES".lower()
RNG = np.random.default_rng(20261107)
N_DRAWS = 20_000
MAX_NEAREST_DIST = 50_000

KNOWN = {
    "AT5G45830": "DOG1", "AT3G24650": "ABI3", "AT2G40220": "ABI4",
    "AT2G36270": "ABI5", "AT1G21970": "LEC1", "AT1G28300": "LEC2",
    "AT3G26790": "FUS3", "AT1G18100": "MFT", "AT2G20180": "PIF1/PIL5",
    "AT2G18790": "PHYB", "AT4G36930": "SPT", "AT3G05120": "GID1A",
    "AT3G63010": "GID1B", "AT5G27320": "GID1C", "AT2G01570": "RGA",
    "AT1G14920": "GAI", "AT1G66350": "RGL1", "AT3G03450": "RGL2",
    "AT5G17490": "RGL3", "AT4G24210": "SLY1", "AT3G24220": "NCED6",
    "AT1G78390": "NCED9", "AT4G19230": "CYP707A1", "AT2G29090": "CYP707A2",
    "AT5G51760": "AHG1", "AT3G11410": "AHG3/PP2CA", "AT1G72770": "HAB1",
    "AT4G26080": "ABI1", "AT5G57050": "ABI2", "AT5G08590": "SNRK2.1",
    "AT3G50500": "SNRK2.2", "AT5G66880": "SNRK2.3", "AT4G25420": "GA20OX1",
    "AT5G51810": "GA20OX2", "AT1G15550": "GA3OX1", "AT1G80340": "GA3OX2",
    "AT5G67030": "ABA1", "AT1G52340": "ABA2", "AT1G16540": "ABA3",
    "AT2G27150": "AAO3", "AT4G17870": "PYR1", "AT5G46790": "PYL1",
    "AT2G26040": "PYL2", "AT2G38310": "PYL4", "AT5G05440": "PYL5",
    "AT2G40330": "PYL6", "AT5G53160": "PYL8/RCAR3", "AT4G37470": "KAI2",
    "AT2G42620": "MAX2", "AT5G57710": "SMAX1", "AT5G35550": "TT2",
    "AT4G09820": "TT8", "AT5G24520": "TTG1", "AT5G48100": "TT10",
    "AT1G61720": "BAN", "AT1G17260": "AHA10", "AT3G47620": "TCP14",
    "AT1G69690": "TCP15",
}


def build_unified_map(genes, blocks):
    """Maximum-overlap mapping with full overlap/distance audit trail."""
    recs = []
    by_chr = {c: blocks[blocks["chrom"] == c].reset_index(drop=True)
              for c in range(1, 6)}
    for _, g in genes.iterrows():
        c = int(g["chr"])
        gs, ge = int(g["start"]), int(g["end"])
        rec = {"gene": g["gene"], "chr": c, "start": gs, "end": ge,
               "name": g.get("name", "")}
        if c not in by_chr:
            rec.update({"block_id": -1, "rule": "none", "overlap_bp": 0,
                        "distance_bp": -1, "n_overlaps": 0, "all_overlaps": ""})
            recs.append(rec)
            continue
        sub = by_chr[c]
        starts, ends = sub["start"].to_numpy(), sub["end"].to_numpy()
        ov = np.maximum(0, np.minimum(ends, ge) - np.maximum(starts, gs))
        hit = np.flatnonzero(ov > 0)
        if len(hit) > 0:
            best = hit[int(np.argmax(ov[hit]))]
            rec.update({"block_id": int(sub.loc[best, "block_id"]),
                        "rule": "max_overlap", "overlap_bp": int(ov[best]),
                        "distance_bp": 0, "n_overlaps": len(hit),
                        "all_overlaps": ";".join(
                            f"{int(sub.loc[k,'block_id'])}:{int(ov[k])}" for k in hit)})
        else:
            dist = np.minimum(np.abs(starts - ge), np.abs(ends - gs))
            best = int(np.argmin(dist))
            if dist[best] <= MAX_NEAREST_DIST:
                rec.update({"block_id": int(sub.loc[best, "block_id"]),
                            "rule": "nearest_50kb", "overlap_bp": 0,
                            "distance_bp": int(dist[best]), "n_overlaps": 0,
                            "all_overlaps": ""})
            else:
                rec.update({"block_id": -1, "rule": "none", "overlap_bp": 0,
                            "distance_bp": int(dist[best]), "n_overlaps": 0,
                            "all_overlaps": ""})
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    t0 = time.time()
    scan = pd.read_csv(TABLES / "block_scan_imhof.csv")
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    genes = pd.read_csv(ROOT / "data" / "annotations" / "genes_tair10.csv")
    n_tested = int(scan["p_imhof"].notna().sum())

    gmap = build_unified_map(genes, blocks)
    gmap.to_csv(TABLES / "gene_block_map_unified.csv", index=False)
    print(f"mapping rules: {gmap['rule'].value_counts().to_dict()}")

    # percentiles (lower = stronger)
    rank = scan.set_index("block_id")["p_imhof"].rank(method="min")
    pct_map = (100.0 * rank / n_tested).to_dict()
    gmap["pct_raw"] = gmap["block_id"].map(pct_map)

    known = gmap[gmap["gene"].isin(KNOWN)].copy()
    known["short"] = known["gene"].map(KNOWN)
    n_missing = int(known["pct_raw"].isna().sum())
    known_valid = known[known["pct_raw"].notna()]
    print(f"known genes: {len(known)}, unmappable: {n_missing}")

    # matched background: gene-length quartile x block gene-count quartile
    bg = gmap[gmap["pct_raw"].notna()].copy()
    bg["gene_len"] = bg["end"] - bg["start"]
    gc = blocks.set_index("block_id")["gene_count"]
    bg["block_gc"] = bg["block_id"].map(gc)
    bg["len_q"] = pd.qcut(bg["gene_len"], 4, labels=False, duplicates="drop")
    bg["gc_q"] = pd.qcut(bg["block_gc"], 4, labels=False, duplicates="drop")
    bg["stratum"] = bg["len_q"].astype(str) + "_" + bg["gc_q"].astype(str)
    kv = bg[bg["gene"].isin(KNOWN)]
    strata = {s: sub["pct_raw"].to_numpy() for s, sub in bg.groupby("stratum")}
    obs_mean = float(kv["pct_raw"].mean())
    null_mean = np.zeros(N_DRAWS)
    covered = 0
    kv_strata = kv["stratum"].tolist()
    kv_pct = kv["pct_raw"].to_numpy()
    for s in kv_strata:
        if s in strata and len(strata[s]) > 1:
            covered += 1
    for i in range(N_DRAWS):
        vals = []
        for s in kv_strata:
            pool = strata.get(s)
            if pool is None or len(pool) == 0:
                vals.append(np.nan)
            else:
                vals.append(RNG.choice(pool))
        null_mean[i] = np.nanmean(vals)
    p_primary = float((1 + (null_mean <= obs_mean).sum()) / (N_DRAWS + 1))

    # secondary endpoints (reported with multiplicity note)
    secondary = {}
    for frac in [0.01, 0.05, 0.10]:
        top = set(scan.nsmallest(int(round(frac * n_tested)), "p_imhof")["block_id"])
        obs = int(kv["block_id"].isin(top).sum())
        # matched null for top-fraction counts
        block_pct = bg.set_index("gene")["pct_raw"]
        null_cnt = np.zeros(N_DRAWS, dtype=int)
        thr = 100.0 * frac
        for i in range(N_DRAWS):
            cnt = 0
            for s in kv_strata:
                pool = strata.get(s)
                if pool is not None and len(pool) > 0:
                    cnt += int(RNG.choice(pool) <= thr)
            null_cnt[i] = cnt
        secondary[f"top{int(frac*100)}"] = {
            "observed": obs, "p_matched": float((1 + (null_cnt >= obs).sum()) / (N_DRAWS + 1))}

    known_out = known_valid.merge(
        scan[["block_id", "p_imhof", "q_bh", "p_rotation", "minp_fwer"]],
        on="block_id", how="left").sort_values("pct_raw")
    known_out.to_csv(TABLES / "known_gene_ranks.csv", index=False)

    # ---------------- candidate evidence table ----------------
    cand = scan[scan["p_imhof"].notna()].sort_values("p_imhof").head(20).copy()
    genes_by_block = {}
    for _, r in gmap[gmap["block_id"] >= 0].iterrows():
        genes_by_block.setdefault(int(r["block_id"]), []).append(r["gene"])
    known_blocks = set(known_valid["block_id"])
    rows = []
    for _, r in cand.iterrows():
        bid = int(r["block_id"])
        if r["minp_fwer"] < 0.05:
            grade = "A"
        elif bid in known_blocks:
            grade = "D"
        else:
            grade = "C"
        rows.append({"block_id": bid, "chrom": int(r["chrom"]),
                     "start": int(r["start"]), "end": int(r["end"]),
                     "p_imhof": float(r["p_imhof"]), "q_bh": float(r["q_bh"]),
                     "p_rotation": float(r["p_rotation"]),
                     "minp_fwer": float(r["minp_fwer"]),
                     "genes": ";".join(genes_by_block.get(bid, [])),
                     "grade": grade})
    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(TABLES / "candidate_evidence.csv", index=False)

    dog1 = known_out[known_out["gene"] == "AT5G45830"]
    summary = {
        "description": "Biological validation (frozen mapping rules, matched background)",
        "date": "2026-11-07",
        "mapping": {"n_genes": int(len(gmap)),
                    "rules": {k: int(v) for k, v in gmap["rule"].value_counts().items()}},
        "known_genes": {"n_total": len(KNOWN), "n_mapped": int(len(known_valid)),
                        "n_unmappable": n_missing,
                        "primary_endpoint": "mean percentile, raw-block p_imhof",
                        "observed_mean_percentile": obs_mean,
                        "p_matched_permutation": p_primary,
                        "background": "gene-length quartile x block-gene-count quartile",
                        "stratum_coverage": f"{covered}/{len(kv_strata)}",
                        "n_draws": N_DRAWS,
                        "secondary_endpoints": secondary,
                        "multiplicity_note": "primary frozen; secondary endpoints "
                                             "exploratory, not independently corrected"},
        "dog1": dog1[["block_id", "pct_raw", "p_imhof", "q_bh", "p_rotation",
                      "minp_fwer"]].to_dict(orient="records"),
        "candidates": {"n_table": len(cand_df),
                       "grades": cand_df["grade"].value_counts().to_dict()},
        "elapsed_seconds": time.time() - t0,
    }
    (TABLES / "gene_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["known_genes"], indent=2))
    print("DOG1:", json.dumps(summary["dog1"]))
    print(cand_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()