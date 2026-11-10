"""Leave-one-chromosome-out block score tests with Imhof calibration and residual-rotation family-wise error control."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
RNG = np.random.default_rng(20261107)

N_ROTATIONS = 1000
DELTA_GRID = np.logspace(-3, 3, 121)


# ----------------------------------------------------------------------
# null model
# ----------------------------------------------------------------------
def reml_delta(K, X, y):
    """REML estimate of delta = sigma_g^2 / sigma_e^2 for V = I + delta*K."""
    ev, U = np.linalg.eigh(K)
    ev = np.maximum(ev, 0)
    XtU, ytU = U.T @ X, U.T @ y
    n, p = X.shape
    best = (-np.inf, 1.0, 1.0)
    for delta in DELTA_GRID:
        v = 1.0 + delta * ev
        Xv = XtU / np.sqrt(v)[:, None]
        yv = ytU / np.sqrt(v)
        beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
        resid = yv - Xv @ beta
        sigma2 = float(resid @ resid) / (n - p)
        if sigma2 <= 0:
            continue
        ll = -0.5 * ((n - p) * np.log(sigma2) + np.log(v).sum()
                     + np.linalg.slogdet(Xv.T @ Xv)[1])
        if ll > best[0]:
            best = (ll, float(delta), sigma2)
    return best[1], best[2]


def fit_null(K, X, y):
    delta, sigma2 = reml_delta(K, X, y)
    V = delta * K + np.eye(len(y))
    Vinv = np.linalg.inv(V)
    XtVinv = np.linalg.inv(X.T @ Vinv @ X)
    P = Vinv - Vinv @ X @ XtVinv @ X.T @ Vinv
    L = np.linalg.cholesky(V)
    return {"delta": delta, "sigma2": sigma2, "P": P, "L": L, "V": V}


# ----------------------------------------------------------------------
# mixture chi-square tail
# ----------------------------------------------------------------------
def imhof_sf(q, lam):
    """P(sum_j lam_j chi2_1 > q) by Imhof's numerical inversion."""
    lam = lam[lam > 1e-12]
    if len(lam) == 0 or q <= 0:
        return 1.0

    def integrand(u):
        if u == 0:
            return 0.5 * (lam.sum() - q)
        theta = 0.5 * np.arctan(lam * u).sum() - 0.5 * q * u
        rho = np.prod((1.0 + (lam * u) ** 2) ** 0.25)
        return np.sin(theta) / (u * rho)

    val, err = integrate.quad(integrand, 0, np.inf,
                              limit=300, epsabs=1e-11, epsrel=1e-8)
    # Imhof: CDF = 0.5 - val/pi, survival function P(Q > q) = 0.5 + val/pi
    p = 0.5 + val / np.pi
    return float(np.clip(p, 0.0, 1.0))


def satt_params(lam):
    c1, c2 = float(lam.sum()), float((lam ** 2).sum())
    if c1 <= 0 or c2 <= 0:
        return 1.0, 1.0
    return c2 / c1, c1 ** 2 / c2


def block_test(Z, null):
    """Score test for one block. Returns Q/sigma2, weights, p-values."""
    P, sigma2 = null["P"], null["sigma2"]
    ZP = Z.T @ P                      # (p_b, n)
    w = ZP @ y_global                 # (p_b,)
    Q = float(w @ w) / sigma2
    M = ZP @ Z                        # Z^T P Z  (P V P = P)
    lam = np.linalg.eigvalsh(M)
    lam = np.maximum(lam, 0)
    scale, df = satt_params(lam)
    p_satt = float(stats.chi2.sf(Q / scale, df))
    return Q, lam, scale, df, p_satt, ZP


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    global y_global
    t0 = time.time()
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y = pheno["dsds50"].to_numpy(float)          # raw days, no standardisation
    y_global = y
    n = len(y)
    pcs = pd.read_csv(TABLES / "accession_pcs.csv")
    X = np.column_stack([np.ones(n), pcs[["PC1", "PC2", "PC3"]].to_numpy()])
    p_cov = X.shape[1]

    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()
    G = (codes[keep, :] == minor[keep, None]).astype(np.float64)   # 0/1 inbred
    chrom = snp["chrom"].to_numpy()[keep]
    pos = snp["pos"].to_numpy()[keep]
    blocks = pd.read_csv(PROCESSED / "haplotype_blocks.csv")
    n_blocks = len(blocks)
    # map original snp rows -> filtered rows for block membership
    row_map = np.full(len(snp), -1, dtype=int)
    row_map[keep] = np.arange(len(keep))

    # ---------------- LOCO null models ----------------
    print("[scan] fitting LOCO null models (raw-scale y)", flush=True)
    nulls = {}
    for c in range(1, 6):
        m = chrom != c
        Gm = G[m, :]
        freq = Gm.mean(axis=1)
        ok = (freq > 0) & (freq < 1)
        Zg = (Gm[ok] - freq[ok, None]) / np.sqrt(freq[ok] * (1 - freq[ok]))[:, None]
        K = Zg.T @ Zg / Zg.shape[0]
        K = 0.5 * (K + K.T)
        ev_min = np.linalg.eigvalsh(K).min()
        null = fit_null(K, X, y)
        null["eig_min"] = float(ev_min)
        null["n_snps"] = int(ok.sum())
        null["at_boundary"] = bool(null["delta"] >= DELTA_GRID[-1] * 0.999
                                   or null["delta"] <= DELTA_GRID[0] * 1.001)
        nulls[c] = null
        print(f"  chr{c}: delta={null['delta']:.4g}, sigma2={null['sigma2']:.4g}, "
              f"min_eig(K)={ev_min:.2e}, boundary={null['at_boundary']}", flush=True)

    # ---------------- SNP-level LMM (EMMAX-style) ----------------
    print("[scan] SNP-level LMM scan", flush=True)
    snp_rows = []
    for c in range(1, 6):
        null = nulls[c]
        P, sigma2 = null["P"], null["sigma2"]
        idx = np.flatnonzero(chrom == c)
        Gc = G[idx, :]
        mu = Gc.mean(axis=1)
        Zc = Gc - mu[:, None]
        Py = P @ y
        num = Zc @ Py
        PZc = P @ Zc.T
        denom = np.einsum("ij,ji->i", Zc, PZc)
        good = denom > 1e-10
        beta = np.full(len(idx), np.nan)
        se = np.full(len(idx), np.nan)
        pv = np.full(len(idx), np.nan)
        beta[good] = num[good] / denom[good]
        se[good] = np.sqrt(sigma2 / denom[good])
        t = beta[good] / se[good]
        pv[good] = 2 * stats.t.sf(np.abs(t), n - p_cov - 1)
        for j, i in enumerate(idx):
            snp_rows.append((int(c), int(pos[i]), float(beta[j]), float(se[j]),
                             float(pv[j]) if good[j] else np.nan))
    snp_df = pd.DataFrame(snp_rows, columns=["chrom", "pos", "beta", "se", "p"])
    ok = snp_df["p"].notna()
    snp_df.loc[ok, "q_bh"] = multipletests(snp_df.loc[ok, "p"], method="fdr_bh")[1]
    snp_df.to_csv(TABLES / "snp_lmm_scan.csv", index=False)
    print(f"  SNPs tested: {ok.sum():,}, min p = {snp_df['p'].min():.3e}, "
          f"n(q<0.05) = {(snp_df['q_bh'] < 0.05).sum()}", flush=True)

    # numerical regression test: 5 SNPs vs direct GLS (statsmodels-free check)
    print("[scan] numerical verification of SNP LMM", flush=True)
    verif = []
    test_idx = np.linspace(0, len(keep) - 1, 5, dtype=int)
    for i in test_idx:
        c = int(chrom[i])
        null = nulls[c]
        z = G[i] - G[i].mean()
        V = null["V"]
        # direct GLS with design [X, z]
        D = np.column_stack([X, z])
        Vi = np.linalg.inv(V)
        b = np.linalg.solve(D.T @ Vi @ D, D.T @ Vi @ y)
        resid = y - D @ b
        s2 = float(resid @ Vi @ resid) / (n - D.shape[1])
        cov = s2 * np.linalg.inv(D.T @ Vi @ D)
        t_direct = b[-1] / np.sqrt(cov[-1, -1])
        row = snp_df[(snp_df["chrom"] == c) & (snp_df["pos"] == int(pos[i]))].iloc[0]
        t_scan = row["beta"] / row["se"]
        verif.append({"chrom": c, "pos": int(pos[i]),
                      "t_scan": float(t_scan), "t_direct_gls": float(t_direct),
                      "rel_diff": float(abs(t_scan - t_direct) / max(abs(t_direct), 1e-12))})
    max_diff = max(v["rel_diff"] for v in verif)
    print(f"  max relative t difference vs direct GLS: {max_diff:.2e}", flush=True)

    # ---------------- block score tests (primary family) ----------------
    print("[scan] block score tests (primary family: raw blocks)", flush=True)
    edges = np.load(PROCESSED / "block_edges.npz")
    src, dst = edges["edge_index"]
    neighbours = [set() for _ in range(n_blocks)]
    for i, j in zip(src, dst):
        neighbours[int(i)].add(int(j))
        neighbours[int(j)].add(int(i))
    graph_nodes = np.load(PROCESSED / "graph_embeddings_83.npy")

    recs = []
    ZP_pad = np.zeros((n_blocks, 25, n))     # padded Z^T P for rotations
    Q_obs = np.full(n_blocks, np.nan)
    scale_arr = np.ones(n_blocks)
    df_arr = np.ones(n_blocks)
    block_chrom = blocks["chrom"].to_numpy()
    for b in range(n_blocks):
        c = int(block_chrom[b])
        null = nulls[c]
        s = int(blocks.loc[b, "snp_start_row"])
        e = int(blocks.loc[b, "snp_end_row"])
        rows = row_map[s:e]
        rows = rows[rows >= 0]
        rec = {"block_id": b, "chrom": c,
               "start": int(blocks.loc[b, "start"]), "end": int(blocks.loc[b, "end"]),
               "n_snps_test": len(rows)}
        if len(rows) == 0:
            recs.append(rec)
            continue
        Z = G[rows, :].T
        Z = Z - Z.mean(axis=0)
        Q, lam, scale, df, p_satt, ZP = block_test(Z, null)
        p_imhof = imhof_sf(Q, lam)
        k = min(ZP.shape[0], 25)
        ZP_pad[b, :k, :] = ZP[:k, :]
        Q_obs[b], scale_arr[b], df_arr[b] = Q, scale, df
        rec.update({"Q": Q, "df_satt": df, "p_satt": p_satt, "p_imhof": p_imhof})
        # secondary exploratory families
        group = sorted({b, *neighbours[b]})
        grows = np.concatenate([row_map[int(blocks.loc[j, "snp_start_row"]):
                                       int(blocks.loc[j, "snp_end_row"])]
                                for j in group])
        grows = np.unique(grows[grows >= 0])
        if len(grows) > 0:
            Zg = G[grows, :].T
            Zg = Zg - Zg.mean(axis=0)
            Qg, lamg, _, _, p_satt_g, _ = block_test(Zg, null)
            rec["p_graph_satt"] = p_satt_g
            rec["p_graph_imhof"] = imhof_sf(Qg, lamg)
        Ze = graph_nodes[:, b, :].astype(np.float64)
        Ze = Ze - Ze.mean(axis=0)
        sd = Ze.std(axis=0)
        okd = sd > 1e-8
        if okd.any():
            Ze = Ze[:, okd] / sd[okd]
            Qe, lame, _, _, p_satt_e, _ = block_test(Ze, null)
            rec["p_emb_satt"] = p_satt_e
            rec["p_emb_imhof"] = imhof_sf(Qe, lame)
        recs.append(rec)
        if (b + 1) % 2000 == 0:
            print(f"  {b + 1}/{n_blocks}", flush=True)
    res = pd.DataFrame(recs)
    for col, qcol in [("p_imhof", "q_bh"), ("p_satt", "q_bh_satt"),
                      ("p_graph_imhof", "q_graph_bh"), ("p_emb_imhof", "q_emb_bh")]:
        ok = res[col].notna()
        res.loc[ok, qcol] = multipletests(res.loc[ok, col], method="fdr_bh")[1]

    # ---------------- Imhof verification (mathematical null) ----------------
    print("[scan] Imhof tail verification by mixture sampling", flush=True)
    imhof_check = []
    top5 = res.nsmallest(5, "p_imhof")
    for _, row in top5.iterrows():
        b = int(row["block_id"])
        c = int(row["chrom"])
        s = int(blocks.loc[b, "snp_start_row"])
        e = int(blocks.loc[b, "snp_end_row"])
        rows = row_map[s:e]
        rows = rows[rows >= 0]
        Z = G[rows, :].T
        Z = Z - Z.mean(axis=0)
        M = Z.T @ nulls[c]["P"] @ Z
        lam = np.maximum(np.linalg.eigvalsh(M), 0)
        lam = lam[lam > 1e-12]
        draws = (RNG.standard_normal((200_000, len(lam))) ** 2) @ lam
        q = float(row["Q"])
        p_mc = float((draws >= q).mean())
        imhof_check.append({"block_id": b, "Q": q,
                            "p_imhof": float(row["p_imhof"]),
                            "p_satt": float(row["p_satt"]),
                            "p_mc_200k": p_mc})
    print("  " + json.dumps(imhof_check), flush=True)

    # ---------------- fixed-norm rotation, full-family minP ----------------
    print(f"[scan] {N_ROTATIONS} residual rotations (full-family minP)", flush=True)
    # whiten per chromosome; residual basis
    rot = {}
    for c in range(1, 6):
        null = nulls[c]
        L = null["L"]
        Li = np.linalg.inv(L)
        Xt = Li @ X
        yt = Li @ y
        Qx, _ = np.linalg.qr(Xt, mode="complete")
        B = Qx[:, p_cov:]                    # n x (n-p), orthogonal residual basis
        r = B.T @ yt
        beta_t = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        rot[c] = {"Li": Li, "L": L, "B": B, "r_norm": float(np.linalg.norm(r)),
                  "Xt_beta": Xt @ beta_t}
    minp_null = np.zeros(N_ROTATIONS)
    exceed = np.zeros(n_blocks, dtype=int)   # per-block rotation exceedances
    chunk = 100
    valid = np.flatnonzero(~np.isnan(Q_obs))
    for start in range(0, N_ROTATIONS, chunk):
        m = min(chunk, N_ROTATIONS - start)
        # Per chromosome: rotate its own whitened residual (fixed norm),
        # evaluate only that chromosome's blocks; minP over all blocks per draw.
        p_star = np.full((n_blocks, m), np.nan)
        for c in range(1, 6):
            rc = rot[c]
            g = RNG.standard_normal((n - p_cov, m))
            g *= rc["r_norm"] / np.linalg.norm(g, axis=0)
            yt_star = rc["Xt_beta"][:, None] + rc["B"] @ g
            y_star = rc["L"] @ yt_star            # n x m
            idx_c = np.flatnonzero((block_chrom == c) & (~np.isnan(Q_obs)))
            V = np.einsum("bpn,nm->bpm", ZP_pad[idx_c], y_star)
            Qs = (V ** 2).sum(axis=1) / nulls[c]["sigma2"]
            p_star[idx_c, :] = stats.chi2.sf(Qs / scale_arr[idx_c, None],
                                             df_arr[idx_c, None])
            exceed[idx_c] += (Qs >= Q_obs[idx_c, None]).sum(axis=1)
        minp_null[start:start + m] = np.nanmin(p_star, axis=0)
        if (start + m) % 200 == 0:
            print(f"  rotations {start + m}/{N_ROTATIONS}", flush=True)
    res["p_rotation"] = np.nan
    res.loc[valid, "p_rotation"] = (1 + exceed[valid]) / (N_ROTATIONS + 1)
    minp_obs_satt = float(res["p_satt"].min())
    minp_adj = float((1 + (minp_null <= minp_obs_satt).sum()) / (N_ROTATIONS + 1))
    res["minp_fwer"] = np.nan
    res.loc[valid, "minp_fwer"] = np.minimum(
        1.0, (1 + (minp_null[None, :] <= res.loc[valid, "p_satt"].to_numpy()[:, None]).sum(axis=1))
        / (N_ROTATIONS + 1))
    np.savez_compressed(TABLES / "rotation_null.npz",
                        minp_null=minp_null, Q_obs=Q_obs, exceed=exceed)

    # ---------------- calibration diagnostics ----------------
    p_valid = res["p_imhof"].dropna().to_numpy()
    lambda_gc = float(stats.chi2.isf(np.median(p_valid), 1) / stats.chi2.ppf(0.5, 1))
    union_bound = float(n_blocks * res["p_imhof"].min())
    res.to_csv(TABLES / "block_scan_imhof.csv", index=False)

    # DOG1 region (known locus recovery, grade D)
    dog1 = res[(res["chrom"] == 5) & (res["start"] <= 18_591_506)
               & (res["end"] >= 18_589_418)]
    dog1_info = []
    for _, row in dog1.iterrows():
        rank = int((res["p_imhof"] < row["p_imhof"]).sum() + 1)
        dog1_info.append({"block_id": int(row["block_id"]),
                          "p_imhof": float(row["p_imhof"]),
                          "rank": rank, "percentile": float(100 * rank / len(p_valid))})

    summary = {
        "description": "Block association scan (raw-scale score test, LOCO-LMM)",
        "date": "2026-11-07",
        "covariates": "intercept + PC1-3 (frozen); PC1-5 sensitivity separate",
        "n_blocks_tested": int(len(p_valid)),
        "n_blocks_no_snps": int(n_blocks - len(p_valid)),
        "null_models": {str(c): {k: nulls[c][k] for k in
                                 ["delta", "sigma2", "eig_min", "n_snps", "at_boundary"]}
                        for c in range(1, 6)},
        "snp_lmm": {"n_tested": int(snp_df["p"].notna().sum()),
                    "min_p": float(snp_df["p"].min()),
                    "n_q005": int((snp_df["q_bh"] < 0.05).sum()),
                    "gls_verification_max_rel_diff": max_diff},
        "primary_raw": {
            "min_p_imhof": float(res["p_imhof"].min()),
            "min_p_satt": minp_obs_satt,
            "n_bh005": int((res["q_bh"] < 0.05).sum()),
            "n_bh010": int((res["q_bh"] < 0.10).sum()),
            "minp_fwer_adjusted": minp_adj,
            "lambda_gc": lambda_gc,
            "union_bound_n_times_minp": union_bound,
            "top_block": int(res.loc[res["p_imhof"].idxmin(), "block_id"]),
        },
        "secondary_graph": {"min_p": float(res["p_graph_imhof"].min()),
                            "n_bh005": int((res["q_graph_bh"] < 0.05).sum())},
        "secondary_embedding": {"min_p": float(res["p_emb_imhof"].min()),
                                "n_bh005": int((res["q_emb_bh"] < 0.05).sum())},
        "imhof_verification": imhof_check,
        "dog1_region": dog1_info,
        "rotation": {"n_draws": N_ROTATIONS,
                     "scheme": "fixed-norm rotation in whitened LOCO residual space; "
                               "conditional on REML estimates (not refitted per draw)"},
        "elapsed_seconds": time.time() - t0,
    }
    (TABLES / "association_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["primary_raw"], indent=2), flush=True)
    print(f"[scan] done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()