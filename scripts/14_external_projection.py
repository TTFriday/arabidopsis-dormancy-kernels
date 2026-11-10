"""Frozen-model projection onto the Dryad seed-chilling cohort (phenotypes independent of training)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from kernel_utils import (
    feature_kernel_centered_scaled,
    solve_kernel_ridge_with_intercept,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXTERNAL = ROOT / "data" / "external"
TABLES = ROOT / "results" / "tables"
RAW = ROOT / "data" / "raw" / "atpolydb"
LAMBDA = 1e-8   # frozen from nested CV (mode for all effective kernels)


def norm_name(s):
    return str(s).strip().lower().replace(" ", "")


def load_dryad():
    wb = xlrd.open_workbook(EXTERNAL / "arabidopsis_germ_phen_chilling_DATA.xls")
    ws = wb.sheet_by_index(0)
    rows = []
    for r in range(1, ws.nrows):
        nm = str(ws.cell_value(r, 2)).strip()
        v = ws.cell_value(r, 21)
        try:
            rows.append({"nativename": nm, "dsds50_dryad": float(v)})
        except (ValueError, TypeError):
            pass
    return pd.DataFrame(rows)


def external29_names():
    df = pd.read_csv(RAW / "phenotype_published_raw.tsv", sep="\t",
                     low_memory=False, encoding="latin-1")
    col = "279_DSDS50"
    sub = df[df[col].notna()]
    name_col = [c for c in df.columns if "name" in c.lower()]
    if name_col:
        return set(sub[name_col[0]].astype(str))
    return set()


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "pearson_r": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman_rho": float(stats.spearmanr(y_true, y_pred).statistic),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def main():
    t0 = time.time()
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    y_train = pheno["dsds50"].to_numpy(float)
    acc = pd.read_csv(PROCESSED / "accessions_all.csv")
    cols83 = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    train_names = set(pheno["nativename"].astype(str))
    ext29 = external29_names()

    dryad = load_dryad()
    print(f"Dryad valid DSDS50: {len(dryad)}")

    # name matching (exact, then normalised alias)
    panel_names = acc["nativename"].astype(str).tolist()
    exact = {nm: i for i, nm in enumerate(panel_names)}
    normed = {}
    for i, nm in enumerate(panel_names):
        normed.setdefault(norm_name(nm), i)
    col_idx, match_rule = [], []
    for nm in dryad["nativename"]:
        if nm in exact:
            col_idx.append(exact[nm]); match_rule.append("exact")
        elif norm_name(nm) in normed:
            col_idx.append(normed[norm_name(nm)]); match_rule.append("normalised")
        else:
            col_idx.append(-1); match_rule.append("unmatched")
    dryad["panel_col"] = col_idx
    dryad["match_rule"] = match_rule
    dryad["in_train83"] = dryad["nativename"].isin(train_names)
    dryad["in_old29"] = dryad["nativename"].isin(ext29)

    matched = dryad[dryad["panel_col"] >= 0].copy()
    same83 = matched[matched["in_train83"]]
    indep = matched[~matched["in_train83"]]
    print(f"matched: {len(matched)} (exact {(dryad['match_rule']=='exact').sum()}, "
          f"normalised {(dryad['match_rule']=='normalised').sum()})")
    print(f"same83: {len(same83)}, independent: {len(indep)}, "
          f"independent also in old29: {int(indep['in_old29'].sum())}, "
          f"unmatched: {(dryad['panel_col'] < 0).sum()}")

    # ---------------- frozen training features ----------------
    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp_meta = pd.read_csv(PROCESSED / "snp_metadata.csv")
    maf = snp_meta["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp_meta["minor_code"].to_numpy() < 4))
    minor = snp_meta["minor_code"].to_numpy()[keep]
    X_snp_train = (codes[keep, :] == minor[:, None]).astype(np.float32).T

    block_feats = np.load(PROCESSED / "block_features.npy", mmap_mode="r")
    block_train = np.asarray(block_feats[cols83, :, 0], dtype=np.float64)

    # frozen standardisation + kernels
    K_block, Xb_s, b_mean, b_std = feature_kernel_centered_scaled(block_train)
    K_snp, Xs_s, s_mean, s_std = feature_kernel_centered_scaled(X_snp_train)
    alpha_b, int_b, _ = solve_kernel_ridge_with_intercept(K_block, y_train, LAMBDA)
    alpha_s, int_s, _ = solve_kernel_ridge_with_intercept(K_snp, y_train, LAMBDA)

    # ---------------- external features & prediction ----------------
    ext_cols = matched["panel_col"].to_numpy(int)
    block_ext = np.asarray(block_feats[ext_cols, :, 0], dtype=np.float64)
    codes_all = np.load(PROCESSED / "genotypes_all_1307.npy", mmap_mode="r")
    X_snp_ext = (np.asarray(codes_all[keep, :][:, ext_cols]) == minor[:, None]).astype(np.float32).T

    Xb_ext_s = (block_ext - b_mean) / b_std
    Xs_ext_s = (X_snp_ext - s_mean) / s_std
    Kb_cross = Xb_ext_s @ Xb_s.T / Xb_s.shape[1]
    Ks_cross = Xs_ext_s @ Xs_s.T / Xs_s.shape[1]
    pred_block = int_b + Kb_cross @ alpha_b
    pred_snp = int_s + Ks_cross @ alpha_s
    matched["pred_block"] = pred_block
    matched["pred_snp"] = pred_snp

    y_ext = matched["dsds50_dryad"].to_numpy(float)
    is_same = matched["in_train83"].to_numpy(bool)
    results = {
        "independent_transductive": {
            "block": metrics(y_ext[~is_same], pred_block[~is_same]),
            "snp": metrics(y_ext[~is_same], pred_snp[~is_same]),
        },
        "same83_new_environment": {
            "block": metrics(y_ext[is_same], pred_block[is_same]),
            "snp": metrics(y_ext[is_same], pred_snp[is_same]),
        },
    }

    # measurement repeatability: local vs Dryad phenotype on the same 38 accessions
    local_map = dict(zip(pheno["nativename"], pheno["dsds50"]))
    same = matched[matched["in_train83"]]
    y_local = np.array([local_map[nm] for nm in same["nativename"]])
    y_dryad_same = same["dsds50_dryad"].to_numpy(float)
    repeatability = {
        "n": int(len(same)),
        "pearson_r": float(stats.pearsonr(y_local, y_dryad_same)[0]),
        "spearman_rho": float(stats.spearmanr(y_local, y_dryad_same).statistic),
        "mean_local": float(y_local.mean()),
        "mean_dryad": float(y_dryad_same.mean()),
    }

    summary = {
        "description": "P2 frozen-model projection, Dryad DOI:10.25338/B8VS4P",
        "date": "2026-11-07",
        "model": {"kernel": "block-mean (primary), SNP (benchmark)",
                  "lambda_frozen": LAMBDA,
                  "standardisation": "frozen training-set mean/std; no joint refit"},
        "cohort": {
            "dryad_valid": int(len(dryad)),
            "matched": int(len(matched)),
            "same83": int(len(same83)),
            "independent": int(len(indep)),
            "independent_in_old29": int(indep["in_old29"].sum()),
            "unmatched": int((dryad["panel_col"] < 0).sum()),
            "phenotype_scale": {
                "local_mean": float(y_train.mean()),
                "dryad_matched_mean": float(y_ext.mean()),
                "dryad_matched_median": float(np.median(y_ext)),
                "dryad_matched_max": float(y_ext.max()),
            },
            "independence_note": "independent accessions never used for phenotype "
                                 "training but their genotypes participated in SSL "
                                 "pretraining (transductive); Dryad phenotypes are a "
                                 "different seed batch / chilling protocol",
        },
        "results": results,
        "measurement_repeatability_same38": repeatability,
        "elapsed_seconds": time.time() - t0,
    }
    matched.to_csv(TABLES / "external_predictions.csv", index=False)
    dryad.to_csv(TABLES / "external_matching.csv", index=False)
    (TABLES / "external_projection.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("repeatability:", json.dumps(repeatability))
    print(f"done in {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()