"""Quick nested-CV baseline: GBLUP and random forest on the 83-accession panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
RNG = np.random.default_rng(20261023)


def load() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    codes = np.load(PROCESSED / "genotypes_83.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    maf = snp["maf_83"].to_numpy()
    keep = np.flatnonzero((maf > 0.05) & (snp["minor_code"].to_numpy() < 4))
    minor = snp["minor_code"].to_numpy()[keep]
    X = (codes[keep, :] == minor[:, None]).astype(np.float32).T  # accessions x SNPs
    return X, pheno["dsds50"].to_numpy(), snp.iloc[keep].reset_index(drop=True)


def gblup_cv(X: np.ndarray, y: np.ndarray, repeats: int = 30) -> dict:
    n = len(y)
    preds = np.zeros((repeats, n))
    chosen_lambda = []
    for rep in range(repeats):
        folds = KFold(5, shuffle=True, random_state=rep)
        for tr, te in folds.split(X):
            Xtr = X[tr]
            Ktr = Xtr @ Xtr.T / X.shape[1]
            Kte = X[te] @ Xtr.T / X.shape[1]
            # inner CV for the ridge penalty
            best = (np.inf, 1.0)
            for lam in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
                inner_pred = np.zeros(len(tr))
                for itr, ite in KFold(5, shuffle=True, random_state=rep + 100).split(tr):
                    Kii = Ktr[np.ix_(itr, itr)]
                    alpha = np.linalg.solve(Kii + lam * np.eye(len(itr)), y[tr][itr] - y[tr][itr].mean())
                    inner_pred[ite] = Ktr[np.ix_(ite, itr)] @ alpha + y[tr][itr].mean()
                mse = np.mean((inner_pred - y[tr]) ** 2)
                if mse < best[0]:
                    best = (mse, lam)
            lam = best[1]
            alpha = np.linalg.solve(Ktr + lam * np.eye(len(tr)), y[tr] - y[tr].mean())
            preds[rep, te] = Kte @ alpha + y[tr].mean()
            chosen_lambda.append(lam)
    r = np.array([stats.pearsonr(preds[i], y)[0] for i in range(repeats)])
    rho = np.array([stats.spearmanr(preds[i], y).statistic for i in range(repeats)])
    mae = np.array([np.mean(np.abs(preds[i] - y)) for i in range(repeats)])
    return {
        "pred_repeats": preds,
        "pearson_mean": float(r.mean()),
        "pearson_sd": float(r.std()),
        "spearman_mean": float(rho.mean()),
        "mae_mean": float(mae.mean()),
        "lambda_mode": float(pd.Series(chosen_lambda).mode()[0]),
    }


def rf_cv(X: np.ndarray, y: np.ndarray, repeats: int = 10) -> dict:
    n = len(y)
    preds = np.zeros((repeats, n))
    for rep in range(repeats):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            model = RandomForestRegressor(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=rep,
            )
            model.fit(X[tr], y[tr])
            preds[rep, te] = model.predict(X[te])
    r = np.array([stats.pearsonr(preds[i], y)[0] for i in range(repeats)])
    rho = np.array([stats.spearmanr(preds[i], y).statistic for i in range(repeats)])
    return {
        "pearson_mean": float(r.mean()),
        "pearson_sd": float(r.std()),
        "spearman_mean": float(rho.mean()),
        "mae_mean": float(np.mean([np.mean(np.abs(preds[i] - y)) for i in range(repeats)])),
    }


def main() -> None:
    X, y, snp = load()
    print(f"X {X.shape}, y mean {y.mean():.1f}")
    gb = gblup_cv(X, y)
    print("GBLUP", {k: v for k, v in gb.items() if k != "pred_repeats"})
    rf = rf_cv(X, y)
    print("RF", rf)

    # permutation null for GBLUP (5 permutations, 10 repeats each)
    perm_r = []
    for s in range(5):
        yp = RNG.permutation(y)
        r = gblup_cv(X, yp, repeats=10)["pearson_mean"]
        perm_r.append(r)
    print("permutation r", perm_r)

    summary = {
        "n": int(len(y)),
        "n_snps": int(X.shape[1]),
        "gblup": {k: v for k, v in gb.items() if k != "pred_repeats"},
        "random_forest": rf,
        "permutation_gblup_r": perm_r,
    }
    (TABLES / "baseline_prediction.json").write_text(json.dumps(summary, indent=2))
    np.save(TABLES / "gblup_cv_predictions.npy", gb["pred_repeats"])


if __name__ == "__main__":
    main()
