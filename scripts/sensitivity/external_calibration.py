"""Protocol-aware analysis of the frozen-model Dryad projection.

The original external projection reported weak agreement (r=0.150) on 168
independent accessions phenotyped under a different seed batch / chilling
protocol. This script adds three legitimate, leakage-free analyses:

1. Scale sensitivity: agreement on log1p(days), since the Dryad distribution
   is strongly right-skewed (median 35, max 725).
2. Protocol calibration: the 38 accessions phenotyped under BOTH protocols
   are used to estimate a local->Dryad mapping (linear and isotonic). The
   mapping is fitted exclusively on the overlap set and applied to the 168
   independent accessions, whose phenotypes never enter training or
   calibration. Linear calibration leaves Pearson r unchanged by construction
   but can substantially reduce MAE/RMSE.
3. Repeatability ceiling: cross-protocol phenotypic correlation on the same
   38 accessions bounds the agreement any genetic predictor can achieve
   across protocols.

All inputs are read-only; outputs go to results/sensitivity/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "sensitivity"


def metrics(pred, obs):
    return {
        "n": int(len(obs)),
        "pearson_r": float(stats.pearsonr(pred, obs)[0]),
        "spearman_rho": float(stats.spearmanr(pred, obs).statistic),
        "mae": float(np.mean(np.abs(pred - obs))),
        "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
    }


def main():
    pred = pd.read_csv(TABLES / "external_predictions.csv")
    local = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    local_map = dict(zip(local["nativename"], local["dsds50"]))

    overlap = pred[pred["in_train83"]].copy()
    indep = pred[~pred["in_train83"]].copy()
    overlap["dsds50_local"] = overlap["nativename"].map(local_map)
    assert overlap["dsds50_local"].notna().all()

    out = {
        "cohort": {"overlap": len(overlap), "independent": len(indep)},
        "baseline_independent": {},
        "log_scale_independent": {},
        "repeatability_overlap": {},
        "calibration": {},
        "sensitivity_independent": {},
    }

    for k in ["block", "snp"]:
        p = indep[f"pred_{k}"].to_numpy()
        y = indep["dsds50_dryad"].to_numpy()
        out["baseline_independent"][k] = metrics(p, y)
        out["log_scale_independent"][k] = metrics(np.log1p(p), np.log1p(y))

    yl = overlap["dsds50_local"].to_numpy()
    yd = overlap["dsds50_dryad"].to_numpy()
    out["repeatability_overlap"] = {
        "pearson_r": float(stats.pearsonr(yl, yd)[0]),
        "spearman_rho": float(stats.spearmanr(yl, yd).statistic),
        "mean_local": float(yl.mean()),
        "mean_dryad": float(yd.mean()),
        "note": "cross-protocol repeatability bounds achievable genetic agreement",
    }

    # --- calibration fitted ONLY on the overlap accessions ---
    lin = stats.linregress(yl, yd)
    iso_pheno = IsotonicRegression(out_of_bounds="clip").fit(yl, yd)
    for k in ["block", "snp"]:
        p_ov = overlap[f"pred_{k}"].to_numpy()
        iso_pred = IsotonicRegression(out_of_bounds="clip").fit(p_ov, yd)
        p = indep[f"pred_{k}"].to_numpy()
        y = indep["dsds50_dryad"].to_numpy()
        out["calibration"][k] = {
            "linear_pheno_map": {
                "slope": float(lin.slope),
                "intercept": float(lin.intercept),
                "overlap_fit_r": float(lin.rvalue),
                "independent": metrics(lin.intercept + lin.slope * p, y),
            },
            "isotonic_pheno_map": {"independent": metrics(iso_pheno.predict(p), y)},
            "isotonic_pred_map": {"independent": metrics(iso_pred.predict(p), y)},
        }

    # --- sensitivity: exclude extreme Dryad values (secondary, descriptive) ---
    y = indep["dsds50_dryad"].to_numpy()
    for cap in [365.0, 200.0]:
        keep = y <= cap
        out["sensitivity_independent"][f"cap_{int(cap)}d"] = {
            "n_kept": int(keep.sum()),
            "n_excluded": int((~keep).sum()),
            "block": metrics(indep["pred_block"].to_numpy()[keep], y[keep]),
        }

    (OUT / "external_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()