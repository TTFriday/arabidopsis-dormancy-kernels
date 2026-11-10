"""Per-repeat prediction curves and kernel diagnostics used by the main figures."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
import nested_cv_prediction as P12

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def pc_regression_preds(X_snp, y, n_pcs=10):
    mean = X_snp.mean(axis=0)
    Xc = X_snp - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PCs = U[:, :n_pcs] * S[:n_pcs]
    preds = np.zeros((P12.N_REPEATS, len(y)))
    for rep in range(P12.N_REPEATS):
        kf = KFold(n_splits=P12.N_FOLDS, shuffle=True, random_state=rep)
        for tr, te in kf.split(PCs):
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            model.fit(PCs[tr], y[tr])
            preds[rep, te] = model.predict(PCs[te])
    return preds


def per_rep_metrics(preds, y):
    r = np.array([stats.pearsonr(preds[i], y)[0] for i in range(len(preds))])
    mae = np.array([np.mean(np.abs(preds[i] - y)) for i in range(len(preds))])
    return r, mae


def main():
    start = time.time()
    data = P12.load_data()
    y = data["y"]
    kernels = P12.build_kernels(data)

    out = {"y": y}

    # PC regression baseline
    pc_preds = pc_regression_preds(data["X_snp"], y)
    r, mae = per_rep_metrics(pc_preds, y)
    out["pc_r"], out["pc_mae"] = r, mae
    print(f"PC regression: r={r.mean():.4f} +/- {r.std():.4f}")

    # Six kernel models
    for key in ["snp", "block", "ssl_graph", "random_graph",
                "combined_w01", "combined_w05"]:
        res = P12.nested_cv_kernel(kernels[key], y)
        r, mae = per_rep_metrics(res["preds"], y)
        out[f"{key}_r"], out[f"{key}_mae"] = r, mae
        out[f"{key}_preds"] = res["preds"]
        print(f"{key}: r={r.mean():.4f} +/- {r.std():.4f}, "
              f"MAE={mae.mean():.2f} +/- {mae.std():.2f}")

    # Ablation sweep: w in [0, 1]; reuse w=0 (block), 0.1, 0.5, 1 (snp)
    weights = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    abl_r_mean, abl_r_sd = [], []
    K_snp, K_block = kernels["snp"], kernels["block"]
    for w in weights:
        if w == 0.0:
            r = out["block_r"]
        elif w == 1.0:
            r = out["snp_r"]
        elif w == 0.1:
            r = out["combined_w01_r"]
        elif w == 0.5:
            r = out["combined_w05_r"]
        else:
            K_w = w * K_snp + (1 - w) * K_block
            res = P12.nested_cv_kernel(K_w, y)
            r, _ = per_rep_metrics(res["preds"], y)
        abl_r_mean.append(float(r.mean()))
        abl_r_sd.append(float(r.std()))
        print(f"ablation w={w:.2f}: r={r.mean():.4f} +/- {r.std():.4f}")
    out["abl_weights"] = np.array(weights)
    out["abl_r_mean"] = np.array(abl_r_mean)
    out["abl_r_sd"] = np.array(abl_r_sd)

    # Consistency check against the published nested-CV JSON
    ref = json.loads((TABLES / "nested_cv_prediction.json").read_text(encoding="utf-8"))
    print("\nConsistency check (regenerated vs published):")
    ok = True
    for key in ["pc_regression", "snp", "block", "ssl_graph", "random_graph",
                "combined_w01", "combined_w05"]:
        arr_key = "pc_r" if key == "pc_regression" else f"{key}_r"
        diff = abs(out[arr_key].mean() - ref["results"][key]["pearson_mean"])
        status = "OK" if diff < 1e-10 else "MISMATCH"
        if diff >= 1e-10:
            ok = False
        print(f"  {key}: diff={diff:.2e} {status}")
    if not ok:
        raise SystemExit("Regenerated values do not match published results!")

    np.savez_compressed(TABLES / "prediction_figure_data.npz", **out)
    print(f"\nSaved: {TABLES / 'prediction_figure_data.npz'}")
    print(f"Elapsed: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()