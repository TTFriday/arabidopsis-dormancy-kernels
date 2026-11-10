"""Bandwidth ablation and regularisation-grid sensitivity.

Two inference gaps in the published analysis:

* The best model is a per-block Gaussian kernel whose bandwidth is selected
  inside the inner CV, and the selected value concentrates on the smallest
  grid entry (gamma=0.05). As gamma -> 0 the doubly centred per-block kernel
  converges to the centred linear block kernel, because
  (z_i - z_j)^2 = z_i^2 + z_j^2 - 2 z_i z_j and the first two terms are
  removed by double centring. So a small advantage over the linear kernel
  could be an artefact of a nearly linear kernel rather than of nonlinearity.
  Here each bandwidth is *fixed* (never selected), the prediction accuracy is
  compared with the linear block kernel, and the Frobenius / Spearman
  agreement between each centred nonlinear kernel and the centred linear
  kernel is reported, which bounds how nonlinear a given bandwidth can be.

* Every effective kernel selects the smallest lambda on the published grid
  (1e-8), so the reported accuracy sits on a grid boundary. Here the grid is
  extended one and two decades downward and the accuracy of fixed lambdas is
  reported, showing whether the boundary value is a genuine interior optimum
  or a monotone limit.

Result: results/sensitivity/gamma_lambda_sensitivity.json"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "sensitivity"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import kernel_experiments as ke  # noqa: E402
from kernel_utils import feature_kernel_centered_scaled  # noqa: E402

RHO = 0.25
GAMMAS_FIXED = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
LAMBDA_GRIDS = {
    "published_logspace(-8,2,21)": np.logspace(-8, 2, 21),
    "extended_logspace(-10,2,25)": np.logspace(-10, 2, 25),
    "extended_logspace(-12,2,29)": np.logspace(-12, 2, 29),
}
LAMBDA_FIXED = [1e-8, 1e-6, 1e-4, 1e-2]


def paired(diff):
    d = np.asarray(diff, float)
    n = len(d)
    m, s2 = float(d.mean()), float(d.var(ddof=1))
    if s2 < 1e-24:
        return {"mean_diff": m, "sd_diff": 0.0, "cohen_dz": None,
                "p_naive": 1.0, "p_nadeau_bengio": 1.0,
                "note": "per-repeat differences are identical (zero variance)"}
    return {
        "mean_diff": m,
        "sd_diff": float(np.sqrt(s2)),
        "cohen_dz": float(m / np.sqrt(s2)),
        "p_naive": float(2 * stats.t.sf(abs(m / np.sqrt(s2 / n)), n - 1)),
        "p_nadeau_bengio": float(
            2 * stats.t.sf(abs(m / np.sqrt(s2 * (1 / n + RHO))), n - 1)
        ),
    }


def kernel_agreement(Ka, Kb):
    """Frobenius cosine and Spearman agreement between two centred kernels
    (off-diagonal entries only)."""
    n = Ka.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = Ka[iu], Kb[iu]
    cos = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    rel_dev = float(np.abs(a - b).sum() / np.abs(a).sum())
    return {
        "frobenius_cosine": cos,
        "one_minus_cosine": 1.0 - cos,
        "relative_abs_deviation_from_linear": rel_dev,
        "spearman_offdiag": float(stats.spearmanr(a, b).statistic),
    }


def nested_cv_joint(candidates, y, lambdas):
    """Joint (kernel, lambda) inner selection that also records the chosen
    lambda, which ke.nested_cv_multi does not report."""
    from kernel_utils import solve_kernel_ridge_with_intercept

    n = len(y)
    preds = np.zeros((ke.N_REPEATS, n))
    chosen = []
    for rep in range(ke.N_REPEATS):
        for tr, te in KFold(n_splits=ke.N_FOLDS, shuffle=True, random_state=rep).split(y):
            ytr = y[tr]
            kfi = KFold(n_splits=ke.INNER_FOLDS, shuffle=True, random_state=rep * 100)
            best = (np.inf, None, None)
            for name, K in candidates:
                Ktr = K[np.ix_(tr, tr)]
                for lam in lambdas:
                    ip = np.zeros(len(ytr))
                    for itr, iye in kfi.split(Ktr):
                        _, _, ip[iye] = solve_kernel_ridge_with_intercept(
                            Ktr[np.ix_(itr, itr)], ytr[itr], lam, Ktr[np.ix_(iye, itr)]
                        )
                    mse = float(np.mean((ip - ytr) ** 2))
                    if mse < best[0]:
                        best = (mse, name, lam)
            name, lam = best[1], best[2]
            K = dict(candidates)[name]
            _, _, preds[rep, te] = solve_kernel_ridge_with_intercept(
                K[np.ix_(tr, tr)], ytr, lam, K[np.ix_(te, tr)]
            )
            chosen.append((name, float(lam)))
    out = ke.metrics_from_preds(preds, y)
    names = pd.Series([c[0] for c in chosen])
    lams = pd.Series([c[1] for c in chosen])
    out["selection_counts"] = {k: int(v) for k, v in names.value_counts().items()}
    out["lambda_mode"] = float(lams.mode()[0])
    out["lambda_median"] = float(lams.median())
    out["lambda_min"] = float(lams.min())
    out["lambda_at_grid_floor_share"] = float((lams <= lambdas[0] * 1.001).mean())
    return out, preds


def nested_cv_fixed_lambda(K, y, lam):
    n = len(y)
    preds = np.zeros((ke.N_REPEATS, n))
    for rep in range(ke.N_REPEATS):
        for tr, te in KFold(n_splits=ke.N_FOLDS, shuffle=True, random_state=rep).split(K):
            from kernel_utils import solve_kernel_ridge_with_intercept
            _, _, preds[rep, te] = solve_kernel_ridge_with_intercept(
                K[np.ix_(tr, tr)], y[tr], lam, K[np.ix_(te, tr)]
            )
    return ke.metrics_from_preds(preds, y), preds


def main():
    t0 = time.time()
    data = ke.load_data()
    y = data["y"]
    published = json.loads((OUT / "kernel_experiments.json").read_text(encoding="utf-8"))
    r_block = np.array(published["block"]["r_per_repeat"])
    r_local = np.array(published["block_rbf_perblock"]["r_per_repeat"])

    results = {"rho_for_correction": RHO}

    # ---------------- fixed bandwidth ablation ----------------
    print("[gamma] fixed-bandwidth per-block RBF ablation", flush=True)
    K_linear_pub = ke.center_trace(
        feature_kernel_centered_scaled(data["block_mean"])[0]
    )
    # published convention inside this script: standardise, linear kernel, centre,
    # trace-normalise (identical to kernel_experiments.block)
    ablation = {}
    preds_store = {}
    for g in GAMMAS_FIXED:
        K_g = ke.center_trace(ke.perblock_rbf(data["block_mean"], g))
        res, preds = ke.nested_cv_single(K_g, y)
        ablation[f"gamma_{g:g}"] = {
            **{k: v for k, v in res.items()},
            "vs_linear_block": paired(np.array(res["r_per_repeat"]) - r_block),
            "agreement_with_linear_kernel": kernel_agreement(K_g, K_linear_pub),
        }
        preds_store[f"gamma_{g:g}"] = preds
        print(
            f"  gamma={g:<6g} r={res['pearson_mean']:.4f} "
            f"1-cos_lin={ablation[f'gamma_{g:g}']['agreement_with_linear_kernel']['one_minus_cosine']:.2e} "
            f"rel_dev={ablation[f'gamma_{g:g}']['agreement_with_linear_kernel']['relative_abs_deviation_from_linear']:.3f} "
            f"p_vs_block={ablation[f'gamma_{g:g}']['vs_linear_block']['p_naive']:.3f} "
            f"pNB={ablation[f'gamma_{g:g}']['vs_linear_block']['p_nadeau_bengio']:.3f}",
            flush=True,
        )
    results["fixed_gamma_ablation"] = ablation

    # how often does the published inner CV pick each bandwidth, and is the
    # selected bandwidth's kernel close to linear? (reported gamma counts from
    # the published run, recomputed agreement here)
    results["published_gamma_selection"] = published["block_rbf_perblock"].get(
        "inner_selection_counts", {}
    )

    # ---------------- lambda grid sensitivity ----------------
    print("[lambda] lambda grid sensitivity", flush=True)
    K_block, _, _, _ = feature_kernel_centered_scaled(data["block_mean"])
    K_local_cand = [(f"g{g:g}", ke.center_trace(ke.perblock_rbf(data["block_mean"], g)))
                    for g in ke.GAMMAS]
    grid_res = {}
    old_lambdas = ke.LAMBDAS
    for label, grid in LAMBDA_GRIDS.items():
        ke.LAMBDAS = grid
        res_b, preds_b = nested_cv_joint([("block", K_block)], y, grid)
        res_l, preds_l = nested_cv_joint(K_local_cand, y, grid)
        ke.LAMBDAS = old_lambdas
        grid_res[label] = {
            "block": {**{k: v for k, v in res_b.items()},
                      "vs_published_grid": paired(np.array(res_b["r_per_repeat"]) - r_block)},
            "local_rbf": {**{k: v for k, v in res_l.items()},
                          "vs_published_grid": paired(np.array(res_l["r_per_repeat"]) - r_local)},
        }
        print(
            f"  {label}: block r={res_b['pearson_mean']:.4f} lam_mode={res_b['lambda_mode']:.1e} "
            f"floor={res_b['lambda_at_grid_floor_share']:.2f} | "
            f"local r={res_l['pearson_mean']:.4f} lam_mode={res_l['lambda_mode']:.1e} "
            f"floor={res_l['lambda_at_grid_floor_share']:.2f}",
            flush=True,
        )
        preds_store[f"grid_block_{label}"] = preds_b
        preds_store[f"grid_local_{label}"] = preds_l
    results["lambda_grid"] = grid_res

    # fixed-lambda flatness for the block kernel
    flat = {}
    for lam in LAMBDA_FIXED:
        res, preds = nested_cv_fixed_lambda(K_block, y, lam)
        flat[f"lambda_{lam:g}"] = {
            **{k: v for k, v in res.items()},
            "vs_linear_block_selected": paired(np.array(res["r_per_repeat"]) - r_block),
        }
        preds_store[f"fixed_lambda_{lam:g}"] = preds
        print(f"  fixed lambda={lam:<8g} r={res['pearson_mean']:.4f} "
              f"mae={res['mae_mean']:.2f} mae_sd={res['mae_sd']:.2f}", flush=True)
    results["fixed_lambda_block"] = flat

    results["elapsed_seconds"] = time.time() - t0
    (OUT / "gamma_lambda_sensitivity.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(OUT / "gamma_lambda_preds.npz", y=y, **preds_store)
    print(f"\nDone in {results['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()