"""Fold-wise kernel centring sensitivity.

Every published kernel in this project is centred and trace-normalised on the
full 83-accession feature matrix before cross-validation, so each fold's
kernel statistics involve the held-out accessions (transductive centring).
The candidate accessions' genotypes are legitimately available in a real
breeding panel, so the published numbers are defensible, but it is worth
asking how much of the accuracy depends on that convention.

This experiment repeats the identical 10 x 5 nested CV protocol with all
centring and scaling constants derived from the training rows of each outer
fold only:

  * features are standardised with training-fold means and SDs,
  * the double centring row means and grand mean come from the current
    training block, and the same constants are applied to the cross block,
  * the trace normalisation uses the training block only,
  * for the nonlinear kernels the inner-loop candidate bandwidths are rebuilt
    per fold under the same rule (the inner CV subsets re-derive their
    constants from the inner training rows).

Kernels tested: SNP linear, block-mean linear, block-mean local RBF (bandwidth
selected inside) and SSL block kernel (fixed, since its node embeddings come
from a panel-wide pretraining and are not re-fittable per fold).

Result: results/sensitivity/foldwise_centering.json, with per-repeat r so the
fold-wise and published conventions can be compared directly."""

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
from kernel_utils import solve_kernel_ridge_with_intercept  # noqa: E402

N_REPEATS, N_FOLDS, INNER_FOLDS = 10, 5, 5
LAMBDAS = ke.LAMBDAS
GAMMAS = ke.GAMMAS
BLOCK_CHUNK = 400


def center_constants(K_raw, anchor):
    """Row means / grand mean / trace scale of the raw kernel on the anchor set."""
    Ka = K_raw[np.ix_(anchor, anchor)]
    row_mean = Ka.mean(axis=1)
    total_mean = float(Ka.mean())
    Kc = Ka - row_mean[:, None] - row_mean[None, :] + total_mean
    tr = np.trace(Kc) / len(anchor)
    scale = tr if tr > 1e-10 else 1.0
    return row_mean, total_mean, scale, anchor


def centered_block(K_raw, rows, cols, consts):
    row_mean, total_mean, scale, anchor = consts
    c_r = K_raw[np.ix_(rows, anchor)].mean(axis=1)
    c_c = K_raw[np.ix_(cols, anchor)].mean(axis=1)
    sub = K_raw[np.ix_(rows, cols)] - c_r[:, None] - c_c[None, :] + total_mean
    return sub / scale


def raw_linear(X, train_idx):
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    return Z @ Z.T / Z.shape[1]


def raw_rbf_global(X, train_idx, gamma):
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    sq = np.maximum(
        (Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * Z @ Z.T, 0.0
    )
    return np.exp(-gamma * sq / Z.shape[1])


def raw_perblock_rbf_all(block_mean, train_idx, gammas):
    """Average per-block 1-D RBF kernels for every bandwidth in one pass.

    The squared standardized differences of each block are computed once per
    training fold and reused for all candidate bandwidths.
    """
    n, B = block_mean.shape
    mu = block_mean[train_idx].mean(axis=0)
    sd = block_mean[train_idx].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    acc = {float(g): np.zeros((n, n)) for g in gammas}
    for start in range(0, B, BLOCK_CHUNK):
        stop = start + BLOCK_CHUNK
        z = (block_mean[:, start:stop] - mu[start:stop]) / sd[start:stop]
        d2 = (z[:, None, :] - z[None, :, :]) ** 2
        for g in gammas:
            acc[float(g)] += np.exp(-g * d2).sum(axis=2)
    return {k: v / B for k, v in acc.items()}


def nested_cv_foldwise(raw_fn, y, select_gamma=False):
    """raw_fn(train_idx) -> {gamma_or_None: raw kernel over all samples}.

    All centring / scaling constants are derived from the training rows of the
    current (outer or inner) fold; test rows only enter through their raw
    kernel values against those training rows.
    """
    n = len(y)
    preds = np.zeros((N_REPEATS, n))
    chosen = []
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        for tr, te in kf.split(y):
            raws = raw_fn(tr)
            kfi = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=rep * 100)
            best = (np.inf, None, None)
            for g, K_raw in raws.items():
                for lam in LAMBDAS:
                    ip = np.zeros(len(tr))
                    for itr, iye in kfi.split(tr):
                        a_tr, a_iy = tr[itr], tr[iye]
                        consts = center_constants(K_raw, a_tr)
                        _, _, p = solve_kernel_ridge_with_intercept(
                            centered_block(K_raw, a_tr, a_tr, consts), y[a_tr], lam,
                            centered_block(K_raw, a_iy, a_tr, consts),
                        )
                        ip[iye] = p
                    mse = float(np.mean((ip - y[tr]) ** 2))
                    if mse < best[0]:
                        best = (mse, g, lam)
            _, g, lam = best
            K_raw = raws[g]
            consts = center_constants(K_raw, tr)
            _, _, preds[rep, te] = solve_kernel_ridge_with_intercept(
                centered_block(K_raw, tr, tr, consts), y[tr], lam,
                centered_block(K_raw, te, tr, consts),
            )
            chosen.append({"gamma": (float(g) if g is not None else None),
                           "lambda": float(lam)})
    out = ke.metrics_from_preds(preds, y)
    lam_series = pd.Series([c["lambda"] for c in chosen])
    out["lambda_mode"] = float(lam_series.mode()[0])
    if select_gamma:
        gs = pd.Series([c["gamma"] for c in chosen])
        out["gamma_selection_counts"] = {f"{v:g}": int(k) for v, k in gs.value_counts().items()}
        out["gamma_median"] = float(gs.median())
    return out, preds, chosen


def fixed_kernel_foldwise(K_raw, y):
    """For a kernel whose raw form is already fixed (SSL block kernel): only the
    centring / scaling constants change per fold."""
    return nested_cv_foldwise(lambda tr: {None: K_raw}, y, select_gamma=False)


def main():
    t0 = time.time()
    data = ke.load_data()
    y = data["y"]

    published = json.loads((OUT / "kernel_experiments.json").read_text(encoding="utf-8"))
    results = {
        "protocol": (
            "identical 10x5 nested CV; feature standardisation, double centring "
            "and trace scaling derived from training rows of each outer fold "
            "(inner subsets re-derive constants from inner training rows)"
        ),
        "comparison_with_published_full_sample_centering": {},
    }

    specs = [
        ("snp", lambda tr: {None: raw_linear(data["X_snp"], tr)}),
        ("block", lambda tr: {None: raw_linear(data["block_mean"], tr)}),
        ("block_rbf_perblock",
         lambda tr: raw_perblock_rbf_all(data["block_mean"], tr, GAMMAS)),
    ]
    store = {}
    for name, fn in specs:
        print(f"[foldwise] {name}", flush=True)
        out, preds, _ = nested_cv_foldwise(fn, y, select_gamma=(name == "block_rbf_perblock"))
        results[name] = out
        store[name + "_preds"] = preds
        pub = published[name]
        fw_r = np.array(out["r_per_repeat"])
        pb_r = np.array(pub["r_per_repeat"])
        t = stats.ttest_rel(fw_r, pb_r)
        results["comparison_with_published_full_sample_centering"][name] = {
            "published_r": pub["pearson_mean"],
            "foldwise_r": out["pearson_mean"],
            "delta_r": float(out["pearson_mean"] - pub["pearson_mean"]),
            "published_mae": pub["mae_mean"],
            "foldwise_mae": out["mae_mean"],
            "paired_t_p": float(t.pvalue),
            "sd_r_published": pub["pearson_sd"],
            "sd_r_foldwise": out["pearson_sd"],
        }
        print(f"  r: published={pub['pearson_mean']:.4f} foldwise={out['pearson_mean']:.4f} "
              f"p={t.pvalue:.3f}", flush=True)

    # SSL block kernel: raw kernel fixed, fold-wise centring only
    K_ssl_raw = ke.graph_block_kernel(data["graph_nodes"])
    print("[foldwise] ssl_graph (fixed raw kernel)", flush=True)
    out, preds, _ = fixed_kernel_foldwise(K_ssl_raw, y)
    results["ssl_graph"] = out
    store["ssl_graph_preds"] = preds
    pub = published["ssl_graph"]
    t = stats.ttest_rel(np.array(out["r_per_repeat"]), np.array(pub["r_per_repeat"]))
    results["comparison_with_published_full_sample_centering"]["ssl_graph"] = {
        "published_r": pub["pearson_mean"], "foldwise_r": out["pearson_mean"],
        "delta_r": float(out["pearson_mean"] - pub["pearson_mean"]),
        "published_mae": pub["mae_mean"], "foldwise_mae": out["mae_mean"],
        "paired_t_p": float(t.pvalue),
        "sd_r_published": pub["pearson_sd"], "sd_r_foldwise": out["pearson_sd"],
    }
    print(f"  r: published={pub['pearson_mean']:.4f} foldwise={out['pearson_mean']:.4f} "
          f"p={t.pvalue:.3f}", flush=True)

    # diagnostic: how different are the centring conventions numerically?
    full_block = ke.center_trace(
        (lambda X: (lambda Z: Z @ Z.T / Z.shape[1])((X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))))(data["block_mean"])
    )
    tr0 = np.arange(len(y))
    foldwise_full = centered_block(raw_linear(data["block_mean"], tr0), tr0, tr0,
                                   center_constants(raw_linear(data["block_mean"], tr0), tr0))
    results["kernel_construction_check"] = {
        "max_abs_diff_full_sample_vs_wholesample_foldwise": float(
            np.abs(full_block - foldwise_full).max()
        ),
        "note": (
            "with the whole panel as the anchor set the two conventions must "
            "coincide; this validates the fold-wise implementation"
        ),
    }

    results["elapsed_seconds"] = time.time() - t0
    (OUT / "foldwise_centering.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(OUT / "foldwise_centering_preds.npz", y=y, **store)
    print(json.dumps(results["comparison_with_published_full_sample_centering"], indent=2))
    print(f"Done in {results['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()