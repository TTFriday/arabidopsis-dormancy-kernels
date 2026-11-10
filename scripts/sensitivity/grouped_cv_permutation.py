"""Permutation calibration of the grouped (cross-subpopulation) CV.

Holding out whole genetic clusters gives correlations around zero or below.
Whether a negative value is beyond the null was never tested, and a
leave-one-group-out prediction vector is not an independent sample, so a
Fisher-z interval alone is not enough. Here the phenotype is permuted 500
times with the cluster structure held fixed: the same kernels, the same Ward
clusters from the kinship matrix, the same GroupKFold partition and the same
fixed lambda as the published grouped experiment. The permutation distribution
of the grouped r therefore isolates what grouped CV gives when genotype and
phenotype are unrelated on this panel with these clusters.

Outputs, for k = 6, 8, 10 clusters and the block-mean and SNP kernels:
  * observed grouped r (reproduced),
  * null mean / sd / 2.5-97.5 percentile band,
  * one-sided p for observing r at or below the null (a genuinely
    anti-predictive partition) and two-sided p for |r|.

Result: results/sensitivity/grouped_permutation.json"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "sensitivity"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import kernel_experiments as ke  # noqa: E402
from kernel_utils import (  # noqa: E402
    feature_kernel_centered_scaled,
    solve_kernel_ridge_with_intercept,
)

N_PERM = 500
PERM_SEED = 20261109
LAMBDA_GROUPED = 1e-4  # fixed value used by the published grouped experiment
K_VALUES = (6, 8, 10)


def ward_groups(kinship, n_groups):
    from scipy.cluster.hierarchy import fcluster, linkage

    dist = 1 - kinship / kinship.max()
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(condensed, method="average")
    return fcluster(Z, t=n_groups, criterion="maxclust")


def grouped_r(K, y, groups, lam=LAMBDA_GROUPED):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(len(np.unique(groups)), 5)).split(K, y, groups):
        _, _, pred[te] = solve_kernel_ridge_with_intercept(
            K[np.ix_(tr, tr)], y[tr], lam, K[np.ix_(te, tr)]
        )
    return float(stats.pearsonr(pred, y)[0]), pred


def null_distribution(K, y, groups, n_perm, seed):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i], _ = grouped_r(K, yp, groups)
    return null


def summarise(observed, null, label):
    n = len(null)
    p_lower = (1 + int((null <= observed).sum())) / (n + 1)
    p_abs = (1 + int((np.abs(null) >= abs(observed)).sum())) / (n + 1)
    return {
        "label": label,
        "observed_r": observed,
        "n_permutations": n,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
        "null_min": float(null.min()),
        "null_max": float(null.max()),
        "p_one_sided_below_null": float(p_lower),
        "p_two_sided_abs": float(p_abs),
        "z_score_vs_null": float((observed - null.mean()) / null.std(ddof=1)),
    }


def main():
    t0 = time.time()
    data = ke.load_data()
    y = data["y"]
    kinship = np.load(ROOT / "data" / "processed" / "kinship_83.npy")
    K_block, _, _, _ = feature_kernel_centered_scaled(data["block_mean"])
    K_snp, _, _, _ = feature_kernel_centered_scaled(data["X_snp"])

    out = {
        "protocol": (
            f"leave-one-cluster-out GroupKFold, fixed lambda={LAMBDA_GROUPED}, "
            f"{N_PERM} phenotype permutations with clusters fixed, seed {PERM_SEED}"
        ),
        "ward": {},
    }

    for k in K_VALUES:
        groups = ward_groups(kinship, k)
        for mname, K in [("block", K_block), ("snp", K_snp)]:
            obs, _ = grouped_r(K, y, groups)
            null = null_distribution(K, y, groups, N_PERM, PERM_SEED)
            out["ward"][f"k{k}_{mname}"] = summarise(obs, null, f"Ward k={k} {mname}")
            print(
                f"[ward k={k} {mname}] obs={obs:+.3f} null={null.mean():+.3f}"
                f"+-{null.std(ddof=1):.3f} p_low={out['ward'][f'k{k}_{mname}']['p_one_sided_below_null']:.3f} "
                f"p_two={out['ward'][f'k{k}_{mname}']['p_two_sided_abs']:.3f}",
                flush=True,
            )
        out["ward"][f"k{k}_group_sizes"] = np.bincount(groups)[1:].tolist()

    out["elapsed_seconds"] = time.time() - t0
    (OUT / "grouped_permutation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone in {out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()