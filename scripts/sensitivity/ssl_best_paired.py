"""Post-sweep analysis: select the pretrained configuration by held-out
reconstruction MSE (an unsupervised criterion that uses no phenotype
information), rebuild its block kernel, and run paired comparisons against
the block-mean and local per-block RBF kernels under the identical
10-repeat x 5-fold nested CV protocol.

Outputs results/ssl_best_paired.json and ssl_best_preds.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import kernel_experiments as ke

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results" / "sensitivity"
SWEEP = RES / "ssl_sweep"


def per_repeat_r(preds, y):
    return np.array([stats.pearsonr(p, y)[0] for p in preds])


def paired(a, b):
    diff = a - b
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    return {
        "mean_diff": float(diff.mean()),
        "t_p": float(t.pvalue),
        "wilcoxon_p": float(w.pvalue),
        "wins": int((diff > 0).sum()),
        "n": int(len(diff)),
    }


def main():
    summary = json.loads((SWEEP / "ssl_sweep_summary.json").read_text(encoding="utf-8"))
    # Unsupervised selection within the primary 15%-masking arm: lowest
    # held-out masked-reconstruction MSE. The 30%-masking arm is a harder
    # pretext task, so its MSE is not comparable across arms.
    primary = [k for k in summary if summary[k]["mask"] == 0.15]
    best_name = min(primary, key=lambda k: summary[k]["best_val_mse"])
    cfg = summary[best_name]
    nodes = np.load(SWEEP / best_name / "graph_embeddings_83.npy").astype(np.float64)

    data = ke.load_data()
    y = data["y"]
    K_best = ke.graph_block_kernel(nodes)
    res, preds = ke.nested_cv_single(K_best, y)

    ref = np.load(RES / "kernel_experiments_preds.npz")
    r_best = per_repeat_r(preds, y)
    r_block = per_repeat_r(ref["block_preds"], y)
    r_local = per_repeat_r(ref["block_rbf_perblock_preds"], y)

    K_block, _, _, _ = ke.feature_kernel_centered_scaled(data["block_mean"])
    tri = np.triu_indices(len(y), k=1)
    sim_block = float(np.corrcoef(
        ke.center_trace(K_best)[tri], ke.center_trace(K_block)[tri])[0, 1])

    # Paired tests for every configuration (exploratory capacity analysis).
    all_paired = {}
    for name in summary:
        nd = np.load(SWEEP / name / "graph_embeddings_83.npy").astype(np.float64)
        _, pr = ke.nested_cv_single(ke.graph_block_kernel(nd), y)
        rr = per_repeat_r(pr, y)
        all_paired[name] = {
            "r_mean": float(rr.mean()),
            "r_sd": float(rr.std()),
            "vs_block_mean": paired(rr, r_block),
            "vs_local_rbf": paired(rr, r_local),
        }

    out = {
        "selection_rule": "lowest held-out masked-reconstruction MSE within the primary 15%-masking arm (no phenotype used); 30%-masking arm reported separately",
        "selected_config": best_name,
        "best_val_mse": cfg["best_val_mse"],
        "prediction": {k: v for k, v in res.items() if k != "r_per_repeat"},
        "paired_vs_block_mean": paired(r_best, r_block),
        "paired_vs_local_rbf": paired(r_best, r_local),
        "kernel_similarity_vs_block_mean": sim_block,
        "all_configs": {
            k: {
                "best_val_mse": v["best_val_mse"],
                "block_r_mean": v["prediction"]["block_kernel"]["pearson_mean"],
                "block_r_sd": v["prediction"]["block_kernel"]["pearson_sd"],
                "block_mae_mean": v["prediction"]["block_kernel"]["mae_mean"],
                "paired": all_paired[k],
            }
            for k, v in summary.items()
        },
    }
    (RES / "ssl_best_paired.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez(RES / "ssl_best_preds.npz", y=y, preds=preds, r_per_repeat=r_best)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()