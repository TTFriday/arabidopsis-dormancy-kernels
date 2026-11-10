"""Paired pretraining-vs-control comparison with per-repeat statistics.

Reproduces the no-pretraining control (random-initialized encoder, same seed,
no gradient updates) and the trained-encoder evaluation under the identical
corrected kernel protocol, retaining per-repeat correlations so the two can
be compared with paired tests. Outputs are written under results/sensitivity/."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "sensitivity"
SEED = 20261024

sys.path.insert(0, str(ROOT / "scripts"))
from kernel_utils import feature_kernel_centered_scaled  # noqa: E402
from kernel_experiments import graph_block_kernel, nested_cv_single  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    t0 = time.time()
    torch.set_num_threads(2)
    ssl = load_module("ssl06", ROOT / "scripts" / "06_self_supervised_pretrain.py")
    feats, src, dst, efeat, _, _ = ssl.build_graph_tensors()
    cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")

    torch.manual_seed(SEED)
    model = ssl.MaskedGraphEncoder(
        in_dim=feats.shape[2], d=16, heads=4, n_layers=2,
        edge_dim=efeat.shape[1], dropout=0.10,
    )
    model.eval()
    nodes, _ = ssl.encode_all(
        model, torch.from_numpy(feats[cols]), src, dst, efeat, batch_size=16,
    )

    import pandas as pd
    y = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")["dsds50"].to_numpy(float)
    trained_nodes = np.load(PROCESSED / "graph_embeddings_83.npy")

    K_control = graph_block_kernel(nodes.astype(np.float64))
    K_trained = graph_block_kernel(trained_nodes.astype(np.float64))
    res_c, preds_c = nested_cv_single(K_control, y)
    res_t, preds_t = nested_cv_single(K_trained, y)

    rc = np.array(res_c["r_per_repeat"])
    rt = np.array(res_t["r_per_repeat"])
    t = stats.ttest_rel(rt, rc)
    w = stats.wilcoxon(rt, rc)
    out = {
        "control": res_c,
        "trained": res_t,
        "paired_trained_vs_control": {
            "mean_diff": float((rt - rc).mean()),
            "ttest_p": float(t.pvalue),
            "wilcoxon_p": float(w.pvalue),
            "n_repeats_trained_higher": int((rt > rc).sum()),
        },
        "elapsed_seconds": time.time() - t0,
    }
    (OUT / "ssl_control_paired.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT / "ssl_control_preds.npz",
                        control_preds=preds_c, trained_preds=preds_t, y=y)
    print(json.dumps(out["paired_trained_vs_control"], indent=2))
    print(f"control r={res_c['pearson_mean']:.4f}, trained r={res_t['pearson_mean']:.4f}")


if __name__ == "__main__":
    main()