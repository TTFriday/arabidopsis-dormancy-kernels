"""No-pretraining (random initialization) encoder control under the corrected kernel pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nested_cv_prediction as P12
from kernel_utils import feature_kernel_centered_scaled

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
SEED = 20261024


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_kernel(nodes):
    n_blocks = nodes.shape[1]
    K = np.zeros((nodes.shape[0], nodes.shape[0]))
    for b in range(n_blocks):
        z = nodes[:, b, :].astype(np.float64)
        Kb, _, _, _ = feature_kernel_centered_scaled(z)
        K += Kb
    return K / n_blocks


def main():
    t0 = time.time()
    torch.set_num_threads(1)
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
    nodes = nodes.astype(np.float64)

    data = P12.load_data()
    y = data["y"]
    K_snp, _, _, _ = feature_kernel_centered_scaled(data["X_snp"])
    K_block, _, _, _ = feature_kernel_centered_scaled(data["block_mean"])
    K_trained = graph_kernel(data["graph_nodes"].astype(np.float64))
    K_random = graph_kernel(nodes)

    res = P12.nested_cv_kernel(K_random, y)
    res_trained = P12.nested_cv_kernel(K_trained, y)
    tri = np.triu_indices(len(y), k=1)

    def similarity(A, B):
        return float(np.corrcoef(A[tri], B[tri])[0, 1])

    out = {
        "control": "random-initialised encoder, no pretraining (corrected kernels)",
        "seed": SEED,
        "n_repeats": 10,
        "pearson_mean": float(res["pearson_mean"]),
        "pearson_sd": float(res["pearson_sd"]),
        "spearman_mean": float(res["spearman_mean"]),
        "mae_mean": float(res["mae_mean"]),
        "trained_ssl_pearson_mean": float(res_trained["pearson_mean"]),
        "trained_ssl_pearson_sd": float(res_trained["pearson_sd"]),
        "kernel_similarity": {
            "snp_vs_block": similarity(K_snp, K_block),
            "block_vs_graph_trained": similarity(K_block, K_trained),
            "block_vs_graph_random": similarity(K_block, K_random),
            "graph_trained_vs_random": similarity(K_trained, K_random),
        },
        "elapsed_seconds": time.time() - t0,
    }
    (TABLES / "no_pretraining_control.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()