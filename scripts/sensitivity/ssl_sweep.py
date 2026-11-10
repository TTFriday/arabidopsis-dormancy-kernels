"""Graph-transformer pretraining configuration sweep (GPU).

The published encoder (d=16, 2 layers, 25 epochs, mask 15%) reached a masked
reconstruction MSE of ~0.90 on standardized features (variance ~1), i.e. it
explained little of the masked signal. This sweep tests whether a longer
schedule and larger capacity produce genuinely better haplotype embeddings,
and whether any improvement transfers to dormancy prediction.

Encoders, embeddings, logs and metrics are written under
results/sensitivity/ssl_sweep/.

For every configuration the downstream evaluation is identical to the formal
prediction experiment: per-block linear kernels averaged over blocks, and an
RBF kernel on pooled sample embeddings, both assessed with 10x5 nested CV."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "sensitivity" / "ssl_sweep"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spec = importlib.util.spec_from_file_location(
    "ssl_pretrain", ROOT / "scripts" / "06_self_supervised_pretrain.py"
)
ssl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssl_mod)

CONFIGS = [
    {"name": "d16_l2_m15_e100", "d": 16, "layers": 2, "mask": 0.15, "epochs": 100},
    {"name": "d32_l2_m15_e100", "d": 32, "layers": 2, "mask": 0.15, "epochs": 100},
    {"name": "d64_l2_m15_e100", "d": 64, "layers": 2, "mask": 0.15, "epochs": 100},
    {"name": "d32_l3_m30_e100", "d": 32, "layers": 3, "mask": 0.30, "epochs": 100},
    {"name": "d64_l3_m30_e100", "d": 64, "layers": 3, "mask": 0.30, "epochs": 100},
]


def make_mask(x, ratio, generator):
    s, n, _ = x.shape
    mask = torch.rand(s, n, generator=generator) < ratio
    for i in range(s):
        if mask[i].sum() < 8:
            idx = torch.randperm(n, generator=generator)[:8]
            mask[i, idx] = True
    return mask


def run_epoch(model, x, src, dst, efeat, optimizer, batch_size, generator,
              train, mask_ratio, fixed_mask=None):
    model.train() if train else model.eval()
    order = torch.randperm(x.shape[0], generator=generator) if train else torch.arange(x.shape[0])
    losses = []
    for start in range(0, x.shape[0], batch_size):
        idx = order[start:start + batch_size]
        xb = x[idx].to(DEVICE)
        if train:
            mask = make_mask(xb, mask_ratio, generator).to(DEVICE)
        else:
            mask = fixed_mask[idx].to(DEVICE)
        if mask.sum() == 0:
            continue
        with torch.set_grad_enabled(train):
            pred = model(xb, mask, src, dst, efeat)
            loss = F.mse_loss(pred[mask], xb[mask])
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else float("nan")


def encode(model, x, src, dst, efeat, dorm_cols, batch_size=16):
    """Pool sample embeddings for all accessions; keep node embeddings only
    for the phenotyped subset to bound memory at large d."""
    model.eval()
    sample_parts = []
    node_parts = []
    col_set = set(int(c) for c in dorm_cols)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(DEVICE)
            mask = torch.zeros(xb.shape[:2], dtype=torch.bool, device=DEVICE)
            _, h = model(xb, mask, src, dst, efeat, return_nodes=True)
            h = h[:, :x.shape[1]]
            sample_parts.append(
                torch.cat([h.mean(1), h.max(1).values, h.std(1)], dim=1).cpu()
            )
            rows = [i for i in range(start, start + xb.shape[0]) if i in col_set]
            if rows:
                local = [i - start for i in rows]
                node_parts.append((rows, h[local].cpu()))
    sample = torch.cat(sample_parts).numpy().astype(np.float32)
    nodes = np.zeros((len(dorm_cols), x.shape[1], model.inp.out_features), dtype=np.float32)
    pos = {int(c): i for i, c in enumerate(dorm_cols)}
    for rows, arr in node_parts:
        for r, a in zip(rows, arr):
            nodes[pos[r]] = a.numpy()
    return nodes, sample


def train_config(cfg, feats, src, dst, efeat, train_idx, val_idx, dorm_cols):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cdir = OUT / cfg["name"]
    cdir.mkdir(parents=True, exist_ok=True)

    model = ssl_mod.MaskedGraphEncoder(
        in_dim=feats.shape[2], d=cfg["d"], heads=4, n_layers=cfg["layers"],
        edge_dim=int(efeat.shape[1]), dropout=0.10,
    ).to(DEVICE)
    train_x = torch.from_numpy(feats[train_idx])
    val_x = torch.from_numpy(feats[val_idx])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    gen = torch.Generator().manual_seed(SEED)
    val_mask = torch.rand(val_x.shape[:2], generator=torch.Generator().manual_seed(SEED + 1)) < cfg["mask"]

    log = []
    best_val, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        tr = run_epoch(model, train_x, src, dst, efeat, optimizer, 32, gen, True, cfg["mask"])
        va = run_epoch(model, val_x, src, dst, efeat, None, 32, gen, False, cfg["mask"], val_mask)
        scheduler.step()
        log.append({"epoch": epoch, "train_mse": tr, "val_mse": va, "elapsed_s": time.time() - t0})
        if va < best_val:
            best_val, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{cfg['name']}] epoch {epoch:3d}: train={tr:.4f} val={va:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)

    nodes, sample = encode(model, torch.from_numpy(feats), src, dst, efeat, dorm_cols)
    np.save(cdir / "graph_embeddings_83.npy", nodes)
    np.save(cdir / "sample_embeddings_1307.npy", sample)
    torch.save({"config": cfg, "state_dict": model.state_dict()}, cdir / "encoder.pt")
    import pandas as pd
    pd.DataFrame(log).to_csv(cdir / "training_log.csv", index=False)
    return {"best_val_mse": float(best_val), "runtime_seconds": time.time() - t0,
            "final_train_mse": log[-1]["train_mse"]}


def evaluate(nodes, sample_emb, y):
    from kernel_experiments import (
        GAMMAS, center_trace, graph_block_kernel, nested_cv_multi,
        nested_cv_single, rbf_kernel,
    )
    K = graph_block_kernel(nodes)
    block_res, _ = nested_cv_single(K, y)
    cand = [(f"g{g:g}", center_trace(rbf_kernel(sample_emb, g))) for g in GAMMAS]
    sample_res, _ = nested_cv_multi(cand, y)
    return {
        "block_kernel": {k: v for k, v in block_res.items() if k != "r_per_repeat"},
        "sample_rbf": {k: v for k, v in sample_res.items() if k != "r_per_repeat"},
    }


def main():
    import pandas as pd
    print(f"device: {DEVICE}", flush=True)
    feats, src, dst, efeat, mu, sd = ssl_mod.build_graph_tensors()
    src, dst, efeat = src.to(DEVICE), dst.to(DEVICE), efeat.to(DEVICE)
    n_acc = feats.shape[0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_acc)
    n_val = max(64, int(round(0.1 * n_acc)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    dorm_cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    y = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")["dsds50"].to_numpy(float)

    summary = {}
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===", flush=True)
        info = train_config(cfg, feats, src, dst, efeat, train_idx, val_idx, dorm_cols)
        cdir = OUT / cfg["name"]
        nodes = np.load(cdir / "graph_embeddings_83.npy")
        sample = np.load(cdir / "sample_embeddings_1307.npy")[dorm_cols]
        info["prediction"] = evaluate(nodes, sample, y)
        summary[cfg["name"]] = {**cfg, **info}
        (OUT / "ssl_sweep_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  val_mse={info['best_val_mse']:.4f} "
              f"block_r={info['prediction']['block_kernel']['pearson_mean']:.4f} "
              f"sample_rbf_r={info['prediction']['sample_rbf']['pearson_mean']:.4f}", flush=True)
    print("\nSweep complete.", flush=True)


if __name__ == "__main__":
    main()