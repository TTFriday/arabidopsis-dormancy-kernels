"""Masked graph self-supervised pretraining on the 1,307-accession panel.

The encoder is an edge-aware graph Transformer over haplotype blocks. One
virtual node per sample carries global haplotype background. Training masks
15% of blocks and reconstructs their four block features (mean dosage + three
sign-stable PCs). Phenotypes are never used in this script.

Outputs
-------
data/processed/ssl_encoder.pt
data/processed/ssl_summary.json
data/processed/ssl_training_log.csv
data/processed/graph_embeddings_83.npy       (83, n_blocks, d)
data/processed/sample_embeddings_1307.npy    (1307, 3*d)
results/figures/fig3_ssl_training.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"

SEED = 20261024
MASK_RATIO = 0.15


class GraphTransformerLayer(nn.Module):
    def __init__(self, d: int, heads: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        assert d % heads == 0
        self.d = d
        self.heads = heads
        self.dh = d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.edge_msg = nn.Linear(edge_dim, d)
        self.out = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        efeat: torch.Tensor,
    ) -> torch.Tensor:
        # h: (S, N, d); src/dst: (E,); efeat: (E, edge_dim)
        s, n, _ = h.shape
        e = src.numel()
        q = self.q(h)[:, src].view(s, e, self.heads, self.dh)
        k = self.k(h)[:, dst].view(s, e, self.heads, self.dh)
        score = (q * k).sum(-1) + self.edge_bias(efeat).unsqueeze(0)
        score = F.leaky_relu(score, negative_slope=0.2)

        dst_idx = dst.view(1, e, 1).expand(s, -1, self.heads)
        max_score = torch.full(
            (s, n, self.heads), -1e9, dtype=score.dtype, device=score.device
        )
        max_score.scatter_reduce_(1, dst_idx, score, reduce="amax", include_self=True)
        score = torch.exp(score - max_score.gather(1, dst_idx))
        denom = torch.zeros_like(max_score)
        denom.scatter_add_(1, dst_idx, score)
        att = score / (denom.gather(1, dst_idx) + 1e-12)

        msg = (self.v(h)[:, src] + self.edge_msg(efeat).unsqueeze(0)).view(
            s, e, self.heads, self.dh
        )
        agg = torch.zeros(s, n, self.heads, self.dh, dtype=h.dtype, device=h.device)
        agg.scatter_add_(
            1,
            dst_idx.unsqueeze(-1).expand(s, e, self.heads, self.dh),
            att.unsqueeze(-1) * msg,
        )
        agg = agg.view(s, n, self.d)

        h = self.norm1(h + self.dropout(self.out(agg)))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h


class MaskedGraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d: int,
        heads: int,
        n_layers: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.virtual_token = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d, heads, edge_dim, dropout) for _ in range(n_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, in_dim),
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        efeat: torch.Tensor,
        return_nodes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        s, n, _ = x.shape
        h = self.inp(x)
        h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = torch.cat([h, self.virtual_token.view(1, 1, -1).expand(s, 1, -1)], dim=1)
        for layer in self.layers:
            h = layer(h, src, dst, efeat)
        h = self.norm(h)
        block_h = h[:, :n]
        pred = self.decoder(block_h)
        if return_nodes:
            return pred, block_h
        return pred


def build_graph_tensors():
    feats = np.load(PROCESSED / "block_features.npy").astype(np.float32)
    n_acc, n_blocks, n_feat = feats.shape
    edges = np.load(PROCESSED / "block_edges.npz")
    src = edges["edge_index"][0].astype(np.int64)
    dst = edges["edge_index"][1].astype(np.int64)
    attr = edges["edge_attr"].astype(np.float32)

    # standardise each block feature across accessions and blocks
    flat = feats.reshape(-1, n_feat)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd < 1e-6] = 1.0
    feats = np.clip((feats - mu) / sd, -6.0, 6.0)

    # bidirectional message passing + self loops + virtual node
    src_bi = np.concatenate([src, dst])
    dst_bi = np.concatenate([dst, src])
    attr_bi = np.concatenate([attr, attr], axis=0)

    loop = np.arange(n_blocks, dtype=np.int64)
    virt = np.full(n_blocks, n_blocks, dtype=np.int64)
    src_all = np.concatenate([src_bi, loop, virt, loop])
    dst_all = np.concatenate([dst_bi, loop, loop, virt])
    attr_all = np.concatenate(
        [
            attr_bi,
            np.tile(np.array([[0.0, 1.0, 2.0]], dtype=np.float32), (n_blocks, 1)),
            np.tile(np.array([[0.0, 1.0, 3.0]], dtype=np.float32), (n_blocks, 1)),
            np.tile(np.array([[0.0, 1.0, 3.0]], dtype=np.float32), (n_blocks, 1)),
        ]
    )

    cont = attr_all[:, :2].copy()
    cont[:, 0] = np.clip(cont[:, 0], 0.0, 1.0)
    etype = attr_all[:, 2].astype(np.int64)
    onehot = np.eye(4, dtype=np.float32)[etype]
    efeat = np.concatenate([cont, onehot], axis=1)

    return (
        feats,
        torch.from_numpy(src_all),
        torch.from_numpy(dst_all),
        torch.from_numpy(efeat),
        mu.astype(np.float32),
        sd.astype(np.float32),
    )


def masked_batch(x: torch.Tensor, ratio: float, generator: torch.Generator):
    s, n, _ = x.shape
    mask = torch.rand(s, n, generator=generator) < ratio
    # make sure every accession contributes at least 8 masked blocks
    for i in range(s):
        if mask[i].sum() < 8:
            idx = torch.randperm(n, generator=generator)[:8]
            mask[i, idx] = True
    return mask


def run_epoch(
    model: nn.Module,
    x: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    efeat: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    generator: torch.Generator | None,
    train: bool,
    fixed_mask: torch.Tensor | None = None,
) -> float:
    if train:
        model.train()
        order = torch.randperm(x.shape[0], generator=generator)
    else:
        model.eval()
        order = torch.arange(x.shape[0])
    losses: list[float] = []
    for start in range(0, x.shape[0], batch_size):
        idx = order[start : start + batch_size]
        xb = x[idx]
        if train:
            mask = masked_batch(xb, MASK_RATIO, generator)
        else:
            assert fixed_mask is not None
            mask = fixed_mask[idx]
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


def encode_all(
    model: nn.Module,
    x: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    efeat: torch.Tensor,
    batch_size: int = 16,
):
    model.eval()
    node_parts = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            mask = torch.zeros(xb.shape[:2], dtype=torch.bool)
            _, h = model(xb, mask, src, dst, efeat, return_nodes=True)
            node_parts.append(h.cpu())
    nodes = torch.cat(node_parts, dim=0)
    mean = nodes.mean(dim=1)
    mx = nodes.max(dim=1).values
    sd = nodes.std(dim=1)
    sample = torch.cat([mean, mx, sd], dim=1).numpy().astype(np.float32)
    return nodes.numpy(), sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="tiny run for checking")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    feats, src, dst, efeat, mu, sd = build_graph_tensors()
    print(f"[06] graph: {feats.shape[0]} accessions x {feats.shape[1]} blocks; "
          f"{src.numel()} directed edges", flush=True)
    n_acc, n_blocks, in_dim = feats.shape

    config = {
        "in_dim": in_dim,
        "d": 16,
        "heads": 4,
        "n_layers": 2,
        "edge_dim": int(efeat.shape[1]),
        "dropout": 0.10,
        "mask_ratio": MASK_RATIO,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "seed": SEED,
        "n_blocks": n_blocks,
        "n_accessions": n_acc,
        "n_edges": int(src.numel()),
    }
    model = MaskedGraphEncoder(
        in_dim=in_dim,
        d=config["d"],
        heads=config["heads"],
        n_layers=config["n_layers"],
        edge_dim=config["edge_dim"],
        dropout=config["dropout"],
    )

    # accession-level split; no phenotype is involved
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_acc)
    n_val = max(64, int(round(0.1 * n_acc)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    if args.smoke:
        train_idx = train_idx[:64]
        val_idx = val_idx[:16]
    train_x = torch.from_numpy(feats[train_idx])
    val_x = torch.from_numpy(feats[val_idx])
    print(f"[06] train {len(train_idx)}, validation {len(val_idx)} accessions", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 2)
    )
    gen = torch.Generator().manual_seed(SEED)

    log: list[dict] = []
    best_val = float("inf")
    best_state = None
    started = time.time()
    val_mask = torch.rand(val_x.shape[:2], generator=torch.Generator().manual_seed(SEED + 1)) < MASK_RATIO
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_x, src, dst, efeat, optimizer, args.batch_size, gen, True
        )
        val_loss = run_epoch(
            model,
            val_x,
            src,
            dst,
            efeat,
            None,
            args.batch_size,
            gen,
            False,
            fixed_mask=val_mask,
        )
        scheduler.step()
        log.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - started,
            }
        )
        print(
            f"[06] epoch {epoch:02d}/{args.epochs}: train={train_loss:.4f} "
            f"val={val_loss:.4f} ({time.time() - started:.1f}s)",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    log_df = pd.DataFrame(log)
    log_df.to_csv(PROCESSED / "ssl_training_log.csv", index=False)
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
            "feat_mu": mu,
            "feat_sd": sd,
            "train_idx": train_idx,
            "val_idx": val_idx,
        },
        PROCESSED / "ssl_encoder.pt",
    )

    nodes, sample_emb = encode_all(
        model,
        torch.from_numpy(feats),
        src,
        dst,
        efeat,
        batch_size=8 if args.smoke else 16,
    )
    np.save(PROCESSED / "sample_embeddings_1307.npy", sample_emb)
    if not args.smoke:
        dorm_cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
        np.save(
            PROCESSED / "graph_embeddings_83.npy",
            nodes[dorm_cols].astype(np.float32),
        )

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=160)
    ax.plot(log_df["epoch"], log_df["train_mse"], label="Train", color="#3b6ea8")
    ax.plot(log_df["epoch"], log_df["val_mse"], label="Validation", color="#c0504d")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked-block MSE")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_ssl_training.png")
    plt.close(fig)

    summary = {
        **config,
        "train_accessions": int(len(train_idx)),
        "val_accessions": int(len(val_idx)),
        "best_val_mse": float(best_val),
        "final_train_mse": float(log_df["train_mse"].iloc[-1]),
        "epochs_run": int(len(log_df)),
        "runtime_seconds": float(time.time() - started),
        "sample_embedding_dim": int(sample_emb.shape[1]),
        "graph_embedding_83_shape": [int(x) for x in nodes[np.load(PROCESSED / "dormancy_cols_in_1307.npy")].shape]
        if not args.smoke
        else None,
    }
    (PROCESSED / "ssl_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
