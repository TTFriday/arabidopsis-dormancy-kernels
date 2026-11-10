"""Seed replication of the graph-transformer pretraining (GPU).

The published SSL block kernel and the capacity sweep each used a single
random seed, so it is worth asking whether the (small) SSL advantage over the
untrained encoder is a seed artefact. This script repeats the *original*
published configuration (d=16, 2 layers, mask 15%, 25 epochs, batch 8) under
five independent seeds, and the best sweep configuration (d=64, 2 layers,
mask 15%, 100 epochs, batch 32) under three seeds. Every seed change re-draws
the encoder initialisation, the accession train/validation split, the batch
order and the mask draws, i.e. it replicates the whole pipeline, not just the
weight init.

For every (configuration, seed) pair:
  * the pretrained encoder and a paired randomly initialised encoder (identical
    seed, no gradient steps) are both used to build the per-block average
    linear kernel, and
  * both kernels are evaluated with the identical 10 x 5 nested CV protocol,
    so the SSL-vs-control difference is paired at the level of CV repeats and
    also comparable to the SNP and block-mean kernels on the same folds.

Reported per pair: mean r, paired difference against the untrained control and
against the block-mean kernel, naive and Nadeau-Bengio corrected p-values.
Reported per configuration: seed-level mean difference +/- SD with a one-sample
t-test over seeds (the seed is the unit of replication for SSL quality).

Outputs go to results/sensitivity/ssl_seed_replication.json and
results/sensitivity/ssl_seed_repl/."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "sensitivity"
RUNDIR = OUT / "ssl_seed_repl"
RUNDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

SPLIT_SEED_BASE = 20261024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RHO = 0.25

spec = importlib.util.spec_from_file_location(
    "ssl_pretrain", ROOT / "scripts" / "06_self_supervised_pretrain.py"
)
ssl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssl_mod)

ORIG = {"name": "orig_d16_e25_b8", "d": 16, "layers": 2, "mask": 0.15,
        "epochs": 25, "batch": 8}
BIG = {"name": "sweep_d64_e100_b32", "d": 64, "layers": 2, "mask": 0.15,
       "epochs": 100, "batch": 32}
SEEDS_ORIG = [20261024, 20261025, 7, 42, 2026]
SEEDS_BIG = [20261024, 7, 42]


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
    order = (torch.randperm(x.shape[0], generator=generator) if train
             else torch.arange(x.shape[0]))
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
    model.eval()
    node_parts, sample_parts = [], []
    col_pos = {int(c): i for i, c in enumerate(dorm_cols)}
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(DEVICE)
            mask = torch.zeros(xb.shape[:2], dtype=torch.bool, device=DEVICE)
            _, h = model(xb, mask, src, dst, efeat, return_nodes=True)
            h = h[:, :x.shape[1]]
            sample_parts.append(
                torch.cat([h.mean(1), h.max(1).values, h.std(1)], dim=1).cpu()
            )
            for j in range(xb.shape[0]):
                pos = col_pos.get(start + j)
                if pos is not None:
                    node_parts.append((pos, h[j].cpu()))
    nodes = torch.zeros(len(dorm_cols), x.shape[1], model.inp.out_features)
    for pos, arr in node_parts:
        nodes[pos] = arr
    return (nodes.numpy().astype(np.float32),
            torch.cat(sample_parts).numpy().astype(np.float32))


def build_encoder(cfg, seed, edge_dim, in_dim):
    torch.manual_seed(seed)
    return ssl_mod.MaskedGraphEncoder(
        in_dim=in_dim, d=cfg["d"], heads=4, n_layers=cfg["layers"],
        edge_dim=edge_dim, dropout=0.10,
    ).to(DEVICE)


def train_one(cfg, seed, feats, src, dst, efeat, dorm_cols):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(feats.shape[0])
    n_val = max(64, int(round(0.1 * feats.shape[0])))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    train_x = torch.from_numpy(feats[train_idx])
    val_x = torch.from_numpy(feats[val_idx])

    model = build_encoder(cfg, seed, int(efeat.shape[1]), feats.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg["epochs"], 2)
    )
    gen = torch.Generator().manual_seed(seed)
    val_mask = (torch.rand(val_x.shape[:2],
                           generator=torch.Generator().manual_seed(seed + 1))
                < cfg["mask"])

    t0 = time.time()
    best_val, best_state, log = float("inf"), None, []
    for epoch in range(1, cfg["epochs"] + 1):
        tr = run_epoch(model, train_x, src, dst, efeat, optimizer, cfg["batch"],
                       gen, True, cfg["mask"])
        va = run_epoch(model, val_x, src, dst, efeat, None, cfg["batch"], gen,
                       False, cfg["mask"], val_mask)
        scheduler.step()
        log.append({"epoch": epoch, "train_mse": tr, "val_mse": va})
        if va < best_val:
            best_val, best_state = va, {k: v.detach().clone()
                                        for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg["epochs"]:
            print(f"  [{cfg['name']} seed={seed}] epoch {epoch:3d}: "
                  f"train={tr:.4f} val={va:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    nodes, sample = encode(model, torch.from_numpy(feats), src, dst, efeat, dorm_cols)
    return nodes, sample, {"best_val_mse": float(best_val),
                           "final_train_mse": log[-1]["train_mse"],
                           "n_val": int(len(val_idx)),
                           "train_seconds": time.time() - t0}


def control_embeddings(cfg, seed, feats, src, dst, efeat, dorm_cols):
    """Randomly initialised encoder, identical seed, no gradient steps."""
    model = build_encoder(cfg, seed, int(efeat.shape[1]), feats.shape[2])
    nodes, sample = encode(model, torch.from_numpy(feats), src, dst, efeat, dorm_cols)
    return nodes, sample


def paired_tests(diff):
    d = np.asarray(diff, float)
    n = len(d)
    m, s2 = float(d.mean()), float(d.var(ddof=1))
    p_naive = float(2 * stats.t.sf(abs(m / np.sqrt(s2 / n)), n - 1))
    p_nb = float(2 * stats.t.sf(abs(m / np.sqrt(s2 * (1 / n + RHO))), n - 1))
    return {"mean_diff": m, "sd_diff": float(np.sqrt(s2)),
            "p_naive": p_naive, "p_nadeau_bengio": p_nb}


def main():
    import kernel_experiments as ke
    smoke = "--smoke" in sys.argv
    orig, big, seeds_orig, seeds_big = (dict(ORIG), dict(BIG),
                                        list(SEEDS_ORIG), list(SEEDS_BIG))
    if smoke:
        orig["epochs"] = big["epochs"] = 2
        seeds_orig, seeds_big = seeds_orig[:1], []
        print("[smoke] 2 epochs, one seed, temp output", flush=True)
    print(f"device: {DEVICE}", flush=True)
    feats, src, dst, efeat, mu, sd = ssl_mod.build_graph_tensors()
    src, dst, efeat = src.to(DEVICE), dst.to(DEVICE), efeat.to(DEVICE)
    dorm_cols = np.load(PROCESSED / "dormancy_cols_in_1307.npy")
    y = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")["dsds50"].to_numpy(float)

    store = np.load(OUT / "kernel_experiments_preds.npz")
    r_block = np.array(json.loads((OUT / "kernel_experiments.json").read_text(
        encoding="utf-8"))["block"]["r_per_repeat"])
    r_snp = np.array(json.loads((OUT / "kernel_experiments.json").read_text(
        encoding="utf-8"))["snp"]["r_per_repeat"])

    summary_path = (RUNDIR / "_smoke_summary.json") if smoke else (OUT / "ssl_seed_replication.json")
    results = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"runs": {}}
    results.setdefault("config_definitions", {"orig": orig, "sweep_best": big})
    results.setdefault("seeds", {"orig": seeds_orig, "sweep_best": seeds_big})
    results["rho_correction"] = RHO
    results["device"] = str(DEVICE)

    for cfg, seeds in [(orig, seeds_orig), (big, seeds_big)]:
        for seed in seeds:
            key = f"{cfg['name']}_seed{seed}"
            if key in results["runs"] and results["runs"][key].get("complete"):
                print(f"[skip] {key} already done", flush=True)
                continue
            print(f"\n=== {key} ===", flush=True)
            nodes, sample, info = train_one(cfg, seed, feats, src, dst, efeat, dorm_cols)
            cdir = RUNDIR / key
            cdir.mkdir(parents=True, exist_ok=True)
            np.save(cdir / "graph_embeddings_83.npy", nodes)
            np.save(cdir / "sample_embeddings_1307.npy", sample)

            K_ssl = ke.graph_block_kernel(nodes)
            ssl_res, ssl_preds = ke.nested_cv_single(K_ssl, y)

            c_nodes, c_sample = control_embeddings(cfg, seed, feats, src, dst, efeat, dorm_cols)
            K_ctl = ke.graph_block_kernel(c_nodes)
            ctl_res, _ = ke.nested_cv_single(K_ctl, y)

            # kernel similarity between pretrained and untrained representations
            zs = ke.center_trace(K_ssl).ravel()
            zc = ke.center_trace(K_ctl).ravel()
            sim = float(np.corrcoef(zs, zc)[0, 1])

            run = {
                **cfg, "seed": seed, **info,
                "ssl": {k: v for k, v in ssl_res.items()},
                "control_untrained": {k: v for k, v in ctl_res.items()
                                     if k != "r_per_repeat"},
                "ssl_vs_control": paired_tests(
                    np.array(ssl_res["r_per_repeat"]) - np.array(ctl_res["r_per_repeat"])),
                "ssl_vs_block_mean": paired_tests(
                    np.array(ssl_res["r_per_repeat"]) - r_block),
                "ssl_vs_snp": paired_tests(
                    np.array(ssl_res["r_per_repeat"]) - r_snp),
                "kernel_similarity_pretrained_vs_untrained": sim,
                "complete": True,
            }
            results["runs"][key] = run
            summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  ssl r={ssl_res['pearson_mean']:.4f} "
                  f"control r={ctl_res['pearson_mean']:.4f} "
                  f"delta={run['ssl_vs_control']['mean_diff']:+.4f} "
                  f"p_naive={run['ssl_vs_control']['p_naive']:.3f} "
                  f"sim={sim:.3f}", flush=True)

    # ---- seed-level aggregation ----
    agg = {}
    for cfg, cfg_seeds in [(orig, seeds_orig), (big, seeds_big)]:
        keys = [f"{cfg['name']}_seed{s}" for s in cfg_seeds
                if f"{cfg['name']}_seed{s}" in results["runs"]]
        if not keys:
            continue
        d_ctl = np.array([results["runs"][k]["ssl_vs_control"]["mean_diff"] for k in keys])
        d_blk = np.array([results["runs"][k]["ssl_vs_block_mean"]["mean_diff"] for k in keys])
        r_ssl = np.array([results["runs"][k]["ssl"]["pearson_mean"] for k in keys])
        val = np.array([results["runs"][k]["best_val_mse"] for k in keys])
        entry = {
            "n_seeds": len(keys),
            "seeds": [int(results["runs"][k]["seed"]) for k in keys],
            "ssl_r_mean": float(r_ssl.mean()), "ssl_r_sd": float(r_ssl.std(ddof=1)) if len(keys) > 1 else None,
            "val_mse_mean": float(val.mean()),
            "delta_vs_untrained_mean": float(d_ctl.mean()),
            "delta_vs_untrained_sd": float(d_ctl.std(ddof=1)) if len(keys) > 1 else None,
            "delta_vs_untrained_positive_seeds": int((d_ctl > 0).sum()),
            "delta_vs_block_mean_mean": float(d_blk.mean()),
            "delta_vs_block_mean_sd": float(d_blk.std(ddof=1)) if len(keys) > 1 else None,
        }
        if len(keys) > 1:
            entry["one_sample_t_vs_untrained"] = {
                "t": float(d_ctl.mean() / (d_ctl.std(ddof=1) / np.sqrt(len(keys)))),
                "p": float(2 * stats.t.sf(abs(d_ctl.mean() / (d_ctl.std(ddof=1) / np.sqrt(len(keys)))), len(keys) - 1)),
                "df": len(keys) - 1,
            }
            entry["one_sample_t_vs_block_mean"] = {
                "t": float(d_blk.mean() / (d_blk.std(ddof=1) / np.sqrt(len(keys)))),
                "p": float(2 * stats.t.sf(abs(d_blk.mean() / (d_blk.std(ddof=1) / np.sqrt(len(keys)))), len(keys) - 1)),
                "df": len(keys) - 1,
            }
        agg[cfg["name"]] = entry
    results["seed_level_summary"] = agg
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== seed-level summary ===", flush=True)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()