"""Build haplotype-block graph from the 1,307-accession AtPolyDB panel.

Blocks are consecutive windows of 25 SNPs (max span 50 kb). This is a pragmatic
definition that keeps blocks comparable in information content; sensitivity to
10/50-SNP windows is handled in downstream scripts.

Outputs (data/processed):
  haplotype_blocks.csv        block metadata
  block_features.npy          (1307, n_blocks, 4) block mean dosage + 3 PCs
  block_edges.npz             edge_index, edge_attr (distance, |r|, type)
  block_snp_map.npz           SNP row ranges per block
  block_summary.json
  dormancy_cols_in_1307.npy   column positions of the 83 phenotyped accessions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "atpolydb"
PROCESSED = ROOT / "data" / "processed"
ANNOT = ROOT / "data" / "annotations"

SNP_PER_BLOCK = 25
MAX_SPAN_BP = 50_000
LD_WINDOW = 50          # candidate blocks downstream
LD_MAX_BP = 1_000_000
LD_MIN_R = 0.5
LD_TOP_K = 3


def main() -> None:
    codes = np.load(PROCESSED / "genotypes_all_1307.npy")
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    genes = pd.read_csv(ANNOT / "genes_tair10.csv")

    counts = snp[["n_A_all", "n_C_all", "n_G_all", "n_T_all"]].to_numpy()
    freq = np.where(counts > 0, counts / np.maximum(counts.sum(1, keepdims=True), 1), np.inf)
    minor = np.argmin(freq, axis=1).astype(np.uint8)
    mono = counts.max(1) == counts.sum(1)
    minor[mono] = 255

    chrom = snp["chrom"].to_numpy()
    pos = snp["pos"].to_numpy()

    # ---- block boundaries ----
    rows: list[dict] = []
    for c in range(1, 6):
        idx = np.flatnonzero(chrom == c)
        start = 0
        b = 0
        while start < len(idx):
            stop = start
            while stop + 1 < len(idx) and (stop - start + 1) < SNP_PER_BLOCK:
                if pos[idx[stop + 1]] - pos[idx[start]] > MAX_SPAN_BP:
                    break
                stop += 1
            block_rows = idx[start : stop + 1]
            rows.append(
                {
                    "block_id": len(rows),
                    "chrom": c,
                    "start": int(pos[block_rows[0]]),
                    "end": int(pos[block_rows[-1]]),
                    "n_snps": int(len(block_rows)),
                    "snp_start_row": int(block_rows[0]),
                    "snp_end_row": int(block_rows[-1]) + 1,
                    "local_block_index": b,
                }
            )
            start = stop + 1
            b += 1
    blocks = pd.DataFrame(rows)
    n_blocks = len(blocks)
    n_acc = codes.shape[1]

    # ---- block features ----
    print("[05] computing block features", flush=True)
    feats = np.zeros((n_acc, n_blocks, 4), dtype=np.float32)
    for b, sl_start, sl_end in zip(
        blocks["block_id"].to_numpy(),
        blocks["snp_start_row"].to_numpy(),
        blocks["snp_end_row"].to_numpy(),
    ):
        sub = (codes[sl_start:sl_end, :] == minor[sl_start:sl_end, None]).astype(np.float32)
        feats[:, b, 0] = sub.mean(axis=0)
        centered = sub - sub.mean(axis=1, keepdims=True)
        # sign-stable PCA via SVD
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        n_pc = min(3, vt.shape[0])
        if n_pc:
            pcs = vt[:n_pc].T * s[:n_pc]
            for k in range(n_pc):
                j = int(np.argmax(np.abs(vt[k])))
                if vt[k, j] < 0:
                    pcs[:, k] *= -1
            feats[:, b, 1 : 1 + n_pc] = pcs

    # ---- annotations (vectorized per chromosome) ----
    print("[05] annotating blocks", flush=True)
    gene_by_chr = {}
    for c in range(1, 6):
        sub = genes[genes["chr"] == c].sort_values("start")
        gene_by_chr[c] = (
            sub["start"].to_numpy(),
            sub["end"].to_numpy(),
            sub["gene"].to_numpy(),
        )
    gene_counts = []
    nearest_gene = []
    nearest_dist = []
    for c, lo, hi in zip(
        blocks["chrom"].to_numpy(),
        blocks["start"].to_numpy(),
        blocks["end"].to_numpy(),
    ):
        starts, ends, names = gene_by_chr[int(c)]
        d = np.where(ends < lo, lo - ends, np.where(starts > hi, starts - hi, 0))
        gene_counts.append(int((d == 0).sum()))
        j = int(np.argmin(d))
        nearest_gene.append(names[j])
        nearest_dist.append(int(d[j]))
    blocks["gene_count"] = gene_counts
    blocks["nearest_gene"] = nearest_gene
    blocks["nearest_dist"] = nearest_dist
    blocks["gene_density"] = blocks["gene_count"] / np.maximum(
        (blocks["end"] - blocks["start"]) / 1000.0, 1.0
    )

    # ---- edges ----
    mean_dosage = feats[:, :, 0].T.astype(np.float32)  # blocks x accessions
    z = (mean_dosage - mean_dosage.mean(axis=1, keepdims=True)) / np.where(
        mean_dosage.std(axis=1, keepdims=True) > 0, mean_dosage.std(axis=1, keepdims=True), 1.0
    )
    edge_set: dict[tuple[int, int], tuple[float, float, int]] = {}

    def add_edge(i: int, j: int, dist: float, r: float, etype: int) -> None:
        key = (min(i, j), max(i, j))
        prev = edge_set.get(key)
        if prev is None or abs(r) > prev[1]:
            edge_set[key] = (dist, abs(r), etype)

    for c in range(1, 6):
        local = blocks.index[blocks["chrom"] == c].to_numpy()
        for k in range(len(local) - 1):
            i, j = int(local[k]), int(local[k + 1])
            add_edge(i, j, float(blocks.loc[j, "start"] - blocks.loc[i, "end"]), 1.0, 0)
        for k, i in enumerate(local):
            cand = local[k + 1 : k + 1 + LD_WINDOW]
            if len(cand) == 0:
                continue
            cand = cand[(blocks.loc[cand, "start"].to_numpy() - blocks.loc[i, "end"]) <= LD_MAX_BP]
            if len(cand) == 0:
                continue
            r = (z[i] @ z[cand].T) / z.shape[1]
            order = np.argsort(-np.abs(r))[:LD_TOP_K]
            for o in order:
                if abs(r[o]) >= LD_MIN_R:
                    j = int(cand[o])
                    add_edge(i, j, float(blocks.loc[j, "start"] - blocks.loc[i, "end"]), float(r[o]), 1)

    edge_index = np.array(list(edge_set.keys()), dtype=np.int32).T
    edge_attr = np.array(
        [
            [d / 1_000_000.0, r, t]
            for (d, r, t) in edge_set.values()
        ],
        dtype=np.float32,
    )

    # ---- 83 accession column positions ----
    with open(RAW / "call_method_75" / "call_method_75_TAIR9.csv", "rb") as handle:
        next(handle)
        header = handle.readline().decode().rstrip("\n").split(",")
    header_ids = np.array([int(x) for x in header[2:]])
    pheno = pd.read_csv(PROCESSED / "dormancy_phenotype.csv")
    col_map = {eid: i for i, eid in enumerate(header_ids)}
    dorm_cols = np.array([col_map[eid] for eid in pheno["ecotype_id"]], dtype=np.int32)

    # ---- save ----
    blocks.to_csv(PROCESSED / "haplotype_blocks.csv", index=False)
    np.save(PROCESSED / "block_features.npy", feats)
    np.savez_compressed(
        PROCESSED / "block_edges.npz",
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    np.savez_compressed(
        PROCESSED / "block_snp_map.npz",
        snp_start=blocks["snp_start_row"].to_numpy(np.int32),
        snp_end=blocks["snp_end_row"].to_numpy(np.int32),
        chrom=blocks["chrom"].to_numpy(np.int8),
        start=blocks["start"].to_numpy(np.int64),
        end=blocks["end"].to_numpy(np.int64),
    )
    np.save(PROCESSED / "dormancy_cols_in_1307.npy", dorm_cols)

    summary = {
        "n_blocks": int(n_blocks),
        "snps": int(len(snp)),
        "accessions": int(n_acc),
        "snps_per_block": int(blocks["n_snps"].median()),
        "span_bp_median": int((blocks["end"] - blocks["start"]).median()),
        "span_bp_p90": int((blocks["end"] - blocks["start"]).quantile(0.9)),
        "blocks_per_chromosome": {str(k): int(v) for k, v in blocks.groupby("chrom").size().items()},
        "edges": int(edge_index.shape[1]),
        "adjacency_edges": int((edge_attr[:, 2] == 0).sum()),
        "ld_edges": int((edge_attr[:, 2] == 1).sum()),
        "mean_abs_r_ld": float(edge_attr[edge_attr[:, 2] == 1, 1].mean()),
        "dormancy_accessions_mapped": int(len(dorm_cols)),
    }
    (PROCESSED / "block_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
