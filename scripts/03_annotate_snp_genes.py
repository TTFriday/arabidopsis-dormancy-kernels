"""Annotate SNPs with TAIR10/Araport11 gene models and check coordinate consistency."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANNOT = ROOT / "data" / "annotations"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

GFF = ANNOT / "Arabidopsis_TAIR10.gff3.gz"


def parse_genes() -> pd.DataFrame:
    rows = []
    with gzip.open(GFF, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            if parts[0] not in {"1", "2", "3", "4", "5"}:
                continue
            attrs = dict(
                item.split("=", 1) for item in parts[8].split(";") if "=" in item
            )
            gene = attrs.get("ID", "").replace("gene:", "")
            rows.append(
                {
                    "chr": int(parts[0]),
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": parts[6],
                    "gene": gene,
                    "name": attrs.get("Name", gene),
                    "biotype": attrs.get("biotype", ""),
                    "description": attrs.get("description", ""),
                }
            )
    genes = pd.DataFrame(rows).sort_values(["chr", "start"]).reset_index(drop=True)
    return genes


def annotate(snp: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    out = []
    for chrom, sub in snp.groupby("chrom"):
        sub = sub.copy()
        g = genes.loc[genes["chr"] == chrom].reset_index(drop=True)
        starts = g["start"].to_numpy()
        ends = g["end"].to_numpy()
        pos = sub["pos"].to_numpy()
        j = np.searchsorted(starts, pos, side="right") - 1
        dist_prev = np.full(len(pos), np.iinfo(np.int64).max)
        dist_next = np.full(len(pos), np.iinfo(np.int64).max)
        prev_ok = j >= 0
        dist_prev[prev_ok] = np.maximum(0, pos[prev_ok] - ends[j[prev_ok]])
        next_ok = j + 1 < len(g)
        dist_next[next_ok] = np.maximum(0, starts[j[next_ok] + 1] - pos[next_ok])
        nearest = np.where(dist_prev <= dist_next, j, np.minimum(j + 1, len(g) - 1))
        nearest = np.clip(nearest, 0, len(g) - 1)
        sub["nearest_gene"] = g["gene"].to_numpy()[nearest]
        sub["nearest_gene_name"] = g["name"].to_numpy()[nearest]
        sub["distance_to_gene"] = np.minimum(dist_prev, dist_next)
        sub["inside_gene"] = (pos >= starts[nearest]) & (pos <= ends[nearest])
        out.append(sub)
    return pd.concat(out).sort_values(["chrom", "pos"]).reset_index(drop=True)


def main() -> None:
    genes = parse_genes()
    genes.to_csv(ANNOT / "genes_tair10.csv", index=False)
    snp = pd.read_csv(PROCESSED / "snp_metadata.csv")
    annotated = annotate(snp, genes)
    annotated.to_csv(PROCESSED / "snp_annotation.csv", index=False)

    summary = {
        "genes": int(len(genes)),
        "genes_per_chromosome": {str(k): int(v) for k, v in genes.groupby("chr").size().items()},
        "snps": int(len(annotated)),
        "genic_fraction": float(annotated["inside_gene"].mean()),
        "within_1kb_of_gene": float((annotated["distance_to_gene"] <= 1000).mean()),
        "within_5kb_of_gene": float((annotated["distance_to_gene"] <= 5000).mean()),
        "median_distance_to_gene": float(annotated["distance_to_gene"].median()),
        "dog1": genes.loc[genes["gene"] == "AT5G45830"].to_dict(orient="records"),
        "snps_in_dog1_window": int(
            (
                (annotated["chrom"] == 5)
                & (annotated["pos"] >= 18_500_000)
                & (annotated["pos"] <= 18_700_000)
            ).sum()
        ),
        "dog1_window_genes": genes.loc[
            (genes["chr"] == 5) & (genes["start"] < 18_700_000) & (genes["end"] > 18_500_000),
            ["gene", "name", "start", "end", "strand"],
        ].to_dict(orient="records"),
    }
    (TABLES / "annotation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # quick sanity: nearest gene for the strongest dormancy SNPs
    top = annotated.loc[
        (annotated["chrom"] == 5) & (annotated["pos"] >= 18_500_000) & (annotated["pos"] <= 18_700_000)
    ].head(5)
    print(top[["chrom", "pos", "nearest_gene", "nearest_gene_name", "distance_to_gene"]].to_string(index=False))


if __name__ == "__main__":
    main()
