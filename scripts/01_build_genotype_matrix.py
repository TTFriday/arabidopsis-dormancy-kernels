"""Build compact genotype matrices from the AtPolyDB call_method_75 TAIR9 table.

The source file is a 562 MB CSV with two header rows:
  row 1: array ids (unused)
  row 2: Chromosome, Positions, then 1,307 ecotype ids
Each SNP row stores one A/C/G/T character per ecotype.

Outputs (data/processed):
  genotypes_all_1307.npy   uint8 allele codes, SNPs x accessions
  genotypes_83.npy         uint8 allele codes, SNPs x 83 dormancy accessions
  snp_metadata.csv         chromosome, position, allele frequencies, minor allele
  accessions_all.csv       ecotype id, array id, native name, in_dormancy_set
  dormancy_phenotype.csv   ecotype id, native name, DSDS50
  build_summary.json       counts and QC figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "atpolydb"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

GENO_CSV = RAW / "call_method_75" / "call_method_75_TAIR9.csv"
INFO_TSV = RAW / "call_method_75" / "call_method_75_info.tsv"
PHENO_TSV = RAW / "phenotype_published_raw.tsv"

TRANS = bytes.maketrans(b"ACGTacgt", b"\x00\x01\x02\x03\x00\x01\x02\x03")


def allele_freq(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-SNP allele counts (n_snp x 4) and number of called genotypes."""
    valid = codes <= 3
    counts = np.stack([(codes == code).sum(axis=1) for code in range(4)], axis=1)
    return counts, valid.sum(axis=1)


def main() -> None:
    info = pd.read_csv(INFO_TSV, sep="\t")
    pheno = pd.read_csv(PHENO_TSV, sep="\t", encoding="latin-1", low_memory=False)
    pheno = pheno.loc[:, ["ecotype id", "nativename", "8_Seed Dormancy"]].rename(
        columns={"ecotype id": "ecotype_id", "8_Seed Dormancy": "dsds50"}
    )
    dorm = pheno.dropna(subset=["dsds50"]).copy()
    dorm["dsds50"] = dorm["dsds50"].astype(float)
    dorm = dorm.sort_values("ecotype_id").reset_index(drop=True)

    with open(GENO_CSV, "rb") as handle:
        next(handle)
        header = handle.readline().decode().rstrip("\n").split(",")
    ecotypes = np.array([int(x) for x in header[2:]])
    n_acc = len(ecotypes)
    n_rows = sum(1 for _ in open(GENO_CSV, "rb")) - 2

    dorm_ids = dorm["ecotype_id"].to_numpy()
    missing_in_geno = sorted(set(dorm_ids) - set(ecotypes.tolist()))
    if missing_in_geno:
        raise SystemExit(f"dormancy accessions absent from genotype: {missing_in_geno}")
    keep_set = np.isin(ecotypes, dorm_ids)
    keep_idx = np.flatnonzero(keep_set)
    order = {eid: i for i, eid in enumerate(ecotypes[keep_idx])}
    keep_idx = keep_idx[[order[eid] for eid in dorm_ids]]

    chr_arr = np.empty(n_rows, dtype=np.int16)
    pos_arr = np.empty(n_rows, dtype=np.int64)
    codes_all = np.empty((n_rows, n_acc), dtype=np.uint8)
    codes_83 = np.empty((n_rows, len(dorm_ids)), dtype=np.uint8)

    with open(GENO_CSV, "rb") as handle:
        next(handle)
        next(handle)
        for i, line in enumerate(handle):
            chrom, pos, rest = line.rstrip(b"\n").split(b",", 2)
            letters = rest.replace(b",", b"")
            if len(letters) != n_acc:
                raise SystemExit(f"row {i + 1}: {len(letters)} genotypes, expected {n_acc}")
            arr = np.frombuffer(letters.translate(TRANS), dtype=np.uint8)
            arr = np.where(arr > 3, np.uint8(4), arr)
            chr_arr[i] = int(chrom)
            pos_arr[i] = int(pos)
            codes_all[i] = arr
            codes_83[i] = arr[keep_idx]
            if (i + 1) % 50000 == 0:
                print(f"  parsed {i + 1:,} SNPs", flush=True)
    if i + 1 != n_rows:
        raise SystemExit(f"parsed {i + 1} rows, expected {n_rows}")

    snp = pd.DataFrame({"chrom": chr_arr, "pos": pos_arr})
    counts_all, called_all = allele_freq(codes_all)
    counts_83, called_83 = allele_freq(codes_83)

    for tag, counts, called, total in (
        ("all", counts_all, called_all, n_acc),
        ("83", counts_83, called_83, len(dorm_ids)),
    ):
        freq = counts / np.maximum(called, 1)[:, None]
        snp[f"maf_{tag}"] = np.where(called > 0, np.minimum(freq.max(axis=1), 1 - freq.max(axis=1)), np.nan)
        snp[f"n_called_{tag}"] = called
        snp[f"missing_frac_{tag}"] = (total - called) / total

    freq83 = counts_83 / np.maximum(called_83, 1)[:, None]
    # least frequent allele among those actually observed
    freq_obs = np.where(counts_83 > 0, freq83, np.inf)
    minor = np.argmin(freq_obs, axis=1).astype(np.uint8)
    monomorphic = (counts_83.max(axis=1) == called_83) | (called_83 == 0)
    minor[monomorphic] = 255
    snp["minor_code"] = minor
    snp["minor_base"] = np.array(["ACGT"[c] if c < 4 else "N" for c in minor])
    for code, base in enumerate("ACGT"):
        snp[f"n_{base}_83"] = counts_83[:, code]
        snp[f"n_{base}_all"] = counts_all[:, code]

    np.save(PROCESSED / "genotypes_all_1307.npy", codes_all)
    np.save(PROCESSED / "genotypes_83.npy", codes_83)
    snp.to_csv(PROCESSED / "snp_metadata.csv", index=False)

    acc_all = info.loc[:, ["ecotype_id", "array_id", "nativename", "country"]].copy()
    acc_all["in_dormancy_set"] = acc_all["ecotype_id"].isin(dorm_ids)
    acc_all = acc_all.sort_values("ecotype_id").reset_index(drop=True)
    acc_all.to_csv(PROCESSED / "accessions_all.csv", index=False)
    dorm.to_csv(PROCESSED / "dormancy_phenotype.csv", index=False)

    summary = {
        "snps": int(n_rows),
        "accessions_genotyped": int(n_acc),
        "dormancy_accessions": int(len(dorm_ids)),
        "chromosomes": sorted(int(c) for c in np.unique(chr_arr)),
        "snp_per_chromosome": {
            str(int(c)): int((chr_arr == c).sum()) for c in np.unique(chr_arr)
        },
        "all_1307": {
            "maf_gt_0.05": int((snp["maf_all"] > 0.05).sum()),
            "monomorphic": int((snp["maf_all"] == 0).sum()),
            "mean_missing_frac": float(snp["missing_frac_all"].mean()),
        },
        "dormancy_83": {
            "maf_gt_0.05": int((snp["maf_83"] > 0.05).sum()),
            "monomorphic": int((snp["maf_83"] == 0).sum()),
            "mean_missing_frac": float(snp["missing_frac_83"].mean()),
            "max_missing_frac": float(snp["missing_frac_83"].max()),
            "complete_snps": int((snp["missing_frac_83"] == 0).sum()),
        },
        "phenotype": {
            "n": int(len(dorm)),
            "min": float(dorm["dsds50"].min()),
            "median": float(dorm["dsds50"].median()),
            "mean": float(dorm["dsds50"].mean()),
            "max": float(dorm["dsds50"].max()),
            "n_zero": int((dorm["dsds50"] == 0).sum()),
            "n_ge_189": int((dorm["dsds50"] >= 189).sum()),
        },
    }
    (PROCESSED / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
