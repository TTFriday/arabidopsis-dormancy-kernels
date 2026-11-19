"""Independent, training-only rice validation; never writes other analysis code.

Run with the observed scientific interpreter: python -s -B rice_validation.py
--stage prepare|test|cv|all --device cuda. Data and results stay in this revision.
The published numeric CSV contains line-mean imputation. Recover missingness
from its raw .75-call-rate HapMap before fitting any model. Allele dosage is
2 for the first declared HapMap allele and 0 for the second (source +1/-1).
This is a marker-density-adapted validation, not identical Arabidopsis windows.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import requests
from scipy import linalg
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
# Core imports NumPy/pandas/SciPy before the observed user-site torch 2.6 CUDA.
# Only its centering, stable eigensolver, metrics and Torch infrastructure are
# reused. GenomicKernels' nucleotide counts are NOT used for numeric dosages.
import kernel_benchmark as core

torch = core.torch
HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
DATA = REVISION / "data"
RESULTS = REVISION / "results"
PROTOCOL_PATH = HERE / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
LAMBDAS = np.asarray(PROTOCOL["lambda_grid"], dtype=np.float64)
GAMMAS = tuple(float(g) for g in PROTOCOL["gamma_grid"])
SCALES = (5, 10, 25)
MAXSPAN = 3_000_000
ENGINE_VERSION = "rice-numeric-raw-hapmap-v2-strict-maf"
FAMILIES = ("snp_linear", "regional_linear", "local_gaussian", "global_snp_rkhs", "train_mean")
URL_BASE = "https://www.ricediversity.org/data/Spindel_2015_PLOS/GS_datasets/"
SOURCES = {
    "genotype": (URL_BASE + "genotypes/binned_subsets_2_14.zip", "rice_binned_subsets_2_14.zip"),
    "phenotype": (URL_BASE + "phenotypes/RYT_plotdata_by_GHID_corrected_PUBLIC_ACCESS.zip",
                  "rice_RYT_plotdata_by_GHID_corrected_PUBLIC_ACCESS.zip"),
    "hapmap": (URL_BASE + "genotypes/MET_crfilt_.75_allchrom.hmp.txt.zip",
               "rice_MET_crfilt_.75_allchrom.hmp.txt.zip"),
    "license": ("https://api.datacite.org/dois/10.5061/dryad.7369p", "rice_datacite_dryad.json"),
    "paper": ("https://doi.org/10.1371/journal.pgen.1004982",
              "rice_paper_schema_evidence.json"),
}
GENO_SUFFIX = "bin_120/geno_sub_120Kb_binned_random_selection_w_maf_filter_1.csv"
PHENO_SUFFIX = "corrected_RYT2011DS_plotdata_by_GHID.csv"
DATASET_PATH = DATA / "rice_dataset.npz"
AUDIT_PATH = RESULTS / "rice_dataset_audit.json"
MANIFEST_PATH = RESULTS / "rice_manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, obj):
    core.save_json(Path(path), obj)


def metrics(y, prediction):
    """Reuse core metrics; exact constants need a stronger guard at kg/ha scale.

    At a mean near 6000, floating summation can give an exactly constant array
    an apparent SD just above core's absolute 1e-12 guard. Its correlations and
    calibration slope are undefined, not NaN/Infinity JSON or enormous slopes.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", core.stats.ConstantInputWarning)
        result = core.metrics(y, prediction)
    if np.ptp(np.asarray(prediction, float)) == 0 or np.ptp(np.asarray(y, float)) == 0:
        for key in ("pearson", "spearman", "calibration_slope"):
            result[key] = None
    return {k: (None if v is not None and not np.isfinite(v) else v) for k, v in result.items()}


def save_npz(path, compresslevel=6, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Standard NumPy-compatible NPZ; cache compression level 1 saves CPU time
    # without rounding, dropping, or otherwise changing any kernel entries.
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as archive:
        for key, value in arrays.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)
    tmp.replace(path)


def obtain(label, url, filename):
    """Verify existing downloads or assemble bounded, CRC-checked byte ranges."""
    path = DATA / filename
    response_path = DATA / ("rice_response_" + label + ".json")
    if path.is_file():
        if filename.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                if z.testzip() is not None:
                    raise RuntimeError('Included source archive failed CRC')
        return {'url': url, 'path': str(path), 'bytes': path.stat().st_size,
                'sha256': sha256(path), 'download_method': 'included source verified locally'}
    if response_path.exists() and path.exists():
        record = json.loads(response_path.read_text(encoding="utf-8"))
        if sha256(path) == record["sha256"]:
            return record
    # Public PLOS download redirects use a GET-signed object URL: HEAD may
    # return 403 despite a successful ordinary GET. Do not misclassify that
    # method mismatch as an inaccessible source or bypass its permissions.
    is_archive = filename.endswith(".zip")
    head = (requests.head(url, allow_redirects=True, timeout=(30, 90)) if is_archive
            else requests.get(url, timeout=(30, 120)))
    head.raise_for_status()
    record = {"url": url, "final_url": head.url, "http_status": head.status_code,
              "request_method": "HEAD" if is_archive else "GET",
              "headers": dict(head.headers), "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not is_archive:
        path.write_bytes(head.content)
        record["download_method"] = "verified public HTTP GET"
    elif not path.exists():
        total = int(head.headers.get("Content-Length", "0"))
        if filename.endswith(".zip") and total > 0:
            if total > 30_000_000:
                raise RuntimeError("Archive exceeds authorized small-download threshold: " + url)
            blob = bytearray(total)
            spans = [(a, min(total, a + 262144) - 1) for a in range(0, total, 262144)]

            def fetch(span):
                a, b = span
                r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=(30, 90))
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != b - a + 1:
                    raise RuntimeError("Server did not honor byte range")
                expected = f"bytes {a}-{b}/{total}"
                if r.headers.get("Content-Range") != expected:
                    raise RuntimeError("Unexpected Content-Range")
                return a, r.content

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                for a, part in pool.map(fetch, spans):
                    blob[a:a + len(part)] = part
            tmp = path.with_suffix(".download")
            tmp.write_bytes(blob)
            # Close ZipFile BEFORE replace: Windows prohibits replacing open files.
            with zipfile.ZipFile(tmp) as z:
                if z.testzip() is not None:
                    raise RuntimeError("Archive CRC failure")
            tmp.replace(path)
            record["download_method"] = "complete archive assembled from verified HTTP byte ranges"
        else:
            r = requests.get(url, timeout=(30, 120))
            r.raise_for_status()
            path.write_bytes(r.content)
            record["download_method"] = "HTTP GET"
            record["get_status"] = r.status_code
    else:
        record["download_method"] = "existing complete file, verified against current source size and archive CRC"
    expected_size = head.headers.get("Content-Length")
    if filename.endswith(".zip"):
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise RuntimeError("Incomplete archive: " + str(path))
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                raise RuntimeError("Archive CRC failure: " + bad)
            record["archive_crc_verified"] = True
    record.update(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
    save_json(response_path, record)
    return record


def selected_member(archive, suffix):
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if n.endswith(suffix) and "__MACOSX" not in n]
        if len(matches) != 1:
            raise ValueError((suffix, matches))
        name = matches[0]
        payload = z.read(name)  # ZipFile verifies selected member's CRC too.
        info = z.getinfo(name)
    return name, payload, {"member": name, "bytes": len(payload), "crc32": f"{info.CRC:08x}",
                           "sha256": hashlib.sha256(payload).hexdigest()}


def physical_groups(metadata, scale, maxspan=MAXSPAN):
    """Greedy disjoint windows in chromosome/position/ID order, NOT CSV order.

    A window stops before exceeding the marker cap or last-first > maxspan.
    Membership is frozen before training MAF filtering; empty windows vanish
    and retained dosages get equal weights within each remaining window.
    """
    ids = metadata.marker_id.to_numpy(str)
    chrom = metadata.chrom.to_numpy(np.int64)
    pos = metadata.pos.to_numpy(np.int64)
    order = np.lexsort((ids, pos, chrom))
    groups = np.empty(len(metadata), np.int64)
    group = -1
    previous_chrom, first_pos, count = None, None, 0
    for i in order:
        if chrom[i] != previous_chrom or count >= scale or pos[i] - first_pos > maxspan:
            group += 1
            previous_chrom, first_pos, count = chrom[i], pos[i], 0
        groups[i] = group
        count += 1
    return groups


def window_audit(metadata, groups):
    values = []
    for group in np.unique(groups):
        rows = metadata.iloc[np.flatnonzero(groups == group)]
        if rows.chrom.nunique() != 1:
            raise AssertionError("Window crosses chromosomes")
        values.append((len(rows), int(rows.pos.max() - rows.pos.min())))
    sizes, spans = np.asarray(values).T
    return {"windows": len(values), "singleton_windows": int((sizes == 1).sum()),
            "markers_per_window_min_median_max": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
            "physical_span_bp_min_median_max": [int(spans.min()), float(np.median(spans)), int(spans.max())],
            "assignment_sha256": array_hash(groups)}


def frozen_files(manifest):
    for record in manifest["frozen_files"]:
        path = Path(record["path"])
        if not path.is_absolute():
            path = REVISION.parent / path
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError("Frozen dataset/source changed: " + str(path))
    if sha256(PROTOCOL_PATH) != manifest["protocol_sha256"]:
        raise RuntimeError("Original protocol differs from the frozen protocol")


def prepare():
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        frozen_files(manifest)
        print("rice frozen dataset verified, n=" + str(manifest["final_n"]), flush=True)
        return manifest
    for directory in (DATA, RESULTS):
        if not directory.is_dir():
            raise RuntimeError("Expected existing authorized directory: " + str(directory))
    responses = {label: obtain(label, *source) for label, source in SOURCES.items()}
    genotype_member, payload, genotype_member_audit = selected_member(DATA / SOURCES["genotype"][1], GENO_SUFFIX)
    columns = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    if len(columns) != len(set(columns)) or columns[0] != "Entry":
        raise ValueError("Duplicate marker columns or unexpected Entry schema")
    genotype_path = DATA / "rice_geno_sub_120Kb_binned_random_selection_w_maf_filter_1.csv"
    genotype_path.write_bytes(payload)
    geno = pd.read_csv(io.BytesIO(payload), dtype={"Entry": str})
    geno.Entry = geno.Entry.str.strip()
    if geno.Entry.isna().any() or geno.Entry.duplicated().any():
        raise ValueError("Genotype Entry IDs are missing or nonunique")
    original_markers = geno.columns[1:].to_numpy(str)
    source_numeric = geno.iloc[:, 1:].to_numpy(np.float64)
    if not np.isfinite(source_numeric[np.isfinite(source_numeric)]).all():
        raise ValueError("Invalid numeric genotype")
    if np.nanmin(source_numeric) < -1 or np.nanmax(source_numeric) > 1:
        raise ValueError("Source is not the documented -1/0/+1 coding")

    phenotype_member, ppayload, phenotype_member_audit = selected_member(DATA / SOURCES["phenotype"][1], PHENO_SUFFIX)
    phenotype_path = DATA / "rice_corrected_RYT2011DS_plotdata_by_GHID.csv"
    phenotype_path.write_bytes(ppayload)
    pheno = pd.read_csv(io.BytesIO(ppayload), dtype={"GHID": str})
    required = {"GHID", "YEAR", "SEASON", "REP", "Kg_ha", "PLOTNO"}
    if not required.issubset(pheno.columns):
        raise ValueError("Unexpected phenotype schema: " + str(pheno.columns.tolist()))
    pheno.GHID = pheno.GHID.str.strip()
    if pheno.GHID.isna().any() or pheno.duplicated(["GHID", "REP"]).any():
        raise ValueError("Missing GHID or duplicate line/replication")
    if not (pheno.YEAR.eq(2011).all() and pheno.SEASON.eq(1).all()):
        raise ValueError("Not the fixed 2011 dry-season response")
    if not set(pheno.REP.unique()).issubset({1, 2, 3}):
        raise ValueError("Unexpected REP values")
    pheno.Kg_ha = pd.to_numeric(pheno.Kg_ha, errors="raise")
    if np.isinf(pheno.Kg_ha).any():
        raise ValueError("Infinite phenotype")
    lines = pheno.groupby("GHID", sort=True).agg(yield_kg_ha=("Kg_ha", "mean"),
        observed_reps=("Kg_ha", "count"), total_reps=("REP", "size"))
    if lines.index.duplicated().any():
        raise ValueError("Aggregated phenotype IDs are nonunique")
    observed_ids = set(lines.index[lines.observed_reps > 0])
    genotype_ids = geno.Entry.to_numpy(str)
    matched_ids = sorted(set(genotype_ids) & observed_ids)

    # Read raw source HapMap: no source line-mean or discretized imputation is
    # inherited. Its first allele is source +1 at every checked called cell.
    wanted = set(original_markers)
    hapmap_rows = {}
    total_hapmap_rows = 0
    with zipfile.ZipFile(DATA / SOURCES["hapmap"][1]) as z:
        names = [n for n in z.namelist() if n.endswith(".hmp.txt") and "__MACOSX" not in n]
        if len(names) != 1:
            raise ValueError("Unexpected HapMap members")
        hapmap_member = names[0]
        with io.TextIOWrapper(z.open(hapmap_member), encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            if header[:4] != ["rs#", "alleles", "chrom", "pos"] or len(set(header[11:])) != len(header[11:]):
                raise ValueError("Unexpected HapMap schema or duplicate sample IDs")
            if not set(genotype_ids).issubset(set(header[11:])):
                raise ValueError("Genotype Entry absent from HapMap")
            for row in reader:
                total_hapmap_rows += 1
                if row[0] in wanted:
                    if row[0] in hapmap_rows or len(row) != len(header):
                        raise ValueError("Duplicate/malformed selected HapMap row")
                    hapmap_rows[row[0]] = row
    if set(hapmap_rows) != wanted:
        raise ValueError("Some published numeric SNPs are absent from HapMap")
    sample_columns = {s: i for i, s in enumerate(header)}
    numeric = np.full(source_numeric.shape, np.nan, np.float64)
    meta_rows, checked_called, missing_tokens = [], 0, {}
    source_discrete_imputations_masked = 0
    for j, marker in enumerate(original_markers):
        row = hapmap_rows[marker]
        parsed = __import__("re").fullmatch(r"S(\d+)_(\d+)", marker)
        if parsed is None:
            raise ValueError("Unparseable reference marker ID: " + marker)
        chrom, pos = map(int, parsed.groups())
        if (chrom, pos) != (int(row[2]), int(row[3])) or not 1 <= chrom <= 12 or pos <= 0:
            raise ValueError("ID disagrees with explicit HapMap chromosome/position")
        alleles = row[1].split("/")
        if len(alleles) != 2 or any(a not in "ACGT" for a in alleles) or alleles[0] == alleles[1]:
            raise ValueError("Unexpected selected biallelic schema")
        calls = np.asarray([row[sample_columns[s]] for s in genotype_ids])
        first, second = calls == alleles[0], calls == alleles[1]
        called = first | second
        # Actual selected HapMap has homozygous A/C/G/T plus NA/X, no called
        # heterozygotes. Fail rather than inventing an unobserved encoding.
        unknown = set(calls[~called]) - {"NA", "X", "N", "NN", "", "-", "."}
        if unknown:
            raise ValueError("Unexpected HapMap call(s): " + str(unknown))
        numeric[first, j], numeric[second, j] = 2., 0.
        if not np.array_equal(source_numeric[first, j], np.ones(first.sum())):
            raise ValueError("First HapMap allele disagrees with source +1 convention")
        if not np.array_equal(source_numeric[second, j], -np.ones(second.sum())):
            raise ValueError("Second HapMap allele disagrees with source -1 convention")
        checked_called += int(called.sum())
        source_discrete_imputations_masked += int((~called & np.isin(source_numeric[:, j], [-1., 0., 1.])).sum())
        for token, count in zip(*np.unique(calls[~called], return_counts=True)):
            missing_tokens[str(token)] = missing_tokens.get(str(token), 0) + int(count)
        meta_rows.append({"marker_id": marker, "chrom": chrom, "pos": pos,
                          "source_column_index": j + 1, "hapmap_alleles": row[1], "strand": row[4],
                          "dosage_2_allele": alleles[0], "dosage_0_allele": alleles[1]})
    metadata = pd.DataFrame(meta_rows).sort_values(["chrom", "pos", "marker_id"], kind="mergesort").reset_index(drop=True)
    if metadata.marker_id.duplicated().any() or metadata.duplicated(["chrom", "pos"]).any():
        raise ValueError("Duplicate marker IDs/coordinates")
    column_order = metadata.source_column_index.to_numpy(int) - 1
    row_by_id = {s: i for i, s in enumerate(genotype_ids)}
    rows = np.asarray([row_by_id[s] for s in matched_ids], int)
    dosage = numeric[np.ix_(rows, column_order)]
    if not np.isfinite(dosage).any(1).all():
        raise ValueError("A phenotype-matched line has no raw genotype observations; audit exclusion explicitly")
    ids = np.asarray(matched_ids, dtype=str)
    y = lines.loc[ids, "yield_kg_ha"].to_numpy(np.float64)
    if not np.isfinite(y).all() or len(set(ids)) != len(ids):
        raise ValueError("Invalid final outcomes/IDs")
    groups = {s: physical_groups(metadata, s) for s in SCALES}
    for s, assignment in groups.items():
        wa = window_audit(metadata, assignment)
        if wa["markers_per_window_min_median_max"][2] > s or wa["physical_span_bp_min_median_max"][2] > MAXSPAN:
            raise AssertionError("Window constraint violated")
        metadata["window_" + str(s)] = assignment
    metadata_path = DATA / "rice_marker_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    lines["matched_genotype"] = lines.index.isin(ids)
    aggregation_path = DATA / "rice_2011DS_line_yield.csv"
    lines.reset_index().to_csv(aggregation_path, index=False)
    save_npz(DATASET_PATH, dosage=dosage, y=y, ids=ids, marker_ids=metadata.marker_id.to_numpy(str),
             chrom=metadata.chrom.to_numpy(np.int64), pos=metadata.pos.to_numpy(np.int64),
             **{"groups_" + str(s): g for s, g in groups.items()})
    rights = json.loads((DATA / SOURCES["license"][1]).read_text(encoding="utf-8"))["data"]["attributes"]["rightsList"]
    if not any(r.get("rightsIdentifier") == "cc0-1.0" for r in rights):
        raise ValueError("CC0 license was not verified")
    quotations = json.loads((DATA / SOURCES["paper"][1]).read_text(encoding="utf-8"))["quotations"]
    if not any("MSU version 6.0" in p for p in quotations):
        raise ValueError("Missing source reference-assembly evidence")
    gaps = np.concatenate([np.diff(metadata.loc[metadata.chrom == c, "pos"].to_numpy())
                           for c in sorted(metadata.chrom.unique())])
    replicate_details = [{"GHID": s, "observed_reps": int(lines.loc[s, "observed_reps"]),
                          "REP": pheno.loc[pheno.GHID == s, "REP"].astype(int).tolist(),
                          "Kg_ha": [None if pd.isna(v) else float(v) for v in pheno.loc[pheno.GHID == s, "Kg_ha"]],
                          "line_mean_kg_ha": float(lines.loc[s, "yield_kg_ha"])} for s in ids]
    audit = {
        "status": "dataset_frozen_before_model_selection", "final_n": len(ids), "snp_count": dosage.shape[1],
        "paper_doi": "10.1371/journal.pgen.1004982", "data_doi": "10.5061/dryad.7369p",
        "license": rights, "source_responses": responses,
        "mirror_policy": "Cornell/ricediversity public mirror; Dryad direct 403 previously reported, not bypassed",
        "genotype_archive_member": genotype_member_audit, "phenotype_archive_member": phenotype_member_audit,
        "actual_genotype_fields": {"id": "Entry", "rows": len(geno), "marker_columns": len(original_markers),
            "source_code_min_max": [float(np.nanmin(source_numeric)), float(np.nanmax(source_numeric))],
            "source_noninteger_imputed_cells": int((np.isfinite(source_numeric) & ~np.isin(source_numeric, [-1., 0., 1.])).sum()),
            "source_nan_cells": int(np.isnan(source_numeric).sum()), "all_missing_source_Entry": geno.loc[np.isnan(numeric).all(1), "Entry"].tolist()},
        "coordinate_evidence": {"reference": "MSU version 6.0 Nipponbare", "hapmap_member": hapmap_member,
            "hapmap_fields": header[:11], "hapmap_rows": total_hapmap_rows, "hapmap_sample_ids": len(header) - 11,
            "hapmap_sample_ids_unique": True, "all_numeric_markers_checked": len(metadata),
            "ID_rule": "Full-match S<chromosome>_<base-pair-position>; both numbers checked against HapMap chrom and pos for EVERY retained source marker",
            "ID_examples": metadata[["marker_id", "chrom", "pos"]].iloc[:5].to_dict("records"),
            "assembly_field": "HapMap assembly# is NA; MSU6 established by paper Genotyping paragraph, not inferred from NA",
            "source_quotations": quotations, "genes": "No gene-annotation assignment; these are physical SNP windows, not gene windows"},
        "raw_numeric_reconstruction": {"dosage": "first declared HapMap allele=2, second=0; fixed source allele convention",
            "source_called_cells_checked": checked_called, "all_called_cells_agree_with_source_plus_minus_one": True,
            "raw_missing_tokens": missing_tokens, "source_discrete_imputations_also_masked": source_discrete_imputations_masked,
            "policy": "Discard all source numeric fill values; use raw called HapMap cells only. NA/X and other declared missing calls become NaN. Fit marker means separately in every training split.",
            "final_raw_called_cells": int(np.isfinite(dosage).sum()), "final_missing_cells": int(np.isnan(dosage).sum()),
            "final_missing_fraction": float(np.isnan(dosage).mean()),
            "called_dosages": np.unique(dosage[np.isfinite(dosage)]).tolist(),
            "upstream_limitations": "Published marker ascertainment/call-rate and full-panel MAF prefilter are frozen upstream; training MAF is re-estimated, but discarded source markers cannot be recovered within this specified panel. No phenotype-based ascertainment is introduced."},
        "phenotype": {"fixed_response": "2011DS grain yield Kg_ha; not selected among traits or seasons by prediction",
            "actual_fields": pheno.columns.tolist(), "rows": len(pheno), "unique_GHID": pheno.GHID.nunique(),
            "missing_Kg_ha_plots": int(pheno.Kg_ha.isna().sum()), "aggregation": "Arithmetic mean of available Kg_ha across distinct REP per GHID; do not impute response; all reps of a line share every fold",
            "observed_rep_count_distribution_final": {str(k): int(v) for k, v in lines.loc[ids, "observed_reps"].value_counts().items()},
            "matched_replication_records": replicate_details},
        "identity": {"Entry_unique": True, "GHID_REP_unique": True, "aggregate_GHID_unique": True,
            "marker_ID_unique": True, "chrom_pos_unique": True, "final_ID_unique": True,
            "join": "Exact stripped Entry == GHID; no name/fuzzy/order-based matching",
            "sample_order": "Lexicographically sorted matched GHID, frozen before CV",
            "genotype_without_observed_2011DS": sorted(set(genotype_ids) - observed_ids),
            "observed_2011DS_without_genotype": sorted(observed_ids - set(genotype_ids)),
            "final_ids": ids.tolist(), "ids_sha256": array_hash(ids)},
        "windows": {"species_dependent_marker_density_scale": True, "source_density": "One randomly selected marker per 120 kb bin in published bin_120 selection 1; gaps are not exactly 120 kb",
            "adjacent_same_chromosome_gap_bp_median": float(np.median(gaps)),
            "scales": list(SCALES), "maxspan_bp": MAXSPAN, "rule": physical_groups.__doc__.strip(),
            "different_search_range_disclosure": "Rice uses only prespecified 5/10/25-marker caps with 3 Mb maximum span. Arabidopsis protocol uses 5/10/25/50/100 and its 50 kb base windows. NOT identical genomic windows or identical window search space.",
            "per_scale": {str(s): window_audit(metadata, g) for s, g in groups.items()}},
        "evaluation": {"original_protocol_unmodified": True, "outer_repeats": 10, "outer_folds": 5,
            "inner_folds": 5, "outer_seeds": list(range(10)), "inner_seed": "1000 + repeat*5 + outer_fold",
            "lambda_grid": LAMBDAS.tolist(), "gamma_grid": list(GAMMAS), "selection_loss": PROTOCOL["selection_loss"],
            "training_only": "MAF=min(mean observed dosage/2, 1-mean/2)>0.05; train-only marker-mean imputation; training population SD (ddof=0); remove constant features; equal mean dosage per physical window; training-only regional standardization and kernel centering/trace normalization",
            "models": list(FAMILIES), "ancestry_limitations": "Protocol-matched random line folds, not the original paper's pedigree-cluster or future-season validation; relatedness can inflate random-fold prediction. Repeats describe split variability, not independent biological replications."},
    }
    frozen_paths = [DATA / filename for _, filename in SOURCES.values()] + [genotype_path, phenotype_path,
                    metadata_path, aggregation_path, DATASET_PATH]
    audit["frozen_files"] = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in frozen_paths]
    audit["protocol_sha256"] = sha256(PROTOCOL_PATH)
    audit["dataset_sha256"] = sha256(DATASET_PATH)
    audit["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_json(AUDIT_PATH, audit)
    save_json(MANIFEST_PATH, {"final_n": len(ids), "snp_count": dosage.shape[1],
        "dataset_sha256": audit["dataset_sha256"], "audit_sha256": sha256(AUDIT_PATH),
        "protocol_sha256": audit["protocol_sha256"], "frozen_files": audit["frozen_files"],
        "frozen_at": audit["frozen_at"], "evaluation_plan": audit["evaluation"], "windows": audit["windows"]})
    print(f"rice dataset FROZEN n={len(ids)} SNPs={dosage.shape[1]} raw missing={np.isnan(dosage).sum()}", flush=True)
    print("rice window audit " + json.dumps(audit["windows"]["per_scale"]), flush=True)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_dataset():
    manifest = prepare()
    if sha256(AUDIT_PATH) != manifest["audit_sha256"]:
        raise RuntimeError("Frozen audit changed")
    with np.load(DATASET_PATH, allow_pickle=False) as f:
        x, y, ids = f["dosage"].copy(), f["y"].copy(), f["ids"].copy()
        groups = {s: f["groups_" + str(s)].copy() for s in SCALES}
    return x, y, ids, groups, manifest


class NumericFeatureKernels:
    """Training-only real-valued dosage preprocessing, with bounded GPU memory."""

    def __init__(self, dosage, groups, device="cpu", feature_chunk=64, local_dtype=torch.float64):
        self.dosage = np.asarray(dosage, np.float64)
        self.groups = groups
        self.device = torch.device(device)
        self.feature_chunk = min(64, max(1, int(feature_chunk)))
        self.local_dtype = local_dtype
        self.last_audit = {}
        if np.isinf(self.dosage).any() or np.any((self.dosage[np.isfinite(self.dosage)] < 0) | (self.dosage[np.isfinite(self.dosage)] > 2)):
            raise ValueError("Expected numeric dosages in [0,2] or NaN")

    def fit_features(self, train, query):
        train, query = np.asarray(train, int), np.asarray(query, int)
        if len(set(train)) != len(train) or len(set(query)) != len(query) or set(train) & set(query):
            raise ValueError("Duplicate/overlapping train/query IDs")
        anchor = self.dosage[train]
        count = np.isfinite(anchor).sum(0)
        total = np.nansum(anchor, axis=0)
        mean = np.divide(total, count, out=np.zeros(anchor.shape[1]), where=count > 0)
        # Avoid 1 - .95 == .050000000000000044 incorrectly PASSING strict
        # MAF > .05. Subtract allele totals BEFORE dividing, for either allele
        # orientation. Called 0/1/2 totals are exact integers in float64.
        minor_total = np.minimum(total, 2. * count - total)
        maf = np.divide(minor_total, 2. * count, out=np.zeros(anchor.shape[1]), where=count > 0)
        keep = np.flatnonzero((count > 0) & (maf > .05))
        if not len(keep):
            raise ValueError("No training MAF-passing markers")
        x = self.dosage[np.ix_(np.r_[train, query], keep)].copy()
        x = np.where(np.isnan(x), mean[keep][None, :], x)
        sd = x[:len(train)].std(0, ddof=0)
        variable = sd > 1e-10
        keep, x, sd = keep[variable], x[:, variable], sd[variable]
        if not len(keep):
            raise ValueError("No training-variable markers")
        z = (x - mean[keep]) / sd
        audit = {"training_n": len(train), "query_n": len(query), "retained_markers": len(keep),
                 "training_maf_min_max": [float(maf[keep].min()), float(maf[keep].max())],
                 "retained_marker_index_sha256": array_hash(keep),
                 "training_imputation_mean_sha256": array_hash(mean[keep]),
                 "training_marker_sd_sha256": array_hash(sd),
                 "training_missing_cells_retained": int(np.isnan(anchor[:, keep]).sum()),
                 "training_constant_MAF_passing_removed": int((~variable).sum())}
        self.last_audit = audit
        return x, z, keep, mean[keep], sd

    def build(self, train, query):
        x, z, keep, means, sds = self.fit_features(train, query)
        n = len(train)
        out = {}

        def pair(raw):
            return core.centered_pair(raw[:n, :n].cpu().numpy(), raw[n:, :n].cpu().numpy())

        tz = torch.as_tensor(z, dtype=torch.float64, device=self.device)
        raw = tz @ tz.T / tz.shape[1]
        out[("snp_linear", 0, 0.)] = pair(raw)
        d = (raw.diag()[:, None] + raw.diag()[None, :] - 2 * raw).clamp(min=0)
        for gamma in GAMMAS:
            out[("global_snp_rkhs", 0, gamma)] = pair(torch.expm1(-gamma * d))
        tx = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        for scale in SCALES:
            _, inverse, counts = np.unique(self.groups[scale][keep], return_inverse=True, return_counts=True)
            inv = torch.as_tensor(inverse, dtype=torch.long, device=self.device)
            b = torch.zeros((len(x), len(counts)), dtype=torch.float64, device=self.device)
            b.scatter_add_(1, inv[None, :].expand(len(x), -1), tx)
            b /= torch.as_tensor(counts, dtype=torch.float64, device=self.device)[None, :]
            mean = b[:n].mean(0)
            sd = b[:n].std(0, unbiased=False)
            valid = sd > 1e-10
            bz = (b[:, valid] - mean[valid]) / sd[valid]
            if bz.shape[1] == 0:
                raise ValueError("No variable regional features")
            out[("regional_linear", scale, 0.)] = pair(bz @ bz.T / bz.shape[1])
            accum = torch.zeros((len(GAMMAS), len(x), len(x)), dtype=torch.float64, device=self.device)
            # Only ONE bandwidth's feature-distance tensor exists at once.
            # Never allocate n x n x 9 gammas x 512 features.
            for start in range(0, bz.shape[1], self.feature_chunk):
                part = bz[:, start:start + self.feature_chunk].to(self.local_dtype)
                distance = (part[:, None, :] - part[None, :, :]).square()
                for i, gamma in enumerate(GAMMAS):
                    values = torch.expm1(-gamma * distance)
                    accum[i] += values.sum(2, dtype=torch.float64)
                    del values
                del distance, part
            accum /= bz.shape[1]
            for i, gamma in enumerate(GAMMAS):
                out[("local_gaussian", scale, gamma)] = pair(accum[i])
            self.last_audit["window_" + str(scale) + "_variable_features"] = int(bz.shape[1])
            del b, bz, accum
        return out, {"retained_indexes": keep, "training_means": means, "training_sds": sds}


class AnchorCache:
    def __init__(self, engine, fingerprint):
        self.engine = engine
        self.fingerprint = fingerprint
        self.directory = RESULTS / "rice_kernel_cache"
        self.directory.mkdir(exist_ok=True)
        self.hits, self.builds = 0, 0

    def get(self, train):
        train = np.asarray(train, np.int64)
        query = np.setdiff1d(np.arange(len(self.engine.dosage)), train)
        key = hashlib.sha256((self.fingerprint + array_hash(train)).encode()).hexdigest()
        path = self.directory / (key + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                if str(f["fingerprint"]) != self.fingerprint or not np.array_equal(f["train"], train) or not np.array_equal(f["query"], query):
                    raise RuntimeError("Anchor cache identity mismatch")
                keys = [tuple(k) for k in json.loads(str(f["keys"]))]
                train_k, cross_k = f["kt"], f["cross"]
                kernels = {k: (a, b) for k, a, b in zip(keys, train_k, cross_k)}
                audit = json.loads(str(f["audit"]))
            self.hits += 1
        else:
            kernels, preprocessor = self.engine.build(train, query)
            audit = dict(self.engine.last_audit)
            keys = sorted(kernels)
            save_npz(path, compresslevel=1, fingerprint=np.asarray(self.fingerprint), train=train, query=query,
                     keys=np.asarray(json.dumps(keys)), audit=np.asarray(json.dumps(audit)),
                     kt=np.stack([kernels[k][0] for k in keys]), cross=np.stack([kernels[k][1] for k in keys]),
                     **preprocessor)
            self.builds += 1
        # Query genotypes are transformed independently, never fitted. Caching
        # all complement rows permits arbitrary validation/test row subsets.
        lookup = np.full(len(self.engine.dosage), -1, int)
        lookup[query] = np.arange(len(query))
        return kernels, lookup, audit, str(path)


def pooled_inner_selection(engine, y, train, repeat, fold, cache=None):
    losses, audits = {}, []
    seed = 1000 + repeat * 5 + fold
    total_validation = 0
    for inner_fold, (it, iv) in enumerate(KFold(5, shuffle=True, random_state=seed).split(train)):
        a, v = train[it], train[iv]
        if cache is None:
            kernels, _ = engine.build(a, v)
            audit, path = dict(engine.last_audit), None
            index = None
        else:
            kernels, lookup, audit, path = cache.get(a)
            index = lookup[v]
            if (index < 0).any():
                raise AssertionError("Validation was in anchor training")
        for key, (kt, cross) in kernels.items():
            kv = cross if index is None else cross[index]
            prediction = core.eig_predictions(kt, kv, y[a], LAMBDAS)
            loss = np.square(prediction - y[v, None]).sum(0)
            losses[key] = losses.get(key, np.zeros(len(LAMBDAS))) + loss
        audits.append({"inner_fold": inner_fold, "train_indices": a.tolist(), "validation_indices": v.tolist(),
                       "preprocessing": audit, "anchor_cache": path})
        total_validation += len(v)
    if total_validation != len(train):
        raise AssertionError("Inner validation coverage mismatch")
    selection = {}
    for family in FAMILIES[:-1]:
        candidates = [(float(v.min() / total_validation), k, int(v.argmin()))
                      for k, v in losses.items() if k[0] == family]
        if not candidates:
            raise AssertionError("Missing model family")
        selection[family] = min(candidates, key=lambda a: (a[0], a[1], a[2]))
    return selection, audits


def selftests(device="cpu", feature_chunk=64):
    dosage, y, ids, groups, manifest = load_dataset()
    rng = np.random.default_rng(PROTOCOL["random_seed"])
    toy = rng.integers(0, 3, size=(32, 80)).astype(float)
    toy[rng.random(toy.shape) < .12] = np.nan
    toy[:, 0] = 0.
    toy[20:, 0] = 2.  # validation-only polymorphism must never pass train MAF.
    toy[:, 1] = 0.
    toy[0, 1] = 2.  # MAF exactly .05 in 20 training lines must be removed.
    toy[20:, 1] = 2.
    toy[:, 2] = 2.
    toy[0, 2] = 0.  # Complement orientation: avoid 1-.95 cancellation.
    toy[20:, 2] = 0.
    toy_meta = pd.DataFrame({"marker_id": [f"S1_{1000 + i * 120000}" for i in range(80)],
                             "chrom": np.ones(80, int), "pos": 1000 + np.arange(80) * 120000})
    toy_groups = {s: physical_groups(toy_meta, s) for s in SCALES}
    tr, qu = np.arange(20), np.arange(20, 32)
    engine = NumericFeatureKernels(toy, toy_groups, device, feature_chunk)
    imp, z, keep, means, sds = engine.fit_features(tr, qu)
    if 0 in keep or 1 in keep or 2 in keep:
        raise AssertionError("Training-only MAF/strict .05 test failed in either allele orientation")
    expected_mean = np.nanmean(toy[tr][:, keep], axis=0)
    if not np.allclose(means, expected_mean, atol=1e-14):
        raise AssertionError("Imputation means use nontraining data")
    missing = np.isnan(toy[np.ix_(np.r_[tr, qu], keep)])
    if not np.allclose(z[missing], 0., atol=1e-14):
        raise AssertionError("Missing genotypes not imputed to training mean")
    base_k, _ = engine.build(tr, qu)
    base_audit = dict(engine.last_audit)
    altered = toy.copy()
    altered[qu] = rng.choice([0., 1., 2., np.nan], size=(len(qu), toy.shape[1]))
    altered_engine = NumericFeatureKernels(altered, toy_groups, device, feature_chunk)
    altered_k, _ = altered_engine.build(tr, qu)
    audit = dict(altered_engine.last_audit)
    if base_audit != audit:
        raise AssertionError("Query genotypes changed preprocessing")
    leakage_error = max(float(np.max(np.abs(base_k[k][0] - altered_k[k][0]))) for k in base_k)
    if leakage_error > 1e-12:
        raise AssertionError("Query genotypes changed training kernel")
    # Reference-position permutation test: sorting is internal to grouping.
    permutation = rng.permutation(len(toy_meta))
    shuffled = toy_meta.iloc[permutation].reset_index(drop=True)
    for s in SCALES:
        recovered = np.empty(len(toy_meta), int)
        recovered[permutation] = physical_groups(shuffled, s)
        if not np.array_equal(recovered, toy_groups[s]):
            raise AssertionError("Windows depend on arbitrary CSV column order")
    toy_y = rng.normal(size=len(toy))
    outer_train = np.arange(26)
    selected, _ = pooled_inner_selection(engine, toy_y, outer_train, 0, 0)
    mutant_y = toy_y.copy()
    mutant_y[26:] = 1e9 + rng.normal(size=6)
    selected_mutant, _ = pooled_inner_selection(engine, mutant_y, outer_train, 0, 0)
    if any(selected[f][1:] != selected_mutant[f][1:] or
           not np.isclose(selected[f][0], selected_mutant[f][0], rtol=1e-10, atol=1e-12)
           for f in selected):
        raise AssertionError(("Outer-test label mutation changed inner selection", selected, selected_mutant))

    # Actual data subset is a test only, never written as rice CV predictions.
    actual_rows = rng.choice(len(y), size=36, replace=False)
    a, v = actual_rows[:26], actual_rows[26:]
    actual = NumericFeatureKernels(dosage, groups, device, feature_chunk)
    kernels, _ = actual.build(a, v)
    imputed, zsnp, retained, _, _ = actual.fit_features(a, v)
    n = len(a)
    reference_error = 0.
    for scale in SCALES:
        _, inverse, counts = np.unique(groups[scale][retained], return_inverse=True, return_counts=True)
        b = np.stack([np.bincount(inverse, weights=row, minlength=len(counts)) / counts for row in imputed])
        sd = b[:n].std(0)
        variable = sd > 1e-10
        bz = (b[:, variable] - b[:n, variable].mean(0)) / sd[variable]
        distance = (bz[:, None, :] - bz[None, :, :]) ** 2
        for gamma in GAMMAS:
            raw = np.exp(-gamma * distance).mean(2)
            kt, kc = core.centered_pair(raw[:n, :n], raw[n:, :n])
            built_t, built_c = kernels[("local_gaussian", scale, gamma)]
            reference_error = max(reference_error, float(np.max(np.abs(kt - built_t))), float(np.max(np.abs(kc - built_c))))
    if reference_error > 1e-7:
        raise AssertionError(("NumPy/Torch numeric Gaussian disagreement", reference_error))
    kt, kc = kernels[("local_gaussian", 5, .2)]
    penalty = .3
    alpha_b = np.linalg.solve(np.block([[kt + penalty * np.eye(n), np.ones((n, 1))],
                                        [np.ones((1, n)), np.zeros((1, 1))]]), np.r_[y[a], 0.])
    direct = kc @ alpha_b[:-1] + alpha_b[-1]
    stable = core.eig_predictions(kt, kc, y[a], [penalty])[:, 0]
    solver_error = float(np.max(np.abs(direct - stable)))
    if solver_error > 1e-7:
        raise AssertionError(("Augmented solver disagreement", solver_error))
    center_error = max(float(np.abs(k.mean())) for k, _ in kernels.values())
    trace_error = max(float(abs(np.trace(k) / n - 1)) for k, _ in kernels.values())
    if max(center_error, trace_error) > 1e-8:
        raise AssertionError("Training kernel centering/trace mismatch")
    # Diagnostic only: production deliberately uses float64 local distances,
    # accumulation, centered kernels AND the shared float64 eigensolver.
    mixed = NumericFeatureKernels(dosage, groups, device, feature_chunk, local_dtype=torch.float32)
    mixed_k, _ = mixed.build(a, v)
    mixed_error = max(float(np.max(np.abs(kernels[k][0] - mixed_k[k][0]))) for k in kernels if k[0] == "local_gaussian")
    mixed_prediction_error = max(float(np.max(np.abs(
        core.eig_predictions(kernels[k][0], kernels[k][1], y[a], [1e-4, .1, 10.]) -
        core.eig_predictions(mixed_k[k][0], mixed_k[k][1], y[a], [1e-4, .1, 10.]))))
        for k in kernels if k[0] == "local_gaussian")
    constant_metrics = metrics(y[v], np.full(len(v), y[a].mean()))
    if any(constant_metrics[k] is not None for k in ("pearson", "spearman", "calibration_slope")):
        raise AssertionError("Exactly constant train-mean correlations must be undefined")
    if not np.isfinite(constant_metrics["rmse"]):
        raise AssertionError("Constant predictor error metrics must remain finite")
    tests = {"passed": True, "exact_constant_metric_guard": True,
        "dataset_sha256": manifest["dataset_sha256"], "final_dataset_n": len(y),
        "actual_data_smalltest_n": len(actual_rows), "smalltest_not_full_CV": True,
        "training_only_MAF_and_exact_threshold": True, "train_only_mean_imputation": True,
        "missing_standardized_to_zero": True, "outer_query_mutation_training_kernel_max_error": leakage_error,
        "outer_outcome_mutation_inner_selection_identical": True, "physical_column_permutation_invariance": True,
        "numeric_local_numpy_reference_max_error": reference_error,
        "augmented_intercept_solver_prediction_max_error": solver_error,
        "training_center_max_error": center_error, "training_trace_max_error": trace_error,
        "float32_diagnostic_centered_train_kernel_max_error": mixed_error,
        "float32_diagnostic_prediction_max_error_lambda_at_least_1e_minus_4": mixed_prediction_error,
        "production_precision": "float64 throughout; float32 comparison diagnostic only",
        "device": str(device), "feature_chunk": min(64, feature_chunk),
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "torch_version": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()), "tested_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_json(RESULTS / "rice_selftests.json", tests)
    print("rice leakage/solver/reference smalltests PASSED " + json.dumps(tests), flush=True)
    return tests


def nested_cv(device, feature_chunk):
    dosage, y, ids, groups, manifest = load_dataset()
    if PROTOCOL["outer_cv"]["repeats"] != 10 or PROTOCOL["outer_cv"]["folds"] != 5 or PROTOCOL["inner_cv"]["folds"] != 5:
        raise RuntimeError("Expected authorized 10-repeat 5-outer/5-inner protocol")
    if PROTOCOL["outer_cv"]["seed"] != "repeat index 0 through 9" or PROTOCOL["inner_cv"]["seed"] != "1000 + repeat*5 + outer_fold":
        raise RuntimeError("Unexpected protocol seed grid")
    test_path = RESULTS / "rice_selftests.json"
    if not test_path.exists():
        raise RuntimeError("Run --stage test before full CV")
    tests = json.loads(test_path.read_text(encoding="utf-8"))
    if not tests["passed"] or tests["dataset_sha256"] != manifest["dataset_sha256"]:
        raise RuntimeError("Missing valid dataset-matched selftests")
    fingerprint_spec = {"dataset_sha256": manifest["dataset_sha256"], "protocol_sha256": sha256(PROTOCOL_PATH),
        "core_sha256": sha256(HERE / "kernel_benchmark.py"), "script_sha256": sha256(Path(__file__)),
        "engine": ENGINE_VERSION, "scales": list(SCALES), "maxspan": MAXSPAN, "gammas": list(GAMMAS),
        "lambdas": LAMBDAS.tolist(), "device": str(device), "dtype": "float64", "feature_chunk": feature_chunk}
    fingerprint = object_hash(fingerprint_spec)
    state_path, predictions_path = RESULTS / "rice_state.json", RESULTS / "rice_predictions.npz"
    repeats, n = 10, len(y)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "status": "running", "complete": False, "fingerprint": fingerprint, "fingerprint_spec": fingerprint_spec,
        "dataset_n": n, "marker_count": dosage.shape[1], "outer_repeats": 10, "outer_folds": 5, "inner_folds": 5,
        "protocol_snapshot": PROTOCOL, "rice_window_override": manifest["windows"], "folds": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_total_seconds": 0.}
    if state["fingerprint"] != fingerprint or state["dataset_n"] != n:
        raise RuntimeError("Resume fingerprint mismatch; do not overwrite prior run")
    if predictions_path.exists():
        with np.load(predictions_path, allow_pickle=False) as f:
            if not np.array_equal(f["ids"], ids) or not np.array_equal(f["y"], y):
                raise RuntimeError("Prediction checkpoint identity mismatch")
            predictions = {k: f[k].copy() for k in FAMILIES}
            fold_id = f["outer_fold"].copy()
    else:
        if state["folds"]:
            raise RuntimeError("State exists without predictions")
        predictions = {k: np.full((repeats, n), np.nan) for k in FAMILIES}
        fold_id = np.full((repeats, n), -1, np.int16)
    for k, p in predictions.items():
        if p.shape != (10, n):
            raise RuntimeError("Checkpoint prediction shape mismatch")
    done = {(r["repeat"], r["fold"]) for r in state["folds"]}
    if len(done) != len(state["folds"]):
        raise RuntimeError("Duplicate completed fold records")
    for rep, fold in done:
        rows = next(r["test_indices"] for r in state["folds"] if (r["repeat"], r["fold"]) == (rep, fold))
        if any(not np.isfinite(p[rep, rows]).all() for p in predictions.values()):
            raise RuntimeError("Committed fold has incomplete predictions")
    engine = NumericFeatureKernels(dosage, groups, device, feature_chunk)
    cache = AnchorCache(engine, fingerprint)
    started, previous_elapsed = time.perf_counter(), state["elapsed_total_seconds"]
    for rep in range(10):
        for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=rep).split(y)):
            if (rep, fold) in done:
                continue
            tick = time.perf_counter()
            state["active_fold"] = {"repeat": rep, "fold": fold, "phase": "inner_selection"}
            save_json(state_path, state)
            selection, inner_audits = pooled_inner_selection(engine, y, train, rep, fold, cache)
            outer, lookup, audit, anchor_path = cache.get(train)
            ix = lookup[test]
            if (ix < 0).any():
                raise AssertionError("Outer test in anchor train")
            record = {"repeat": rep, "fold": fold, "outer_seed": rep, "inner_seed": 1000 + rep * 5 + fold,
                "train_indices": train.tolist(), "test_indices": test.tolist(), "inner_splits": inner_audits,
                "preprocessing": audit, "anchor_cache": anchor_path, "selection": {}}
            for family, (loss, key, ilambda) in selection.items():
                kt, cross = outer[key]
                kv = cross[ix]
                prediction = core.eig_predictions(kt, kv, y[train], [LAMBDAS[ilambda]])[:, 0]
                if not np.isfinite(prediction).all():
                    raise RuntimeError("Nonfinite outer prediction")
                predictions[family][rep, test] = prediction
                q = np.maximum(linalg.eigvalsh(kt), 0.)
                positive = q[q > max(1., q.max()) * 1e-10]
                record["selection"][family] = {"window_cap": int(key[1]), "gamma": float(key[2]),
                    "lambda": float(LAMBDAS[ilambda]), "pooled_inner_mse": loss,
                    "test_metrics": metrics(y[test], prediction), "spectrum": {
                        "largest_eigenvalue": float(q.max()), "positive_rank": len(positive),
                        "effective_dimension": float(np.sum(q / (q + LAMBDAS[ilambda]))),
                        "participation_rank": float(q.sum() ** 2 / np.sum(q * q))}}
            predictions["train_mean"][rep, test] = float(y[train].mean())
            fold_id[rep, test] = fold
            record["train_mean_test_metrics"] = metrics(y[test], predictions["train_mean"][rep, test])
            record["seconds"] = time.perf_counter() - tick
            state["folds"].append(record)
            state["completed_outer_folds"] = len(state["folds"])
            state["elapsed_total_seconds"] = previous_elapsed + time.perf_counter() - started
            state["active_fold"] = None
            state["cache_current_process"] = {"hits": cache.hits, "builds": cache.builds}
            # Predictions first, then state: an interrupted uncommitted fold is
            # safely recomputed using its saved outcome-independent anchor cache.
            save_npz(predictions_path, ids=ids, y=y, outer_fold=fold_id, **predictions)
            save_json(state_path, state)
            print(f"rice repeat={rep + 1}/10 fold={fold + 1}/5 completed={len(state['folds'])}/50 n={n} seconds={record['seconds']:.2f} " +
                  " ".join(f"{f}:w={k[1]},g={k[2]:g},lambda={LAMBDAS[i]:g}" for f, (_, k, i) in selection.items()), flush=True)
    if len(state["folds"]) != 50 or any(not np.isfinite(p).all() for p in predictions.values()):
        raise AssertionError("Full 10x5 CV has not completed")
    if not np.all(np.sort(fold_id, axis=1)[:, 0] == 0) or (fold_id < 0).any():
        raise AssertionError("Missing outer fold coverage")
    report = {family: core.summarize(y, p) for family, p in predictions.items()}
    paired = {}
    for first, second in (("regional_linear", "snp_linear"), ("local_gaussian", "regional_linear"),
                          ("local_gaussian", "global_snp_rkhs")):
        a, b = report[first]["per_repeat"], report[second]["per_repeat"]
        paired[first + "_minus_" + second] = {}
        for name in a[0]:
            values = [float(x[name] - z[name]) for x, z in zip(a, b) if x[name] is not None and z[name] is not None]
            paired[first + "_minus_" + second][name] = {"per_repeat": values,
                "mean": float(np.mean(values)) if values else None,
                "split_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    report["paired_differences"] = paired
    report["evaluation"] = {"complete": True, "n": n, "snp_count": dosage.shape[1], "outer_repeats": 10,
        "outer_folds": 5, "inner_folds": 5, "completed_outer_folds": 50,
        "prediction_shape_each_model": [10, n], "units": "kg/ha", "metric_aggregation": "metrics on all line-level out-of-fold predictions per repeat; mean and sample SD across 10 repeats; averaged-prediction metrics separately labelled",
        "dataset_sha256": manifest["dataset_sha256"], "fingerprint": fingerprint,
        "window_disclosure": manifest["windows"]["different_search_range_disclosure"],
        "uncertainty_limitation": "Split SD and paired repeat differences are descriptive, not independent biological replication or a model-superiority p-value. No ancestry-blocked or future-season validation claimed.",
        "source": {"paper_doi": "10.1371/journal.pgen.1004982", "data_doi": "10.5061/dryad.7369p",
                   "license": "CC0-1.0", "mirror": "https://www.ricediversity.org/data/Spindel_2015_PLOS/"}}
    save_json(RESULTS / "rice_metrics.json", report)
    state.update(status="complete", complete=True, completed_outer_folds=50,
                 completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), predictions_sha256=sha256(predictions_path),
                 metrics_sha256=sha256(RESULTS / "rice_metrics.json"))
    save_json(state_path, state)
    print("rice FULL CV COMPLETE " + json.dumps({k: {"pearson_mean": report[k]["pearson_mean"],
        "rmse_mean": report[k]["rmse_mean"], "predictive_r2_mean": report[k]["predictive_r2_mean"]} for k in FAMILIES}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "test", "cv", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--feature-chunk", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.feature_chunk <= 64:
        parser.error("feature chunk must be between 1 and 64")
    if args.threads < 1:
        parser.error("threads must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Observed CUDA installation unavailable")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"rice stage={args.stage} device={device} numpy={np.__version__} pandas={pd.__version__} torch={torch.__version__} chunk={args.feature_chunk} float64", flush=True)
    with threadpool_limits(args.threads):
        if args.stage == "prepare":
            prepare()
        elif args.stage == "test":
            selftests(device, args.feature_chunk)
        elif args.stage == "cv":
            nested_cv(device, args.feature_chunk)
        else:
            prepare()
            selftests(device, args.feature_chunk)
            nested_cv(device, args.feature_chunk)


if __name__ == "__main__":
    main()
