# Arabidopsis seed-dormancy genomic prediction

This archive contains software, plant data and numerical results for genomic
prediction of DSDS50 (days of dry storage to 50% germination) in Arabidopsis.
The training cohort has 83 accessions; the genotype panel has 1,307 accessions.

## Audited evaluation and legacy evidence

`revision/analysis/` implements the audited evaluation: training-only nested
cross-validation, matched-input ablations, strict competing baselines,
canonical-ID external validation, rice validation, flowering-time checks,
independent-reference graph training, and numerical and robustness diagnostics.
`revision/results/` contains the completed results, all final out-of-fold (OOF)
predictions and fold records, including all six flowering cohorts and all twelve
trained/untrained models for each of the 25- and 100-epoch graph experiments.

`results/legacy/` contains only frozen numerical inputs needed to reproduce the
legacy identity and rank-one audits. The legacy projection used sorted metadata
positions as raw genotype columns; 205 of 206 projections had mismatched
identities. Its previously described independent cohort and prediction accuracy
must not be interpreted as validated external performance. The audited pipeline
uses canonical ecotype IDs, excludes ambiguous and duplicate IDs, and separates
training overlaps from independent measurements. The legacy rank-one formula
also lacked training-fold feature centering. Both problems are documented in the
included numerical audits. Historical representations and their results do not
establish haplotype or pretrained-encoder superiority. All prespecified positive,
zero and negative outcomes are retained. Repeated folds describe split variability,
not independent biological replication.

The numbered preprocessing scripts and `scripts/legacy_*` are historical source
or forensic evidence; `scripts/reference_encoder.py` preserves the exact two
encoder class definitions. They are distinct from the audited evaluation.

## Layout and data

- `data/processed/`: original nucleotide codes, complete 1,307-accession arrays,
  legacy block features, coordinate windows, phenotype and accession metadata.
- `data/raw/atpolydb/`: full original TAIR9 genotype CSV, accession information
  and published phenotype TSV; no download is needed for Arabidopsis rebuilds.
- `data/annotations/genes_tair10.csv`: source-derived gene coordinates, TAIR10.
- `data/external/`: original Dryad chilling-study phenotype spreadsheet.
- `revision/data/`: rice source archives, raw-missingness-aware numeric dataset,
  marker metadata and source/license evidence.
- `revision/results/package_*.json`: file checksums and packaging provenance.

The original TAIR9 and TAIR10 nuclear assemblies use compatible coordinates,
so the release change does not require liftover. Annotation versions can still
differ in gene models. Rice marker ascertainment and panel filtering remain
upstream limitations.
See `DATA_SOURCES.txt` for citations and dataset-specific license boundaries.

## Environment and checks

Use Python 3.10 in a dedicated environment:

```bash
python -m pip install -r requirements.txt
python -s -B revision/analysis/package_checks.py --output ./local_checks
```

The code uses ordinary installed PyTorch, on CPU or optional CUDA; it does not
search a Windows user profile for a different installation. CPU imports and unit
checks do not require a GPU. Provision an appropriate PyTorch 2.6 CUDA wheel if
GPU training is wanted. Full graph training and nested CV are long-running;
CUDA scatter reductions need not be bitwise reproducible across devices.

## Recompute without mixing historical checkpoints

Packaged historical code hashes describe the original computations, not the
path-adapted software. `package_manifest.json` is authoritative for the actual
archive bytes. The reconstruction driver copies the inputs into a NEW workspace
and starts with fresh outputs. It refuses an existing output directory, never
reuses incompatible historical signatures, and never edits the recorded arrays:

```bash
python -s -B revision/analysis/rebuild.py --analysis numerical_audit --output ./recomputed_audit
python -s -B revision/analysis/rebuild.py --analysis kernel_benchmark --output ./recomputed_kernels -- --mode strict
python -s -B revision/analysis/rebuild.py --analysis strong_baselines --output ./recomputed_baselines -- --mode run
python -s -B revision/analysis/rebuild.py --analysis rice_validation --output ./recomputed_rice -- --stage all --device cpu
python -s -B revision/analysis/rebuild.py --analysis graph_validation --output ./recomputed_graph25 -- --mode run
python -s -B revision/analysis/rebuild.py --analysis graph_validation_long --output ./recomputed_graph100 -- --mode run
```

The same driver accepts any audited analysis name. Frozen primary comparator
outputs are copied only when needed by regional-RKHS, geometry, streaming and
conditional-uncertainty diagnostics; this is recorded by the driver. Graph
reference feature arrays and node embeddings are intentionally omitted: they
are regenerated from the included nucleotide panel by `--mode run`. All encoder
weights, training logs and final OOF arrays are included. Rice kernel caches and
BayesC preprocessing/task caches are also regenerable, not required inputs.
Rice source schema evidence is a short attributed excerpt, not an article copy.

## License and distribution

`LICENSE` applies to original software only (MIT), not to third-party data.
The AtPolyDB and Dryad chilling datasets are CC0-1.0, TAIR public annotation
data are CC-BY-4.0, and the rice dataset is CC0-1.0. No new MIT grant is made
for those datasets or their derived representations.

This is a local archive prepared for author-controlled deposit. No external
publication or GitHub write is performed by the package builder. The web staging
batches are each at most 90 files and every individual file is below 24 MiB.
Larger inputs are separately staged as release assets AND present at their final
paths in the complete ZIP, so the ZIP needs no release-asset downloads.
