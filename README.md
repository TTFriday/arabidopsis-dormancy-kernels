# Genomic kernels for Arabidopsis seed dormancy prediction

This repository contains the analysis code, processed genotype matrices and result tables for a study of seed dormancy in *Arabidopsis thaliana*. The phenotype is DSDS50, the number of days of dry storage required for 50% germination. The prediction panel contains 83 natural accessions with both genotype and phenotype data; haplotype blocks and the graph pretraining use a larger panel of 1,307 accessions.

The analysis compares four genomic kernels (SNP, haplotype-block mean, self-supervised graph, and weighted combinations) under kernel ridge regression with nested cross-validation, and asks whether the same block representation supports locus prioritization. The main conclusions are:

1. The block-mean kernel is the most accurate predictor (Pearson r = 0.715 +/- 0.013 over 10 repeats of 5-fold nested CV); combining kernels does not improve on it.
2. A graph Transformer pretrained by masked block reconstruction adds little over a randomly initialized encoder (r = 0.712 versus 0.703), so the self-supervised representation is reported as an ablation rather than as a source of predictive gain.
3. Generalization is narrow: grouped leave-one-out cross-validation across genetic subpopulations gives negative correlations, and frozen-model projection onto an independent Dryad cohort gives only weak rank agreement (r = 0.150).
4. A leave-one-chromosome-out block-level association scan with Imhof-calibrated score tests and rotation-based family-wise error control finds no experiment-wide significant locus (minimum P = 6.4e-5, family-wise P = 0.205), and known dormancy genes are not enriched among top ranks. The scan is candidate-generating, not discovery-grade, at this sample size.

Robustness of these conclusions is examined in `scripts/sensitivity/`: bandwidth and regularization grids, fold-wise kernel centering, permutation calibration of grouped CV, pretraining seed replication, replication on a second trait (flowering time), and ablations of the DOG1 locus.

## Data

The raw genotype table is the AtPolyDB `call_method_75` release in TAIR9 coordinates:

- Repository: <https://github.com/Gregor-Mendel-Institute/atpolydb>
- File: `call_method_75.tar.gz` (92,088,117 bytes), containing `call_method_75/call_method_75_TAIR9.csv`
- Content: 214,051 SNPs x 1,307 accessions, A/C/G/T calls, no missing genotypes

The phenotype file `data/raw/atpolydb/phenotype_published_raw.tsv` is included. The DSDS50 values are the `8_Seed Dormancy` column, originally from Atwell et al. (2010).

The external cohort is the Dryad deposit of Martinez-Berdeja et al. (2020, PNAS), DOI [10.25338/B8VS4P](https://doi.org/10.25338/B8VS4P). `data/external/` contains the downloaded archive, the phenotype spreadsheet and the DOG1 genotype list. Of 523 accessions with valid DSDS50, 206 match the genotype panel by name: 38 belong to the 83 training accessions (repeatability set) and 168 are independent accessions whose phenotypes were never used for training.

Gene models are TAIR10/Araport11. The source GFF3 and the derived `genes_tair10.csv` are included in `data/annotations/`. SNP positions are TAIR9; no formal liftover was applied. The consistency between the two coordinate sets is checked in `scripts/03_annotate_snp_genes.py` and described in `docs/data_provenance.md`.

## Repository layout

```
scripts/                 analysis scripts, numbered in run order; shared modules unnumbered
scripts/sensitivity/     robustness and sensitivity analyses
data/annotations/        TAIR10/Araport11 gene models
data/processed/          processed matrices and model artefacts
data/external/           Dryad chilling cohort (phenotypes and DOG1 genotypes)
data/raw/atpolydb/       phenotype table and download notes
results/tables/          result tables and JSON summaries
results/sensitivity/     JSON/NPZ summaries of the sensitivity analyses
docs/                    data provenance and output descriptions
```

Large intermediate files are not included:

- `genotypes_all_1307.npy` (267 MB) can be rebuilt with `scripts/01_build_genotype_matrix.py`.
- `block_features.npy` (172 MB, 1,307 x 8,589 x 4) can be rebuilt with `scripts/05_build_haplotype_graph.py`.

Two processed files are distributed as assets of the [latest release](https://github.com/TTFriday/arabidopsis-dormancy-kernels/releases/latest) because they exceed the 25 MB limit of the GitHub web uploader:

- `graph_embeddings_83.npy` (44 MB)
- `snp_annotation.csv` (25 MB)

Run the download helper once before the analysis:

```bash
python scripts/00_download_large_files.py
```

It places both files in `data/processed/` and verifies their SHA-256 checksums. They can also be downloaded manually from the latest release page and moved to that directory.

The repository includes the 83-accession subset and the external-29 genotype caches so that prediction, association, candidate and most sensitivity scripts run without the full matrices; the two release assets above complete the processed data. The pretraining script, its random-initialization control, the external projection and the sensitivity scripts that use the full panel still need the full files, because they use all 1,307 accessions.

## Environment

Python 3.10 with:

```
pip install -r requirements.txt
```

The reported runs used NumPy 2.2, pandas 2.3, SciPy 1.15, scikit-learn 1.6, statsmodels 0.15, PyTorch 2.6 and Matplotlib 3.9. The main pipeline runs on CPU; no GPU is required for it. The pretraining configuration sweep and seed-replication experiments (`scripts/sensitivity/ssl_sweep.py`, `scripts/sensitivity/ssl_seed_replication.py`) were run on a CUDA GPU and are best reproduced on one.

## Running the analysis

### From the included processed data

```bash
python scripts/00_download_large_files.py        # fetch release assets if absent
python scripts/02_qc_population_structure.py     # QC, PCA, LD decay, kinship
python scripts/04_baseline_prediction.py         # optional GBLUP / random-forest baseline
python scripts/07_kernel_validation.py           # short validation of the kernel implementation
python scripts/nested_cv_prediction.py           # 10 x 5-fold nested CV, all kernels, grouped LOO
python scripts/08_no_pretraining_control.py      # random-initialization encoder control
python scripts/09_prediction_figure_data.py      # per-repeat curves for the figures
python scripts/10_block_granularity_sensitivity.py   # 2x / 4x block merging
python scripts/block_association_scan.py         # LOCO block scan, Imhof + rotation FWER
python scripts/11_gene_recovery_candidates.py    # gene mapping, known genes, candidate grades
python scripts/12_covariate_sensitivity.py       # PC1-5 versus PC1-3 null
python scripts/13_phenotype_transform_sensitivity.py  # Blom rank-normal transform
python scripts/14_external_projection.py         # frozen-model projection onto Dryad cohort
python scripts/15_prediction_figures.py          # Figures 1-3
python scripts/16_external_figure.py             # Figure 4
python scripts/17_association_figure.py          # Figure 5
```

`nested_cv_prediction.py` and `block_association_scan.py` are also imported as modules by the sensitivity and figure scripts. On a laptop CPU the nested cross-validation takes roughly 25 minutes; the association scan with 1,000 residual rotations a few minutes more.

### Sensitivity analyses

`scripts/sensitivity/` contains the robustness checks. The scripts resolve the repository root from their own location, share `scripts/kernel_utils.py` and the data loaders in `scripts/sensitivity/kernel_experiments.py`, and write all outputs to `results/sensitivity/`. The stored JSON/NPZ summaries are included, so the analyses can be inspected without re-running them.

```bash
python scripts/sensitivity/kernel_experiments.py        # nonlinear kernels, finer blocks, phenotype transforms, permutation
python scripts/sensitivity/inference_reanalysis.py      # corrected paired tests, Holm adjustment, Fisher-z and bootstrap CIs
python scripts/sensitivity/gamma_lambda_sensitivity.py  # fixed-bandwidth ablation, extended lambda grids
python scripts/sensitivity/foldwise_centering.py        # training-fold-only centering and scaling
python scripts/sensitivity/grouped_cv_permutation.py    # permutation null for grouped leave-one-out CV
python scripts/sensitivity/second_trait.py              # flowering-time replication (FT10 / FT16 / FT22)
python scripts/sensitivity/dog1_mechanism.py            # DOG1 containment, ablation and single-block null
python scripts/sensitivity/external_calibration.py      # Dryad scale sensitivity and protocol calibration
python scripts/sensitivity/external29_primary.py        # frozen-model validation on the legacy 29 accessions
python scripts/sensitivity/statistical_audit.py         # multiplicity accounting across comparison families
python scripts/sensitivity/ssl_control_paired.py        # paired trained-versus-random encoder comparison
```

Requirements beyond the included data:

- Most scripts need only the 83-accession matrices and the two release assets.
- `second_trait.py` and `external29_primary.py` additionally need the full 1,307-accession matrices (`genotypes_all_1307.npy`, `block_features.npy`, rebuilt by scripts 01 and 05) and the raw `call_method_75_TAIR9.csv` table. The external-29 genotype caches (`genotypes_external29.npy`, `block_features_external29.npy`) are included.
- `external29_primary.py` creates its three output files exclusively and refuses to overwrite them; move the stored `external29_primary*` files out of `results/sensitivity/` before a re-run.
- `results/tables/external_validation.json` and `results/tables/external_validation_predictions.npz` record an older external-projection protocol on the 29-accession cohort (legacy lambda grid, mean-only intercept). They are kept as fixed comparator inputs read by `inference_reanalysis.py` and `external29_primary.py`; they are not regenerated by the included scripts.

GPU re-training:

- `scripts/sensitivity/ssl_sweep.py` (five encoder configurations) and `scripts/sensitivity/ssl_seed_replication.py` (eight configuration-seed pairs) retrain the graph encoder; a CUDA GPU is strongly recommended.
- Their raw per-run outputs (`results/sensitivity/ssl_sweep/<config>/` and `results/sensitivity/ssl_seed_repl/`, together about 1.3 GB of encoders and embeddings) are not included; only the summary JSONs are.
- `ssl_best_paired.py` reads the per-configuration `graph_embeddings_83.npy` files produced by `ssl_sweep.py`, so re-run the sweep first to regenerate them:

```bash
python scripts/sensitivity/ssl_sweep.py            # GPU re-training, regenerates per-config embeddings
python scripts/sensitivity/ssl_best_paired.py      # paired comparisons from the regenerated embeddings
python scripts/sensitivity/ssl_seed_replication.py # GPU re-training across seeds
```

- `ssl_seed_replication.py` resumes from its stored summary JSON and skips runs marked complete; delete the corresponding run entries (or the summary file) to force re-training and regenerate the per-seed embeddings.

### From the raw data

1. Download `call_method_75.tar.gz` from the AtPolyDB repository and unpack it to `data/raw/atpolydb/call_method_75/`.
2. Run the pipeline. The TAIR10 GFF3 and the derived gene table are already in `data/annotations/`.

```bash
python scripts/01_build_genotype_matrix.py
python scripts/02_qc_population_structure.py
python scripts/03_annotate_snp_genes.py
python scripts/05_build_haplotype_graph.py
python scripts/06_self_supervised_pretrain.py
```

then continue with the block above starting at `07_kernel_validation.py`.

All scripts write to `data/processed/`, `results/tables/`, `results/sensitivity/` and `results/figures/`. Paths are resolved relative to the repository root, so the checkout can be placed anywhere.

The pretraining script takes about 17 minutes on a 16-core CPU with single-threaded BLAS. Setting `OMP_NUM_THREADS=1` is recommended when reproducing the reported run:

```bash
OMP_NUM_THREADS=1 python scripts/06_self_supervised_pretrain.py
```

## Key result files

| File | Content |
|---|---|
| `results/tables/nested_cv_prediction.json` | nested-CV metrics for all kernels, grouped leave-one-out results |
| `results/tables/kernel_validation.json` | kernel centering / scaling validation |
| `results/tables/known_gene_ranks.csv` | block percentiles and P values for known dormancy genes |
| `results/tables/candidate_evidence.csv` | top candidate blocks with evidence grades |
| `results/tables/block_scan_imhof.csv` | block-level statistics for all 8,589 blocks, three test families |
| `results/tables/snp_lmm_scan.csv` | SNP-level LOCO linear-mixed-model scan |
| `results/tables/rotation_null.npz` | the 1,000 residual-rotation null draws (seeded, regenerable) |
| `results/tables/external_projection.json` | Dryad projection metrics (independent and repeatability sets) |
| `results/tables/covariate_sensitivity.csv` | PC1-5 covariate sensitivity of the block scan |
| `results/tables/phenotype_transform_sensitivity.csv` | rank-normal phenotype sensitivity |
| `results/sensitivity/` | sensitivity-analysis summaries; see `docs/outputs.md` for the full list |

Tables 1 and 2 are assembled from `qc_summary.json`, `build_summary.json`, `block_summary.json` and `nested_cv_prediction.json`; `docs/outputs.md` lists every output with its producing script.

## Notes on interpretation

- The panel has 83 phenotyped accessions. The block-level scan is calibrated but has limited power; the smallest block P value does not reach experiment-wide significance after rotation calibration, and known dormancy genes (including DOG1) are not enriched among top ranks. These negative results are reported as such.
- The graph kernel is highly correlated with the block kernel, and a randomly initialized encoder performs almost as well. The self-supervised representation is therefore an ablation, not a source of major predictive gain.
- Block definitions and pretraining use the genotypes of all 1,307 accessions, so the external projection is transductive with respect to genotypes but fully independent with respect to phenotypes.
- SNP positions are TAIR9 and gene models are TAIR10/Araport11. Key loci were checked manually, but a formal liftover was not performed.

## Contact

Please open an issue in this repository for questions about the code or data.