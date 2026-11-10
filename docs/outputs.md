# Output files

Every file in `results/tables/` with the script that produces it. Regenerating a
file overwrites it in place; all scripts are seeded, so repeated runs reproduce
the stored values.

| File | Producing script | Content |
|---|---|---|
| `qc_summary.json` | `02_qc_population_structure.py` | MAF filter counts, PCA variance shares, LD decay summary |
| `accession_pcs.csv` | `02_qc_population_structure.py` | PC1-5 scores for the 83 accessions |
| `kinship_83.csv` | `02_qc_population_structure.py` | VanRaden kinship matrix |
| `ld_decay_pairs.csv` | `02_qc_population_structure.py` | binned LD decay used for graph edges |
| `annotation_summary.json` | `03_annotate_snp_genes.py` | SNP-to-gene annotation counts and coordinate checks |
| `baseline_prediction.json` | `04_baseline_prediction.py` | GBLUP and random-forest baseline (exploratory) |
| `kernel_validation.json` | `07_kernel_validation.py` | short-repeat validation of centering, scaling and intercept |
| `nested_cv_prediction.json` | `nested_cv_prediction.py` | 10 x 5-fold nested CV for all kernels; grouped leave-one-out (6/8/10 groups) |
| `no_pretraining_control.json` | `08_no_pretraining_control.py` | random-initialization encoder control, per-repeat correlations |
| `prediction_figure_data.npz` | `09_prediction_figure_data.py` | per-repeat observed/predicted values and kernel diagnostics |
| `block_granularity_sensitivity.json` | `10_block_granularity_sensitivity.py` | nested-CV r for 1x / 2x / 4x block merging |
| `block_scan_imhof.csv` | `block_association_scan.py` | per-block Q, Satterthwaite and Imhof P, BH q, rotation P and minP FWER for raw / graph / embedding families |
| `snp_lmm_scan.csv` | `block_association_scan.py` | SNP-level LOCO LMM P values (202,747 markers) |
| `rotation_null.npz` | `block_association_scan.py` | 1,000 fixed-norm residual rotation draws per family |
| `association_summary.json` | `block_association_scan.py` | lambda_GC, minimum P, BH counts, FWER, DOG1 block rank |
| `gene_block_map_unified.csv` | `11_gene_recovery_candidates.py` | gene-to-block mapping (overlap, nearest within 50 kb, none) |
| `known_gene_ranks.csv` | `11_gene_recovery_candidates.py` | percentiles and P values for the known dormancy gene list |
| `candidate_evidence.csv` | `11_gene_recovery_candidates.py` | top blocks with evidence grades A-D |
| `gene_recovery_summary.json` | `11_gene_recovery_candidates.py` | matched-background enrichment test and top-k recovery counts |
| `covariate_sensitivity.csv` / `.json` | `12_covariate_sensitivity.py` | block scan refitted with PC1-5 covariates |
| `phenotype_transform_sensitivity.csv` / `.json` | `13_phenotype_transform_sensitivity.py` | block scan on Blom rank-normal phenotype |
| `external_predictions.csv` | `14_external_projection.py` | frozen-model predictions for the 206 matched Dryad accessions |
| `external_matching.csv` | `14_external_projection.py` | name matching audit trail (matched / unmatched records) |
| `external_projection.json` | `14_external_projection.py` | r, rho, MAE, RMSE for independent (168) and repeatability (38) sets |
| `external_validation.json` / `external_validation_predictions.npz` | not regenerated (fixed comparator) | earlier external-projection protocol on the 29-accession cohort (legacy lambda grid, mean-only intercept); read by `inference_reanalysis.py` and `external29_primary.py` |

Figures are written to `results/figures/` by `15_prediction_figures.py`
(Figures 1-3), `16_external_figure.py` (Figure 4) and
`17_association_figure.py` (Figure 5). Figure files are not committed; the
scripts regenerate them from the tables above.

Processed artefacts in `data/processed/` are produced by
`01_build_genotype_matrix.py` (matrices, metadata, phenotype),
`05_build_haplotype_graph.py` (blocks, edges, features) and
`06_self_supervised_pretrain.py` (encoder weights, embeddings, training log).

## Sensitivity analyses

The scripts in `scripts/sensitivity/` write their summaries to
`results/sensitivity/`. The stored JSON/NPZ files are included; re-running a
script overwrites its outputs in place (except `external29_primary.py`, which
creates its three outputs exclusively and refuses to overwrite). All scripts are
seeded.

| File | Producing script | Content |
|---|---|---|
| `kernel_experiments.json` / `kernel_experiments_preds.npz` | `kernel_experiments.py` | nonlinear kernels, finer blocks, phenotype transforms, permutation null; per-repeat predictions |
| `inference_reanalysis.json` / `coarser_granularity_preds.npz` | `inference_reanalysis.py` | corrected paired tests, Holm adjustment, Fisher-z and bootstrap CIs; coarser-block predictions |
| `gamma_lambda_sensitivity.json` / `gamma_lambda_preds.npz` | `gamma_lambda_sensitivity.py` | fixed-bandwidth ablation and extended lambda grids |
| `foldwise_centering.json` / `foldwise_centering_preds.npz` | `foldwise_centering.py` | training-fold-only centering and scaling |
| `grouped_permutation.json` | `grouped_cv_permutation.py` | permutation null for grouped leave-one-out CV |
| `second_trait.json` / `second_trait_preds.npz` | `second_trait.py` | flowering-time replication (FT10 / FT16 / FT22) |
| `dog1_mechanism.json` | `dog1_mechanism.py` | DOG1 containment, ablation and single-block null |
| `external_calibration.json` | `external_calibration.py` | Dryad scale sensitivity and protocol calibration |
| `external29_primary.json` / `external29_primary.npz` / `external29_primary_frozen.json` | `external29_primary.py` | frozen-model validation on the legacy 29 accessions (exclusive-create outputs) |
| `statistical_audit.json` | `statistical_audit.py` | multiplicity accounting across comparison families |
| `ssl_control_paired.json` / `ssl_control_preds.npz` | `ssl_control_paired.py` | paired trained-versus-random encoder comparison |
| `ssl_best_paired.json` / `ssl_best_preds.npz` | `ssl_best_paired.py` | paired comparisons from the selected sweep configuration |
| `ssl_seed_replication.json` | `ssl_seed_replication.py` | pretraining seed replication (summary only) |
| `ssl_sweep/ssl_sweep_summary.json` | `ssl_sweep.py` | per-configuration reconstruction MSE and prediction metrics (summary only) |

The raw per-run outputs of `ssl_sweep.py` (`ssl_sweep/<config>/`) and
`ssl_seed_replication.py` (`ssl_seed_repl/`) are not included; they are
regenerated by re-running those scripts on a GPU. `ssl_best_paired.py` reads the
per-configuration embeddings produced by `ssl_sweep.py`, so re-run the sweep
first to regenerate them.