# Data provenance and processing notes

## Genotype data

Source: AtPolyDB, `call_method_75` release, TAIR9 coordinates.

- Repository: <https://github.com/Gregor-Mendel-Institute/atpolydb>
- Archive: `call_method_75.tar.gz`, 92,088,117 bytes
- Table: 214,051 SNPs × 1,307 accessions
- Calls: A/C/G/T, homozygous inbred lines; no missing values
- Encoding: A=0, C=1, G=2, T=3 (`genotypes_all_1307.npy`, SNPs × accessions, uint8)

The minor allele is redefined within the 83 phenotyped accessions. Because all lines are inbred, a SNP has two observed alleles in this subset and the dosage is coded 0 or 1, not 0/1/2. SNPs with minor allele frequency ≤ 0.05 are excluded from prediction and association, leaving 202,747 SNPs.

The 1,307-accession matrix is used to define haplotype blocks, pretrain the graph encoder, and supply genotypes for the 29 external accessions. The 83-accession matrix is used to fit all phenotype models; the external-29 subset is used only for validation.

## Phenotype data

Source: AtPolyDB `phenotype_published_raw.tsv`; `8_Seed Dormancy` is DSDS50 in days, originally reported by Atwell et al. (2010).

- 83 accessions with non-missing DSDS50
- Range 0–189 days; median 36 days; mean 43.1 days
- Seven accessions with 0 days; one accession (Kas-2) with 189 days

The external set uses the `279_DSDS50` column. That column has 109 non-missing entries, 80 of which overlap the 83 accessions. The remaining 29 accessions are used as an exploratory external set. The two phenotype columns have a Spearman correlation of 0.792 on the 80 overlapping accessions, which supports rank transfer but does not make the measurements equivalent.

## Gene annotation

Source: TAIR10/Araport11 gene models.

- 27,445 protein-coding genes on chromosomes 1–5
- `data/annotations/genes_tair10.csv` contains chromosome, start, end, strand, gene identifier, short name and description
- SNPs were annotated with the nearest gene and with an inside-gene flag

SNP coordinates are TAIR9 and have not been lifted over. The coordinate consistency check in `scripts/03_annotate_snp_genes.py` found that 61.8% of SNPs fall inside a TAIR10 gene and 93.0% are within 5 kb of one. Key loci (DOG1, RGA and the chromosome 1 candidate region) were checked manually. A genome-wide coordinate offset of a few hundred base pairs cannot be excluded.

## Population structure

LD pruning used 100 kb windows and r² > 0.2, retaining 9,245 SNPs. The first five principal components explain 4.13%, 3.55%, 3.20%, 2.87% and 2.67% of the genotype variance. The VanRaden kinship matrix has a mean off-diagonal value of −0.010.

LD decays quickly in this panel: mean r² is 0.348 below 10 kb, 0.126 at 10–20 kb, 0.155 at 20–30 kb and 0.052 at 30–40 kb. The 20–30 kb bin is noisy because only 83 accessions are available.

## Haplotype blocks and graph

Blocks are consecutive runs of at most 25 SNPs, with a maximum span of 50 kb. The definition uses the 1,307 accessions, not the 83 phenotyped accessions, to avoid phenotype-dependent boundaries.

- 8,589 blocks
- Median span 11.5 kb; 90th percentile 21.8 kb
- Node features: mean minor-allele dosage plus the first three principal components of the within-block dosage matrix
- Principal-component signs are fixed by the SNP with the largest absolute loading

Edges:

- Adjacency between consecutive blocks on the same chromosome (8,584 edges)
- LD edges when the absolute correlation of block mean dosages is at least 0.5 and the genomic distance is at most 1 Mb; at most three strongest LD edges per block (3,008 edges)
- Mean absolute LD correlation of LD edges: 0.662

The undirected graph has 11,592 edges. For pretraining, edges are expanded in both directions, self-loops and a virtual node are added, giving 48,951 directed edges.

## Graph encoder and pretraining

The encoder is a two-layer edge-aware graph Transformer with hidden dimension 16, four attention heads and dropout 0.10. Edge features are normalized genomic distance, absolute LD correlation, and a one-hot edge type. A virtual node connects to every block.

Pretraining masks 15% of blocks in each accession and reconstructs the four block features from the remaining graph. The 1,307 accessions are split 1,176/131 into training and validation. Training uses AdamW, learning rate 3×10⁻³, cosine decay, weight decay 1×10⁻⁴, batch size 8 and 25 epochs. No phenotype is used.

The best validation MSE is 0.9052 and the final training MSE is 0.9056. The 83 phenotyped accessions are then encoded into an 83 × 8,589 × 16 block embedding; the 1,307-accession sample representation is 1,307 × 48 (mean, maximum and standard deviation of block embeddings).

## Prediction and association settings

Prediction uses kernel ridge regression in the GBLUP form. The SNP kernel is `XXᵀ/p` with p = 202,747. The block kernel is the average linear kernel of standardized block mean dosages. The graph kernel is the average linear kernel of standardized 16-dimensional block embeddings. The dual kernel is `0.5 K_SNP + 0.5 K_block`; the triple kernel gives equal weight to all three.

The outer loop is 5-fold cross-validation repeated 10 times. The inner loop selects the ridge parameter from 10⁻⁸ to 10² (21 grid points) by 5-fold cross-validation within each training fold. Per-repeat correlations are summarized in `nested_cv_prediction.json`; the paired model comparisons corrected for resampling overlap (Nadeau-Bengio), the Bayesian correlated t and Holm adjustment are computed by `scripts/sensitivity/inference_reanalysis.py`.

The association model is an OLS null with intercept and PC1–3, fitted to standardized DSDS50. For each block, the score statistic is `Q = ||Zᵀr||²`, where Z is the residualized block genotype matrix and r is the residualized phenotype. P values use an Imhof inversion of the mixture chi-square distribution, with a Satterthwaite approximation as a cross-check. The rotation null uses 1,000 draws from the orthogonal complement of the covariate space; the experiment-wide error rate is based on the minimum P value across blocks after standardization. BH q values are reported alongside the rotation results.

## Included processed files

| File | Shape / content |
|---|---|
| `genotypes_83.npy` | 214,051 × 83, uint8 allele codes |
| `genotypes_external29.npy` | 214,051 × 29, uint8 allele codes |
| `block_features_83.npy` | 83 × 8,589 × 4, float32 |
| `block_features_external29.npy` | 29 × 8,589 × 4, float32 |
| `graph_embeddings_83.npy` | 83 × 8,589 × 16, float32 (release asset) |
| `sample_embeddings_1307.npy` | 1,307 × 48, float32 |
| `haplotype_blocks.csv` | block id, chromosome, start, end, SNP row range |
| `block_edges.npz` | edge index and edge attributes |
| `block_snp_map.npz` | SNP row ranges per block |
| `snp_metadata.csv` | position, allele frequencies, minor allele |
| `snp_annotation.csv` | nearest gene and inside-gene flag (release asset) |
| `dormancy_phenotype.csv` | ecotype id, name and DSDS50 |
| `accessions_all.csv` | accession metadata for the 1,307 panel |
| `ssl_encoder.pt` | trained encoder state dict |
| `ssl_summary.json`, `ssl_training_log.csv` | pretraining settings and loss trace |
| `block_summary.json`, `build_summary.json` | build counts and QC summaries |

Files marked "release asset" are attached to the latest GitHub release and can be fetched with `scripts/00_download_large_files.py`.

## External chilling cohort (Dryad)

The external validation uses the Dryad deposit DOI 10.25338/B8VS4P (Martinez-Berdeja et al. 2020, PNAS). `data/external/` holds the original archive `doi_10_25338_B8VS4P__v20191217.zip`, the phenotype spreadsheet `arabidopsis_germ_phen_chilling_DATA.xls` and the DOG1 genotype list `dog1.may22.uniq.txt`. DSDS50 was measured under a different seed batch and chilling protocol than the AtPolyDB phenotype. Of 523 records with valid DSDS50, 206 accessions match the 1,307-accession genotype panel by exact or normalized native name (38 of them among the 83 training accessions, 168 independent); 317 unmatched records are excluded. Matching is audited in `results/tables/external_matching.csv`. Phenotypes of the independent set were never used for training; genotypes of all 1,307 accessions enter block definitions and pretraining, so the projection is transductive in genotypes only.

## Sensitivity and robustness analyses

The scripts in `scripts/sensitivity/` reuse the same processed matrices, phenotype table, kinship and graph embeddings as the main pipeline. They write their summaries to `results/sensitivity/`. The bandwidth and regularization grids, fold-wise centering, grouped-CV permutation null, second-trait replication and DOG1 ablations are all seeded, so the stored JSON/NPZ values reproduce on a re-run. The pretraining configuration sweep and seed replication retrain the graph encoder and require a GPU; only their summary JSONs are included, not the per-run encoders and embeddings.

`results/tables/external_validation.json` and `results/tables/external_validation_predictions.npz` record an earlier external-projection protocol on the 29-accession cohort (legacy lambda grid, mean-only intercept). They are retained as fixed comparator inputs for the calibration and frozen-model validation scripts; they are not regenerated by the current projection script.