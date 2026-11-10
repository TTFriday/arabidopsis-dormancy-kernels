# Raw data

The genotype table is not included because of its size (562 MB uncompressed).

Download `call_method_75.tar.gz` from the AtPolyDB repository:

<https://github.com/Gregor-Mendel-Institute/atpolydb>

The archive is 92,088,117 bytes and unpacks to:

```
data/raw/atpolydb/call_method_75/call_method_75_TAIR9.csv
```

The phenotype table `phenotype_published_raw.tsv` is included in this directory. Its `8_Seed Dormancy` column contains the 83 DSDS50 values; `279_DSDS50` supplies the 29 external accessions.

After unpacking the genotype table, run `scripts/01_build_genotype_matrix.py` to create the compact NumPy matrices used by the rest of the pipeline.

The TAIR10 GFF3 file is already included at `data/annotations/Arabidopsis_TAIR10.gff3.gz`, together with the derived gene table. It is only needed if the annotation has to be rebuilt.
