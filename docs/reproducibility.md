# Reproducibility notes

## Tested environments

- Python syntax/CLI audit: Python 3.13.7 on macOS (August 2026).
- Recommended training environment: Python 3.10–3.12 with PyTorch 2.x.
- R data/DEG scripts: R 4.5 with Bioconductor.
- CUDA training was used for the complete repeated experiments; CPU/MPS are suitable for dry runs and small smoke tests.

The dependency ranges in `requirements.txt` describe the maintained Python pipeline. `requirements-optional.txt` contains literature baselines, generative augmentation, legacy T-GEM, and PDF tooling.

## Leakage controls

- diagnosis-stratified train/validation/test partitions;
- feature selection and scaling fitted on training data only in the main model protocol;
- no augmentation of validation or test samples;
- validation-only checkpoint selection;
- shared manifests across reproduced methods where applicable.

Global ComBat correction used in some historical pooled experiments is an explicit exception: it is fitted before splitting and therefore allows validation/test distributions to affect harmonization. Raw-cohort and train-aware alternatives should be preferred for fully inductive evaluation.

## Cost tiers

1. **Data-free:** compilation, unit tests, and `--help`/`--dry-run` commands.
2. **Smoke:** one short CPU run using `--smoke`; requires prepared local data.
3. **Full:** repeated ten-seed CUDA training, PPI/node2vec initialization, SOTA grids, and attention export.

## R packages

Install from Bioconductor/CRAN as needed:

```r
install.packages("BiocManager")
BiocManager::install(c("GEOquery", "Biobase", "limma", "sva"))
install.packages(c("enrichR", "forcats", "ggplot2", "openxlsx", "pheatmap", "stringr"))
```

`sva` is necessary only for `--combat`. Network-backed enrichment services can change over time; preserve the returned database version and date with any published result.

## Result provenance checklist

For every reported experiment retain locally:

- exact command and arguments;
- Git commit and dirty-tree status;
- Python/R and package versions;
- dataset accession, preprocessing mode, and feature list;
- train/validation/test sample IDs and seeds;
- checkpoint selection metric;
- per-seed metrics before aggregation.
