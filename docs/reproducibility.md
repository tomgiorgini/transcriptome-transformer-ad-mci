# Reproducibility notes

## Tested environments

- Python syntax/CLI audit: Python 3.13.7 on macOS (August 2026).
- Recommended training environment: Python 3.10–3.12 with PyTorch 2.x.
- GEO data preparation: R 4.5 with Bioconductor.
- CUDA training was used for the complete repeated experiments; CPU/MPS are suitable for dry runs and small smoke tests.

The dependency ranges in `requirements.txt` describe the maintained Python pipeline. `requirements-optional.txt` contains only generative-augmentation backends and an optional TLS certificate bundle.

## Leakage controls

- diagnosis-stratified train/validation/test partitions;
- feature selection and scaling fitted on training data only in the main model protocol;
- no augmentation of validation or test samples;
- validation-only checkpoint selection;
- machine-readable split and configuration manifests for each run.

Global ComBat correction used in some historical pooled experiments is an explicit exception: it is fitted before splitting and therefore allows validation/test distributions to affect harmonization. Raw-cohort and train-aware alternatives should be preferred for fully inductive evaluation.

## Cost tiers

1. **Data-free:** compilation and the three `--help` commands.
2. **Smoke:** one epoch with bounded train/validation batches; requires prepared local data.
3. **Full:** repeated-seed CUDA training and PPI/node2vec initialization.

## R packages

Install from Bioconductor/CRAN as needed:

```r
install.packages("BiocManager")
BiocManager::install(c("GEOquery", "Biobase", "sva"))
```

`sva` is necessary only for the explicit `--combat` reproduction mode.

## Result provenance checklist

For every reported experiment retain locally:

- exact command and arguments;
- Git commit and dirty-tree status;
- Python/R and package versions;
- dataset accession, preprocessing mode, and feature list;
- train/validation/test sample IDs and seeds;
- checkpoint selection metric;
- per-seed metrics before aggregation.
