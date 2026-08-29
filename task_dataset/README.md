# Local task data

This directory is a local data workspace. Expression matrices and generated splits are ignored by Git.

## Source cohorts

- NCBI GEO [GSE63060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63060)
- NCBI GEO [GSE63061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63061)

The expected retained cohort is 711 samples (284 AD, 189 MCI, 238 controls). Probe annotation, aggregation to gene symbols, and intersection across platforms yield 19,460 shared genes in the version used for the thesis.

## Generate local inputs

From the repository root:

```bash
Rscript pretraining_dataset/scripts/prepare_addneuromed_geo.R \
  --output-dir=task_dataset --install
```

This creates:

```text
task_dataset/matrix.txt
task_dataset/AD.txt
task_dataset/MCI.txt
task_dataset/CTL.txt
task_dataset/geo_preparation_manifest.tsv
```

Use `--combat` only for the experiments that explicitly require the original two-batch global ComBat matrix. Since that correction is fitted before splitting, it is transductive and must not be described as a fully inductive held-out protocol.

Build the canonical Python datasets:

```bash
python experiments/scripts/baseline/build_alzheimer_dataset.py
python experiments/scripts/paper_comparison/build_txt_pairwise_datasets.py
```

The first command creates a diagnosis-stratified 70/10/20 split at seed 42. Repeated evaluation protocols create their own seed-specific splits from the same aligned cohort.

Do not commit generated matrices, individual-level metadata, or split manifests containing local absolute paths.
