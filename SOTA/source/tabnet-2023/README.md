# TabNet 2023 DGS-TabNet Reproduction

This folder reproduces the core method from Jin et al. 2023, "Classification of Alzheimer's disease using robust TabNet neural networks on genetic data", adapted to the local AD vs MCI benchmark.

## Benchmark

Default input:

- `task_dataset/processed/ad_mci_binary/X_ad_mci.csv`
- `task_dataset/processed/ad_mci_binary/y_ad_mci.csv`

The task is AD vs MCI with `MCI=0` and `AD=1`.

## Protocol

The default run uses the same batch-holdout scenarios used for the other SOTA reproductions:

- `shared_test`
- `test_gse63060`
- `test_gse63061`

DGS can be run in two modes:

- `--artifact-scope full_dataset`: paper-like DGS on the full expression matrix, marked as `intentional_leakage`.
- `--artifact-scope train_inner`: leakage-safe DGS fitted separately inside each training split.

## Method

- limma DGS is fitted on the unscaled expression matrix, not on ML-scaled features.
- The primary DGS threshold is `adj.P.Value < 0.01`.
- If a split selects zero genes, optional relaxed FDR thresholds can be tried with `--dgs-fallback-thresholds`.
- If no relaxed FDR threshold selects genes, optional nominal p-value thresholds can be tried with `--dgs-fallback-p-thresholds`; this still does not force a fixed gene count.
- Median imputation and MinMax scaling are fitted after gene selection, on selected genes only.
- No top-k cap: all genes passing the effective FDR threshold are used.
- DGS-TabNet only, implemented with `pytorch-tabnet`.
- Fixed paper-space TabNet hyperparameters, no Bayesian tuning.
- Feature importances and globally important genes are saved per scenario.

## Usage

Install dependencies:

```powershell
python -m pip install -r SOTA\source\tabnet-2023\requirements-tabnet-2023.txt
```

Smoke test:

```powershell
python SOTA\source\tabnet-2023\run_experiments.py --smoke
```

Default run:

```powershell
python SOTA\source\tabnet-2023\run_experiments.py
```

Outputs are written to:

```text
results/SOTA/tabnet-2023
```

Main aggregate files:

- `all_metrics.csv`
- `summary_by_method.csv`
- `ranking_by_pr_auc.csv`
- `leakage_audit.csv`
- `dgs_manifest.json`

Per-run outputs include metrics, predictions, confusion matrix, selected genes, DGS table, feature importances, globally important genes, hyperparameters, and a run manifest.
