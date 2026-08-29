# Nature 2026 Reproduction: Hariharan & Jothi

Leakage-safe local adaptation of **Alzheimer's disease prediction using deep learning and XAI based interpretable feature selection from blood gene expression data** for the AD vs MCI benchmark.

## Local Benchmark

- Input: `task_dataset/processed/ad_mci_binary/X_ad_mci.csv`
- Labels: `task_dataset/processed/ad_mci_binary/y_ad_mci.csv`
- Task: `MCI=0`, `AD=1`
- Protocol: batch-holdout with `shared_test`, `test_gse63060`, `test_gse63061`
- Default repeats: `10`
- Result root: `results/SOTA/nature-2026`

## Paper Methods Implemented

Feature selection is fit only on `train_inner` after train-only median imputation and MinMax scaling:

- `all_genes`: paper no-feature-selection baseline
- `chi2`, default `k=10814`, after train-fitted quantile discretization (`--chi2-bins`, default 10 because the paper omits the count)
- `anova`, default `k=514`
- `rfe`, default `k=1258`, DecisionTreeClassifier RFE
- `elasticnet`, default `k=1000`, logistic ElasticNet coefficient ranking

Optional comparison selectors from the paper introduction are also available:

- `lasso`, paper comparison `k=500`
- `rf_importance`, paper comparison `k=500`

Models:

- `dnn`: dense `7-6-6-6-5-sigmoid` with dropout, adapted from Table 1
- `cnn`: 1D-CNN `Conv4-Conv3-Conv3-Dense4-Dense2-softmax`, following Table 2
- classical models: RBF `svm`, `rf` with 100 trees, `adaboost` with 200 estimators, and `xgboost` with 100 estimators
- deep default: 100 epochs, as reported by the paper

Training balance:

- `--training-balance undersample` (default): deterministic majority undersampling on `train_inner` only
- `--training-balance none`: no class resampling

Validation and test samples are never resampled. Per-split before/after counts are written to `training_balance_manifest.json`.

Augmentation:

- `none` by default
- `ctgan`: external `ctgan.CTGAN`, fit and sampled only on `train_inner`
- `gan`: local Keras conditional GAN approximation, retained for comparison and explicitly labelled `approximation_not_ctgan`
- both generator modes expose the paper-reported setup:
  - latent dimension `128`
  - generator/discriminator hidden layers `[256, 256]`
  - batch size `64`
  - epochs `200`
  - Adam learning rate `0.001`

External CTGAN uses `pac=1` by default because the reported batch size 64 is not divisible by the package default PAC 10; PAC was not disclosed in the paper. The paper's post-generation filtering criterion is also undisclosed and is not invented here. Both limitations are recorded in `augmentation_manifest.json`.

## Smoke Test

```powershell
python SOTA\source\nature-2026\run_experiments.py --smoke --result-root results\SOTA\_smoke_nature-2026
```

## Full Deep Run Without GAN

```powershell
python SOTA\source\nature-2026\run_experiments.py `
  --repeats 10 `
  --feature-selectors all_genes chi2 anova rfe elasticnet lasso rf_importance `
  --models dnn cnn `
  --augmentations none `
  --result-root results\SOTA\nature-2026 `
  --skip-existing
```

## Paper-Like Run With GAN

This is much slower because the GAN is trained per split and feature representation.

```powershell
python SOTA\source\nature-2026\run_experiments.py `
  --repeats 10 `
  --feature-selectors chi2 anova rfe elasticnet `
  --models dnn cnn `
  --augmentations none ctgan gan `
  --training-balance undersample `
  --result-root results\SOTA\nature-2026 `
  --skip-existing
```

## Outputs

Each run writes:

- `metrics.csv`
- `predictions.csv`
- `confusion_matrix.csv`
- `classification_report.csv`
- `selected_genes.txt`
- `feature_ranking.csv`
- `feature_selection_manifest.json`
- `augmentation_manifest.json`
- `hyperparameters.json`
- `run_manifest.json`

Global outputs:

- `all_metrics.csv`
- `summary_by_method.csv`
- `ranking_by_roc_auc.csv`
- `ranking_by_macro_f1.csv`
- `ranking_by_accuracy.csv`
- `leakage_audit.csv`

## Main Differences From The Paper

- The original paper evaluates AD vs CTL and includes ADNI; this pipeline adapts the method to local AD vs MCI on GSE63060/GSE63061.
- The paper reports nested 5-fold CV; this pipeline uses the same batch-holdout protocol used for the other local SOTA comparisons.
- `ctgan` uses the external package; `gan` is a separately named local approximation. The paper does not disclose PAC or its synthetic-sample filtering criterion, so even the external mode is a best-effort reproduction.
- SHAP ranking and GO/KEGG enrichment are not implemented in this first version.
