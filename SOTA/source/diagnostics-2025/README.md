# Diagnostics 2025 Leakage-Safe Reproduction

This pipeline adapts Sarma and Chatterjee 2025 to the local AD vs MCI benchmark.

## Method

- Dataset: `task_dataset/processed/ad_mci_binary/X_ad_mci.csv` and `y_ad_mci.csv`.
- Task: `MCI=0`, `AD=1`.
- Protocol: batch-holdout with `shared_test`, `test_gse63060`, and `test_gse63061`.
- Feature selection: train-only MinMax scaling, XGBoost ranking, top 300 genes, then actual floating backward selection with a fixed 95-gene target (`--sfbs-mode true --sfbs-target-genes 95`).
- Models: dense DL, SVM RBF, Gradient Boosting, Random Forest.
- Sampling: both `no_smote` and train-only `borderline_smote`.

## Usage

```powershell
python SOTA\source\diagnostics-2025\run_experiments.py --smoke
```

Full run:

```powershell
python SOTA\source\diagnostics-2025\run_experiments.py `
  --protocol batch_holdout `
  --batch-scenarios shared_test test_gse63060 test_gse63061 `
  --repeats 10 `
  --models dl svm gbm rf `
  --sampling no_smote borderline_smote `
  --sfbs-mode true `
  --sfbs-target-genes 95 `
  --sfbs-cv-folds 5 `
  --result-root results\SOTA\diagnostics-2025 `
  --skip-existing
```

Expected full run size: `3 scenarios x 10 repeats x 4 models x 2 sampling = 240` metric files.

True SFBS uses `mlxtend.SequentialFeatureSelector` with an L2 logistic-regression estimator and the local strict-v2 score. Those are explicit best-effort assumptions because the paper does not disclose the SFBS estimator, scorer, stopping/tie-breaking rules, XGBoost hyperparameters, selected genes, or seeds. Exact reproduction is therefore not possible from the publication alone.

SFBS failures abort by default. `--allow-sfbs-fallback` explicitly permits an XGBoost-ranking-only fallback, recorded as `fallback_not_sfbs`. `--sfbs-mode approximate_lr` remains available for exploratory runs and is recorded as `approximation_not_sfbs`; neither mode is silently relabelled as exact SFBS.

## Outputs

- Per run: metrics, predictions, confusion matrix, classification report, selected genes, XGBoost ranking, SFBS trace, sampling manifest, hyperparameters, run manifest.
- Aggregates: `all_metrics.csv`, `summary_by_method.csv`, `ranking_by_roc_auc.csv`, `ranking_by_macro_f1.csv`, `ranking_by_accuracy.csv`, `leakage_audit.csv`.
