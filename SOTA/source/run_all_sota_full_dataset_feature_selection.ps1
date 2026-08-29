$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

Write-Host "Running Lee 2020 full-dataset feature selection..."
python SOTA\source\nature-2020\run_experiments.py `
  --repeats 10 `
  --batch-scenarios shared_test test_gse63060 test_gse63061 `
  --artifact-scope full_dataset `
  --feature-sets deg vae tf_genes cfg_genes hub_genes `
  --models lr l1_lr svm rf dnn `
  --fdr-threshold 0.1 `
  --deg-fallback-top-k 300 `
  --dnn-max-epochs 300 `
  --dnn-patience 30 `
  --result-root results\SOTA\nature-2020-fullfs `
  --skip-existing

Write-Host "Running Kelly 2023 full-dataset feature selection..."
python SOTA\source\nature-2023\run_experiments.py `
  --protocol batch_holdout `
  --batch-scenarios shared_test test_gse63060 test_gse63061 `
  --repeats 10 `
  --artifact-scope full_dataset `
  --feature-sets lasso vssrfe_lr vae_latent `
  --models lr svm rf xgboost mlp vae_classifier cnn `
  --deep-feature-models `
  --hyperparameter-mode fixed_paper `
  --disable-fixed-vssrfe-n-genes `
  --vssrfe-min-genes 5 `
  --vssrfe-max-genes 200 `
  --vssrfe-step-genes 5 `
  --deep-epochs 100 `
  --vae-epochs 100 `
  --deep-patience 3 `
  --result-root results\SOTA\nature-2023-fullfs `
  --skip-existing

Write-Host "Running One2MFusion 2023 full-dataset feature selection..."
python SOTA\source\one2mfusion-2023\run_experiments.py `
  --protocol batch_holdout `
  --batch-scenarios shared_test test_gse63060 test_gse63061 `
  --repeats 10 `
  --artifact-scope full_dataset `
  --hyperparameter-mode fixed_paper `
  --paper-lasso-alpha 1e-6 `
  --gene-selection-mode nonzero `
  --image-gene-order input `
  --models fnn cnn one2mfusion `
  --epochs 300 `
  --patience 10 `
  --result-root results\SOTA\one2mfusion-fullfs-lambda1e-6 `
  --skip-existing

Write-Host "Running TabNet 2023 full-dataset DGS..."
python SOTA\source\tabnet-2023\run_experiments.py `
  --protocol batch_holdout `
  --batch-scenarios shared_test test_gse63060 test_gse63061 `
  --repeats 10 `
  --artifact-scope full_dataset `
  --dgs-method limma `
  --dgs-adj-p-threshold 0.01 `
  --dgs-fallback-thresholds 0.05 0.1 0.2 `
  --dgs-fallback-p-thresholds 0.01 0.05 `
  --models dgs_tabnet `
  --max-epochs 300 `
  --patience 50 `
  --result-root results\SOTA\tabnet-fullfs `
  --skip-existing

Write-Host "Running Diagnostics 2025 full-dataset feature selection..."
python SOTA\source\diagnostics-2025\run_experiments.py `
  --protocol batch_holdout `
  --batch-scenarios shared_test test_gse63060 test_gse63061 `
  --repeats 10 `
  --artifact-scope full_dataset `
  --models dl svm gbm rf `
  --sampling no_smote borderline_smote `
  --result-root results\SOTA\diagnostics-2025-fullfs `
  --skip-existing

Write-Host "Running Nature 2026 full-dataset feature selection with automatic gene counts..."
python SOTA\source\nature-2026\run_experiments.py `
  --protocol batch_holdout `
  --batch-scenarios shared_test test_gse63060 test_gse63061 `
  --repeats 10 `
  --artifact-scope full_dataset `
  --feature-count-mode auto `
  --auto-k-values 100 200 500 1000 2000 `
  --auto-k-scoring roc_auc `
  --auto-k-cv-folds 5 `
  --feature-selectors chi2 anova rfe elasticnet `
  --models dnn cnn `
  --augmentations none gan `
  --deep-epochs 300 `
  --deep-patience 50 `
  --gan-target-size 2000 `
  --gan-epochs 200 `
  --result-root results\SOTA\nature-2026-fullfs-autok `
  --skip-existing

Write-Host "All full-dataset feature-selection SOTA runs finished."
