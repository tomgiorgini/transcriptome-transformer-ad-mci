$ErrorActionPreference = "Stop"

Write-Host "Running Nature 2023 strict-v2: VSSRFE+RF, StandardScaler, leakage-free..."
python SOTA\source\nature-2023\run_experiments.py --strict-v2

Write-Host "Running simple baselines top1000 with StandardScaler on AD vs MCI..."
python SOTA\source\simple-baselines\run_experiments.py `
  --x-file task_dataset\processed\ad_mci_binary\X_ad_mci.csv `
  --y-file task_dataset\processed\ad_mci_binary\y_ad_mci.csv `
  --top-variance-genes 1000 `
  --scaler standard `
  --models lr l1_lr svm_rbf rf xgboost gnb `
  --result-root results\SOTA\simple-baselines-top1000-ad-mci-standard `
  --overwrite-results

Write-Host "Done."
