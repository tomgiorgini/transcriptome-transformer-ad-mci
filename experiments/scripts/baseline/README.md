# Baseline and Dataset Selection Scripts

## XGBoost gene selection

`select_genes_xgboost.py` selects genes for the AD/MCI task from the full AD/MCI gene matrix, not from the DEG list.

Install the dependency in the active Python environment:

```powershell
python -m pip install xgboost
```

Default run, automatically choosing the panel size from candidate K values using train-only CV AUROC and the one-standard-error rule:

```powershell
python experiments\scripts\baseline\select_genes_xgboost.py `
  --x-file task_dataset\processed\ad_mci_binary\X_ad_mci.csv `
  --y-file task_dataset\processed\ad_mci_binary\y_ad_mci.csv `
  --split-file task_dataset\processed\ad_mci_binary\splits\official_seed42.csv `
  --output-dir task_dataset\processed\ad_mci_xgboost_auto
```

To select by another criterion:

```powershell
python experiments\scripts\baseline\select_genes_xgboost.py `
  --output-dir task_dataset\processed\ad_mci_xgboost_auto_macro_f1 `
  --auto-select-by cv_macro_f1
```

To force a fixed DEG-sized comparison panel:

```powershell
python experiments\scripts\baseline\select_genes_xgboost.py `
  --output-dir task_dataset\processed\ad_mci_xgboost_top788 `
  --selection-mode fixed `
  --top-k 788
```

Outputs:

```text
task_dataset/processed/ad_mci_xgboost_auto/X_xgboost_top{selected_k}_ad_mci.csv
task_dataset/processed/ad_mci_xgboost_auto/y_ad_mci.csv
task_dataset/processed/ad_mci_xgboost_auto/splits/official_seed42.csv
task_dataset/processed/ad_mci_xgboost_auto/xgboost_gene_ranking.csv
task_dataset/processed/ad_mci_xgboost_auto/xgboost_cv_metrics.csv
task_dataset/processed/ad_mci_xgboost_auto/xgboost_candidate_k_metrics.csv
task_dataset/processed/ad_mci_xgboost_auto/xgboost_selection_report.json
```

The ranking is computed using only the official train split, with internal stratified CV, to avoid selecting genes from validation/test information.
