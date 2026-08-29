# Final TxT Multitask Pipeline

This pipeline targets three binary tasks with one shared TxT encoder and one shared selected gene set:

- `AD_vs_MCI`
- `AD_vs_CTL`
- `MCI_vs_CTL`

The primary optimization target is `AD_vs_MCI`; the other heads are auxiliary tasks that regularize the shared representation.

## Design Decisions

- Use a single shared encoder and separate classification heads, adapting the [TxT multitask design](https://doi.org/10.1093/bib/bbaf628).
- Keep `--task-specific-pooling off` for the final thesis pipeline so all heads consume the same encoded shared gene set.
- Initialize gene embeddings from HIPPIE/PPI node2vec with `--embed-file`; keep `--embed-dim == --d-model`, matching the TxT supplementary setup.
- Prefer `ppi_node_embedding.csv` plus `--embedding-gene-policy mapped_only` when testing the TxT-style PPI inductive bias.
- Use simple train-only gene selection because AD/MCI has weak split-level differential signal:
  - `mad`: unsupervised top-k median absolute deviation.
  - `pairwise_anova_union`: equal round-robin union from the three binary ANOVA rankings.
  - `ad_mci_priority_anova_union`: shared union with a protected AD/MCI quota.
- Use `borderline_smote` as the main HDLSS augmentation candidate, aligned with the local Diagnostics 2025 reproduction, applied only after train-only feature selection/scaling and never to validation/test. SMOTE variants clip generated values only under MinMax scaling. Keep `ctgan` as a secondary ablation, aligned with the local Nature 2026 CTGAN/TGAN-style reproduction, because previous local runs did not show consistent gains; CTGAN is restricted to `--scaler minmax`.
- Use `val_primary_auc_minus_025_loss` as the default checkpoint metric: it prioritizes `AD_vs_MCI` validation ROC-AUC, keeps 25% auxiliary-task AUC support, and penalizes validation loss. If AD/MCI AUC is undefined in a split, the checkpoint falls back to AD/MCI macro-F1 before falling back to negative validation loss.

## PPI Init

The one-command entry point for the final workflow is:

It assumes the canonical multiclass files already exist under
`task_dataset/processed/alzheimer_multiclass/`. If only `task_dataset/AD.txt`,
`task_dataset/MCI.txt`, and `task_dataset/CTL.txt` are present, first place the
raw expression matrix as `task_dataset/matrix.txt` and run
`experiments/scripts/baseline/build_alzheimer_dataset.py`.

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_final_pipeline.py `
  --preset balanced `
  --run-mode cv `
  --grid-skip-existing `
  --skip-existing-ppi
```

Dry-run it first to inspect the dataset, PPI, and grid commands without executing them:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_final_pipeline.py `
  --preset balanced `
  --dry-run
```

Use `--skip-existing-ppi` when rerunning after an embedding dimension has already been built; otherwise the wrapper will rebuild that PPI embedding before launching the grid.

Build the shared multitask dataset first, if it is not already present:

```powershell
python experiments\scripts\paper_comparison\build_txt_pairwise_datasets.py
```

Build one HIPPIE/PPI embedding per TxT dimension:

```powershell
.\experiments\scripts\pretraining\run_ppi_init.ps1 -EmbeddingDim 64
.\experiments\scripts\pretraining\run_ppi_init.ps1 -EmbeddingDim 128
.\experiments\scripts\pretraining\run_ppi_init.ps1 -EmbeddingDim 256
```

The default `--score-threshold 0.73` is intentional: HIPPIE defines high confidence as the third quartile of its score distribution, `0.73`. Do not lower this to `0.70` for the final high-quality PPI run unless explicitly running an ablation.

Expected node embedding paths:

```text
results/pretraining/ppi_init/hippie_highconf_dim64_score0p73_seed42/ppi_node_embedding.csv
results/pretraining/ppi_init/hippie_highconf_dim128_score0p73_seed42/ppi_node_embedding.csv
results/pretraining/ppi_init/hippie_highconf_dim256_score0p73_seed42/ppi_node_embedding.csv
```

## Final Grid

The grid has four presets:

- `pilot`: small PPI grid for quick validation.
- `balanced`: default final search, PPI-first, AD/MCI-priority.
- `wide`: larger ablation grid including random embeddings, variance, SMOTE, and more top-k sizes.
- `random_sanity`: one small random-init control.

Dry-run the default planned grid:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_final_grid.py `
  --preset balanced `
  --dry-run `
  --skip-missing-ppi
```

Run a quick PPI pilot after PPI embeddings exist:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_final_grid.py `
  --preset pilot `
  --run-mode cv `
  --skip-existing
```

Run the default PPI-first balanced grid:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_final_grid.py `
  --preset balanced `
  --run-mode cv `
  --skip-existing
```

Override any list explicitly when needed. For example, this narrows the balanced preset to two dimensions and one AD/MCI quota:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_final_grid.py `
  --preset balanced `
  --run-mode cv `
  --embedding-sources ppi `
  --ppi-gene-policy mapped_only `
  --gene-selections mad ad_mci_priority_anova_union pairwise_anova_union `
  --max-genes-list 512 1000 1500 2000 `
  --d-models 128 256 `
  --dropouts 0.3 0.4 0.5 `
  --ad-mci-gene-fractions 0.5 `
  --augmentations none borderline_smote `
  --skip-existing
```

Then validate the best CV configurations with 10 repeated 70/15/15 seed splits by rerunning the same script with `--run-mode seeds` and a narrowed grid.

Or select the top CV configurations automatically from `grid_ranked_summary.csv` and rerun only those on repeated seed splits:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_seed_followup.py `
  --grid-root results\paper_comparison\txt_multitask_final_grid `
  --top-n 5 `
  --split val `
  --evaluation cv5 `
  --device cuda `
  --skip-existing
```

Dry-run the seed follow-up first:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_seed_followup.py `
  --grid-root results\paper_comparison\txt_multitask_final_grid `
  --top-n 5 `
  --dry-run
```

The grid writes:

- `grid_jobs.csv`: planned commands and skipped jobs.
- `grid_ranked_summary.csv`: ranking sorted first by `AD_vs_MCI` ROC-AUC, then `AD_vs_MCI` macro-F1, then auxiliary ROC-AUC.

The seed follow-up writes:

- `seed_followup_jobs.csv`: selected CV rows and exact seed-validation commands.
- `top_seed_validation/<job_name>/seeds10/seeds10_summary.csv`: repeated-split validation for the selected configuration.

Rebuild the ranking from existing completed runs without launching training:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_final_grid.py `
  --result-root results\paper_comparison\txt_multitask_final_grid `
  --collect-only
```
