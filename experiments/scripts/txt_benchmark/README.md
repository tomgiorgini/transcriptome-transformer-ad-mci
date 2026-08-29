# TxT Benchmark

Batch-holdout TxT benchmark for the thesis TxT model. This code is intentionally outside `SOTA/source` because TxT is the proposed model, not one of the reproduced SOTA papers.

Default input:

- `task_dataset/processed/ad_mci_binary/X_ad_mci.csv`
- `task_dataset/processed/ad_mci_binary/y_ad_mci.csv`

Default benchmark runner protocol:

- `shared_test`
- `test_gse63060`
- `test_gse63061`
- 1 repeat
- train-only imputation/scaling
- default gene source: train-only top 1000 variance genes
- checkpoint selected on inner validation score: `0.5 * val_roc_auc + 0.5 * val_macro_f1`

Optuna search protocol:

- one seed: `42`
- stratified `70/10/20` train/validation/test split
- top 1000 genes selected once at the start from the seed-42 train split
- all Optuna trials reuse the same fixed 1000-gene input matrix
- aggregation function: `Flatten`
- objective: `0.5 * val_roc_auc + 0.5 * val_macro_f1 - 0.25 * val_loss`

Supported train-only scalers: `minmax`, `standard`, `robust`, `none`.

Supported gene embedding initializers:

- `--embedding-source random` (default): run-local random gene embeddings.
- `--embedding-source ppi`: precomputed HIPPIE/node2vec high-confidence graph-node embeddings. Gene selection is restricted to genes mapped in `ppi_node_embedding.csv` before top-variance ranking.

When `--strict-v2 --embedding-source ppi` is used, gene selection follows the TxT supplement more closely:

- keep only PPI-mapped genes;
- rank genes on `train_inner` by median absolute deviation (MAD);
- retain the top 1000 genes;
- fit MinMax scaling on `train_inner` only.

Run:

```powershell
python experiments\scripts\txt_benchmark\run_experiments.py
```

Run the Optuna architecture search:

```powershell
python experiments\scripts\txt_benchmark\optuna_search.py `
  --n-trials 256 `
  --result-root results\TxT\optuna
```

This covers the full grid:

- `n_heads`: `2, 4`
- `n_layers`: `1, 2`
- `d_model = embed_dim`: `16, 32, 64, 128`
- `d_ff`: `4 * d_model`
- `dropout`: `0.2, 0.3, 0.4, 0.5`
- `batch_size`: `16, 32`
- `aggfunc`: `Flatten, Avgpool`
- fixed `lr=1e-4`, `weight_decay=1e-4`
- fixed `epochs=200`, `early_stopping_patience=70`

Run strict-v2 TxT with train-only GAN augmentation on top 1000 train-inner variance genes:

```powershell
python experiments\scripts\txt_benchmark\run_experiments.py `
  --strict-v2 `
  --augmentation gan `
  --gan-targets double 1000 2000 `
  --gan-epochs 200 `
  --gan-sampling-strategy balanced `
  --baseline-result-root results\TxT\benchmark `
  --result-root results\TxT\TxT_gan_top1000_strict_v2
```

`double` means the augmented training set target is `2 * n_train_inner`.
GAN augmentation is fit only on preprocessed `train_inner`; validation and outer test samples are never augmented.
The default `balanced` sampling strategy uses the conditional GAN to push the augmented train set toward class balance.

Run strict-v2 TxT with train-only Borderline SMOTE instead:

```powershell
python experiments\scripts\txt_benchmark\run_experiments.py `
  --strict-v2 `
  --augmentation borderline_smote `
  --smote-kind borderline-1 `
  --baseline-result-root results\TxT\benchmark `
  --result-root results\TxT\TxT_borderline_smote_top1000_strict_v2
```

Run strict-v2 TxT with HIPPIE high-confidence PPI embeddings and top 1000 train-inner MAD genes, selected only from PPI-mapped genes:

```powershell
python experiments\scripts\txt_benchmark\run_experiments.py `
  --strict-v2 `
  --embedding-source ppi `
  --ppi-embedding-file results\pretraining\ppi_init\hippie_highconf_allgenes_seed42\ppi_node_embedding.csv `
  --baseline-result-root results\TxT\benchmark `
  --result-root results\TxT\TxT_ppi_top1000_strict_v2
```

Resume:

```powershell
python experiments\scripts\txt_benchmark\run_experiments.py --skip-existing
```

To reproduce the older macro-F1-only checkpoint rule:

```powershell
python experiments\scripts\txt_benchmark\run_experiments.py --checkpoint-metric macro_f1
```

Run TxT on the genes selected by the leakage-safe Lee & Lee 2020 reproduction:

```powershell
python experiments\scripts\txt_benchmark\run_experiments.py `
  --x-file task_dataset\processed\ad_mci_binary\X_ad_mci.csv `
  --y-file task_dataset\processed\ad_mci_binary\y_ad_mci.csv `
  --gene-source nature2020 `
  --nature2020-result-root results\SOTA\nature-2020 `
  --nature2020-feature-set deg `
  --batch-size 16 `
  --lr 3e-4 `
  --early-stopping-patience 70 `
  --result-root results\TxT\txt-nature2020-deg-fdr0.1-bs16-lr3e-4-es70
```

Outputs are written to:

```text
results/TxT/benchmark
```
