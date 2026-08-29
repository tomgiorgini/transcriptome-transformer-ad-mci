# One2MFusion 2023 — local leakage-safe reproduction

This runner adapts Akkaya and Kalkan's One2MFusion method (Biomolecules, 2023) to the thesis cohort while keeping preprocessing, LASSO, Fisher grouping, and LDA mapping inside `train_inner`.

## Implemented models and feature routing

The three models compared in the article are all available:

- `fnn`: standalone FNN on **all genes** (`128, 64, 32, 32`);
- `cnn`: CNN on a 90×90×3 image built from **LASSO-selected genes**;
- `one2mfusion`: FNN and CNN branches both originating from **LASSO-selected genes**, followed by `Dense(32) → Dropout(0.4) → Dense(32) → sigmoid`.

LASSO is the paper's only gene feature-selection method. Fisher-distance grouping and LDA are the image representation, not additional feature selectors. `fixed_k` and `fixed_input` remain available only as explicitly labelled local sensitivity variants; the paper configuration is `--gene-selection-mode nonzero`.

The last two 32-unit layers in each branch include the kernel, bias, and activity regularizers present in the released notebook. Pixel collisions are averaged, matching the public image transformer.

## Paper and public-code profiles

The article and notebook conflict. The runner makes the choice explicit:

| Setting | `paper` (default) | `public_code` |
|---|---:|---:|
| LASSO alpha | `1e-6` | `2e-4` |
| Image gene order | Fisher distance | input order used by notebook |
| Learning rate | `1e-4` for all models | `1e-3` FNN/fusion; `1e-4` CNN |
| Batch size | `30` for all models | `32` FNN/CNN; `64` fusion |
| Epoch caps | `1003` | `1002` FNN, `1001` CNN, `1003` fusion |

Both profiles default to patience 10 and training-loss early stopping after epoch 250, as reported. `--early-stopping-monitor val_loss` is available as a leakage-safe training adaptation. The notebook's global hard-coded 488-gene AD-vs-control list is not reused on other tasks.

## TxT shared-test command for one task

Use the same canonical `seed_101.csv` … `seed_110.csv` manifests as TxT. They encode exact shared-cohort 70/10/20 membership and avoid independently resplitting each binary task.

```powershell
python SOTA\source\one2mfusion-2023\run_experiments.py `
  --x-file task_dataset\processed\txt_pairwise_multitask\ad_vs_mci\X.csv `
  --y-file task_dataset\processed\txt_pairwise_multitask\ad_vs_mci\y.csv `
  --split-manifest-dir <canonical-70-10-20-splits> `
  --result-root results\SOTA_reproduction\one2mfusion-2023\ad_vs_mci `
  --protocol batch_holdout `
  --batch-scenarios shared_test `
  --repeats 10 `
  --models fnn cnn one2mfusion `
  --implementation-profile paper `
  --hyperparameter-mode fixed_paper `
  --paper-lasso-alpha 1e-6 `
  --gene-selection-mode nonzero `
  --image-gene-order fisher `
  --fisher-groups 15 `
  --artifact-scope train_inner `
  --threshold-mode fixed_0_5
```

## Verification

```powershell
python SOTA\source\one2mfusion-2023\test_one2mfusion_reproduction.py
python SOTA\source\one2mfusion-2023\run_experiments.py --smoke --models fnn cnn one2mfusion
```

The original experiment pooled GSE63060, GSE63061, and GSE140829 and used five-fold CV. The local two-batch 70/10/20×10 experiment is therefore a leakage-safe method reproduction, not a numerical replication of the published table.
