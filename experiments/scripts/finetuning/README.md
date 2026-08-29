# TxT fine-tuning

Single fine-tuning entry point for TxT experiments on AD vs MCI DEG genes.

Main scripts:

```text
finetune_txt.py              single fine-tuning run
run_current_optuna_combo.py  current 8-combination Optuna 5-CV launcher
run_fixed_cv.py              fixed-parameter k-fold CV launcher
run_ppi_best_optuna_10seeds.ps1
                             10-seed validation for the best PPI Optuna configs
optuna_finetuning.py         Optuna search over fine-tuning hyperparameters
optuna_finetuning_cv.py      Optuna search with k-fold CV objective
refine_top_optuna_trials.py  refine the best Optuna trials on multiple split seeds
```

Current fine-tuning policy:

- class weighting is configurable and currently defaults to `on`;
- validation-threshold tuning is optional;
- no min-recall constraint;
- no anti-collapse checkpoint score;
- predictions use standard `argmax` over class probabilities;
- checkpoint selection uses one explicit metric;
- train/validation loss and train/val/test metrics are saved, including accuracy, macro F1, balanced accuracy, and one-vs-rest AUC.

Main parameters to tune:

```text
--transfer-mode full|embedding_only|random_init
--batch-size
--lr-encoder
--lr-head
--weight-decay
--dropout
--freeze-encoder-layers
```

Example single run:

```powershell
python .\experiments\scripts\finetuning\finetune_txt.py `
  --pretrained-checkpoint "results\pretraining\self_supervised\txt_gexbert\iter21_overnight_two_layer_2head_with_reference_deg_only_500ep_mask25\best_checkpoint.pt" `
  --transfer-mode full `
  --result-dir "results\pretraining\finetuning\P1_2l2h_full_split101" `
  --dataset-mode deg_pretrained_overlap `
  --split-seed 101 `
  --n-layers 2 `
  --n-heads 2 `
  --checkpoint-metric val_loss
```

Freeze the first Transformer encoder layer during fine-tuning:

```powershell
--freeze-encoder-layers 1
```

Tune transfer mode and freezing with Optuna:

```powershell
python .\experiments\scripts\finetuning\optuna_finetuning.py `
  --pretrained-checkpoint "results\pretraining\self_supervised\txt_gexbert\iter21_overnight_two_layer_2head_with_reference_deg_only_500ep_mask25\best_checkpoint.pt" `
  --result-root "results\pretraining\finetuning_optuna\P1_2l2h_clean" `
  --transfer-modes full embedding_only random_init `
  --freeze-encoder-layer-options "" 1 `
  --fixed-n-layers 2 `
  --fixed-n-heads 2
```

Outputs are written under `results/pretraining/finetuning/` unless `--result-dir` is provided.

Run one current Optuna combo:

```powershell
python .\experiments\scripts\finetuning\run_current_optuna_combo.py `
  --dataset legacy `
  --task ad_mci `
  --arch 1l2h
```

Run fixed-parameter CV:

```powershell
python .\experiments\scripts\finetuning\run_fixed_cv.py --help
```
