param(
    [string]$PythonExe = "python",
    [string]$PretrainedCheckpoint = "results\pretraining\ppi_init\hippie_highconf_deg_seed42_20ep\ppi_embedding_checkpoint.pt",
    [string]$ResultRoot = "results\pretraining\finetuning_ppi_best10seeds",
    [string]$XFile = "task_dataset\processed\ad_mci_deg\X_deg_ad_mci.csv",
    [string]$YFile = "task_dataset\processed\ad_mci_deg\y_ad_mci.csv",
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda",
    [int[]]$Seeds = @(101, 102, 103, 104, 105, 106, 107, 108, 109, 110),
    [int]$Epochs = 180,
    [int]$EarlyStoppingPatience = 45,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptPath "..\..\..")
Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"

if (-not (Test-Path -LiteralPath $PretrainedCheckpoint)) {
    throw "Pretrained checkpoint not found: $PretrainedCheckpoint"
}
if (-not (Test-Path -LiteralPath $XFile)) {
    throw "X file not found: $XFile"
}
if (-not (Test-Path -LiteralPath $YFile)) {
    throw "Y file not found: $YFile"
}

$Configs = @(
    [pscustomobject]@{
        Name = "hippie_deg_20ep_1l2h_d128_best"
        NLayers = 1
        NHeads = 2
        DModel = 128
        DFF = 256
        BatchSize = 16
        LearningRate = 0.0001746088296566736
        Dropout = 0.2
        DHidden1 = 128
        DHidden2 = 64
        SourceOptuna = "results\pretraining\finetuning_ppi_optuna\hippie_deg_20ep_1l2h_d128_20trials"
        SourceTrial = 20
        SourceValue = 0.7183518211683426
    },
    [pscustomobject]@{
        Name = "hippie_deg_20ep_2l2h_d128_best"
        NLayers = 2
        NHeads = 2
        DModel = 128
        DFF = 256
        BatchSize = 16
        LearningRate = 0.00012298149529318397
        Dropout = 0.4
        DHidden1 = 128
        DHidden2 = 64
        SourceOptuna = "results\pretraining\finetuning_ppi_optuna\hippie_deg_20ep_2l2h_d128_20trials_v2"
        SourceTrial = 7
        SourceValue = 0.7221657375232164
    }
)

foreach ($config in $Configs) {
    $configRoot = Join-Path $ResultRoot $config.Name
    New-Item -ItemType Directory -Force -Path $configRoot | Out-Null

    $configJson = @{
        name = $config.Name
        pretrained_checkpoint = $PretrainedCheckpoint
        x_file = $XFile
        y_file = $YFile
        seeds = $Seeds
        epochs = $Epochs
        early_stopping_patience = $EarlyStoppingPatience
        source_optuna = $config.SourceOptuna
        source_trial = $config.SourceTrial
        source_value = $config.SourceValue
        transfer_mode = "embedding_only"
        batch_size = $config.BatchSize
        lr = $config.LearningRate
        weight_decay = 0.0001
        dropout = $config.Dropout
        n_layers = $config.NLayers
        n_heads = $config.NHeads
        d_model = $config.DModel
        d_ff = $config.DFF
        d_hidden1 = $config.DHidden1
        d_hidden2 = $config.DHidden2
        aggfunc = "Avgpool"
    } | ConvertTo-Json -Depth 5
    $configJson | Set-Content -LiteralPath (Join-Path $configRoot "run_config.json") -Encoding UTF8

    foreach ($seed in $Seeds) {
        $resultDir = Join-Path $configRoot "split_seed_$seed"
        $metricsFile = Join-Path $resultDir "metrics_summary.csv"
        if ($SkipExisting -and (Test-Path -LiteralPath $metricsFile)) {
            Write-Host "Skipping existing run: $resultDir"
            continue
        }

        Write-Host ""
        Write-Host "================================================================================"
        Write-Host "PPI best Optuna | $($config.Name) | split_seed=$seed"
        Write-Host "================================================================================"

        & $PythonExe -u "experiments\scripts\finetuning\finetune_txt.py" `
            --pretrained-checkpoint $PretrainedCheckpoint `
            --transfer-mode embedding_only `
            --x-file $XFile `
            --y-file $YFile `
            --result-dir $resultDir `
            --seed $seed `
            --split-seed $seed `
            --split-mode stratified `
            --val-ratio 0.15 `
            --test-ratio 0.15 `
            --dataset-mode deg `
            --max-genes 0 `
            --scaler minmax `
            --scaler-fit-scope train `
            --batch-size $config.BatchSize `
            --epochs $Epochs `
            --early-stopping-patience $EarlyStoppingPatience `
            --lr-encoder $config.LearningRate `
            --lr-head $config.LearningRate `
            --weight-decay 0.0001 `
            --label-smoothing 0.0 `
            --class-weighting off `
            --checkpoint-metric val_auc_f1_50_50 `
            --threshold-tuning off `
            --final-threshold-mode fixed `
            --final-threshold 0.5 `
            --evaluate-test on `
            --grad-clip-norm 1.0 `
            --device $Device `
            --n-layers $config.NLayers `
            --n-heads $config.NHeads `
            --d-model $config.DModel `
            --d-ff $config.DFF `
            --dropout $config.Dropout `
            --aggfunc Avgpool `
            --d-hidden1 $config.DHidden1 `
            --d-hidden2 $config.DHidden2

        if ($LASTEXITCODE -ne 0) {
            throw "Run failed: $($config.Name), split_seed=$seed"
        }
    }
}

$summaryScript = @"
from pathlib import Path
import json
import pandas as pd

root = Path(r"$ResultRoot")
all_rows = []
for config_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    config_path = config_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"name": config_dir.name}
    rows = []
    for metrics_path in sorted(config_dir.glob("split_seed_*/metrics_summary.csv")):
        run_dir = metrics_path.parent
        seed = int(run_dir.name.replace("split_seed_", ""))
        metrics = pd.read_csv(metrics_path).set_index("split")
        model_summary_path = run_dir / "model_summary.json"
        training_summary = {}
        if model_summary_path.exists():
            training_summary = json.loads(model_summary_path.read_text(encoding="utf-8")).get("training_summary", {})
        for split in metrics.index:
            row = {
                "config": config.get("name", config_dir.name),
                "seed": seed,
                "split": split,
                "best_epoch": training_summary.get("best_epoch"),
                "source_trial": config.get("source_trial"),
                "source_value": config.get("source_value"),
                "n_layers": config.get("n_layers"),
                "n_heads": config.get("n_heads"),
                "d_model": config.get("d_model"),
                "d_ff": config.get("d_ff"),
                "batch_size": config.get("batch_size"),
                "lr": config.get("lr"),
                "dropout": config.get("dropout"),
                "macro_f1": metrics.loc[split, "macro_f1"],
                "accuracy": metrics.loc[split, "accuracy"],
                "balanced_accuracy": metrics.loc[split, "balanced_accuracy"],
                "roc_auc_ovr_macro": metrics.loc[split, "roc_auc_ovr_macro"],
                "recall_mci": metrics.loc[split, "recall_mci"],
                "recall_ad": metrics.loc[split, "recall_ad"],
            }
            rows.append(row)
            all_rows.append(row)
    if rows:
        runs = pd.DataFrame(rows).sort_values(["seed", "split"])
        runs.to_csv(config_dir / "per_seed_metrics.csv", index=False)
        summary = (
            runs.groupby(["config", "split"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                auc_mean=("roc_auc_ovr_macro", "mean"),
                auc_std=("roc_auc_ovr_macro", "std"),
                accuracy_mean=("accuracy", "mean"),
                balanced_accuracy_mean=("balanced_accuracy", "mean"),
                recall_mci_mean=("recall_mci", "mean"),
                recall_ad_mean=("recall_ad", "mean"),
            )
        )
        summary.to_csv(config_dir / "summary_metrics.csv", index=False)
        print("")
        print(config_dir)
        print(summary.to_string(index=False))

if all_rows:
    all_runs = pd.DataFrame(all_rows).sort_values(["config", "seed", "split"])
    all_runs.to_csv(root / "all_per_seed_metrics.csv", index=False)
    all_summary = (
        all_runs.groupby(["config", "split"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auc_mean=("roc_auc_ovr_macro", "mean"),
            auc_std=("roc_auc_ovr_macro", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            recall_mci_mean=("recall_mci", "mean"),
            recall_ad_mean=("recall_ad", "mean"),
        )
    )
    all_summary.to_csv(root / "all_summary_metrics.csv", index=False)
    print("")
    print("Combined summary:")
    print(all_summary.to_string(index=False))
else:
    raise SystemExit("No metrics_summary.csv files found.")
"@

$summaryScript | & $PythonExe -

Write-Host ""
Write-Host "Completed PPI best Optuna 10-seed runs. Results: $ResultRoot"
