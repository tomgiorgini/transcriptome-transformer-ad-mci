param(
    [string]$Python = "python",
    [ValidateSet("nature-2020", "nature-2023", "one2mfusion-2023", "tabnet-2023", "diagnostics-2025", "nature-2026", "txt")]
    [string]$StartAt = "nature-2020",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$XFile = "task_dataset\processed\ad_ctl_binary\X_ad_ctl.csv"
$YFile = "task_dataset\processed\ad_ctl_binary\y_ad_ctl.csv"
$LogRoot = Join-Path $Root "results\SOTA_AD_CTL\logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$experiments = @(
    @{
        Name = "nature-2020"
        Args = @(
            "SOTA\source\nature-2020\run_experiments.py",
            "--x-file", $XFile,
            "--y-file", $YFile,
            "--result-root", "results\SOTA_AD_CTL\nature-2020",
            "--protocol", "batch_holdout",
            "--batch-scenarios", "shared_test", "test_gse63060", "test_gse63061",
            "--repeats", "10",
            "--seed", "42",
            "--artifact-scope", "train_inner",
            "--feature-sets", "deg",
            "--models", "lr", "rf", "svm",
            "--fdr-threshold", "0.01",
            "--p-value-threshold", "0.01",
            "--deg-fallback-top-k", "0",
            "--scaler", "none",
            "--threshold-mode", "fixed_0_5",
            "--overwrite-results"
        )
    },
    @{
        Name = "nature-2023"
        Args = @(
            "SOTA\source\nature-2023\run_experiments.py",
            "--x-file", $XFile,
            "--y-file", $YFile,
            "--result-root", "results\SOTA_AD_CTL\nature-2023",
            "--protocol", "batch_holdout",
            "--batch-scenarios", "shared_test", "test_gse63060", "test_gse63061",
            "--repeats", "10",
            "--seed", "42",
            "--artifact-scope", "train_inner",
            "--feature-sets", "vssrfe_lr",
            "--models", "rf",
            "--scaler", "standard",
            "--bayes-iter", "25",
            "--cv-folds", "5",
            "--hyperparameter-mode", "bayes",
            "--disable-fixed-vssrfe-n-genes",
            "--vssrfe-min-genes", "10",
            "--vssrfe-max-genes", "50",
            "--vssrfe-step-genes", "10",
            "--vssrfe-extra-gene-counts", "75", "100", "125", "150", "175", "200",
            "--threshold-mode", "fixed_0_5",
            "--overwrite-results"
        )
    },
    @{
        Name = "one2mfusion-2023"
        Args = @(
            "SOTA\source\one2mfusion-2023\run_experiments.py",
            "--x-file", $XFile,
            "--y-file", $YFile,
            "--result-root", "results\SOTA_AD_CTL\one2mfusion",
            "--protocol", "batch_holdout",
            "--batch-scenarios", "shared_test", "test_gse63060", "test_gse63061",
            "--repeats", "10",
            "--seed", "42",
            "--models", "one2mfusion",
            "--artifact-scope", "train_inner",
            "--hyperparameter-mode", "fixed_paper",
            "--paper-lasso-alpha", "1e-5",
            "--gene-selection-mode", "nonzero",
            "--image-gene-order", "input",
            "--paper-like-preprocessing",
            "--epochs", "300",
            "--patience", "10",
            "--early-stopping-monitor", "strict_v2",
            "--threshold-mode", "fixed_0_5",
            "--overwrite-results"
        )
    },
    @{
        Name = "tabnet-2023"
        Args = @(
            "SOTA\source\tabnet-2023\run_experiments.py",
            "--x-file", $XFile,
            "--y-file", $YFile,
            "--result-root", "results\SOTA_AD_CTL\tabnet",
            "--protocol", "batch_holdout",
            "--batch-scenarios", "shared_test", "test_gse63060", "test_gse63061",
            "--repeats", "10",
            "--seed", "42",
            "--artifact-scope", "train_inner",
            "--dgs-adj-p-threshold", "0.01",
            "--dgs-min-genes", "100",
            "--dgs-fallback-thresholds", "0.05", "0.1", "0.2",
            "--dgs-fallback-p-thresholds", "0.001", "0.005", "0.01", "0.05", "0.1",
            "--skip-below-min-genes",
            "--max-epochs", "200",
            "--patience", "5",
            "--batch-size", "10",
            "--threshold-mode", "fixed_0_5",
            "--overwrite-results"
        )
    },
    @{
        Name = "diagnostics-2025"
        Args = @(
            "SOTA\source\diagnostics-2025\run_experiments.py",
            "--x-file", $XFile,
            "--y-file", $YFile,
            "--result-root", "results\SOTA_AD_CTL\diagnostics-2025",
            "--protocol", "batch_holdout",
            "--batch-scenarios", "shared_test", "test_gse63060", "test_gse63061",
            "--repeats", "10",
            "--seed", "42",
            "--artifact-scope", "train_inner",
            "--models", "dl",
            "--sampling", "borderline_smote",
            "--xgb-top-k", "300",
            "--sfbs-min-genes", "20",
            "--sfbs-max-genes", "95",
            "--sfbs-step", "5",
            "--sfbs-cv-folds", "5",
            "--sfbs-mode", "approximate_lr",
            "--threshold-mode", "fixed_0_5",
            "--overwrite-results"
        )
    },
    @{
        Name = "nature-2026"
        Args = @(
            "SOTA\source\nature-2026\run_experiments.py",
            "--x-file", $XFile,
            "--y-file", $YFile,
            "--result-root", "results\SOTA_AD_CTL\nature-2026",
            "--protocol", "batch_holdout",
            "--batch-scenarios", "shared_test", "test_gse63060", "test_gse63061",
            "--repeats", "10",
            "--seed", "42",
            "--artifact-scope", "train_inner",
            "--feature-selectors", "elasticnet",
            "--feature-count-mode", "natural",
            "--models", "dnn",
            "--augmentations", "gan",
            "--gan-target-size", "2000",
            "--gan-epochs", "200",
            "--deep-epochs", "300",
            "--deep-patience", "50",
            "--deep-batch-size", "32",
            "--deep-learning-rate", "0.001",
            "--threshold-mode", "fixed_0_5",
            "--overwrite-results"
        )
    },
    @{
        Name = "txt"
        Args = @(
            "experiments\scripts\txt_benchmark\run_experiments.py",
            "--x-file", $XFile,
            "--y-file", $YFile,
            "--result-root", "results\TxT\benchmark_ad_ctl_trial083_topvariance1000_10repeat",
            "--gene-source", "top_variance",
            "--top-variance-genes", "1000",
            "--batch-scenarios", "shared_test", "test_gse63060", "test_gse63061",
            "--split-protocol", "batch_holdout",
            "--repeats", "10",
            "--seed", "101",
            "--device", "cuda",
            "--scaler", "minmax",
            "--class-weighting", "off",
            "--batch-size", "16",
            "--epochs", "200",
            "--early-stopping-patience", "70",
            "--lr", "0.0001",
            "--weight-decay", "0.0001",
            "--checkpoint-metric", "strict_v2",
            "--threshold-mode", "fixed_0_5",
            "--augmentation", "none",
            "--n-layers", "1",
            "--n-heads", "2",
            "--d-model", "256",
            "--embed-dim", "256",
            "--d-ff", "1024",
            "--dropout", "0.4",
            "--aggfunc", "Avgpool",
            "--overwrite-results"
        )
    }
)

$started = [string]::IsNullOrWhiteSpace($StartAt)
foreach ($experiment in $experiments) {
    if (-not $started) {
        if ($experiment.Name -eq $StartAt) {
            $started = $true
        } else {
            continue
        }
    }

    $arguments = @($experiment.Args)
    if ($SkipExisting) {
        $arguments = $arguments | Where-Object { $_ -ne "--overwrite-results" }
        $arguments += "--skip-existing"
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $LogRoot "$($timestamp)_$($experiment.Name).log"
    Write-Host "===== START $($experiment.Name) ====="
    Write-Host "Log: $logFile"
    Write-Host "Command: $Python $($arguments -join ' ')"

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        throw "Experiment '$($experiment.Name)' failed with exit code $exitCode. See log: $logFile"
    }
    Write-Host "===== END $($experiment.Name) ====="
}
