param(
    [string]$Python = "python",
    [string]$StartAt = "",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$LogRoot = Join-Path $Root "results\SOTA\fivecv_shared_seed108\logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$experiments = @(
    @{
        Name = "nature-2020"
        Args = @(
            "SOTA\source\nature-2020\run_experiments.py",
            "--protocol", "stratified_5cv",
            "--seed", "108",
            "--repeats", "1",
            "--feature-sets", "deg",
            "--models", "lr", "rf", "svm",
            "--p-value-threshold", "0.01",
            "--fdr-threshold", "0.01",
            "--deg-fallback-top-k", "0",
            "--scaler", "minmax",
            "--threshold-mode", "fixed_0_5",
            "--result-root", "results\SOTA\fivecv_shared_seed108\nature-2020",
            "--overwrite-results"
        )
    },
    @{
        Name = "nature-2023"
        Args = @(
            "SOTA\source\nature-2023\run_experiments.py",
            "--protocol", "nested_leakage_safe",
            "--seed", "108",
            "--repeats", "1",
            "--outer-folds", "5",
            "--feature-sets", "vssrfe_lr",
            "--models", "rf",
            "--scaler", "minmax",
            "--hyperparameter-mode", "fixed_paper",
            "--paper-vssrfe-n-genes", "50",
            "--threshold-mode", "fixed_0_5",
            "--result-root", "results\SOTA\fivecv_shared_seed108\nature-2023",
            "--overwrite-results"
        )
    },
    @{
        Name = "one2mfusion-2023"
        Args = @(
            "SOTA\source\one2mfusion-2023\run_experiments.py",
            "--protocol", "nested_leakage_safe",
            "--seed", "108",
            "--repeats", "1",
            "--outer-folds", "5",
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
            "--result-root", "results\SOTA\fivecv_shared_seed108\one2mfusion",
            "--overwrite-results"
        )
    },
    @{
        Name = "tabnet-2023"
        Args = @(
            "SOTA\source\tabnet-2023\run_experiments.py",
            "--protocol", "stratified_5cv",
            "--seed", "108",
            "--repeats", "1",
            "--artifact-scope", "train_inner",
            "--dgs-adj-p-threshold", "0.01",
            "--dgs-fallback-thresholds", "0.05", "0.1", "0.2",
            "--dgs-fallback-p-thresholds", "0.001", "0.005", "0.01", "0.05", "0.1",
            "--dgs-min-genes", "100",
            "--skip-below-min-genes",
            "--max-epochs", "100",
            "--patience", "20",
            "--batch-size", "10",
            "--threshold-mode", "fixed_0_5",
            "--result-root", "results\SOTA\fivecv_shared_seed108\tabnet",
            "--overwrite-results"
        )
    },
    @{
        Name = "diagnostics-2025"
        Args = @(
            "SOTA\source\diagnostics-2025\run_experiments.py",
            "--protocol", "stratified_5cv",
            "--seed", "108",
            "--repeats", "1",
            "--artifact-scope", "train_inner",
            "--models", "svm",
            "--sampling", "borderline_smote",
            "--xgb-top-k", "300",
            "--sfbs-min-genes", "20",
            "--sfbs-max-genes", "95",
            "--sfbs-step", "5",
            "--sfbs-cv-folds", "5",
            "--sfbs-mode", "approximate_lr",
            "--threshold-mode", "fixed_0_5",
            "--result-root", "results\SOTA\fivecv_shared_seed108\diagnostics-2025",
            "--overwrite-results"
        )
    },
    @{
        Name = "nature-2026"
        Args = @(
            "SOTA\source\nature-2026\run_experiments.py",
            "--protocol", "stratified_5cv",
            "--seed", "108",
            "--repeats", "1",
            "--artifact-scope", "train_inner",
            "--feature-selectors", "elasticnet",
            "--feature-count-mode", "natural",
            "--models", "dnn",
            "--augmentations", "none",
            "--deep-epochs", "200",
            "--deep-patience", "50",
            "--deep-batch-size", "32",
            "--deep-learning-rate", "0.001",
            "--threshold-mode", "fixed_0_5",
            "--result-root", "results\SOTA\fivecv_shared_seed108\nature-2026",
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
