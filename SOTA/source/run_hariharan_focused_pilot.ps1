param(
    [int]$WaitForPid = 0,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $repoRoot "SOTA\source\nature-2026\run_experiments.py"
$resultRoot = Join-Path $repoRoot "results\SOTA\shared_test_70_10_20_r10\hariharan-2026\focused_pilot\ad_vs_mci_ctgan_r3"
$stdout = Join-Path $resultRoot "pilot_stdout.log"
$stderr = Join-Path $resultRoot "pilot_stderr.log"

if ($WaitForPid -gt 0) {
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$arguments = @(
    "-u",
    $runner,
    "--x-file", (Join-Path $repoRoot "task_dataset\processed\txt_pairwise_multitask\ad_vs_mci\X.csv"),
    "--y-file", (Join-Path $repoRoot "task_dataset\processed\txt_pairwise_multitask\ad_vs_mci\y.csv"),
    "--result-root", $resultRoot,
    "--protocol", "batch_holdout",
    "--batch-scenarios", "shared_test",
    "--split-manifest-dir", (Join-Path $repoRoot "results\SOTA\shared_test_70_10_20_r10\_txt_shared_splits"),
    "--repeats", "3",
    "--inner-val-ratio", "0.125",
    "--shared-test-size", "0.20",
    "--seed", "101",
    "--artifact-scope", "train_inner",
    "--threshold-mode", "fixed_0_5",
    "--skip-existing",
    "--feature-selectors", "rfe", "elasticnet",
    "--models", "svm", "dnn",
    "--augmentations", "ctgan",
    "--feature-count-mode", "fixed",
    "--training-balance", "undersample",
    "--deep-epochs", "100",
    "--n-jobs", "2",
    "--ctgan-cuda"
)

$process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -Wait

$process.ExitCode | Set-Content -Path (Join-Path $resultRoot "pilot_exit_code.txt") -Encoding ascii
exit $process.ExitCode
