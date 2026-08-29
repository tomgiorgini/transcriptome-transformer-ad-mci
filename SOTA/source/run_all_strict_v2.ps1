param(
    [string]$Python = "python",
    [string]$One2MFusionAlpha = "1e-5",
    [ValidateSet("nature-2020", "nature-2023", "one2mfusion-2023", "tabnet-2023", "diagnostics-2025", "nature-2026")]
    [string]$StartAt = "nature-2020"
)

$ErrorActionPreference = "Continue"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$LogRoot = Join-Path $Root "results\SOTA\logs_strict_v2"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$ExperimentOrder = @("nature-2020", "nature-2023", "one2mfusion-2023", "tabnet-2023", "diagnostics-2025", "nature-2026")
$StartIndex = [array]::IndexOf($ExperimentOrder, $StartAt)

function ShouldRun {
    param([Parameter(Mandatory = $true)][string]$Name)
    return ([array]::IndexOf($ExperimentOrder, $Name) -ge $StartIndex)
}

function Invoke-Experiment {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $started = Get-Date
    $logFile = Join-Path $LogRoot ("{0}_{1}.log" -f $started.ToString("yyyyMMdd_HHmmss"), $Name)
    Write-Host ""
    Write-Host "===== START $Name =====" -ForegroundColor Cyan
    Write-Host "Log: $logFile"
    Write-Host ("Command: {0} {1}" -f $Python, ($Arguments -join " "))

    & $Python @Arguments 2>&1 | Tee-Object -FilePath $logFile
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Experiment '$Name' failed with exit code $exitCode. See log: $logFile"
    }

    $elapsed = (Get-Date) - $started
    Write-Host "===== END $Name in $($elapsed.ToString()) =====" -ForegroundColor Green
}

Push-Location $Root
try {
    if (ShouldRun "nature-2020") {
        Invoke-Experiment "nature-2020" @(
            "SOTA\source\nature-2020\run_experiments.py",
            "--strict-v2"
        )
    }

    if (ShouldRun "nature-2023") {
        Invoke-Experiment "nature-2023" @(
            "SOTA\source\nature-2023\run_experiments.py",
            "--strict-v2",
            "--skip-existing"
        )
    }

    if (ShouldRun "one2mfusion-2023") {
        Invoke-Experiment "one2mfusion-2023" @(
            "SOTA\source\one2mfusion-2023\run_experiments.py",
            "--strict-v2",
            "--paper-lasso-alpha", $One2MFusionAlpha
        )
    }

    if (ShouldRun "tabnet-2023") {
        Invoke-Experiment "tabnet-2023" @(
            "SOTA\source\tabnet-2023\run_experiments.py",
            "--strict-v2"
        )
    }

    if (ShouldRun "diagnostics-2025") {
        Invoke-Experiment "diagnostics-2025" @(
            "SOTA\source\diagnostics-2025\run_experiments.py",
            "--strict-v2"
        )
    }

    if (ShouldRun "nature-2026") {
        Invoke-Experiment "nature-2026" @(
            "SOTA\source\nature-2026\run_experiments.py",
            "--strict-v2"
        )
    }

    Write-Host ""
    Write-Host "All strict-v2 SOTA experiments finished." -ForegroundColor Green
}
finally {
    Pop-Location
}
