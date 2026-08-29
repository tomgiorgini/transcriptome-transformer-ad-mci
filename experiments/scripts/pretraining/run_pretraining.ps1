param(
    [string]$PythonExe = "python",
    [ValidateSet("with_reference")]
    [string]$ReferencePolicy = "with_reference",
    [ValidateSet("baseline_arch", "two_layer_2head", "larger_4layer")]
    [string]$ModelSize = "larger_4layer",
    [string]$RunName = "",
    [string]$MatrixFile = "",
    [int]$Epochs = 500,
    [int]$BatchSize = 8,
    [double]$LearningRate = 1e-4,
    [double]$WeightDecay = 1e-4,
    [int]$GeneSubsetSize = 512,
    [double]$MaskRatio = 0.15,
    [int]$EarlyStoppingPatience = 0,
    [int]$KeepTopKCheckpoints = 10,
    [int]$SaveEveryEpochs = 25,
    [int]$Seed = 42,
    [string]$ManifestArch = "",
    [string]$CheckpointManifest = "experiments\scripts\pretraining\pretraining_checkpoint_manifest.csv",
    [switch]$SkipMatrixValidation,
    [switch]$SkipManifestUpdate
)

$ErrorActionPreference = "Stop"

function Update-ManifestRow {
    param(
        [string]$ManifestPath,
        [string]$PretrainingArch,
        [string]$Policy,
        [string]$CheckpointPath
    )

    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "Checkpoint manifest not found: $ManifestPath"
    }

    $UpdatedLine = "true,$PretrainingArch,$Policy,$CheckpointPath"
    $Lines = Get-Content -LiteralPath $ManifestPath
    $Matched = $false
    $UpdatedLines = foreach ($Line in $Lines) {
        if ($Line -match "^[^,]+,$PretrainingArch,$Policy,") {
            $Matched = $true
            $UpdatedLine
        } else {
            $Line
        }
    }

    if (-not $Matched) {
        $UpdatedLines += $UpdatedLine
    }

    $UpdatedLines | Set-Content -LiteralPath $ManifestPath
}

function Invoke-PythonLogged {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$LogFile
    )

    $oldErrorActionPreference = $ErrorActionPreference
    $hasNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativePreference) {
        $oldNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments *>&1 | Tee-Object -FilePath $LogFile
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $oldNativePreference
        }
    }
    if ($exitCode -ne 0) {
        throw "$Executable failed with exit code $exitCode. See log: $LogFile"
    }
}

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptPath "..\..\..")
Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"

if ([string]::IsNullOrWhiteSpace($MatrixFile)) {
    $MatrixFile = "pretraining_dataset\geo_downloads\merged_pretraining\pretraining_greedy_iter_21_with_reference_global_zscore_matrix_samples_x_genes.csv"
}

if (-not (Test-Path -LiteralPath $MatrixFile)) {
    throw "Pretraining matrix not found: $MatrixFile"
}

$cudaCheck = @'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available for this Python environment.")
print(torch.cuda.get_device_name(0))
'@
$cudaDevice = $cudaCheck | & $PythonExe -
Write-Host "CUDA device: $cudaDevice"

if (-not $SkipMatrixValidation) {
    & $PythonExe experiments\scripts\pretraining\check_pretraining_matrix.py --matrix-file $MatrixFile
}

if ($ModelSize -eq "baseline_arch") {
    $NHeads = 2
    $NLayers = 1
    $DModel = 256
    $DFF = 1024
    $EmbedDim = 128
    $Dropout = 0.4
} elseif ($ModelSize -eq "two_layer_2head") {
    $NHeads = 2
    $NLayers = 2
    $DModel = 256
    $DFF = 1024
    $EmbedDim = 128
    $Dropout = 0.4
} else {
    $NHeads = 4
    $NLayers = 4
    $DModel = 256
    $DFF = 800
    $EmbedDim = 200
    $Dropout = 0.15
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "iter21_${ReferencePolicy}_${ModelSize}_${Epochs}ep_$Stamp"
}
if ([string]::IsNullOrWhiteSpace($ManifestArch)) {
    $ManifestArch = $ModelSize
}

$OutDir = Join-Path "results\pretraining\self_supervised\txt_gexbert" $RunName
$LogFile = Join-Path $OutDir "pretraining.log"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Reference policy: $ReferencePolicy"
Write-Host "Model size:       $ModelSize"
Write-Host "Matrix:           $MatrixFile"
Write-Host "Result dir:       $OutDir"
Write-Host "Architecture:     layers=$NLayers heads=$NHeads d_model=$DModel d_ff=$DFF embed_dim=$EmbedDim dropout=$Dropout"
Write-Host "Masking:          gene_subset_size=$GeneSubsetSize mask_ratio=$MaskRatio"
Write-Host "Early stopping:   patience=$EarlyStoppingPatience"
if ($GeneSubsetSize -le 0) {
    Write-Host "Masking mode:     all genes from the matrix are used for each sample"
}

$ArgsList = @(
    "-u",
    "experiments\scripts\pretraining\pretrain_txt_gexbert.py",
    "--matrix-file", $MatrixFile,
    "--result-dir", $OutDir,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--n-heads", "$NHeads",
    "--n-layers", "$NLayers",
    "--d-model", "$DModel",
    "--d-ff", "$DFF",
    "--embed-dim", "$EmbedDim",
    "--dropout", "$Dropout",
    "--gene-subset-size", "$GeneSubsetSize",
    "--mask-ratio", "$MaskRatio",
    "--early-stopping-patience", "$EarlyStoppingPatience",
    "--keep-top-k-checkpoints", "$KeepTopKCheckpoints",
    "--save-every-epochs", "$SaveEveryEpochs",
    "--seed", "$Seed",
    "--device", "cuda"
)
Invoke-PythonLogged -Executable $PythonExe -Arguments $ArgsList -LogFile $LogFile

$BestCheckpoint = Join-Path $OutDir "best_checkpoint.pt"
if (-not (Test-Path -LiteralPath $BestCheckpoint)) {
    throw "Pretraining finished but best checkpoint was not found: $BestCheckpoint"
}

if (-not $SkipManifestUpdate) {
    Update-ManifestRow `
        -ManifestPath $CheckpointManifest `
        -PretrainingArch $ManifestArch `
        -Policy $ReferencePolicy `
        -CheckpointPath $BestCheckpoint
    Write-Host "Manifest updated: $CheckpointManifest"
}

Write-Host "Best checkpoint: $BestCheckpoint"
