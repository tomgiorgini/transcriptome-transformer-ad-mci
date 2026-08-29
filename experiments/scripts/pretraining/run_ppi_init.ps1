param(
    [string]$PythonExe = "python",
    [string]$HippieFile = "",
    [string]$GeneListFile = "task_dataset\processed\txt_pairwise_multitask\shared_ad_mci_ctl\X.csv",
    [string]$ResultRoot = "results\pretraining\ppi_init",
    [string]$RunName = "",
    [int]$Seed = 42,
    [double]$ScoreThreshold = 0.73,
    [ValidateSet("largest", "all")]
    [string]$ComponentPolicy = "largest",
    [int]$EmbeddingDim = 128,
    [int]$Epochs = 20,
    [int]$BatchSize = 1024,
    [double]$LearningRate = 0.01,
    [int]$WalkLength = 20,
    [int]$ContextSize = 10,
    [int]$WalksPerNode = 10,
    [int]$NegativeSamples = 5,
    [int]$MaxPairsPerEpoch = 1000000,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda",
    [switch]$SkipNode2Vec
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path (Get-Location) $PathValue)
}

if (-not $RunName.Trim()) {
    $ScoreTag = $ScoreThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
    $RunName = "hippie_highconf_dim${EmbeddingDim}_score${ScoreTag}_seed${Seed}"
}
$ResultDir = Join-Path $ResultRoot $RunName
$EdgeFile = "pretraining_dataset\ppi_networks\hippie_highconf_edges.csv"

if (-not (Test-Path -LiteralPath $GeneListFile)) {
    throw "Gene list file not found: $GeneListFile. Build the shared AD/MCI/CTL multitask dataset first, or pass -GeneListFile."
}

$Command = @(
    "experiments\scripts\pretraining\build_ppi_embedding.py",
    "--source", "hippie",
    "--gene-list-file", $GeneListFile,
    "--result-dir", $ResultDir,
    "--edge-output-file", $EdgeFile,
    "--score-threshold", $ScoreThreshold,
    "--component-policy", $ComponentPolicy,
    "--embedding-dim", $EmbeddingDim,
    "--seed", $Seed,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--lr", $LearningRate,
    "--walk-length", $WalkLength,
    "--context-size", $ContextSize,
    "--walks-per-node", $WalksPerNode,
    "--negative-samples", $NegativeSamples,
    "--max-pairs-per-epoch", $MaxPairsPerEpoch,
    "--device", $Device
)

if ($HippieFile.Trim()) {
    $Command += @("--hippie-file", $HippieFile)
}
if ($SkipNode2Vec) {
    $Command += "--skip-node2vec"
}

New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$LogStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $ResultDir "ppi_init_$LogStamp.log"

Write-Host "Running HIPPIE PPI node2vec init..."
Write-Host "Result dir: $ResultDir"
Write-Host "Log file: $LogFile"

& $PythonExe @Command 2>&1 | Tee-Object -FilePath $LogFile
if ($LASTEXITCODE -ne 0) {
    throw "PPI init failed with exit code $LASTEXITCODE"
}

Write-Host "PPI init complete."
Write-Host "PPI node embedding for mapped-only TxT: $(Resolve-RepoPath (Join-Path $ResultDir 'ppi_node_embedding.csv'))"
Write-Host "Target-gene embedding with random fallback: $(Resolve-RepoPath (Join-Path $ResultDir 'gene_embedding.csv'))"
