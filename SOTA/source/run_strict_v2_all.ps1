$ErrorActionPreference = "Stop"

$commands = @(
  @("python", "SOTA\source\nature-2020\run_experiments.py", "--strict-v2"),
  @("python", "SOTA\source\nature-2023\run_experiments.py", "--strict-v2"),
  @("python", "SOTA\source\one2mfusion-2023\run_experiments.py", "--strict-v2"),
  @("python", "SOTA\source\tabnet-2023\run_experiments.py", "--strict-v2"),
  @("python", "SOTA\source\diagnostics-2025\run_experiments.py", "--strict-v2"),
  @("python", "SOTA\source\nature-2026\run_experiments.py", "--strict-v2")
)

foreach ($cmd in $commands) {
  Write-Host ""
  Write-Host "Running: $($cmd -join ' ')"
  $exe = $cmd[0]
  $argv = $cmd[1..($cmd.Length - 1)]
  & $exe @argv
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

Write-Host ""
Write-Host "All strict-v2 SOTA runs finished."
