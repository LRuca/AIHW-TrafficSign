$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_in_pytorch.ps1"
$experimentRunner = Join-Path $PSScriptRoot "run_experiment.py"

$configs = @(
    "configs\experiments\backbone_mobilenetv2.yaml",
    "configs\experiments\backbone_deit_tiny.yaml",
    "configs\experiments\finetune_head_only.yaml",
    "configs\experiments\finetune_last_stage.yaml",
    "configs\experiments\finetune_full.yaml",
    "configs\experiments\finetune_freeze_then_full.yaml",
    "configs\experiments\activation_relu.yaml",
    "configs\experiments\activation_gelu.yaml",
    "configs\experiments\resolution_32.yaml",
    "configs\experiments\resolution_64.yaml",
    "configs\experiments\resolution_96.yaml"
)

$logDir = Join-Path $projectRoot "outputs\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$sessionLog = Join-Path $logDir ("night_batch_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

"Night batch started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $sessionLog -Encoding utf8

foreach ($cfg in $configs) {
    $line = "Running $cfg at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host $line
    $line | Out-File -FilePath $sessionLog -Append -Encoding utf8

    & powershell -NoProfile -ExecutionPolicy Bypass -File $runner $experimentRunner --config $cfg 2>&1 |
        Tee-Object -FilePath $sessionLog -Append

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $cfg"
    }
}

"Night batch finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $sessionLog -Append -Encoding utf8
Write-Host "Night batch finished. Log saved to $sessionLog"
