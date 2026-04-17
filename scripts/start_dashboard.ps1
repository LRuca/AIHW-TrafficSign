$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_in_pytorch.ps1"
$dashboardScript = Join-Path $PSScriptRoot "launch_dashboard.py"
$dashboardHost = "127.0.0.1"
$port = 8765
$logDir = Join-Path $projectRoot "outputs\logs"
$stdoutLog = Join-Path $logDir "dashboard_stdout.log"
$stderrLog = Join-Path $logDir "dashboard_stderr.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($existing) {
    $listeningPids = $existing | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $listeningPids) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Output "Stopped existing dashboard process: $procId"
        }
        catch {
            Write-Output "Failed to stop process ${procId}: $($_.Exception.Message)"
        }
    }
    Start-Sleep -Seconds 1
}

$runnerQuoted = '"' + $runner + '"'
$dashboardQuoted = '"' + $dashboardScript + '"'
$stdoutQuoted = '"' + $stdoutLog + '"'
$stderrQuoted = '"' + $stderrLog + '"'
$projectQuoted = '"' + $projectRoot + '"'
$command = 'cd /d ' + $projectQuoted + ' && start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File ' + $runnerQuoted + ' ' + $dashboardQuoted + ' --host ' + $dashboardHost + ' --port ' + $port + ' 1>' + $stdoutQuoted + ' 2>' + $stderrQuoted
cmd /c $command | Out-Null

Start-Sleep -Seconds 3

$listening = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($listening) {
    Write-Output ""
    Write-Output "Dashboard is running:"
    Write-Output "  http://$dashboardHost`:$port/"
    Write-Output ""
    Write-Output "Press Ctrl+F5 in the browser if the page still shows old content."
}
else {
    Write-Output ""
    Write-Output "Dashboard failed to start. Check logs:"
    Write-Output "  $stdoutLog"
    Write-Output "  $stderrLog"
    throw "Dashboard failed to start on port $port."
}
