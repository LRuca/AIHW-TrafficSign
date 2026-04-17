$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "C:\Users\lenovo\.conda\envs\pytorch\python.exe"
$envRoot = Split-Path -Parent $pythonExe

$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"
$env:CONDA_DLL_SEARCH_MODIFICATION_ENABLE = "1"
$env:PYTHONPATH = "$projectRoot\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

if ($args.Count -eq 0) {
    & $pythonExe -c "import numpy, torch, torchvision; print('python ok'); print('numpy', numpy.__version__); print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
    exit $LASTEXITCODE
}

& $pythonExe @args
exit $LASTEXITCODE
