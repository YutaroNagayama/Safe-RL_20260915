param(
    [string]$PythonCommand = "py",
    [string]$TorchIndexUrl = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    if ($PythonCommand -eq "py") {
        & py -3.11 -m venv $VenvRoot
    } else {
        & $PythonCommand -m venv $VenvRoot
    }
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($TorchIndexUrl) {
    & $VenvPython -m pip install torch --index-url $TorchIndexUrl
}
Push-Location $ProjectRoot
try {
    & $VenvPython -m pip install -e ".[learn,plot,dev]"
    & $VenvPython -m ipykernel install --user --name uav-safe-marl --display-name "UAV Safe MARL (.venv)"
    & $VenvPython -m uav_safe_marl doctor
} finally {
    Pop-Location
}

Write-Host "Setup complete. In VSCode select kernel: UAV Safe MARL (.venv)"

