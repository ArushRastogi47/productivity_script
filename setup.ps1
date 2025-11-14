Param(
    [switch]$Recreate
)

Write-Host "FocusVision setup starting..." -ForegroundColor Cyan

if ($Recreate -and (Test-Path ".venv")) {
    Write-Host "Removing existing .venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment at .venv" -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& .\.venv\Scripts\python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
& .\.venv\Scripts\pip install -r requirements.txt

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate the environment and run:" -ForegroundColor Green
Write-Host " .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host " python main.py" -ForegroundColor Green


