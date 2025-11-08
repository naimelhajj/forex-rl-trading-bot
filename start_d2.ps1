# FIX PACK D2 RESTART
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "FIX PACK D2 RESTART - Rolling Window Anti-Collapse" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Clear old checkpoints
Write-Host "Clearing old checkpoints..." -ForegroundColor Yellow
if (Test-Path "checkpoints\*.pt") {
    Remove-Item "checkpoints\*.pt" -Force
    Write-Host "  Done" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "STARTING 80-EPISODE RUN (seed 42)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor with: .\monitor_fixpack_d2.ps1 -Episode 10" -ForegroundColor Yellow
Write-Host "TARGET: long_ratio 0.35-0.65 (NOT 0.98!)" -ForegroundColor Yellow
Write-Host ""

python main.py --episodes 80 --seed 42
