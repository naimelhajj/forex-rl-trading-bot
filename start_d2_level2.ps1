# FIX PACK D2 LEVEL 2 ESCALATION
# Episode 50 showed continued collapse (93% long)
# Penalties increased 8-10x, margin tightened to 40-60%

Write-Host ""
Write-Host "================================================================" -ForegroundColor Red
Write-Host "FIX PACK D2 LEVEL 2 ESCALATION - 8-10x Stronger Penalties" -ForegroundColor Red
Write-Host "================================================================" -ForegroundColor Red
Write-Host ""
Write-Host "REASON: Episode 50 collapsed to 93% long" -ForegroundColor Yellow
Write-Host "OLD penalties too weak vs trading rewards (0.0005 vs 1-5)" -ForegroundColor Yellow
Write-Host ""

# Clear checkpoints
Write-Host "Clearing checkpoints..." -ForegroundColor Yellow
if (Test-Path "checkpoints\*.pt") {
    Remove-Item "checkpoints\*.pt" -Force
    Write-Host "  Done" -ForegroundColor Green
}

Write-Host ""
Write-Host "NEW PARAMETERS:" -ForegroundColor Cyan
Write-Host "  ls_balance_lambda:  0.006 -> 0.050 (8x stronger)" -ForegroundColor White
Write-Host "  hold_balance_lambda: 0.002 -> 0.020 (10x stronger)" -ForegroundColor White
Write-Host "  Margin:              15% -> 10% (40-60% range, was 35-65%)" -ForegroundColor White
Write-Host ""
Write-Host "EXPECTED PENALTY (93% long):" -ForegroundColor Cyan
Write-Host "  Imbalance: |0.93 - 0.5| - 0.10 = 0.33" -ForegroundColor Gray
Write-Host "  Penalty: 0.050 * (0.33)^2 = 0.00545 per step" -ForegroundColor Gray
Write-Host "  Over 500 bars: ~2.7 reward units (now SIGNIFICANT!)" -ForegroundColor Yellow
Write-Host ""

Write-Host "================================================================" -ForegroundColor Red
Write-Host "STARTING 80-EPISODE RUN (seed 42)" -ForegroundColor Red
Write-Host "================================================================" -ForegroundColor Red
Write-Host ""
Write-Host "TARGET: long_ratio 0.40-0.60 (NOT 0.93!)" -ForegroundColor Yellow
Write-Host ""

python main.py --episodes 80 --seed 42
