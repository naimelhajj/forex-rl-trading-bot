# SIMPLE RESTART SCRIPT - Fix Pack D2
# Clears checkpoints and starts fresh 80-episode run

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  FIX PACK D2 RESTART - Rolling Window Anti-Collapse" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Clear old checkpoints (force fresh start)
Write-Host "[1/2] Clearing old checkpoints..." -ForegroundColor Yellow
if (Test-Path "checkpoints\*.pt") {
    Remove-Item "checkpoints\*.pt" -Force
    Write-Host "  ✓ Old checkpoints removed" -ForegroundColor Green
} else {
    Write-Host "  ℹ No old checkpoints found" -ForegroundColor Gray
}

# Verify key parameters
Write-Host ""
Write-Host "[2/2] Verifying Fix Pack D2 parameters..." -ForegroundColor Yellow
$config = Get-Content "config.py" -Raw

$params = @{
    "ls_balance_lambda" = "0.006"
    "hold_balance_lambda" = "0.002"
    "cooldown_bars" = "12"
}

$allOk = $true
foreach ($key in $params.Keys) {
    $val = $params[$key]
    if ($config -match "$key.*$val") {
        Write-Host "  ✓ $key = $val" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $key NOT $val" -ForegroundColor Red
        $allOk = $false
    }
}

if (-not $allOk) {
    Write-Host ""
    Write-Host "⚠ WARNING: Parameters not configured correctly!" -ForegroundColor Red
    Write-Host "Fix config.py before continuing" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Start training
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  STARTING 80-EPISODE RUN (seed 42)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "MONITORING:" -ForegroundColor Yellow
Write-Host "  Episode 10: .\monitor_fixpack_d2.ps1 -Episode 10" -ForegroundColor Gray
Write-Host "  Episode 20: .\monitor_fixpack_d2.ps1 -Episode 20" -ForegroundColor Gray
Write-Host "  Episode 40: .\monitor_fixpack_d2.ps1 -Episode 40" -ForegroundColor Gray
Write-Host "  Episode 80: .\monitor_fixpack_d2.ps1 -Episode 80" -ForegroundColor Gray
Write-Host ""
Write-Host "TARGET: long_ratio 0.35-0.65 (NOT 0.98!)" -ForegroundColor Yellow
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Run training
python main.py --episodes 80 --seed 42
