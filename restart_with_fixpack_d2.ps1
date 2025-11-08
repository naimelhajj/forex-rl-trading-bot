# OPTION A: Stop current run and restart with Fix Pack D2
# Phase 2.8d - Rolling Window Anti-Collapse Validation

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "RESTARTING WITH FIX PACK D2 (Rolling Window Anti-Collapse)" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# Step 1: Backup old checkpoints (Fix Pack D1)
Write-Host "`n[1/4] Backing up old checkpoints (Fix Pack D1)..." -ForegroundColor Yellow

if (-not (Test-Path "checkpoints_backup_fixpack_d1")) {
    New-Item -ItemType Directory -Path "checkpoints_backup_fixpack_d1" | Out-Null
}

$checkpoints = Get-ChildItem -Path "checkpoints" -Filter "*.pt" -ErrorAction SilentlyContinue
if ($checkpoints) {
    Write-Host "  Moving $($checkpoints.Count) checkpoint(s) to backup folder..." -ForegroundColor Gray
    foreach ($file in $checkpoints) {
        $destPath = Join-Path "checkpoints_backup_fixpack_d1" $file.Name
        if (Test-Path $destPath) {
            Remove-Item $destPath -Force
        }
        Move-Item -Path $file.FullName -Destination "checkpoints_backup_fixpack_d1\" -Force
    }
    Write-Host "  ✓ Checkpoints backed up" -ForegroundColor Green
} else {
    Write-Host "  ℹ No checkpoints to backup (fresh start)" -ForegroundColor Gray
}

# Step 2: Backup old validation logs (optional)
Write-Host "`n[2/4] Backing up validation logs..." -ForegroundColor Yellow

if (Test-Path "logs\validation_summaries") {
    $valFiles = Get-ChildItem -Path "logs\validation_summaries" -Filter "*.json" -ErrorAction SilentlyContinue
    if ($valFiles) {
        if (-not (Test-Path "logs_backup_seed42_fixpack_d1")) {
            New-Item -ItemType Directory -Path "logs_backup_seed42_fixpack_d1" | Out-Null
        }
        Write-Host "  Copying $($valFiles.Count) validation file(s) to backup..." -ForegroundColor Gray
        Copy-Item -Path "logs\validation_summaries\*.json" -Destination "logs_backup_seed42_fixpack_d1\" -Force
        Write-Host "  ✓ Validation logs backed up" -ForegroundColor Green
    }
}

# Step 3: Verify Fix Pack D2 parameters
Write-Host "`n[3/4] Verifying Fix Pack D2 parameters..." -ForegroundColor Yellow

$configContent = Get-Content "config.py" -Raw
$checks = @(
    @{Name="ls_balance_lambda"; Pattern='ls_balance_lambda:\s*float\s*=\s*0\.006'; Value="0.006"},
    @{Name="hold_balance_lambda"; Pattern='hold_balance_lambda:\s*float\s*=\s*0\.002'; Value="0.002"},
    @{Name="cooldown_bars"; Pattern='cooldown_bars:\s*int\s*=\s*12'; Value="12"},
    @{Name="hold_tie_tau"; Pattern='hold_tie_tau:\s*float\s*=\s*0\.030'; Value="0.030"},
    @{Name="eval_tie_tau"; Pattern='eval_tie_tau:\s*float\s*=\s*0\.035'; Value="0.035"}
)

$allGood = $true
foreach ($check in $checks) {
    if ($configContent -match $check.Pattern) {
        Write-Host "  ✓ $($check.Name) = $($check.Value)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($check.Name) NOT FOUND (expected $($check.Value))" -ForegroundColor Red
        $allGood = $false
    }
}

if (-not $allGood) {
    Write-Host "`n⚠ WARNING: Fix Pack D2 parameters not fully applied!" -ForegroundColor Red
    Write-Host "Run the implementation again or manually edit config.py" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 4: Start fresh 80-episode run with Fix Pack D2
Write-Host "`n[4/4] Starting fresh 80-episode run (seed 42)..." -ForegroundColor Yellow
Write-Host "`nCommand: python main.py --episodes 80 --seed 42`n" -ForegroundColor Cyan

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "CHECKPOINT MONITORING SCHEDULE" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Episode 10: Check for early directional collapse signal"
Write-Host "Episode 20: Validate rolling window effectiveness"
Write-Host "Episode 40: Mid-point behavioral assessment"
Write-Host "Episode 60: Pre-final stability check"
Write-Host "Episode 80: Full validation (COMPLETION)"
Write-Host ""
Write-Host "TARGET: long_ratio 0.35-0.65 (NOT 0.98!)" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Start the training run
python main.py --episodes 80 --seed 42
