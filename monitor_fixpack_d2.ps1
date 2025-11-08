# Monitor Fix Pack D2 validation run
# Checks Episode 10, 20, 40, 60, 80 for behavioral targets

param(
    [int]$Episode = 10  # Default to Episode 10
)

Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "FIX PACK D2 CHECKPOINT MONITOR - Episode $Episode" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

$valFile = "logs\validation_summaries\val_ep{0:D3}.json" -f $Episode

if (-not (Test-Path $valFile)) {
    Write-Host "`n⏳ Episode $Episode not yet completed" -ForegroundColor Yellow
    Write-Host "   Waiting for: $valFile`n" -ForegroundColor Gray
    exit 0
}

# Load validation JSON
$val = Get-Content $valFile | ConvertFrom-Json

# Extract key metrics
$spr = [math]::Round($val.score, 3)
$trades = $val.trades
$entropy = [math]::Round($val.action_entropy_bits, 2)
$hold = [math]::Round($val.hold_rate, 2)
$longRatio = [math]::Round($val.long_short.long_ratio, 2)
$switchRate = [math]::Round($val.switch_rate, 2)
$penaltyRate = [math]::Round($val.penalty_rate, 2)

# Display metrics
Write-Host "`nVALIDATION METRICS (Episode $Episode):" -ForegroundColor White
Write-Host "  SPR:           $spr" -ForegroundColor Gray
Write-Host "  Trades:        $trades" -ForegroundColor Gray
Write-Host "  Entropy:       $entropy bits" -ForegroundColor Gray
Write-Host "  Hold rate:     $hold" -ForegroundColor Gray
Write-Host "  Switch rate:   $switchRate" -ForegroundColor Gray
Write-Host "  Penalty rate:  $penaltyRate" -ForegroundColor Gray

# Critical metric: Long ratio
Write-Host "`n  Long ratio:    $longRatio" -ForegroundColor $(
    if ($longRatio -ge 0.35 -and $longRatio -le 0.65) { "Green" }
    elseif ($longRatio -ge 0.25 -and $longRatio -le 0.75) { "Yellow" }
    else { "Red" }
)

# Behavioral assessment
Write-Host "`n" + ("-" * 70) -ForegroundColor Gray
Write-Host "BEHAVIORAL ASSESSMENT:" -ForegroundColor White

$issues = @()
$warnings = @()
$passes = @()

# Check long ratio (CRITICAL)
if ($longRatio -ge 0.35 -and $longRatio -le 0.65) {
    $passes += "✓ Long ratio BALANCED ($longRatio in 0.35-0.65 range)"
} elseif ($longRatio -ge 0.25 -and $longRatio -le 0.75) {
    $warnings += "⚠ Long ratio BORDERLINE ($longRatio, target 0.35-0.65)"
} else {
    $issues += "✗ Long ratio COLLAPSED ($longRatio, expected 0.35-0.65)"
}

# Check entropy
if ($entropy -ge 0.95 -and $entropy -le 1.05) {
    $passes += "✓ Entropy balanced ($entropy bits)"
} elseif ($entropy -ge 0.80 -and $entropy -le 1.20) {
    $warnings += "⚠ Entropy borderline ($entropy bits, target 0.95-1.05)"
} else {
    $issues += "✗ Entropy out of range ($entropy bits, expected 0.95-1.05)"
}

# Check hold rate
if ($hold -ge 0.65 -and $hold -le 0.78) {
    $passes += "✓ Hold rate healthy ($hold)"
} elseif ($hold -ge 0.60 -and $hold -le 0.85) {
    $warnings += "⚠ Hold rate borderline ($hold, target 0.65-0.78)"
} else {
    $issues += "✗ Hold rate out of range ($hold, expected 0.65-0.78)"
}

# Check trades
if ($trades -ge 24 -and $trades -le 32) {
    $passes += "✓ Trade count moderate ($trades)"
} elseif ($trades -ge 18 -and $trades -le 40) {
    $warnings += "⚠ Trade count borderline ($trades, target 24-32)"
}

# Check penalty rate
if ($penaltyRate -le 0.10) {
    $passes += "✓ Penalty rate acceptable ($penaltyRate)"
} else {
    $warnings += "⚠ High penalty rate ($penaltyRate, target ≤0.10)"
}

# Display results
foreach ($pass in $passes) {
    Write-Host "  $pass" -ForegroundColor Green
}
foreach ($warn in $warnings) {
    Write-Host "  $warn" -ForegroundColor Yellow
}
foreach ($issue in $issues) {
    Write-Host "  $issue" -ForegroundColor Red
}

# Overall verdict
Write-Host "`n" + ("-" * 70) -ForegroundColor Gray

if ($issues.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "VERDICT: FIX PACK D2 WORKING PERFECTLY ✅" -ForegroundColor Green
    Write-Host "Continue monitoring at Episode 20, 40, 60, 80" -ForegroundColor Green
} elseif ($issues.Count -eq 0) {
    Write-Host "VERDICT: FIX PACK D2 WORKING (minor warnings) ✓" -ForegroundColor Yellow
    Write-Host "Continue monitoring - adjustments may be needed" -ForegroundColor Yellow
} else {
    Write-Host "VERDICT: FIX PACK D2 NEEDS ESCALATION ⚠" -ForegroundColor Red
    Write-Host "`nRECOMMENDED ACTIONS:" -ForegroundColor Red
    
    if ($longRatio -gt 0.75 -or $longRatio -lt 0.25) {
        Write-Host "  1. STOP current run (not improving)" -ForegroundColor Red
        Write-Host "  2. Increase ls_balance_lambda: 0.006 → 0.010 (+67%)" -ForegroundColor Red
        Write-Host "  3. Increase hold_balance_lambda: 0.002 → 0.004 (2x)" -ForegroundColor Red
        Write-Host "  4. Restart with stronger penalties" -ForegroundColor Red
    }
}

Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""
