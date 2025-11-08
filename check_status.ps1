# Quick status check for current episode
$latest = Get-ChildItem "logs\validation_summaries\val_ep*.json" -ErrorAction SilentlyContinue | 
    Sort-Object Name -Descending | 
    Select-Object -First 1

if ($latest) {
    $epNum = [regex]::Match($latest.Name, "val_ep(\d+)").Groups[1].Value
    $val = Get-Content $latest.FullName | ConvertFrom-Json
    
    Write-Host ""
    Write-Host "LATEST COMPLETED: Episode $epNum" -ForegroundColor Cyan
    Write-Host "  SPR:         $([math]::Round($val.score, 3))" -ForegroundColor Gray
    Write-Host "  Trades:      $($val.trades)" -ForegroundColor Gray
    Write-Host "  Entropy:     $([math]::Round($val.action_entropy_bits, 2))" -ForegroundColor Gray
    Write-Host "  Hold rate:   $([math]::Round($val.hold_rate, 2))" -ForegroundColor Gray
    
    $lr = [math]::Round($val.long_short.long_ratio, 2)
    $color = if ($lr -ge 0.35 -and $lr -le 0.65) { "Green" } 
             elseif ($lr -ge 0.25 -and $lr -le 0.75) { "Yellow" } 
             else { "Red" }
    Write-Host "  Long ratio:  $lr" -ForegroundColor $color
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "No validation files yet - training in progress..." -ForegroundColor Yellow
    Write-Host ""
}
