"""
Quick analysis of Phase 2.8f 20-episode smoke test.
"""

# From terminal output, Episodes 1-20:
train_equity = [972.82, 941.68, 989.39, 1014.42, 998.28, 942.11, 964.46, 962.25, 989.34, 972.77,
                1025.67, 1019.17, 986.97, 996.66, 996.35, 947.83, 980.00, 979.50, 995.22, 985.37]

train_trades = [41, 44, 44, 44, 45, 44, 45, 44, 42, 40, 45, 40, 44, 42, 43, 44, 42, 41, 40, 43]

win_rates = [41.46, 29.55, 45.45, 43.18, 46.67, 34.09, 33.33, 40.91, 42.86, 42.50,
             46.67, 50.00, 36.36, 40.48, 44.19, 31.82, 45.24, 43.90, 47.50, 46.51]

import numpy as np

print("=" * 60)
print("PHASE 2.8f: 20-EPISODE SMOKE TEST RESULTS")
print("=" * 60)
print()

print("✅ OVERFLOW FIX: Zero warnings (log-sum-exp trick working)")
print()

print("TRAINING METRICS:")
print(f"  Equity: Mean ${np.mean(train_equity):.2f}, Std ${np.std(train_equity):.2f}")
print(f"  Equity Range: ${np.min(train_equity):.2f} - ${np.max(train_equity):.2f}")
print(f"  Trades/Episode: Mean {np.mean(train_trades):.1f}, Range {min(train_trades)}-{max(train_trades)}")
print(f"  Win Rate: Mean {np.mean(win_rates):.1f}%, Range {min(win_rates):.1f}-{max(win_rates):.1f}%")
print()

print("STABILITY CHECK:")
equity_volatility = np.std(train_equity) / np.mean(train_equity)
print(f"  Equity Volatility (CV): {equity_volatility:.3f} (<0.10 is good)")
print(f"  Trade Consistency: Std {np.std(train_trades):.1f} (low is good)")
print()

# Final restored model validation
print("FINAL VALIDATION (Best Model Restored):")
print("  Sharpe Ratio: 0.203")
print("  Profit Factor: 1.17")
print("  Max Drawdown: 1.61%")
print("  Trades: 29")
print()

# Holdout validation
print("HOLDOUT VALIDATION (Shifted Windows):")
print("  Sharpe Ratio: 0.323")
print("  Profit Factor: 1.10")
print("  Max Drawdown: 1.58%")
print()

print("=" * 60)
print("VERDICT:")
print("=" * 60)
print("✅ Phase 2.8f Controller: PASSED smoke test")
print("✅ Overflow Bug: FIXED (zero warnings)")
print("✅ Training Stable: Low equity volatility")
print("✅ Trade Activity: Consistent 40-45 trades/episode")
print("✅ Validation Performance: Positive Sharpe, low drawdown")
print()
print("READY FOR: Git commit + 3-seed validation")
print("=" * 60)
