# State Size Fix - From 65 to 69 Dimensions

## Issue
After applying hardening patch #1 (enhanced portfolio features from 19 to 23), the system crashed with:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x69 and 65x256)
```

## Root Cause
The `state_size` calculation in `environment.py` line 178 still used the old value:
```python
self.state_size = len(feature_columns) + 19  # OLD - wrong!
```

But `_portfolio_features()` now returns **23 features**, not 19.

## Fix Applied
Updated `environment.py` line 178:
```python
self.state_size = len(feature_columns) + 23  # NEW - correct!
```

Also updated docstring in `_get_state()` to reflect 23 portfolio features.

## State Composition
**Total: 69 dimensions**
- **Market features: 46** (from feature engineering)
- **Portfolio features: 23** (balance-invariant ratios)

### Portfolio Features Breakdown (23 total):
1. `side` - Position direction normalized (-1, 0, 1)
2. `pos_dir` - Position direction: 0=flat, 1=long, -1=short
3. `long_on` - Binary flag: 1 if long position
4. `short_on` - Binary flag: 1 if short position
5. `lots_norm` - Lot size / hard cap
6. `size_frac` - Lot size as fraction of max allowed
7. `entry_diff_atr` - Entry distance from current price / ATR
8. `unrealized_pct` - Unrealized PnL as % of equity
9. `sl_dist` - SL distance / ATR
10. `sl_dist_norm` - Normalized SL distance / ATR
11. `tp_dist` - TP distance / ATR
12. `tp_dist_norm` - Normalized TP distance / ATR
13. `dd_pct` - Drawdown from peak as %
14. `hold_left` - Remaining hold period / max hold
15. `cool_left` - Remaining cooldown / max cooldown
16. `trades_frac` - Trades taken / max trades
17. `equity_pct` - Equity / initial balance
18. `balance_pct` - Balance / initial balance
19. `margin_used_pct` - Margin used / equity
20. `drawdown_pct` - Max drawdown %
21. `last_action[0]` - One-hot: HOLD
22. `last_action[1]` - One-hot: LONG
23. `last_action[2]` - One-hot: SHORT

## Verification
```bash
python main.py --episodes 2
```

Expected output:
```
State size: 69
Agent created:
  State size: 69
```

✅ **VERIFIED - Training runs without errors**

## Related Files
- `environment.py` (line 178, 508) - Fixed
- `HARDENING_PATCHES_SUMMARY.md` - Documents the 23 portfolio features
