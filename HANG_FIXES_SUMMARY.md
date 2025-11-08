# Training Hang Fixes - Complete Summary

## Issues Found and Fixed

### Issue 1: Slow `_rolling_lr_slope()` - FIXED ✅
**Problem**: Used `pandas.rolling().apply()` which hung on 10k data points  
**Solution**: Manual loop with pre-computed x statistics  
**Result**: 0.16s instead of infinite hang  
**Files**: `features.py`

### Issue 2: Slow `rolling_percentile()` - FIXED ✅  
**Problem**: Also used `pandas.rolling().apply()`  
**Solution**: Manual loop with NumPy arrays  
**Files**: `features.py`

### Issue 3: Infinite Recursion in `DuelingDQN.reset_noise()` - FIXED ✅
**Problem**: Called `self.modules()` which includes self, causing infinite recursion  
**Root Cause**:
```python
def reset_noise(self):
    for m in self.modules():  # includes self!
        if hasattr(m, 'reset_noise'):
            m.reset_noise()  # calls self infinitely!
```

**Solution**: Explicitly iterate only through known layer containers:
```python
def reset_noise(self):
    """Reset noise for all NoisyLinear modules in the network."""
    for layer_name in ['feature', 'value_stream', 'adv_stream']:
        if hasattr(self, layer_name):
            layer = getattr(self, layer_name)
            if isinstance(layer, nn.Sequential):
                for module in layer:
                    if hasattr(module, 'reset_noise'):
                        module.reset_noise()
```

**Files**: `agent.py` (DuelingDQN class)

## Performance Impact

### Before All Fixes:
- Feature computation: **HUNG FOREVER**  
- Training: **Never started**

### After All Fixes:
- Feature computation: **0.33s** (from output)
- Training: Should proceed normally

## Files Modified

1. **features.py**:
   - `_rolling_lr_slope()`: Optimized with manual loop
   - `rolling_percentile()`: Optimized with manual loop
   
2. **agent.py**:
   - `DuelingDQN.reset_noise()`: Fixed infinite recursion

3. **trainer.py**:
   - Added debug logging (can be removed after verification)

## Testing

Run smoke test to verify all fixes:
```bash
python main.py --episodes 5
```

Expected behavior:
- Features compute in <1s
- Training starts and progresses
- Episodes complete without hanging
- Steps print every 50: "Completed step X"

## Next Steps After Verification

1. Remove debug logging from `trainer.py`
2. Run full training session
3. Update documentation
4. Commit changes with message: "Fix training hangs: optimize features, fix NoisyNet recursion"
