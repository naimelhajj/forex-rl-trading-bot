# Environment.py Corruption Fix Script
# Run this to create a clean minimal environment.py for testing

import re

# Read the corrupted file
with open('environment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Critical fixes to apply
fixes = [
    # Fix 1: Clean up the reset() method around line 264
    (r'self\.balance = self\.initial_balance by trainer\)', 'self.balance = self.initial_balance'),
    (r'self\.trades_this_ep = 0 Dict with position info', 'self.trades_this_ep = 0'),
    (r'self\.bars_since_close = 00\.0', 'self.bars_since_close = 0'),
    (r'self\.last_action = \[0, 0, 0, 0\]dget tracking', 'self.last_action = [0, 0, 0, 0]'),
    
    # Fix 2: Clean up duplicated return statements
    (r'return self\._get_state\(\)k_side = None', 'return self._get_state()'),
    (r'return self\._get_state\(\)lse', ''),
    (r'self\._frame_stack = Noneive = False', 'self._frame_stack = None'),
    
    # Fix 3: Fix step() method header
    (r'def step\(self, action: int\) -> Tuple\[np\.ndarray, float, bool, Dict\]:[\s\S]{0,200}"""f\.bars_in_position = 0',
     'def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:\n        """'),
    
    # Fix 4: Clean up market data section
    (r'current_data = self\.data\.iloc\[self\.current_step\][\s\S]{0,50}current_price = current_data\[\'close\'\]# Track position state',
     "current_data = self.data.iloc[self.current_step]\n        current_price = current_data['close']"),
    
    # Fix 5: Clean up reward initialization
    (r'reward = 0\.0dget_pct', 'reward = 0.0'),
    (r'did_trade = Falsetrol state', 'did_trade = False'),
    
    # Fix 6: Clean up position check
    (r'if self\.position is not None:0\]', 'if self.position is not None:'),
    (r'self\.position = Noneuple\[np\.ndarray', 'self.position = None'),
    
    # Fix 7: Clean up LONG action section
    (r'if self\._can_modify\(\):close\'\]', 'if self._can_modify():'),
    (r'did_trade = Truen before weekend', 'did_trade = True'),
    (r'reward -= float\(self\.flip_penalty\)\(current_data\)', 'reward -= float(self.flip_penalty)'),
    (r'self\.trades_this_ep \+= 1s action changed', 'self.trades_this_ep += 1'),
]

# Apply fixes
for pattern, replacement in fixes:
    content = re.sub(pattern, replacement, content)

# Write the fixed file
with open('environment_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Created environment_fixed.py - review and rename to environment.py if correct")
print("Backup your current environment.py first!")
