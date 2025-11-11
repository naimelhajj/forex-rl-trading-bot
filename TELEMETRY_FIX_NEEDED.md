# Telemetry Export Fix Needed

## Issue
The confirmation suite now accepts the `--telemetry extended` and `--output-dir` arguments, but the extended telemetry data (episode_metrics.json) is not being exported.

## Root Cause
The `_export_extended_telemetry()` method in trainer.py is not being called, or is failing silently. Debug logging shows the parameters are being passed correctly.

## Quick Fix Required

### Option 1: Post-process from episode_events.jsonl
The episode_events.jsonl file is being created successfully with episode-level data. We can create a simple script to convert this to the format expected by analyze_confirmation_results.py:

```python
import json
from pathlib import Path

def convert_episode_events_to_metrics(seed_dir):
    """Convert episode_events.jsonl to episode_metrics.json format."""
    events_file = Path(seed_dir) / 'logs' / 'episode_events.jsonl'
    output_file = Path(seed_dir) / 'episode_metrics.json'
    
    episodes = []
    episode_data = {}
    
    with open(events_file, 'r') as f:
        for line in f:
            event = json.loads(line)
            
            if event['event_type'] == 'episode_end':
                ep = event['episode']
                
                # Collect basic metrics
                metric = {
                    'episode': ep,
                    'p_long_smoothed': 0.5,  # TODO: Extract from agent state
                    'p_hold_smoothed': 0.7,  # TODO: Extract from agent state
                    'lambda_long': 0.0,      # TODO: Extract from agent state
                    'lambda_hold': 0.0,      # TODO: Extract from agent state
                    'tau': 1.0,              # TODO: Extract from agent state
                    'H_bits': 1.0,           # TODO: Extract from agent state
                    'run_len_max': 0,        # TODO: Calculate from action history
                    'trades': event['trades'],
                    'SPR': event['reward'],  # Using reward as SPR proxy
                    'switch_rate': 0.0,      # TODO: Calculate from action history
                }
                
                episodes.append(metric)
    
    # Save in expected format
    with open(output_file, 'w') as f:
        json.dump({'episodes': episodes}, f, indent=2)

# Usage:
# convert_episode_events_to_metrics('confirmation_results/seed_42')
```

### Option 2: Fix Trainer Export  (RECOMMENDED)
The _export_extended_telemetry() method needs to actually collect the controller state variables during training, not just at the end.

**Required Changes:**
1. **Add episode-level tracking to Agent** - Store p_long_smoothed, p_hold_smoothed, lambda_long, lambda_hold, tau, H_bits after each episode
2. **Store in training_history** - Append these values to each episode's stats dictionary
3. **Export at end** - _export_extended_telemetry() should read from training_history

**Implementation:**

In `agent.py`, after each episode reset or at episode boundaries:
```python
def reset_episode_tracking(self):
    """Reset controller state tracking for new episode."""
    self.episode_metrics = {
        'p_long_smoothed': 0.5,
        'p_hold_smoothed': 0.7,
        'lambda_long': 0.0,
        'lambda_hold': 0.0,
        'tau': 1.0,
        'H_bits': 1.0,
        'run_len_max': 0,
        'switch_rate': 0.0,
    }
    self.action_history_episode = []

def get_episode_telemetry(self):
    """Get controller telemetry for current episode."""
    # Calculate switch rate
    switches = sum(1 for i in range(1, len(self.action_history_episode)) 
                   if self.action_history_episode[i] != self.action_history_episode[i-1])
    switch_rate = switches / max(len(self.action_history_episode), 1)
    
    # Calculate max run length
    if self.action_history_episode:
        current_run = 1
        max_run = 1
        for i in range(1, len(self.action_history_episode)):
            if self.action_history_episode[i] == self.action_history_episode[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        run_len_max = max_run
    else:
        run_len_max = 0
    
    return {
        'p_long_smoothed': float(getattr(self, 'p_long_smoothed', 0.5)),
        'p_hold_smoothed': float(getattr(self, 'p_hold_smoothed', 0.7)),
        'lambda_long': float(getattr(self, 'lambda_long', 0.0)),
        'lambda_hold': float(getattr(self, 'lambda_hold', 0.0)),
        'tau': float(getattr(self, 'tau', 1.0)),
        'H_bits': float(getattr(self, 'H_bits', 1.0)),
        'run_len_max': int(run_len_max),
        'switch_rate': float(switch_rate),
    }
```

In `trainer.py train_episode()`, after episode ends:
```python
# Collect extended telemetry if needed
if telemetry_mode == 'extended':
    telemetry = self.agent.get_episode_telemetry()
    train_stats.update(telemetry)
```

Then _export_extended_telemetry() can simply read from training_history which will have all the data.

## Status
- ✅ CLI arguments added (--telemetry, --output-dir)
- ✅ Confirmation suite passes arguments correctly
- ✅ Training runs successfully
- ✅ episode_events.jsonl is created
- ❌ episode_metrics.json NOT created (export function not working)
- ⏳ Agent instrumentation needed for full controller tracking

## Next Steps
1. Implement Option 2 (agent instrumentation + proper export)
2. Test with minimal run (1 seed, 5 episodes)
3. Verify episode_metrics.json is created with all 11 required fields
4. Run full confirmation suite

## Workaround for Immediate Testing
Create a simple converter script to transform episode_events.jsonl into episode_metrics.json with placeholder values for the missing controller variables. This will at least let the analyzer run, even if some gates can't be properly checked yet.
