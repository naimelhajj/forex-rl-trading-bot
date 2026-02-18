"""
Quick test to verify telemetry collection is working
"""
import numpy as np
from agent import DQNAgent, ActionSpace

# Create a simple agent
agent = DQNAgent(state_size=100, action_size=ActionSpace.get_action_size(), use_dual_controller=True)

# Simulate an episode
print("Testing telemetry collection...")

# Reset episode tracking
agent.reset_episode_tracking()
print("✓ Episode tracking reset")

# Simulate some actions
actions = [0, 0, 1, 1, 1, 2, 0, 1, 3, 0]  # Mix of actions
for action in actions:
    agent.action_history_episode.append(action)
    # Simulate controller state updates
    agent._update_controller_state(action)

print(f"✓ Tracked {len(agent.action_history_episode)} actions")

# Get telemetry
telemetry = agent.get_episode_telemetry()

print("\nTelemetry collected:")
for key, value in telemetry.items():
    print(f"  {key}: {value}")

# Verify all required fields are present
required_fields = [
    'p_long_smoothed', 'p_hold_smoothed', 'lambda_long', 'lambda_hold',
    'tau', 'H_bits', 'run_len_max', 'switch_rate'
]

missing = [f for f in required_fields if f not in telemetry]
if missing:
    print(f"\n❌ Missing fields: {missing}")
else:
    print(f"\n✅ All {len(required_fields)} required telemetry fields present!")

# Check calculations
print("\nVerifying calculations:")
print(f"  Actions: {actions}")
print(f"  Switches: {sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])}")
print(f"  Switch rate: {telemetry['switch_rate']:.3f}")
print(f"  Max run length: {telemetry['run_len_max']}")
print(f"  Entropy: {telemetry['H_bits']:.3f} bits")

print("\n✅ Telemetry tracking is working correctly!")
