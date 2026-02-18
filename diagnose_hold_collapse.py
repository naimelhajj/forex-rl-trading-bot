"""
Diagnostic script to analyze why agent is stuck on HOLD.
"""
import torch
import numpy as np
from agent import DQNAgent, ActionSpace
from config import Config

print("=" * 60)
print("PHASE 2.8d HOLD COLLAPSE DIAGNOSTIC")
print("=" * 60)

# Load config
cfg = Config()
print(f"\n1. CONFIG CHECK")
print(f"   entropy_beta: {cfg.environment.entropy_beta}")
print(f"   hold_tie_tau: {cfg.agent.hold_tie_tau}")
print(f"   flip_penalty: {cfg.environment.flip_penalty}")
print(f"   learning_rate: {cfg.agent.learning_rate}")
print(f"   use_noisy: {cfg.agent.use_noisy}")
print(f"   epsilon_start: {cfg.agent.epsilon_start}")

# Load agent
print(f"\n2. LOADING BEST MODEL")
agent = DQNAgent(state_size=93, action_size=ActionSpace.get_action_size(), 
                 learning_rate=cfg.agent.learning_rate,
                 use_noisy=cfg.agent.use_noisy)
try:
    agent.load('checkpoints/best_model.pt')
    print(f"   ✓ Model loaded successfully")
    print(f"   Epsilon from checkpoint: {agent.epsilon:.4f}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    exit(1)

# Test Q-values on multiple random states
print(f"\n3. Q-VALUE ANALYSIS (10 random states)")
hold_wins = 0
total_tests = 10

for i in range(total_tests):
    state = np.random.randn(93)
    q = agent.get_q_values(state)
    best_action = int(np.argmax(q))
    
    if best_action == 0:  # HOLD
        hold_wins += 1
    
    if i < 3:  # Print first 3 examples
        print(f"   State {i+1}: Q(HOLD)={q[0]:7.4f} Q(LONG)={q[1]:7.4f} Q(SHORT)={q[2]:7.4f} Q(MOVE_SL)={q[3]:7.4f} → action={best_action}")

hold_percentage = (hold_wins / total_tests) * 100
print(f"\n   HOLD chosen: {hold_wins}/{total_tests} times ({hold_percentage:.0f}%)")

# Check Q-value statistics
all_q_values = []
for _ in range(100):
    state = np.random.randn(93)
    q = agent.get_q_values(state)
    all_q_values.append(q)

all_q = np.array(all_q_values)
q_means = all_q.mean(axis=0)
q_stds = all_q.std(axis=0)

print(f"\n4. Q-VALUE STATISTICS (100 random states)")
print(f"   HOLD:    mean={q_means[0]:7.4f}  std={q_stds[0]:.4f}")
print(f"   LONG:    mean={q_means[1]:7.4f}  std={q_stds[1]:.4f}")
print(f"   SHORT:   mean={q_means[2]:7.4f}  std={q_stds[2]:.4f}")
print(f"   MOVE_SL: mean={q_means[3]:7.4f}  std={q_stds[3]:.4f}")

q_diff = q_means[0] - q_means[[1,2,3]].max()
print(f"\n   Q(HOLD) advantage over best alternative: {q_diff:.4f}")

# Diagnosis
print(f"\n5. DIAGNOSIS")
if hold_percentage > 90:
    print(f"   ✗ CRITICAL: Agent has {hold_percentage:.0f}% HOLD bias")
    print(f"   → Q-network has learned to always prefer HOLD")
    if q_diff > 0.1:
        print(f"   → Q(HOLD) is {q_diff:.4f} higher than alternatives")
        print(f"   → This is a LARGE gap - agent is strongly biased")
    elif q_diff > 0.01:
        print(f"   → Q(HOLD) is slightly higher ({q_diff:.4f})")
        print(f"   → Combined with hold_tie_tau={cfg.agent.hold_tie_tau}, this locks in HOLD")
    else:
        print(f"   → Q-values are close, but hold_tie_tau={cfg.agent.hold_tie_tau} tips to HOLD")
else:
    print(f"   ✓ Q-values show some diversity ({hold_percentage:.0f}% HOLD)")

# Recommendations
print(f"\n6. RECOMMENDATIONS")
if hold_percentage > 95:
    print(f"   URGENT: Need to break Q-value bias")
    print(f"   Option A: Reinitialize Q-network (start fresh)")
    print(f"   Option B: Set hold_tie_tau=0.0 (remove HOLD preference)")
    print(f"   Option C: Add epsilon-greedy exploration (force random actions)")
elif q_diff > 0.05:
    print(f"   Q(HOLD) significantly higher than alternatives")
    print(f"   → Increase entropy_beta to {cfg.environment.entropy_beta * 3:.4f}")
    print(f"   → Decrease hold_tie_tau to {cfg.agent.hold_tie_tau * 0.5:.4f}")
else:
    print(f"   Q-values relatively balanced - problem may be elsewhere")
    print(f"   → Check if entropy bonus is being applied during training")
    print(f"   → Check if eval_mode is preventing exploration")

print(f"\n" + "=" * 60)
