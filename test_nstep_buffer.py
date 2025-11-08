"""Quick test of n-step buffer fix"""
import numpy as np
from agent import ReplayBuffer, PrioritizedReplayBuffer

print("Testing ReplayBuffer with n-step=3...")
buffer = ReplayBuffer(capacity=1000, n_step=3, gamma=0.99)

# Simulate some transitions
for i in range(10):
    state = np.random.randn(74)
    action = np.random.randint(0, 4)
    reward = np.random.randn()
    next_state = np.random.randn(74)
    done = (i == 9)  # Last one is done
    
    buffer.push(state, action, reward, next_state, done)
    print(f"Step {i+1}: Buffer size = {len(buffer.buffer)}, n_step_buffer = {len(buffer.n_step_buffer)}")

print(f"\nFinal buffer size: {len(buffer.buffer)}")
print(f"Expected: ~10 transitions (sliding window should push on each step after first 3)")

# Test sampling
if len(buffer.buffer) >= 4:
    states, actions, rewards, next_states, dones, n_steps = buffer.sample(4)
    print(f"\nSample batch:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  N-steps: {n_steps}")
    print("\n✅ Buffer working correctly!")
else:
    print(f"\n❌ Buffer too small: {len(buffer.buffer)}")

print("\n" + "="*50)
print("Testing PrioritizedReplayBuffer with n-step=3...")
per_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, n_step=3, gamma=0.99)

for i in range(10):
    state = np.random.randn(74)
    action = np.random.randint(0, 4)
    reward = np.random.randn()
    next_state = np.random.randn(74)
    done = (i == 9)
    
    per_buffer.push(state, action, reward, next_state, done)
    print(f"Step {i+1}: Buffer size = {len(per_buffer.buffer)}, n_step_buffer = {len(per_buffer.n_step_buffer)}")

print(f"\nFinal PER buffer size: {len(per_buffer.buffer)}")

if len(per_buffer.buffer) >= 4:
    states, actions, rewards, next_states, dones, n_steps, indices, weights = per_buffer.sample(4, beta=0.4)
    print(f"\nSample batch:")
    print(f"  States shape: {states.shape}")
    print(f"  N-steps: {n_steps}")
    print(f"  Weights: {weights}")
    print("\n✅ PER buffer working correctly!")
else:
    print(f"\n❌ PER buffer too small: {len(per_buffer.buffer)}")
