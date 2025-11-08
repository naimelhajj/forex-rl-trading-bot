"""Test NoisyNet reset_noise() for infinite recursion"""
import torch
import time
from agent import DuelingDQN

print("Creating DuelingDQN with NoisyNet...")
net = DuelingDQN(input_dim=74, action_dim=4, hidden_sizes=[256, 256, 128], use_noisy=True)

print("Testing reset_noise()...")
start = time.time()
for i in range(100):
    net.reset_noise()
    if i == 0:
        print(f"First reset_noise(): {time.time()-start:.3f}s")
elapsed = time.time() - start

print(f"100 calls to reset_noise(): {elapsed:.3f}s")
print(f"Average per call: {elapsed/100*1000:.2f}ms")

if elapsed < 1.0:
    print("\n✅ reset_noise() is FAST!")
else:
    print(f"\n⚠️  reset_noise() is SLOW ({elapsed:.1f}s for 100 calls)")
