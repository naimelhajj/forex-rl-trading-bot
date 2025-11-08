import sys
import os
# Ensure repo root is on sys.path so tests can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Lightweight unit tests for NoisyLinear and PrioritizedReplayBuffer
import numpy as np
import torch
from agent import NoisyLinear, PrioritizedReplayBuffer


def test_noisy_linear():
    torch.manual_seed(0)
    n_in, n_out = 8, 4
    nl = NoisyLinear(n_in, n_out, sigma_init=0.02)

    # Evaluate forward shapes in train/eval modes
    nl.train()
    x = torch.randn(3, n_in)
    out1 = nl(x)
    nl.reset_noise()
    out2 = nl(x)
    assert out1.shape == (3, n_out)
    assert out2.shape == (3, n_out)
    # In training mode outputs should vary after noise reset
    if torch.allclose(out1, out2):
        raise AssertionError('NoisyLinear outputs identical after reset in train mode')

    nl.eval()
    out_eval1 = nl(x)
    nl.reset_noise()
    out_eval2 = nl(x)
    # In eval mode outputs should be deterministic (noise not applied)
    assert torch.allclose(out_eval1, out_eval2), 'NoisyLinear eval outputs should be deterministic'

    print('NoisyLinear tests passed')


def test_prioritized_replay():
    np.random.seed(1)
    buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

    # push simple transitions
    for i in range(50):
        s = np.random.randn(5)
        a = np.random.randint(0, 4)
        r = float(np.random.randn())
        ns = np.random.randn(5)
        done = False
        buf.push(s, a, r, ns, done)

    # sample
    batch_size = 16
    states, actions, rewards, next_states, dones, indices, weights = buf.sample(batch_size, beta=0.4)
    assert states.shape[0] == batch_size
    assert len(indices) == batch_size
    assert weights.shape[0] == batch_size

    # update priorities and verify they change
    old_prios = buf.priorities[indices].copy()
    new_prios = np.abs(np.random.randn(batch_size)) + 1e-6
    buf.update_priorities(indices, new_prios)
    assert not np.allclose(buf.priorities[indices], old_prios), 'Priorities did not update'

    print('PrioritizedReplayBuffer tests passed')


if __name__ == '__main__':
    test_noisy_linear()
    test_prioritized_replay()
    print('All unit tests passed')
