import numpy as np
import random

from agent import DQNAgent


def run_smoke():
    state_size = 30
    action_size = 4
    agent = DQNAgent(state_size=state_size, action_size=action_size, use_noisy=True, buffer_type='simple', replay_batch_size=8, grad_steps=1)
    # populate replay buffer with random transitions
    for _ in range(64):
        s = np.random.randn(state_size).astype(np.float32)
        a = random.randrange(action_size)
        r = float(np.random.randn())
        ns = np.random.randn(state_size).astype(np.float32)
        done = False
        agent.store_transition(s, a, r, ns, done)
    print('Replay buffer size:', len(agent.replay_buffer))
    # select a few actions
    for _ in range(3):
        s = np.random.randn(state_size).astype(np.float32)
        a = agent.select_action(s)
        print('sample action:', a)
    loss = agent.train_step(beta=0.4)
    print('train_step loss:', loss)


if __name__ == '__main__':
    run_smoke()
