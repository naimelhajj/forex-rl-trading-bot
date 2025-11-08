"""
DQN Agent Module
Implements Deep Q-Network with experience replay and target network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Tuple, Optional
import math


class NoisyLinear(nn.Module):
    """Factorized NoisyNet linear layer (Fortunato et al.)."""
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _f(self, x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        # sample factorized noise and compute outer product for weight epsilon
        device = self.weight_mu.device
        eps_in = torch.randn(self.in_features, device=device)
        eps_out = torch.randn(self.out_features, device=device)
        eps_in = self._f(eps_in)
        eps_out = self._f(eps_out)
        # use torch.outer for clarity (fallback to ger for older torch versions)
        try:
            outer = torch.outer(eps_out, eps_in)
        except Exception:
            outer = eps_out.ger(eps_in)
        # copy into registered buffers
        self.weight_epsilon.copy_(outer)
        self.bias_epsilon.copy_(eps_out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# Dueling DQN head example (small and extensible)
class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes=(256,256), use_noisy: bool = False, sigma_init: float = 0.017):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self._use_noisy = use_noisy
        # shared body
        layers = []
        last = input_dim
        for h in hidden_sizes:
            lin = NoisyLinear(last, h, sigma_init=sigma_init) if use_noisy else nn.Linear(last, h)
            layers.append(lin)
            layers.append(nn.ReLU())
            last = h
        self.feature = nn.Sequential(*layers)
        # value and advantage streams
        if use_noisy:
            self.value_stream = nn.Sequential(NoisyLinear(last, 128, sigma_init=sigma_init), nn.ReLU(), NoisyLinear(128, 1, sigma_init=sigma_init))
            self.adv_stream = nn.Sequential(NoisyLinear(last, 128, sigma_init=sigma_init), nn.ReLU(), NoisyLinear(128, action_dim, sigma_init=sigma_init))
        else:
            self.value_stream = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128,1))
            self.adv_stream   = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        # Combine streams: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        """Reset noise for all NoisyLinear modules in the network."""
        # Explicitly iterate through known layer containers only (not self)
        for layer_name in ['feature', 'value_stream', 'adv_stream']:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                # Iterate through Sequential container
                if isinstance(layer, nn.Sequential):
                    for module in layer:
                        if hasattr(module, 'reset_noise'):
                            module.reset_noise()


class ReplayBuffer:
    """
    Experience replay buffer with 3-step returns for faster credit assignment.
    PATCH 2: N-step TD to propagate reward signals faster with SL/TP delays.
    """
    
    def __init__(self, capacity: int = 100000, n_step: int = 3, gamma: float = 0.99):
        """
        Initialize replay buffer with n-step returns.
        
        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step returns (default 3)
            gamma: Discount factor for computing n-step returns
        """
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)  # Temporary buffer for n-step accumulation
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer with n-step return computation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Push to main buffer when we have enough steps OR episode ends
        # Also push single-step transitions to avoid buffer starvation
        if len(self.n_step_buffer) >= self.n_step or done:
            # Compute n-step return from the OLDEST transition in buffer
            n_step_reward = 0.0
            n_step_state = self.n_step_buffer[0][0]  # s_t
            n_step_action = self.n_step_buffer[0][1]  # a_t
            
            # Sum discounted rewards: R = r_t + γ*r_{t+1} + γ²*r_{t+2}
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break
            
            # Get final state s_{t+n} and done flag
            final_next_state = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]
            actual_n = len(self.n_step_buffer)
            
            # Store (s_t, a_t, R^n, s_{t+n}, done_n, n)
            self.buffer.append((n_step_state, n_step_action, n_step_reward, 
                              final_next_state, final_done, actual_n))
            
            # Remove the oldest transition (sliding window)
            if not done:
                self.n_step_buffer.popleft()
        
        # Clear n-step buffer if episode done
        if done:
            self.n_step_buffer.clear()
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of n-step transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, n_step_rewards, next_states, dones, n_steps)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])  # n-step rewards
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        n_steps = np.array([t[5] for t in batch])  # actual n for each transition
        
        return states, actions, rewards, next_states, dones, n_steps
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer with 3-step returns.
    PATCH 2: N-step TD for faster credit assignment.
    """
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, n_step: int = 3, gamma: float = 0.99):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        """Add transition with n-step return computation."""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Push when we have enough steps OR episode ends
        if len(self.n_step_buffer) >= self.n_step or done:
            # Compute n-step return from OLDEST transition
            n_step_reward = 0.0
            n_step_state = self.n_step_buffer[0][0]
            n_step_action = self.n_step_buffer[0][1]
            
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break
            
            final_next_state = self.n_step_buffer[-1][3]
            final_done = self.n_step_buffer[-1][4]
            actual_n = len(self.n_step_buffer)
            
            # Store with n-step info
            max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
            transition = (n_step_state, n_step_action, n_step_reward, 
                         final_next_state, final_done, actual_n)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity
            
            # Remove oldest transition (sliding window)
            if not done:
                self.n_step_buffer.popleft()
        
        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with n-step transitions."""
        if len(self.buffer) == 0:
            raise IndexError('Sampling from empty buffer')
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs = probs / (probs.sum() + 1e-12)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        n_steps = np.array([s[5] for s in samples])
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-12)
        return states, actions, rewards, next_states, dones, n_steps, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, pr in zip(indices, priorities):
            self.priorities[idx] = pr

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with epsilon-greedy exploration and target network.
    
    Actions:
        0: HOLD
        1: LONG
        2: SHORT
        3: MOVE_SL_CLOSER
    """
    
    def __init__(self, state_size: int, action_size: int, device: Optional[torch.device]=None, **kwargs):
        """Initialize DQNAgent. q_network_factory should be a callable returning a network instance."""
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # hyperparams with sensible defaults (can be overridden)
        self.gamma = kwargs.get('gamma', 0.99)
        self.lr = kwargs.get('lr', 1e-4)
        self.replay_batch_size = kwargs.get('replay_batch_size', 256)
        self.learning_starts = kwargs.get('learning_starts', 5000)  # gate backprop until buffer has enough data
        self.update_every = kwargs.get('update_every', 4)
        self.grad_steps = kwargs.get('grad_steps', 4)
        self.grad_clip = kwargs.get('grad_clip', 5.0)
        self.polyak_tau = kwargs.get('polyak_tau', 0.005)
        self.use_double = kwargs.get('use_double', True)

        # create networks (optionally use NoisyNet for exploration)
        use_noisy = kwargs.get('use_noisy', False)
        sigma_init = kwargs.get('noisy_sigma_init', 0.017)
        self.policy_net = DuelingDQN(self.state_size, self.action_size, use_noisy=use_noisy, sigma_init=sigma_init).to(self.device)
        self.target_net = DuelingDQN(self.state_size, self.action_size, use_noisy=use_noisy, sigma_init=sigma_init).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # PHASE-2: EMA model for stable evaluation
        self.use_param_ema = kwargs.get('use_param_ema', False)
        self.ema_decay = kwargs.get('ema_decay', 0.999)
        if self.use_param_ema:
            self.ema_net = DuelingDQN(self.state_size, self.action_size, use_noisy=use_noisy, sigma_init=sigma_init).to(self.device)
            self.ema_net.load_state_dict(self.policy_net.state_dict())
            self.ema_net.eval()
        else:
            self.ema_net = None

        # aliases expected elsewhere
        self.q_net = self.policy_net
        self.target_q = self.target_net

        # Optimizer with optional weight decay
        weight_decay = kwargs.get('weight_decay', 1e-6)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, weight_decay=weight_decay)

        # Replay buffer with n-step returns (PATCH 2)
        buffer_type = kwargs.get('buffer_type', 'simple')
        buffer_capacity = kwargs.get('buffer_capacity', 100000)
        n_step = kwargs.get('n_step', 3)  # PATCH 2: 3-step returns
        
        if buffer_type == 'prioritized':
            alpha = kwargs.get('prioritized_replay_alpha', 0.6)
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=alpha, n_step=n_step, gamma=self.gamma)
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_capacity, n_step=n_step, gamma=self.gamma)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Initialize other parameters
        self.epsilon = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        # If using NoisyNet, prefer noisy exploration and disable epsilon-greedy
        self.use_noisy = use_noisy
        if self.use_noisy:
            # keep epsilon attributes but neutralize greedy ε-exploration
            self.epsilon = 0.0
        self.batch_size = kwargs.get('batch_size', 64)
        self.target_update_freq = kwargs.get('target_update_freq', 1000)
        
        # internal step counter
        self._train_step = 0

    @property
    def replay_size(self):
        """Current number of transitions in replay buffer."""
        return len(self.replay_buffer)

    def select_action(self, state: np.ndarray, explore: bool = True, mask: list = None, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy or NoisyNet exploration.
        PATCH #1: Added eval_mode parameter for deterministic evaluation (eps=0, freeze NoisyNet).
        PHASE-2: Use EMA model for evaluation if available.
        
        Args:
            state: Current state
            explore: Whether to use exploration (epsilon-greedy or noisy net)
            mask: Optional boolean mask [HOLD, LONG, SHORT, MOVE_SL_CLOSER]
            eval_mode: If True, use deterministic policy (epsilon=0, freeze noise, use EMA model)
            
        Returns:
            Selected action index
        """
        import random
        # infer action_size if missing
        if not hasattr(self, 'action_size'):
            try:
                self.action_size = self.q_net(torch.zeros(1, self.state_size, device=self.device)).shape[1]
            except Exception:
                self.action_size = 4
        
        # PHASE-2: Select which network to use for action selection
        if eval_mode and self.use_param_ema and self.ema_net is not None:
            # Use EMA model for stable evaluation
            eval_net = self.ema_net
        else:
            # Use online policy network
            eval_net = self.q_net
        
        # PATCH #1: Deterministic evaluation mode
        if eval_mode:
            eps = 0.0
            if getattr(self, "use_noisy", False):
                # Freeze noise during eval - set eval_net to eval mode
                eval_net.eval()
        else:
            eps = self.epsilon if hasattr(self, 'epsilon') else 0.0
        
        # NoisyNet handles exploration internally (stochastic weights) when enabled.
        if self.use_noisy and not eval_mode:
            # reset noise before selecting action to ensure stochasticity
            self.reset_noise()
            # Get Q-values
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = eval_net(s).squeeze(0).cpu().numpy()
            
            # Apply mask if provided
            if mask is not None:
                for i, ok in enumerate(mask):
                    if not ok:
                        q[i] = -1e9
            
            return int(np.argmax(q))
        else:
            # Epsilon-greedy exploration
            if explore and not eval_mode and hasattr(self, 'epsilon') and random.random() < eps:
                # Random exploration but respect mask
                if mask is not None:
                    valid_actions = [i for i, ok in enumerate(mask) if ok]
                    if valid_actions:
                        return random.choice(valid_actions)
                return random.randrange(self.action_size)
            
            # Greedy action selection
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = eval_net(s).squeeze(0).cpu().numpy()
            
            # Apply mask if provided
            if mask is not None:
                for i, ok in enumerate(mask):
                    if not ok:
                        q[i] = -1e9
            
            return int(np.argmax(q))

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, beta: float = 0.4):
        """Perform a training update using Double DQN with Huber loss and Polyak updates.
        PATCH 2: Supports n-step returns for faster credit assignment.
        Accepts `beta` for importance-sampling when using PER. Returns loss value (float) when updated, otherwise None.
        """
        if not hasattr(self, 'replay_buffer'):
            return None
        buffer_len = len(self.replay_buffer)
        if buffer_len < self.replay_batch_size:
            return None

        total_loss = 0.0
        updates = 0
        # if using noisy nets, reset noise before training updates
        if getattr(self, 'use_noisy', False):
            self.reset_noise()
        for _ in range(self.grad_steps):
            # Sample batch depending on buffer type
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                states, actions, rewards, next_states, dones, n_steps, indices, weights = self.replay_buffer.sample(self.replay_batch_size, beta=beta)
                is_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            else:
                states, actions, rewards, next_states, dones, n_steps = self.replay_buffer.sample(self.replay_batch_size)
                indices = None
                is_weights = torch.ones(self.replay_batch_size, dtype=torch.float32, device=self.device)

            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # n-step rewards
            next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
            n_steps_tensor = torch.tensor(n_steps, dtype=torch.float32, device=self.device)

            # current Q
            q_vals = self.q_net(states).gather(1, actions).squeeze(1)

            with torch.no_grad():
                if self.use_double:
                    next_q_online = self.q_net(next_states)
                    next_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = self.target_q(next_states).gather(1, next_actions).squeeze(1)
                else:
                    next_q_target = self.target_q(next_states).max(dim=1)[0]
                
                # PATCH 2: n-step target with gamma^n discount
                # target = R^n + (gamma^n) * Q_target(s_{t+n}, ...) * (1 - done)
                gamma_n = torch.pow(self.gamma, n_steps_tensor)
                target = rewards + (1.0 - dones) * gamma_n * next_q_target

            # Huber loss (smooth_l1) with IS weights if PER
            td_errors = (q_vals - target).detach()
            loss_per_sample = F.smooth_l1_loss(q_vals, target, reduction='none')
            loss = (loss_per_sample * is_weights).mean()

            self.optimizer.zero_grad()
            loss.backward()
            
            # PATCH 10: Compute and store grad norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
            if not hasattr(self, '_last_grad_norm'):
                self._last_grad_norm = 0.0
            self._last_grad_norm = float(grad_norm)
            
            self.optimizer.step()

            # PHASE-2: Update EMA model after optimizer step
            if self.use_param_ema and self.ema_net is not None:
                with torch.no_grad():
                    for p_ema, p_online in zip(self.ema_net.parameters(), self.policy_net.parameters()):
                        p_ema.mul_(self.ema_decay).add_(p_online, alpha=1.0 - self.ema_decay)

            # soft update target
            for p, tp in zip(self.q_net.parameters(), self.target_q.parameters()):
                tp.data.copy_(self.polyak_tau * p.data + (1.0 - self.polyak_tau) * tp.data)

            total_loss += float(loss.detach().cpu().numpy())
            updates += 1
            self._train_step += 1

            # Update priorities in PER
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and indices is not None:
                new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
                self.replay_buffer.update_priorities(indices, new_priorities)

        if updates == 0:
            return None
        return total_loss / updates
    
    def reset_noise(self):
        """Reset noise on policy and target networks if they support it."""
        try:
            if hasattr(self.policy_net, 'reset_noise'):
                self.policy_net.reset_noise()
        except Exception:
            pass
        try:
            if hasattr(self.target_net, 'reset_noise'):
                self.target_net.reset_noise()
        except Exception:
            pass

    def get_q_values(self, state):
        """Return Q-values for a single state as numpy array."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(s).cpu().numpy().squeeze(0)
            return q

    def save(self, path: str):
        state = {
            'policy_net_state_dict': getattr(self, 'policy_net', self.q_net).state_dict(),
            'target_net_state_dict': getattr(self, 'target_net', self.target_q).state_dict(),
            'optimizer_state_dict': getattr(self, 'optimizer', None).state_dict() if hasattr(self, 'optimizer') else None,
            'epsilon': getattr(self, 'epsilon', None)
        }
        # PHASE-2: Save EMA model if using parameter EMA
        if self.use_param_ema and self.ema_net is not None:
            state['ema_net_state_dict'] = self.ema_net.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        if 'policy_net_state_dict' in data and hasattr(self, 'policy_net'):
            self.policy_net.load_state_dict(data['policy_net_state_dict'])
        if 'target_net_state_dict' in data and hasattr(self, 'target_net'):
            self.target_net.load_state_dict(data['target_net_state_dict'])
        if 'optimizer_state_dict' in data and hasattr(self, 'optimizer') and data['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(data['optimizer_state_dict'])
        if 'epsilon' in data and data['epsilon'] is not None:
            self.epsilon = data['epsilon']
        # PHASE-2: Load EMA model if available
        if 'ema_net_state_dict' in data and self.use_param_ema and self.ema_net is not None:
            self.ema_net.load_state_dict(data['ema_net_state_dict'])


class ActionSpace:
    """
    Action space definition for the trading agent.
    """
    
    HOLD = 0
    LONG = 1
    SHORT = 2
    MOVE_SL_CLOSER = 3
    
    @staticmethod
    def get_action_name(action: int) -> str:
        """Get human-readable action name."""
        names = {
            0: "HOLD",
            1: "LONG",
            2: "SHORT",
            3: "MOVE_SL_CLOSER"
        }
        return names.get(action, "UNKNOWN")
    
    @staticmethod
    def get_action_size() -> int:
        """Get number of actions."""
        return 4


if __name__ == "__main__":
    print("DQN Agent Module")
    print("=" * 50)
    
    # Test agent initialization
    state_size = 30
    action_size = 4
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    print(f"\nAgent initialized:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Device: {agent.device}")
    
    # Test action selection
    state = np.random.randn(state_size)
    action = agent.select_action(state)
    print(f"\nSample action: {action} ({ActionSpace.get_action_name(action)})")
    
    # Test Q-values
    q_values = agent.get_q_values(state)
    print(f"\nQ-values: {q_values}")
    
    # Test storing transitions
    for i in range(100):
        state = np.random.randn(state_size)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_size)
        done = False
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"\nReplay buffer size: {len(agent.replay_buffer)}")
    
    # Test training
    loss = agent.train_step()
    print(f"Training loss: {loss}")

