"""
Sign-Flip Augmentation for Direction-Equivariant Learning

Prevents Q-network from learning systematic directional bias by creating
mirror samples where bullish patterns at state s map to bearish patterns
at flipped state s', with LONG ↔ SHORT actions swapped.

This teaches the network that directional signals should be equivariant:
if pattern P at state s suggests LONG, then pattern -P at state -s should
suggest SHORT with equal confidence.
"""

import numpy as np
import torch
from typing import Tuple, List


# Direction-sensitive feature indices (will flip sign)
# Based on features.py feature list:
DIRECTION_SENSITIVE_INDICES = [
    # OHLC returns (indices 0-3): open, high, low, close
    0, 1, 2, 3,
    # lr_slope (index 18): linear regression slope
    18,
    # Currency strength deltas (indices 23-34 if present): strength lags encode momentum
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
]

# Level features (will NOT flip): time, volatility, RSI, percentiles
# These are direction-neutral


def get_direction_sensitive_mask(state_size: int) -> np.ndarray:
    """
    Get boolean mask for direction-sensitive features.
    
    Args:
        state_size: Total number of features
        
    Returns:
        Boolean array where True = direction-sensitive
    """
    mask = np.zeros(state_size, dtype=bool)
    for idx in DIRECTION_SENSITIVE_INDICES:
        if idx < state_size:
            mask[idx] = True
    return mask


def flip_state(state: np.ndarray, sensitive_mask: np.ndarray = None) -> np.ndarray:
    """
    Create sign-flipped version of state.
    
    Args:
        state: Original state vector or batch [batch_size, state_size]
        sensitive_mask: Boolean mask of features to flip (optional, will auto-detect)
        
    Returns:
        Flipped state with direction-sensitive features negated
    """
    if sensitive_mask is None:
        state_size = state.shape[-1] if len(state.shape) > 1 else len(state)
        sensitive_mask = get_direction_sensitive_mask(state_size)
    
    flipped = state.copy()
    flipped[..., sensitive_mask] *= -1
    return flipped


def flip_action(action: int) -> int:
    """
    Swap LONG ↔ SHORT actions for flipped state.
    
    Action mapping:
    - 0 (HOLD) → 0 (HOLD)
    - 1 (LONG) → 2 (SHORT)
    - 2 (SHORT) → 1 (LONG)
    - 3 (MOVE_SL) → 3 (MOVE_SL)
    
    Args:
        action: Original action index
        
    Returns:
        Flipped action index
    """
    if action == 1:
        return 2
    elif action == 2:
        return 1
    else:
        return action


def flip_action_batch(actions: np.ndarray) -> np.ndarray:
    """
    Flip batch of actions.
    
    Args:
        actions: Array of action indices
        
    Returns:
        Array with LONG ↔ SHORT swapped
    """
    flipped = actions.copy()
    flipped[actions == 1] = 2
    flipped[actions == 2] = 1
    return flipped


def create_augmented_batch(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    n_steps: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sign-flipped augmented batch.
    
    Args:
        states: State batch [batch_size, state_size]
        actions: Action batch [batch_size]
        rewards: Reward batch [batch_size]
        next_states: Next state batch [batch_size, state_size]
        dones: Done flags [batch_size]
        n_steps: N-step counts [batch_size] (optional)
        
    Returns:
        Tuple of (flipped_states, flipped_actions, rewards, flipped_next_states, dones, n_steps)
        Note: Rewards and dones are unchanged (PnL is symmetric)
    """
    state_size = states.shape[1]
    sensitive_mask = get_direction_sensitive_mask(state_size)
    
    # Flip states
    flipped_states = flip_state(states, sensitive_mask)
    flipped_next_states = flip_state(next_states, sensitive_mask)
    
    # Flip actions
    flipped_actions = flip_action_batch(actions)
    
    # CRITICAL FIX: Flip rewards for directional actions
    flipped_rewards = rewards.copy()
    directional_mask = (actions == 1) | (actions == 2)
    flipped_rewards[directional_mask] = -rewards[directional_mask]
    
    if n_steps is None:
        n_steps = np.ones_like(actions)
    
    return flipped_states, flipped_actions, flipped_rewards, flipped_next_states, dones, n_steps


def compute_symmetry_loss(
    q_net: torch.nn.Module,
    states: torch.Tensor,
    flipped_states: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute symmetry loss: L_sym = MSE(Q(s', LONG), Q(s, SHORT)) + MSE(Q(s', SHORT), Q(s, LONG))
    
    This enforces that flipped states should have swapped Q-values for directional actions.
    
    Args:
        q_net: Q-network
        states: Original states [batch_size, state_size]
        flipped_states: Sign-flipped states [batch_size, state_size]
        device: Torch device
        
    Returns:
        Symmetry loss scalar
    """
    # Get Q-values for original and flipped states
    q_orig = q_net(states)  # [batch_size, 4]
    q_flip = q_net(flipped_states)  # [batch_size, 4]
    
    # Extract Q-values for LONG (1) and SHORT (2)
    q_orig_long = q_orig[:, 1]  # Q(s, LONG)
    q_orig_short = q_orig[:, 2]  # Q(s, SHORT)
    q_flip_long = q_flip[:, 1]  # Q(s', LONG)
    q_flip_short = q_flip[:, 2]  # Q(s', SHORT)
    
    # Symmetry constraint: Q(s', LONG) should equal Q(s, SHORT) and vice versa
    loss_long_short = torch.nn.functional.mse_loss(q_flip_long, q_orig_short)
    loss_short_long = torch.nn.functional.mse_loss(q_flip_short, q_orig_long)
    
    return loss_long_short + loss_short_long


# Configuration
SYMMETRY_LOSS_WEIGHT = 0.5  # Increased from 0.2 - bias persists, need stronger symmetry enforcement

