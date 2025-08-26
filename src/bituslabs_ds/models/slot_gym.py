import os
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gymnasium import spaces
from imitation.rewards.reward_nets import BasicRewardNet
from tensorboard.plugins.hparams.summary_v2 import Discrete

from .rewardnets import Termination_Reward


# --------------------------------------
# Custom Gym Environment (For archive purpose)
class SlotBetEnv(gym.Env):
    """
    A custom Gym environment for simulating player behavior in slot machine sessions.

    The environment receives a list of observed betting steps (`obs`) and simulates
    agent actions over the sequence. The agent can choose a bet amount and a delay time (delta_t).

    Observations include information about the session, such as profit, streaks,
    line results, and more. The agent must learn to match real behavior or stop correctly.
    """

    def __init__(self, bet_dict, currency, obs=None, seed=None):
        super().__init__()
        assert isinstance(obs, list) and all(isinstance(o, dict) for o in obs), "obs must be a list of dicts"

        self.observations = []
        self.trajectory = []
        self.bets = bet_dict  # the dictonary
        self.currency = currency  # the original currency in string
        self.pos = 0
        self.done = False
        self.max_len = 1
        self.seed(seed)
        # self.raylib = ray # this control whether we need to use the action wrapper cuz raylib handles tuple action well.

        # Define observation space (REMOVED delta_bet)
        self.observation_space = spaces.Dict(
            {
                "bet": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
                "profit": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
                #'currency': spaces.Box(low=0, high=1, shape=(13,), dtype=np.int32),  # One-hot encoded
                "slottype": spaces.Discrete(2),
                "basepoint": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "delta_t": spaces.Discrete(1),
                # REMOVED: 'delta_bet': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
                "delta_profit": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
                "delta_payout": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
                "streak": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
                "result": spaces.Box(low=0, high=np.inf, shape=(15,), dtype=np.float32),
                "line_win": spaces.Box(low=0, high=np.inf, shape=(25,), dtype=np.int32),  # 25 paylines
                "prev_bet": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
                "prev_basepoint": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "prev_profit": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
                "total_profit": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
            }
        )

        # Define combined action space for bet and delta_t
        self.action_space = spaces.Tuple(
            (spaces.Discrete(max([len(bet_array) for bet_array in self.bets.values()])), spaces.Discrete(1))
        )

        self.set_start(obs=obs)

    def set_start(self, obs):
        """Initialize environment with a list of observations."""
        assert isinstance(obs, list) and all(isinstance(o, dict) for o in obs), "obs must be a list of dicts"
        self.observations = obs
        self.trajectory = [self.observations[0]]
        self.max_len = len(obs)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def preprocess_observation(self, obs_dict):
        """Flatten and one-hot encode observation for model consumption. REMOVED delta_bet."""

        features = []

        # Add bet index
        features.append(obs_dict["bet"].reshape(-1))

        # Add profit
        features.append(obs_dict["profit"].reshape(-1))
        features.append(obs_dict["slottype"].reshape(-1))

        # Add remaining features (REMOVED delta_bet)
        features.append(obs_dict["basepoint"].reshape(-1))
        # features.append(obs_dict['delta_t'].reshape(-1))
        # REMOVED: features.append(obs_dict['delta_bet'].reshape(-1))
        features.append(obs_dict["delta_profit"].reshape(-1))
        features.append(obs_dict["delta_payout"].reshape(-1))
        features.append(obs_dict["streak"].reshape(-1))
        # features.append(obs_dict['result'].reshape(-1))
        # features.append(obs_dict['line_win'])
        features.append(obs_dict["prev_bet"].reshape(-1))
        features.append(obs_dict["prev_basepoint"].reshape(-1))
        features.append(obs_dict["prev_profit"].reshape(-1))
        features.append(obs_dict["total_profit"].reshape(-1))

        return np.concatenate([f.flatten() for f in features])

    def reset(self, seed=None, options=None):
        """Reset the environment to the beginning of a new session."""
        super().reset(seed=seed)
        self.pos = self.pos + 1 if self.pos + 1 < len(self.observations) else 0
        self.max_len = len(self.observations)
        self.done = False
        curr_state = self.observations[self.pos]
        self.trajectory = [curr_state]
        print(f"DEBUG - Start a new episode in position {self.pos}")
        return curr_state, {}

    # ACTION to Trajectory
    def action_2_trajectory(self, action, pos):
        """Create the NEXT observation based on the agent's action. REMOVED delta_bet."""
        bet = action[0]
        delta_t = 0
        next_obs = self.observations[pos + 1]
        curr_obs = self.observations[pos]

        # need raw bet amount for this line
        next_bet_id = next_obs["bet"]
        next_raw_bet = get_bet_value(next_bet_id, currency=self.currency[self.pos], bet_dictionary=self.bets)
        current_raw_bet = get_bet_value(bet, currency=self.currency[self.pos], bet_dictionary=self.bets)

        profit = next_obs["profit"] / next_raw_bet * current_raw_bet

        streak = curr_obs["streak"][0]  # Extract scalar from numpy array
        if bet > 0:
            streak += 1

        return {
            "bet": np.array([bet], dtype=np.int32),
            "profit": np.array([profit], dtype=np.int32),
            #'currency': curr_obs['currency'],
            "slottype": next_obs["slottype"],
            "basepoint": np.array([curr_obs["basepoint"][0] + profit], dtype=np.float32),
            "delta_t": np.array([delta_t], dtype=np.int32),
            # REMOVED: 'delta_bet': np.array([bet - curr_obs['bet'][0]], dtype=np.int32),
            "delta_profit": np.array([profit - curr_obs["profit"][0]], dtype=np.float32),
            "delta_payout": np.array([tot_win - curr_obs["profit"][0] - curr_obs["bet"][0]], dtype=np.int32),
            "streak": np.array([streak], dtype=np.int32),
            #'result': next_obs['result'],
            #'line_win': np.array(line_wins, dtype=np.int32),
            "prev_bet": curr_obs["bet"],
            "prev_basepoint": curr_obs["basepoint"],
            "prev_profit": curr_obs["profit"],
            "total_profit": np.array([curr_obs["total_profit"][0] + profit], dtype=np.int32),
        }, next_bet_id

    def step(self, action):
        """Execute an agent action and transition to the next state."""

        correct_length = len(self.observations)
        bet_idx, delta_t_value = action

        early_terminate = False
        late_terminate = False

        terminate = 0

        if bet_idx == 0:
            terminate = 1

        if terminate == 1:
            delta_t_value = 0
            if self.pos < correct_length:
                # Terminating too early
                early_terminate = True
                late_terminate = False
            elif self.pos > correct_length:
                # Terminating too late
                early_terminate = False
                late_terminate = True
            else:
                # Terminating at correct position
                early_terminate = False
                late_terminate = False

        elif self.pos >= correct_length:
            early_terminate = False
            late_terminate = True

        bet_amount = get_bet_value(bet_idx, currency=self.currency[self.pos], bet_dictionary=self.bets)
        reward = 0.0

        if self.pos >= len(self.observations):
            raise IndexError(f"Position {self.pos} exceeds available observations.")

        current_obs = self.observations[self.pos]
        self.pos += 1

        self.done = (terminate == 1) or (self.pos >= self.max_len)

        if not self.done:
            next_trajectory, expert_bet = self.action_2_trajectory(action, self.pos - 1)
            # print(f"DEBUG - Next_trajectory: {action} - {next_trajectory}")
            self.trajectory.append(next_trajectory)
            next_obs = next_trajectory
        else:
            next_obs = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0 for k, v in current_obs.items()}
            next_obs["slottype"] = 0
            expert_bet = 0

        # Calculate length difference when episode ends
        length_difference = 0
        if self.done:
            episode_length = self.pos
            length_difference = abs(episode_length - correct_length)

        info = {
            "ground_bet": current_obs.get("bet", 0),  # Use get() for safety since delta_bet removed
            "pred_bet": bet_amount,
            "ground_delta_t": current_obs["delta_t"],
            "pred_delta_t": delta_t_value,
            "is_terminal": self.done,
            "early_terminate": early_terminate,
            "late_terminate": late_terminate,
            "correct_length": correct_length,
            "episode_length": self.pos if self.done else None,
            "length_difference": length_difference if self.done else None,
            "expert_action": expert_bet,
        }

        print(f"DEBUG - step!! at {self.pos} and info: {info}, and traj: {next_obs}")

        return next_obs, reward, self.done, False, info


# -------------------------------------
# Tuple Actions Handler because baseline3 can only handle box action
class TupleActionWrapper(gym.ActionWrapper):
    def __init__(self, env, num_discrete_actions=16, num_delta_t_categories=1):
        super().__init__(env)
        self.num_discrete_actions = num_discrete_actions
        self.num_delta_t_categories = num_delta_t_categories

        # note that bet_idx = 0 = zerobet = terminate
        self.total_actions = (num_discrete_actions - 1) * num_delta_t_categories + 1

        # Use discrete action space
        self.action_space = gym.spaces.Discrete(self.total_actions)

    def action(self, action):
        """Convert discrete action to (bet_idx, delta_t_category) tuple"""
        # Handle None env case
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action[0]

        # Ensure action is an integer
        action = int(action)

        # Special case: action 0 represents termination
        if action == 0:
            return (0, np.array([0]))  # Termination with delta_t=2 (>50 seconds)

        # Regular actions (1 to total_actions)
        # Subtract 1 from action to account for the termination action
        adjusted_action = action - 1

        # Extract bet_idx (add 1 because bet_idx=0 is special termination case)
        bet_idx = (adjusted_action // self.num_delta_t_categories) + 1
        delta_t_category = adjusted_action % self.num_delta_t_categories

        return (bet_idx, np.array([delta_t_category]))

    def reverse_action(self, action_tuple):
        """Convert (bet_idx, delta_t_category) tuple to discrete action"""
        bet_idx, delta_t_arr = action_tuple

        # Special case: bet_idx=0 means termination regardless of delta_t
        if bet_idx == 0:
            return 0  # Always return 0 for any (0, [x]) tuple

        # Get delta_t category
        delta_t_category = max(int(delta_t_arr[0]), self.num_delta_t_categories - 1)

        # Calculate flat action index
        # Subtract 1 from bet_idx because action=0 is reserved
        # Add 1 to final result to account for termination action
        action = ((bet_idx - 1) * self.num_delta_t_categories + delta_t_category) + 1

        return int(action)


# Wrap the observation too cause MLPPolicy is such a pain in the ass


class DictToFlatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, normalize=True, normalizers=None, clip_range=(-3, 3)):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "Expected Dict observation space"
        self.original_space = env.observation_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._get_flattened_dim(),), dtype=np.float32
        )

        self.normalize = normalize
        self.normalizers = normalizers
        self.clip_range = clip_range

    def _get_flattened_dim(self):
        """Calculate the dimension of the flattened observation space"""

        total_dim = 0
        for space_name, space in self.original_space.spaces.items():
            if isinstance(space, spaces.Box):
                total_dim += np.prod(space.shape)
            elif isinstance(space, spaces.Discrete):
                total_dim += 1
            # Add other space types as needed
        return 28 - 15 - 1 - 1  # Updated: removed delta_bet (was 28-15-1, now 27-15-1)

    def observation(self, observation):
        """Convert Dict observation to flattened vector with normalization and clipping. REMOVED delta_bet."""
        features = []

        bet = np.array(observation["bet"]).reshape(-1)
        features.append(np.array(observation["bet"]).reshape(-1))

        # Add profit with normalization
        features.append(self._normalize_and_clip(np.array(observation["profit"]).reshape(-1), "adjusted_profit"))

        # Add currency one-hot encoding - ensure it's a numpy array
        # features.append(np.array(observation['currency']))

        features.append(np.array([observation["slottype"]]))

        # Add remaining numeric features with normalization (REMOVED delta_bet)
        features.append(self._normalize_and_clip(np.array(observation["basepoint"]).reshape(-1), "basepoint"))
        # features.append(np.array(observation['delta_t'] / 3))
        # REMOVED: features.append(self._normalize_and_clip(np.array(observation['delta_bet']).reshape(-1), 'delta_bet'))
        features.append(self._normalize_and_clip(np.array(observation["delta_profit"]).reshape(-1), "delta_profit"))
        features.append(self._normalize_and_clip(np.array(observation["delta_payout"]).reshape(-1), "delta_payout"))
        features.append(self._normalize_and_clip(np.array(observation["streak"]).reshape(-1), "streak"))

        # Add array features - ensure they're numpy arrays
        # features.append(self._normalize_and_clip_array(np.array(observation['result']), 'result'))
        # features.append(self._normalize_and_clip_array(np.array(observation['line_win']), 'line_win'))

        # Add remaining features
        features.append(self._normalize_and_clip(np.array(observation["prev_bet"]).reshape(-1), "prev_bet"))
        features.append(self._normalize_and_clip(np.array(observation["prev_basepoint"]).reshape(-1), "prev_basepoint"))
        features.append(self._normalize_and_clip(np.array(observation["prev_profit"]).reshape(-1), "prev_profit"))
        features.append(self._normalize_and_clip(np.array(observation["total_profit"]).reshape(-1), "total_profit"))

        # Process each feature before concatenation
        processed_features = []
        for f in features:
            # Make sure each feature is a numpy array
            if not isinstance(f, np.ndarray):
                f = np.array(f)
            # Now flatten it
            processed_features.append(f.flatten())

        result = np.concatenate(processed_features).astype(np.float32)

        # print(f"DEBUG: {result.shape}")

        # If bet is 0, return an all-zero observation vector of the same size
        if bet[0] == 0:
            return np.zeros_like(result)

        # Concatenate all features into a single flat array

        return result

    def _normalize_and_clip(self, value, feature_name):
        """Normalize and clip a scalar or array value"""
        if self.normalize and self.normalizers and feature_name in self.normalizers:
            mean, std = self.normalizers[feature_name]
            normalized_value = (value - mean) / std
            return np.clip(normalized_value, self.clip_range[0], self.clip_range[1])
        return value

    def _normalize_and_clip_array(self, array, feature_name):
        """Normalize and clip each element in an array"""
        if self.normalize and self.normalizers and feature_name in self.normalizers:
            mean, std = self.normalizers[feature_name]
            normalized_array = [(x - mean) / std for x in array]
            clipped_array = [np.clip(x, self.clip_range[0], self.clip_range[1]) for x in normalized_array]
            return np.array(clipped_array, dtype=np.float32)
        return np.array(array, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset with normalization applied to the initial observation"""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info


def find_bet_index(bet_amount: float, currency: str, bet_dictionary: dict) -> int:
    """
    Find the index of the closest bet value for a given currency in the bet_dictionary.

    This function determines which denomination index in a currency's available denominations
    is closest to the requested bet amount.

    Parameters:
    -----------
    bet_amount : float
        The target bet amount to find the closest denomination for
    currency : str
        The currency code (e.g., 'USD', 'EUR', 'CNY')
    bet_dictionary : dict
        Dictionary with currencies as keys and lists of available bet values as values

    Returns:
    --------
    index : int
        The index of the closest bet value in the currency's list of denominations

    Raises:
    -------
    ValueError
        If the specified currency is not found in the bet_dictionary

    Examples:
    ---------
    >>> bet_dict = {'USD': [0.0, 0.05, 0.1, 0.25, 0.5]}
    >>> find_bet_index(0.22, 'USD', bet_dict)
    3  # Index of 0.25, which is closest to 0.22
    """
    if currency not in bet_dictionary:
        raise ValueError(f"Currency '{currency}' not found in the bet dictionary")

    # Get the bet values array for the specified currency
    bet_values_array = np.array(bet_dictionary[currency])

    # Find the index of the closest value
    index = np.argmin(np.abs(bet_values_array - bet_amount))

    return index


def get_bet_value(bet_index: int, currency: str, bet_dictionary: dict) -> float:
    """
    Convert a bet index to the corresponding bet value for a specific currency.

    This function maps an action index (potentially from an ML model output) to an
    actual bet value, handling out-of-bounds indices by clamping to the maximum
    available denomination.

    Parameters:
    -----------
    bet_index : int
        The index of the bet value to retrieve
    currency : str
        The currency code (e.g., 'USD', 'EUR', 'CNY')
    bet_dictionary : dict
        Dictionary with currencies as keys and lists of available bet values as values

    Returns:
    --------
    bet : float
        The bet value corresponding to the index for the specified currency.
        If the index exceeds the available denominations, returns the maximum
        available denomination for that currency.

    Examples:
    ---------
    >>> bet_dict = {'USD': [0.0, 0.05, 0.1, 0.25, 0.5]}
    >>> get_bet_value(2, 'USD', bet_dict)
    0.1  # Value at index 2
    >>> get_bet_value(10, 'USD', bet_dict)
    0.5  # Max value since index 10 is out of bounds
    """
    if isinstance(bet_index, (np.ndarray, list)):
        bet_index = int(bet_index[0])
    elif not isinstance(bet_index, int):
        # Handle any other type by converting to int
        bet_index = int(bet_index)

    currency_dictionary = bet_dictionary[currency]
    if bet_index >= len(currency_dictionary):
        bet = max(currency_dictionary)
    else:
        # print(f"DEBUG - bet_index:  {bet_index}")
        bet = currency_dictionary[bet_index]

    return bet


def trajectory_to_obs_dict(obs_array, normalizers=None):
    """
    Maps a flattened trajectory observation back to observation dictionary format.
    Modified to exclude delta_bet.

    Args:
        obs_array: Flattened observation array from the trajectory
        normalizers: Dictionary containing normalization parameters (mean, std) for each feature

    Returns:
        Dictionary in the original observation format
    """

    # Denormalize if normalizers provided
    def denorm(value, feature):
        if normalizers and feature in normalizers:
            mean, std = normalizers[feature]
            return (value * std) + mean
        return value

    # Extract features in order: bet_idx, adjusted_profit, slottype, basepoint,
    # delta_profit, delta_payout, streak, prev_bet, prev_basepoint, prev_profit, total_profit
    # NOTE: delta_bet has been REMOVED from this list

    return {
        "bet": np.array([int(obs_array[0])], dtype=np.int32),
        "profit": np.array([int(denorm(obs_array[1], "adjusted_profit"))], dtype=np.int32),
        "slottype": int(obs_array[2]),
        "basepoint": np.array([denorm(obs_array[3], "basepoint")], dtype=np.float32),
        "delta_t": np.array([0.0], dtype=np.float32),
        # REMOVED: 'delta_bet' field
        "delta_profit": np.array([int(denorm(obs_array[4], "delta_profit"))], dtype=np.int32),
        "delta_payout": np.array([int(denorm(obs_array[5], "delta_payout"))], dtype=np.int32),
        "streak": np.array([int(denorm(obs_array[6], "streak"))], dtype=np.int32),
        "result": np.array([], dtype=np.float32),
        "line_win": np.array([], dtype=np.int32),
        "prev_bet": np.array([int(denorm(obs_array[7], "prev_bet"))], dtype=np.int32),
        "prev_basepoint": np.array([denorm(obs_array[8], "prev_basepoint")], dtype=np.float32),
        "prev_profit": np.array([int(denorm(obs_array[9], "prev_profit"))], dtype=np.int32),
        "total_profit": np.array([int(denorm(obs_array[10], "total_profit"))], dtype=np.int32),
    }


# Main Environment
class FlatSlotEnv(gym.Env):
    """
    Minimal slot environment that handles bet/profit/balance logic with expert transitions.
    """

    def __init__(self, expert_transitions, bet_dict, currency_list=None, seed=None):
        super().__init__()

        self.transitions = expert_transitions
        self.bet_dict = bet_dict
        self.currency_list = currency_list  # List of currencies corresponding to each transition
        self.current_idx = 0
        self.total_transitions = len(expert_transitions.obs)

        # Create spaces based on flattened observation dimensions
        obs_dim = expert_transitions.obs[0].shape[0]

        # debug
        print(
            f"DEBUG! -  observation dimension: {expert_transitions.obs[0]}, next_obs: {expert_transitions.next_obs[0]}"
        )

        max_action = int(expert_transitions.acts.max()) + 1

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(max_action)

        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """Reset to next episode starting point"""
        super().reset(seed=seed)

        # Move to next transition (next episode start)
        self.current_idx += 1

        # Wrap around if at end
        if self.current_idx >= self.total_transitions:
            self.current_idx = 0

        obs = self.transitions.obs[self.current_idx].copy()
        expert_action = int(self.transitions.acts[self.current_idx])

        # Get currency with bounds checking
        if self.currency_list:
            if self.current_idx < len(self.currency_list):
                currency = self.currency_list[self.current_idx]
            else:
                currency = self.currency_list[-1]  # Use last element if out of bounds
        else:
            currency = "CNY"  # Default fallback

        info = {"transition_idx": self.current_idx, "expert_action": expert_action, "currency": currency}

        return obs, info

    def step(self, action):
        """Step with proper bet/profit/balance handling"""
        if self.current_idx >= self.total_transitions:
            raise ValueError("No more transitions, call reset()")

        # Handle action format - it might be a tuple from TupleActionWrapper or a direct integer
        if isinstance(action, tuple):
            # If it's a tuple (bet_idx, delta_t), extract bet_idx
            agent_bet_idx = int(action[0])
        elif isinstance(action, (list, np.ndarray)):
            # If it's an array/list, take the first element
            agent_bet_idx = int(action[0])
        else:
            # If it's a direct integer
            agent_bet_idx = int(action)

        # Get current transition data
        current_obs = self.transitions.obs[self.current_idx]
        next_obs = self.transitions.next_obs[self.current_idx].copy()
        expert_action = int(self.transitions.acts[self.current_idx])
        expert_info = self.transitions.infos[self.current_idx]

        # Get currency for this transition with bounds checking
        if self.currency_list:
            if self.current_idx < len(self.currency_list):
                currency = self.currency_list[self.current_idx]
            else:
                currency = self.currency_list[-1]  # Use last element if out of bounds
        else:
            currency = "USD"  # Default fallback

        # Determine if episode should end (only when expert bet = 0)
        expert_bet_idx = int(next_obs[0])
        done = expert_bet_idx == 0

        # If agent chooses to terminate (bet = 0), return zero observation
        if agent_bet_idx == 0:
            next_obs = np.zeros_like(next_obs)
        else:
            # Handle bet/profit/balance adjustments for non-zero bets

            if expert_bet_idx > 0 and agent_bet_idx > 0:  # Both non-zero bets
                # Get actual bet values
                expert_bet_value = get_bet_value(expert_bet_idx, currency, self.bet_dict)
                agent_bet_value = get_bet_value(agent_bet_idx, currency, self.bet_dict)

                # Scale profit by bet ratio
                if expert_bet_value > 0:
                    profit_ratio = agent_bet_value / expert_bet_value

                    # Calculate profit difference for balance adjustment
                    original_profit = next_obs[1]
                    scaled_profit = original_profit * profit_ratio
                    profit_difference = scaled_profit - original_profit

                    # Update all profit/bet related fields
                    next_obs[0] = agent_bet_idx  # bet_idx
                    next_obs[1] *= profit_ratio  # adjusted_profit
                    next_obs[3] += profit_difference  # basepoint (balance)
                    # delta_bet was removed, so indices after basepoint shift down by 1
                    next_obs[4] *= profit_ratio  # delta_profit
                    next_obs[5] *= profit_ratio  # delta_payout
                    next_obs[10] *= profit_ratio  # total_profit

            else:  # expert_bet_idx == 0 or other edge cases
                next_obs[0] = agent_bet_idx

        # Simple reward (can be customized)
        reward = 1.0 if agent_bet_idx == expert_action else 0.0

        # Create info dict
        info = {
            "transition_idx": self.current_idx,
            "expert_action": expert_action,
            "agent_action": agent_bet_idx,
            "expert_bet_idx": expert_bet_idx,
            "agent_bet_idx": agent_bet_idx,
            "currency": currency,
            "action_match": agent_bet_idx == expert_action,
            **expert_info,
        }

        # Move to next transition
        self.current_idx += 1

        return next_obs, reward, done, False, info


# Environment with Binary Actions, for training termination
class BinarySlotEnv(gym.Env):
    """
    Simplified slot environment with binary actions: 0 (stop) or 1 (play).
    No bet value computations - just uses observations as-is.
    Uses Termination_Reward for sophisticated reward computation.
    """

    def __init__(self, expert_transitions, bet_dict=None, currency_list=None, seed=None, reward_config=None):
        super().__init__()

        self.transitions = expert_transitions
        self.bet_dict = bet_dict  # Keep for compatibility, but won't use
        self.currency_list = currency_list
        self.current_idx = 0
        self.total_transitions = len(expert_transitions.obs)

        # Initialize termination reward system
        if reward_config is None:
            reward_config = {"correct_stop": 2, "incorrect_stop": -4, "correct_cont": 1, "incorrect_cont": -2}
        self.termination_reward = Termination_Reward(reward_config)

        # Create spaces based on flattened observation dimensions
        obs_dim = expert_transitions.obs[0].shape[0]

        # Debug
        print(
            f"DEBUG! - observation dimension: {expert_transitions.obs[0]}, next_obs: {expert_transitions.next_obs[0]}"
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Simplified binary action space: 0 (stop), 1 (play)
        self.action_space = spaces.Discrete(2)

        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """Reset to next episode starting point"""
        super().reset(seed=seed)

        # Move to next transition (next episode start)
        self.current_idx += 1

        # Wrap around if at end
        if self.current_idx >= self.total_transitions:
            self.current_idx = 0

        obs = self.transitions.obs[self.current_idx].copy()
        expert_action = int(self.transitions.acts[self.current_idx])

        # Convert expert action to binary: 0 stays 0 (stop), anything else becomes 1 (play)
        expert_binary_action = 0 if expert_action == 0 else 1

        info = {
            "transition_idx": self.current_idx,
            "expert_action": expert_binary_action,
            "original_expert_action": expert_action,  # Keep original for reference
            "currency": self.currency_list[self.current_idx] if self.currency_list else "CNY",
        }

        return obs, info

    def step(self, action):
        """Step with binary actions - no bet computations needed"""
        if self.current_idx >= self.total_transitions:
            raise ValueError("No more transitions, call reset()")

        # Get current transition data
        current_obs = self.transitions.obs[self.current_idx]
        next_obs = self.transitions.next_obs[self.current_idx].copy()
        expert_action = int(self.transitions.acts[self.current_idx])
        expert_info = self.transitions.infos[self.current_idx]

        # Convert expert action to binary
        expert_binary_action = 0 if expert_action == 0 else 1

        # Get currency for this transition
        currency = self.currency_list[self.current_idx] if self.currency_list else "USD"

        # Determine if episode should end based on expert's original action
        expert_bet_idx = int(next_obs[0])
        done = expert_bet_idx == 0

        # Handle binary actions
        if action == 0:  # Stop/Terminate
            # Return zero observation to indicate termination
            next_obs = np.zeros_like(next_obs)
        else:  # Play (action == 1)
            # Use next_obs as-is from expert data - no modifications needed
            # The expert's bet values, profits, etc. are already computed
            pass

        # Compute sophisticated reward using Termination_Reward
        agent_stops = action == 0
        expert_stops = expert_binary_action == 0
        reward = self.termination_reward.get_reward(agent_stops, expert_stops)

        # Create info dict
        info = {
            "transition_idx": self.current_idx,
            "expert_action": expert_binary_action,
            "original_expert_action": expert_action,
            "agent_action": action,
            "expert_bet_idx": expert_bet_idx,
            "currency": currency,
            "action_match": action == expert_binary_action,
            "agent_stops": agent_stops,
            "expert_stops": expert_stops,
            "reward_breakdown": {
                "agent_stops": agent_stops,
                "expert_stops": expert_stops,
                "reward_type": self._get_reward_type(agent_stops, expert_stops),
            },
            **expert_info,
        }

        # Move to next transition
        self.current_idx += 1

        return next_obs, reward, done, False, info

    def _get_reward_type(self, agent_stops, expert_stops):
        """Helper method to get readable reward type for debugging."""
        if agent_stops and expert_stops:
            return "correct_stop"
        elif agent_stops and not expert_stops:
            return "incorrect_stop"
        elif not agent_stops and expert_stops:
            return "incorrect_cont"
        else:  # both continue
            return "correct_cont"
