from curses import wrapper

import gymnasium as gym

# from .SLOTGYM import find_bet_index, TupleActionWrapper, get_bet_value
import numpy as np
import torch

# import torch.nn as nn
from imitation.rewards.reward_nets import BasicRewardNet


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


class RewardWithR1(BasicRewardNet):
    """
    Reward network with R1 regularization and BC regularization to stabilize GAIL training
    after behavior cloning.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        venv,
        r1_gamma=5.0,  # Reduced from 10.0
        bc_coef=1,  # New parameter for BC loss weight
        bc_decay_steps=20000,  # Steps to gradually reduce BC influence
        **kwargs,
    ):
        """
        Initialize the reward network with BC regularization.

        Parameters:
        -----------
        observation_space : gym.spaces.Space
            The observation space of the environment.
        action_space : gym.spaces.Space
            The action space of the environment.
        venv : VecEnv
            Vectorized environment reference used to access environment-specific information.
        r1_gamma : float
            Strength of the R1 gradient regularization (reduced from default).
        bc_coef : float
            Initial weight for the BC loss term.
        bc_decay_steps : int
            Number of steps to gradually reduce BC influence.
        **kwargs : dict
            Additional keyword arguments passed to the parent BasicRewardNet class.
        """
        super().__init__(observation_space=observation_space, action_space=action_space, **kwargs)
        self.venv = venv
        self.r1_gamma = r1_gamma
        self.training = True
        self.num_discrete_actions = 15  # REMINDER: change if needed!!

        # BC regularization parameters
        self.bc_coef = bc_coef
        self.bc_decay_steps = bc_decay_steps
        self.current_step = 0
        self.bc_policy = None  # Will store reference to BC-trained policy

        # Create a wrapper for action decoding
        self.wrapper = TupleActionWrapper(None)

    def set_bc_policy(self, policy):
        """
        Set the BC-trained policy for regularization.

        Parameters:
        -----------
        policy : torch.nn.Module
            The policy trained using behavior cloning.
        """
        self.bc_policy = policy
        print("BC policy reference set for reward regularization")

    def compute_r1_penalty(self, state, action, next_state, done):
        """
        Compute R1 gradient penalty to stabilize training.
        """
        # Create a copy of state that requires gradient
        state_with_grad = state.detach().clone().requires_grad_(True)

        # Forward pass with state_with_grad to get reward
        with torch.enable_grad():
            # Pass through the parent BasicRewardNet directly
            reward_for_grad = super().forward(state_with_grad, action, next_state, done)

            # Compute gradients w.r.t. state_with_grad
            grad_outputs = torch.ones_like(reward_for_grad)
            grads = torch.autograd.grad(
                outputs=reward_for_grad,
                inputs=state_with_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Compute the R1 penalty: sum of squared gradients
            r1_penalty = grads.pow(2).sum(dim=1).mean()

        return r1_penalty

    def compute_bc_reward(self, state, action):
        """
        BC-based rewards (negative KL divergence or negative cross-entropy)
        """
        if self.bc_policy is None:
            return torch.zeros_like(action[:, 0] if action.ndim > 1 else action)

        # Ensure state is detached to avoid backprop through BC policy
        state_detached = state.detach()

        try:
            with torch.no_grad():
                # Get BC policy distribution
                bc_dist = self.bc_policy.get_distribution(state_detached)

                # If action is one-hot encoded, convert to indices
                if action.ndim == 2 and action.shape[1] > 1:
                    action_indices = torch.argmax(action, dim=1)
                else:
                    action_indices = action.long().view(-1)

                # Compute log probabilities of current actions under BC policy
                if hasattr(bc_dist, "logits"):
                    bc_logits = bc_dist.logits
                    # Per-sample cross-entropy loss (don't reduce to mean yet)
                    bc_losses = torch.nn.functional.cross_entropy(bc_logits, action_indices, reduction="none")
                    # Convert losses to rewards (negative loss)
                    rewards = -bc_losses
                else:
                    # Use log probability directly as reward
                    rewards = bc_dist.log_prob(action_indices)

                return rewards
        except Exception as e:
            print(f"Error in computing BC reward: {e}")
            return torch.zeros_like(action[:, 0] if action.ndim > 1 else action)

    def compute_bc_loss(self, state, action):
        """
        Returns:
        --------
        torch.Tensor
            BC loss (KL divergence or cross-entropy)
        """
        if self.bc_policy is None:
            return torch.tensor(0.0, device=state.device)

        # Ensure state is detached to avoid backprop through BC policy
        state_detached = state.detach()

        try:
            with torch.no_grad():
                # Get BC policy distribution
                bc_dist = self.bc_policy.get_distribution(state_detached)

                # If action is one-hot encoded, convert to indices
                if action.ndim == 2 and action.shape[1] > 1:
                    action_indices = torch.argmax(action, dim=1)
                else:
                    action_indices = action.long().view(-1)

                # Compute log probabilities of current actions under BC policy
                if hasattr(bc_dist, "logits"):
                    bc_logits = bc_dist.logits
                    # Cross-entropy loss
                    bc_loss = torch.nn.functional.cross_entropy(bc_logits, action_indices, reduction="mean")
                else:
                    # Fallback to negative log probability
                    log_probs = bc_dist.log_prob(action_indices)
                    bc_loss = -log_probs.mean()

                return bc_loss

        except Exception as e:
            print(f"Error computing BC loss: {e}")
            return torch.tensor(0.0, device=state.device)

    def get_current_bc_coef(self):
        """
        Get the current BC coefficient with decay.

        Returns:
        --------
        float
            Current BC coefficient value
        """
        if self.current_step >= self.bc_decay_steps:
            return 1

        # Linear decay
        decay_factor = 1.0 - (self.current_step / self.bc_decay_steps)

        # DEBUG
        decay_factor = 1

        return self.bc_coef * decay_factor

    def forward(self, state, action, next_state, done, infos=None):
        """
        Calculate reward values with BC regularization and R1 regularization.
        """
        # Increment step counter for BC decay
        if self.training:
            self.current_step += 1

        # Process action to one-hot format
        to_onehot = False
        if action.ndim == 1:
            # If 1D tensor [batch_size], reshape to [batch_size, 1] for scatter
            # Clamp to ensure indices are in valid range
            action_idx = action.long().clamp(0, self.num_discrete_actions - 1).view(-1, 1)
            to_onehot = True
        elif action.ndim == 2 and action.shape[1] == 1:
            # If already [batch_size, 1], use as is but ensure valid range
            action_idx = action.long().clamp(0, self.num_discrete_actions - 1)
            to_onehot = True
        elif action.ndim == 2 and action.shape[1] == self.num_discrete_actions:
            # If already one-hot encoded with correct dimensions
            parent_action = action
        else:
            # Unexpected format, log and try to use as is
            print(f"WARNING: Unexpected action shape: {action.shape}, ndim: {action.ndim}")
            parent_action = action

        if to_onehot:
            device = action.device
            batch_size = action.shape[0]
            # Create one-hot encoded tensor
            one_hot_action = torch.zeros((batch_size, self.num_discrete_actions), device=device)
            try:
                one_hot_action.scatter_(1, action_idx, 1)
                parent_action = one_hot_action
            except Exception as e:
                print(
                    f"Error in one-hot conversion: {e}, action_idx shape: {action_idx.shape}, min: {action_idx.min()}, max: {action_idx.max()}"
                )
                # Fallback in case of error
                parent_action = action

        # Get base reward from parent network
        try:
            base_reward = super().forward(state, parent_action, next_state, done)
        except Exception as e:
            print(f"Error in parent forward: {e}")
            print(
                f"State shape: {state.shape}, Parent action shape: {parent_action.shape}, Next state shape: {next_state.shape}"
            )
            base_reward = torch.zeros(state.shape[0], device=state.device)

        # Apply R1 regularization during training
        r1_penalty = torch.tensor(0.0, device=state.device)
        if self.training and self.r1_gamma > 0:
            r1_penalty = self.compute_r1_penalty(state, parent_action, next_state, done)

        # Apply BC regularization during training
        bc_rew = torch.tensor(0.0, device=state.device)
        current_bc_coef = 0.0
        if self.training and self.bc_policy is not None:
            current_bc_coef = self.get_current_bc_coef()
            if current_bc_coef > 0:
                bc_loss = self.compute_bc_reward(state, action)

        # Compute total reward with all components
        total_reward = base_reward - self.r1_gamma * r1_penalty + current_bc_coef * bc_rew

        # Debug info
        if torch.rand(1).item() < 0.01:  # Only print occasionally to avoid flooding logs
            print(
                f"DEBUG - Reward components: base={base_reward[0].item():.4f}, "
                f"r1={(-self.r1_gamma * r1_penalty).item() if self.r1_gamma > 0 else 0:.4f}, "
                # f"bc={(-current_bc_coef * bc_loss).item() if current_bc_coef > 0 else 0:.4f}, "
                f"total={total_reward[0].item():.4f}, "
                f"bc_coef={current_bc_coef:.4f}, step={self.current_step}"
            )

        return total_reward

    def train(self, mode=True):
        """
        Sets the module in training mode.
        """
        self.training = mode
        super().train(mode)
        return self


class PrioritizedBuffer:
    """
    Prioritized buffer that emphasizes transitions with termination issues and action imbalance.
    Supports individual transitions and handles multiple transition formats.
    """

    def __init__(
        self,
        capacity=10000,
        alpha=0.6,
        beta_start=0.4,
        beta_steps=100000,
        term_priority_factor=3.0,
        bet_priority_factor=3.0,
        initial_transitions=None,
    ):
        self.capacity = capacity
        self.alpha = alpha  # decide priority strength. 0 = not usding
        self.beta = beta_start
        self.beta_step = (1.0 - beta_start) / beta_steps  # Annealing rate

        # Storage
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.wrapper = TupleActionWrapper(None)

        # Prioritization parameters for termination
        self.term_priority_factor = term_priority_factor  # Higher priority for termination issues
        self.bet_priority_factor = bet_priority_factor  # Higher priority for incorrect bets

        if initial_transitions is not None:
            print(f"Adding {len(initial_transitions)} initial transitions to buffer")
            for transition in initial_transitions:
                if len(transition) == 5:  # (obs, act, next_obs, done, info)
                    obs, act, next_obs, done, info = transition
                    self.add((obs, act, next_obs, done), info=info)
                else:
                    self.add(transition)

    def add(self, transition, info=None):
        """
        Add a new transition to the buffer with termination-aware prioritization.

        Parameters:
        -----------
        transition : tuple or object
            Transition data. Can be:
            - (obs, act, next_obs, done) tuple
            - (obs, act, next_obs, done, info) tuple
            - TransitionsWithRew-like object
        info : dict or None
            Additional information about the transition
        """
        # Handle TransitionsWithRew-like objects
        if hasattr(transition, "obs") and hasattr(transition, "acts"):
            print(f"Adding batch of {len(transition.obs)} transitions from TransitionsWithRew")

            for i in range(len(transition.obs)):
                obs = transition.obs[i]
                act = transition.acts[i]
                next_obs = transition.next_obs[i] if hasattr(transition, "next_obs") else None
                done = transition.dones[i] if hasattr(transition, "dones") else False

                # Get info if available
                transition_info = {}
                if hasattr(transition, "infos") and transition.infos:
                    transition_info = transition.infos[i]

                    # Add expert action to info
                    if transition_info is None:
                        transition_info = {}

                # Add individual transition
                self.add((obs, act, next_obs, done), transition_info)
            return

        # Handle various tuple formats
        if isinstance(transition, tuple):
            # Process based on tuple length
            if len(transition) == 5:
                # (obs, act, next_obs, done, info_dict) format
                obs, act, next_obs, done, transition_info = transition
                # Merge info dictionaries
                if info is None:
                    info = transition_info
                else:
                    # Update info with transition_info, preserving existing keys
                    for k, v in transition_info.items():
                        if k not in info:
                            info[k] = v
            elif len(transition) == 4:
                # Standard (obs, act, next_obs, done) format
                obs, act, next_obs, done = transition
            elif len(transition) == 2:
                # Minimal (obs, act) format
                obs, act = transition
                next_obs, done = None, False
            else:
                print(f"WARNING: Unexpected transition format with {len(transition)} elements")
                try:
                    # Try to adapt with best guess
                    obs = transition[0]
                    act = transition[1]
                    next_obs = transition[2] if len(transition) > 2 else None
                    done = transition[3] if len(transition) > 3 else False
                except Exception as e:
                    print(f"ERROR: Could not parse transition: {e}")
                    return
        else:
            print(f"WARNING: Unsupported transition type: {type(transition)}")
            return

        # Ensure info is a dictionary
        if info is None:
            info = {}

        # Standardize transition format for storage
        buffer_transition = (obs, act, next_obs, done, info)

        # Calculate priority based on transition info
        priority = 1.0  # Default priority

        # Apply termination-aware prioritization
        if info.get("early_terminate", False) or info.get("late_terminate", False):
            priority *= self.term_priority_factor

            length_diff = info.get("length_difference", None)

            if length_diff is not None:
                severity = 1.0 + min(2.0, 0.5 * abs(length_diff) / max(1, info.get("correct_length", 1)))
                priority *= severity

        act_id, _ = self.wrapper.action(int(act))
        # Check for bet correctness if expert action is available
        if "expert_action" in info:
            # Simple check if actions match
            if info["expert_action"] != act_id:
                priority *= self.bet_priority_factor

        # Use max priority if buffer isn't empty (ensures new transitions get sampled)
        max_priority = np.max(self.priorities[: self.size]) if self.size > 0 else 1.0
        priority = max(priority, max_priority)

        # RING BUFFER IMPLEMENTATION - Add or overwrite
        if self.size < self.capacity:
            # Buffer not full yet - append new transition
            self.buffer.append(buffer_transition)
        else:
            # Buffer is full - overwrite oldest transition (ring behavior)
            self.buffer[self.position] = buffer_transition

        # Update priority at current position
        self.priorities[self.position] = priority

        # ring buffer logic
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch based on priorities.

        Parameters:
        -----------
        batch_size : int
            Number of transitions to sample

        Returns:
        --------
        tuple
            (transitions, indices, weights) - the sampled transitions, their indices,
            and importance sampling weights
        """
        # Clamp batch size to current buffer size
        batch_size = min(batch_size, self.size)

        # Calculate sampling probabilities
        if self.alpha == 0:
            # Uniform sampling
            probs = np.ones(self.size) / self.size
        else:
            # Prioritized sampling
            probs = self.priorities[: self.size] ** self.alpha
            probs /= np.sum(probs)

        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights

        # Advance beta for next time
        self.beta = min(1.0, self.beta + self.beta_step)

        # Get samples and return everything
        transitions = [self.buffer[idx] for idx in indices]

        # Debug info (occasionally)
        if np.random.random() < 0.01:
            print(
                f"Buffer size: {self.size}, beta: {self.beta:.4f}, "
                f"min weight: {np.min(weights):.4f}, max priority: {np.max(self.priorities[:self.size]):.4f}"
            )

        return transitions, indices, weights

    def update_priorities(self, indices, priorities):

        for idx, priority in zip(indices, priorities):
            # Handle index errors gracefully
            if 0 <= idx < self.size:
                self.priorities[idx] = priority + 1e-6  # Add small constant for stability
            else:
                print(f"WARNING: Index {idx} out of range [0, {self.size - 1}]")

    def get_stats(self):

        if self.size == 0:
            return {"size": 0}

        active_priorities = self.priorities[: self.size]

        return {
            "size": self.size,
            "min_priority": float(np.min(active_priorities)),
            "max_priority": float(np.max(active_priorities)),
            "mean_priority": float(np.mean(active_priorities)),
            "beta": float(self.beta),
        }


class Termination_Reward:
    """A class to specifically handle termination reward in the termination game setting."""

    def __init__(
        self, reward_config={"correct_stop": 2, "incorrect_stop": -4, "correct_cont": 1, "incorrect_cont": -2}
    ):
        self.correct_stop = reward_config["correct_stop"]
        self.incorrect_stop = reward_config["incorrect_stop"]
        self.correct_cont = reward_config["correct_cont"]
        self.incorrect_cont = reward_config["incorrect_cont"]

    def get_reward(self, agent_stops, expert_stops):
        """Get reward based on agent and expert actions."""
        if agent_stops and expert_stops:
            return self.correct_stop
        elif agent_stops and not expert_stops:
            return self.incorrect_stop
        elif not agent_stops and expert_stops:
            return self.incorrect_cont
        else:  # both continue
            pass
