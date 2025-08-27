import copy
import gc
import os
import pickle
import shutil
import tempfile
import zipfile

# Plot BC training losses if needed
import matplotlib.pyplot as plt

# import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import TransitionsWithRew
from sklearn.preprocessing import OneHotEncoder

# from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .rewardnets import PrioritizedBuffer, RewardWithR1
from .slot_gym import DictToFlatObsWrapper, FlatSlotEnv, SlotBetEnv, TupleActionWrapper, find_bet_index, get_bet_value

# train_test_split


paylines = [
    [0, 1, 2, 3, 4],  # Line 1: 11111 (top row)
    [5, 6, 7, 8, 9],  # Line 2: 00000 (middle row)
    [10, 11, 12, 13, 14],  # Line 3: 22222 (bottom row)
    [0, 6, 12, 8, 4],  # Line 4: 01210 (V shape)
    [10, 6, 2, 8, 14],  # Line 5: 21012 (inverted V)
    [5, 1, 2, 3, 9],  # Line 6: 10001
    [5, 11, 12, 13, 9],  # Line 7: 12221
    [0, 1, 7, 13, 14],  # Line 8: 00122
    [10, 11, 7, 3, 4],  # Line 9: 22100
    [5, 1, 7, 13, 9],  # Line 10: 10121
    [5, 11, 7, 3, 9],  # Line 11: 12101
    [0, 6, 7, 8, 4],  # Line 12: 01110
    [10, 6, 7, 8, 14],  # Line 13: 21112
    [0, 6, 2, 8, 4],  # Line 14: 01010
    [10, 6, 12, 8, 14],  # Line 15: 21212
    [5, 6, 2, 8, 9],  # Line 16: 11011
    [5, 6, 12, 8, 9],  # Line 17: 11211
    [0, 1, 12, 3, 4],  # Line 18: 00200
    [10, 11, 2, 13, 14],  # Line 19: 22022
    [0, 11, 12, 13, 4],  # Line 20: 02220
    [10, 1, 2, 3, 14],  # Line 21: 20002
    [5, 1, 12, 3, 9],  # Line 22: 10201
    [5, 11, 2, 13, 9],  # Line 23: 12021
    [0, 11, 2, 13, 4],  # Line 24: 02020
    [10, 1, 12, 3, 14],  # Line 25: 20202
]

payout = {
    1: [100, 200, 500],
    2: [80, 150, 400],
    3: [60, 200, 300],
    4: [40, 100, 250],
    5: [30, 80, 200],
    6: [20, 60, 180],
    7: [10, 60, 150],
    8: [10, 40, 200],
    9: [10, 40, 100],
    10: [5, 30, 100],
    11: [5, 30, 100],
    101: [10, 40, 100],
}

bet_dictionary = {
    # Australian Dollar (AUD)
    "AUD": [0.0, 0.1, 0.2, 0.5, 1.0, 1.6, 3.0, 5.0],
    # Brazilian Real (BRL)
    "BRL": [0.0, 0.25, 0.5, 1.25],
    # Chinese Yuan (CNY)
    "CNY": [0.0, 0.5, 1.0, 2.5, 5.0, 8.0, 15.0, 25.0, 50.0, 70.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0],
    # Euro (EUR)
    "EUR": [0.25],
    # Indonesian Rupiah (IDR)
    "IDR": [0.0, 1.0, 2.0, 5.0, 10.0, 16.0, 30.0, 50.0, 200.0, 500.0],
    # Indian Rupee (INR)
    "INR": [0.0, 5.0, 10.0, 25.0, 50.0, 80.0, 150.0],
    # Japanese Yen (JPY)
    "JPY": [10.0, 50.0, 160.0],
    # Korean Won (KER)
    "KER": [0.0, 100.0, 200.0, 500.0, 1000.0, 1600.0, 3000.0, 5000.0, 10000.0, 14000.0, 20000.0],
    # Myanmar Kyat (MMK)
    "MMK": [1.0, 2.0, 5.0, 10.0, 500.0],
    # Malaysian Ringgit (MYR)
    "MYR": [0.0, 0.25, 0.5, 1.25, 2.5, 4.0, 7.5, 12.5, 25.0, 35.0, 50.0],
    # Thai Baht (THB)
    "THB": [0.0, 2.5, 5.0, 12.5, 25.0, 40.0, 75.0, 125.0, 250.0],
    # US Dollar (USD)
    "USD": [0.0, 0.05, 0.1, 0.25, 0.5, 0.8, 1.5, 2.5, 5.0, 7.0, 10.0, 25.0, 50.0, 100.0, 200.0],
    # Vietnamese Dong (VND)
    "VND": [0.0, 1.0, 2.0, 5.0, 10.0, 16.0, 30.0, 50.0, 100.0],
}


def train_test_split_features(features, test_size):
    n_samples = features.shape[0]
    test_points = int(n_samples * test_size)
    train_features = features[:-test_points]
    test_features = features[-test_points:]
    return train_features, test_features


def SlotPlayer_Simulator(
    data_path,
    # Training vs Evaluation
    train=True,
    result_csv_path="bc_training_results.csv",
    test_size=0.2,
    bet_dict=bet_dictionary,
    model_output_path="bc_model.pth",
    # Data portion control
    data_fraction=1.0,
    # Behavior cloning parameters
    bc_lr=1e-4,
    bc_weight_decay=1e-5,
    bc_batch_size=64,
    bc_epochs=100,  # Increased epochs since we're only doing BC
    # PPO parameters for environment setup
    learning_rate=2e-4,
    batch_size=64,
    n_steps=2048,
    entropy_coef=0.01,
    vf_coef=0.5,
    ppo_clip_range=0.1,
    # Save/load functionality
    save_expert_data=True,
    expert_data_path="expert_data.pkl",
    load_expert_data=False,
    ppo_kwargs={},
    verbose=1,
):
    """
    Train a behavioral cloning model on slot machine betting data (no GAIL).

    Parameters:
    -----------
    data_path : str
        Path to CSV file containing slot machine betting data.
    data_fraction : float
        Fraction of most recent data to load (0.0 to 1.0).
        Default 1.0 loads all data. 0.33 would load latest 1/3.
    bet_dict : dict
        Dictionary mapping currencies to available bet amounts.
    model_output_path : str
        Path to save the trained model.
    bc_epochs : int
        Number of epochs to train behavioral cloning.
    bc_lr : float
        Learning rate for behavioral cloning.
    bc_batch_size : int
        Batch size for behavioral cloning.
    bc_weight_decay : float
        Weight decay for behavioral cloning.
    [... other parameters ...]

    Returns:
    --------
    PPO agent
        The trained behavioral cloning agent
    """
    # Initialize expert_transitions, normalizers, and clip_range
    expert_transitions = None
    normalizers = None
    clip_range = None

    # Load expert data if requested (and file exists)
    if load_expert_data and os.path.exists(expert_data_path):
        if verbose > 0:
            print(f"Loading expert data from {expert_data_path}")
        try:
            with open(expert_data_path, "rb") as f:
                expert_data = pickle.load(f)
                test_expert_transitions = expert_data["test_expert_transitions"]
                train_expert_transitions = expert_data["train_expert_transitions"]
                normalizers = expert_data["normalizers"]
                clip_range = expert_data["clip_range"]
                sample_obs = expert_data["sample_obs"]
                currency_og = expert_data["currency_og"]

            if verbose > 0:
                print(f"Successfully loaded expert data with {len(train_expert_transitions.obs)} training transitions")
                print(f"Loaded normalizers: {list(normalizers.keys())}")
                print(f"Loaded clip range: {clip_range}")

        except Exception as e:
            print(f"Error loading expert data: {e}")
            print("Proceeding with data processing instead")
            load_expert_data = False
    else:
        load_expert_data = False

    # Process data if not loading expert data
    if not load_expert_data:
        # Pre-processing with data_fraction parameter
        df, state_cols = load_and_process_data(data_path, bet_dict=bet_dict, data_fraction=data_fraction)
        if verbose > 0:
            print(f"Loaded data with {len(df)} rows (fraction: {data_fraction})")
            print(f"Sample row: {df.iloc[3, :]}")
            print(f"State columns: {state_cols}")

        if "action_idx" not in df.columns:
            df["action_idx"] = df.apply(
                lambda row: find_bet_index(
                    bet_amount=row["bet"], currency=row["original_currency"], bet_dictionary=bet_dict
                ),
                axis=1,
            )

        # Select a sample observation sequence for environment initialization
        if len(df) > 0:
            sample_player = df.iloc[0]["loginname"]
            player_data = df[df["loginname"] == sample_player]

            # If no rows match the player, use another player
            if len(player_data) == 0:
                player_data = df[df["loginname"] == df.iloc[0]["loginname"]]

            # Sort player data by time
            player_data = player_data.sort_values("billtime").reset_index(drop=True)

            # Find the first termination (either explicit termination or end of session)
            termination_idx = None
            for i in range(len(player_data)):
                # Look for termination signal (bet_idx = 0 or other termination indicator)
                if player_data.iloc[i]["bet_idx"] == 0:
                    termination_idx = i
                    break

                # Also consider session breaks as termination
                if i > 0 and player_data.iloc[i]["delta_t"] > 50:  # Assuming 50 is the session break threshold
                    termination_idx = i
                    break

            # If no termination found or it's too far, take a reasonable number
            if termination_idx is None or termination_idx > 20:
                termination_idx = min(10, len(player_data) - 1)

            # Take sequence up to and including termination
            sample_session = player_data.iloc[: termination_idx + 1]

            # Ensure we have at least a few observations
            if len(sample_session) < 3:
                sample_session = player_data.head(min(10, len(player_data)))

        else:
            raise ValueError("DataFrame is empty, cannot create sample observations")

        # Create sample observations for env initialization
        sample_obs = []
        currency_og = []
        for _, row in sample_session.iterrows():
            obs_dict = {
                "bet": np.array([row["bet_idx"]], dtype=np.int32),
                "profit": np.array([row["adjusted_profit"]], dtype=np.int32),
                "slottype": int(row["slottype"]),
                "basepoint": np.array([row["basepoint"]], dtype=np.float32),
                "delta_t": np.array([row["delta_t"]], dtype=np.float32),
                "delta_bet": np.array([row["delta_bet"]], dtype=np.int32),
                "delta_profit": np.array([row["delta_profit"]], dtype=np.int32),
                "delta_payout": np.array([row["delta_payout"]], dtype=np.int32),
                "streak": np.array([row["streak"]], dtype=np.int32),
                "prev_bet": np.array([row["prev_bet"]], dtype=np.int32),
                "prev_basepoint": np.array([row["prev_basepoint"]], dtype=np.float32),
                "prev_profit": np.array([row["prev_profit"]], dtype=np.int32),
                "total_profit": np.array([row["total_profit"]], dtype=np.int32),
            }

            currency_og.append(row["original_currency"])
            sample_obs.append(obs_dict)

        if verbose > 0:
            print(f"Created {len(sample_obs)} sample observations with currencies: {set(currency_og)}")

        # Display a few sample observations for verification
        if verbose > 0:
            print("\n===== SAMPLE OBSERVATIONS FOR VERIFICATION =====")
            for i, obs in enumerate(sample_obs[:3]):  # Show first 3 observations
                print(f"Observation {i}:")
                print(f"  Currency: {currency_og[i]}")
                print(f"  Bet index: {obs['bet'][0]}")
                print(f"  Bet amount: {get_bet_value(obs['bet'][0], currency_og[i], bet_dict)}")
                print(f"  Profit: {obs['profit'][0]}")
                print(f"  Slot type: {obs['slottype']}")
                print(f"  Basepoint: {obs['basepoint'][0]}")
                print(f"  Delta time: {obs['delta_t'][0]}")
                print(f"  Streak: {obs['streak'][0]}")
                print()

        # Create flattened transitions instead of trajectories
        train_expert_transitions, test_expert_transitions, normalizers, clip_range = get_flattened_trajectories(df)

        # Save expert data with both train and test transitions
        if save_expert_data:
            print(f"Saving expert data - to {expert_data_path}")
            expert_data = {
                "train_expert_transitions": train_expert_transitions,
                "test_expert_transitions": test_expert_transitions,
                "normalizers": normalizers,
                "clip_range": clip_range,
                "sample_obs": sample_obs,
                "currency_og": currency_og,
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(expert_data_path)) or ".", exist_ok=True)

            # Save the data
            with open(expert_data_path, "wb") as f:
                pickle.dump(expert_data, f)

            if verbose > 0:
                print(f"Saved expert data to {expert_data_path}")
                print(f"  - Training transitions: {len(train_expert_transitions.obs)}")
                print(f"  - Test transitions: {len(test_expert_transitions.obs)}")

    # Calculate observation and action dimensions
    obs_dim = train_expert_transitions.obs.shape[1]
    if verbose > 0:
        print(f"Observation dimension: {obs_dim}")

    # --------------------------------------------------
    # Create environment function with the new minimal gym
    def env_fn():
        # Create the minimal environment directly using expert transitions
        env = FlatSlotEnv(
            expert_transitions=train_expert_transitions,
            bet_dict=bet_dict,
            currency_list=currency_og,  # Use the currency list from expert data
        )

        # Apply TupleActionWrapper for baseline3 compatibility
        max_bet_actions = max([len(bet_array) for bet_array in bet_dict.values()])
        env = TupleActionWrapper(env, num_discrete_actions=max_bet_actions)

        return env

    # Create vectorized environment
    venv = DummyVecEnv([env_fn])

    if verbose > 0:
        print(f"Created vectorized environment with:")
        print(f"  - Expert transitions: {len(train_expert_transitions.obs)}")
        print(f"  - Observation space: {venv.observation_space}")
        print(f"  - Action space: {venv.action_space}")
        print(f"  - Max bet actions: {max([len(bet_array) for bet_array in bet_dict.values()])}")

    # Create agent with customizable parameters
    agent = PPO(
        "MlpPolicy",
        venv,
        verbose=verbose,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        ent_coef=entropy_coef,
        clip_range=ppo_clip_range,
        vf_coef=vf_coef,
        policy_kwargs=ppo_kwargs,
    )

    if train:
        # Train with behavior cloning only
        if verbose > 0:
            print(f"Starting behavioral cloning training for {bc_epochs} epochs")

        bc_losses, _ = pretrain_with_behavior_cloning(
            policy=agent.policy,
            expert_transitions=train_expert_transitions,
            venv=venv,
            learning_rate=bc_lr,
            batch_size=bc_batch_size,
            epochs=bc_epochs,
            weight_decay=bc_weight_decay,
            save_path=model_output_path,
            collect_transitions=False,  # We don't need transitions for BC-only
        )

        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(bc_losses)
        plt.title("Behavioral Cloning Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("bc_training_loss.png", dpi=150, bbox_inches="tight")
        plt.show()

        if verbose > 0:
            print(f"Training complete, model saved to {model_output_path}")
            print(f"Final training loss: {bc_losses[-1]:.6f}")

    else:
        if verbose > 0:
            print(f"Loading model from: {model_output_path}")

        # Load the trained model
        load_pretrained_model(agent.policy, model_output_path, eval_mode=True)

    # Verify model performance
    verify(
        agent=agent,
        test_expert_transitions=test_expert_transitions,
        output_csv_path=result_csv_path,
        num_samples=100000,  # or adjust as needed
        verbose=verbose,
    )

    return agent


class ComparisonCallback:
    """Ultra minimal callback that can be called directly as a function"""

    def __init__(self, expert_transitions, gail_agent, display_interval=20):
        self.expert_transitions = expert_transitions
        self.gail_agent = gail_agent  # Store reference to the GAIL object
        self.display_interval = display_interval
        self.iteration = 0

    def __call__(self, round_num):
        """This makes the object callable like a function"""
        self.iteration += 1
        if self.iteration % self.display_interval == 0:
            # Access the model directly from the GAIL object
            model = self.gail_agent.gen_algo

            # Pick a random example
            if len(self.expert_transitions.obs) > 0:
                sample_idx = np.random.randint(0, len(self.expert_transitions.obs))
                obs = self.expert_transitions.obs[sample_idx]
                expert_action = self.expert_transitions.acts[sample_idx]
                expert_info = self.expert_transitions.infos[sample_idx]

                # Get model prediction
                model_action, _ = model.predict(obs[np.newaxis, :], deterministic=True)

                # Get model's wrapper to decode actions (similar to your reward function)
                wrapper = TupleActionWrapper(None, num_discrete_actions=self.gail_agent.venv.action_space.n)

                # Decode both expert and model actions
                model_bet_idx, model_delta_t = wrapper.action(int(model_action))
                expert_bet_idx, expert_delta_t = wrapper.action(expert_action)

                # Print comparison with transition info
                print(f"\nStep {self.iteration} (Round {round_num}):")
                print(
                    f"Expert action: {int(expert_action)} (bet={int(expert_bet_idx)}, delta_t={int(expert_delta_t):.4f})"
                )
                print(f"Model action: {int(model_action)} (bet={int(model_bet_idx)}, delta_t={int(model_delta_t):.4f})")
                print(
                    f"Difference: bet={abs(expert_bet_idx - model_bet_idx)}, delta_t={abs(expert_delta_t - model_delta_t)}"
                )

                # Option to store recent model info if it's not already being stored
                if not hasattr(self.gail_agent, "recent_model_infos"):
                    # Initialize this attribute if it doesn't exist
                    self.gail_agent.recent_model_infos = []

                # If you have model transition info available, print it
                if hasattr(self.gail_agent, "recent_model_infos") and sample_idx < len(
                    self.gail_agent.recent_model_infos
                ):
                    model_info = self.gail_agent.recent_model_infos[sample_idx]
                    print(f"Model info: {model_info}")
                else:
                    # If model info isn't available, you could generate it here
                    # For example:
                    model_info = {"action": model_action, "bet": model_bet_idx, "delta_t": model_delta_t}
                    print(f"Generated model info: {model_info}")

                    if hasattr(self.gail_agent, "recent_model_infos"):
                        if len(self.gail_agent.recent_model_infos) > 1000:  # Set a reasonable limit
                            self.gail_agent.recent_model_infos = self.gail_agent.recent_model_infos[-1000:]
                        self.gail_agent.recent_model_infos.append(model_info)

        return True


def calculate_line_wins(result_vector, paylines_flattened, payout_table, wild_base=200):
    """
    Calculate the win amount for each payline based on the result vector.

    Args:
        result_vector: A 1x15 list or array containing the symbols in the 3x5 grid (flattened)
        paylines_flattened: List of paylines, each containing indices in the flattened grid
        payout_table: Dictionary mapping symbols to payout lists [3-in-a-row, 4-in-a-row, 5-in-a-row]
        wild_base: Base number for wild symbols (all symbols > wild_base are considered wild)

    Returns:
        List of win amounts for each payline
    """
    line_wins = []

    for line_idx, line in enumerate(paylines_flattened):
        # Get symbols on this payline
        line_symbols = [result_vector[idx] for idx in line]

        # First symbol cannot be wild
        first_symbol = line_symbols[0]
        # print(line_symbols)
        is_wild = lambda s: s > wild_base

        second_symbol = line_symbols[1]
        second_is_match = second_symbol == first_symbol or is_wild(second_symbol)

        # Skip this line if the first two don't match
        if not second_is_match:
            line_wins.append(0)
            continue

        # Count consecutive matching symbols and track multiplier
        consecutive_count = 1
        total_multiplier = 1  # Start with default multiplier

        # Count consecutive matching symbols from left to right
        for i in range(1, 5):
            symbol = line_symbols[i]  # at least in this game the first won't be wild.
            # if in the future the first can be wild, change the logic

            if symbol == first_symbol:
                consecutive_count += 1
            elif is_wild(symbol):
                consecutive_count += 1
                # Extract multiplier from wild symbol (mod 100)
                multiplier = symbol % 100

                total_multiplier *= multiplier
            else:
                break

        if consecutive_count >= 3:
            # Subtract 3 cuz 3 in a row is 0 4 in a row is 1 etc...
            payout_index = consecutive_count - 3
            win_amount = payout_table[first_symbol][payout_index] * total_multiplier
            line_wins.append(win_amount)
        else:
            line_wins.append(0)

    return line_wins


def load_and_process_data(csv_path, bet_dict=None, data_fraction=0.3):
    """
    Simplified data processing function that excludes 'delta_bet' column.
    """
    print(f"Loading data from {csv_path}...")

    # Basic dtypes
    dtype_dict = {
        "loginname": "category",
        "currency": "category",
        "slottype": "int8",
        "account": "float32",
        "cus_account": "float32",
        "basepoint": "float32",
        "delta_t": "float32",
        "delta_profit": "float32",
        "delta_payout": "float32",
        "streak": "int16",
    }

    # Get file size and load only what we need
    row_count = sum(1 for _ in open(csv_path)) - 1
    target_rows = int(row_count * data_fraction)
    skip_rows = max(0, row_count - target_rows)

    print(f"Loading {target_rows} rows (skipping {skip_rows})...")

    # Load data
    df = pd.read_csv(csv_path, dtype=dtype_dict, skiprows=range(1, skip_rows + 1))

    print(f"Loaded {len(df)} rows")

    # Sort by player and time
    df.sort_values(["loginname", "billtime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Process free spins - simplified version
    df["free_spin_total"] = 0.0
    rows_to_drop = []

    for player in df["loginname"].unique():
        group = df[df["loginname"] == player].sort_values("billtime")
        indices = group.index.tolist()

        i = 0
        while i < len(indices):
            idx = indices[i]
            if df.at[idx, "slottype"] == 2:  # Free spin
                start = i
                while i < len(indices) and df.at[indices[i], "slottype"] == 2:
                    i += 1
                end = i

                # Find trigger spin (last normal spin before free spins)
                trigger_idx = start - 1
                while trigger_idx >= 0 and df.at[indices[trigger_idx], "slottype"] != 1:
                    trigger_idx -= 1

                if trigger_idx >= 0:
                    bonus_indices = indices[start:end]
                    bonus_total = df.loc[bonus_indices, "cus_account"].sum()
                    df.at[indices[trigger_idx], "free_spin_total"] = bonus_total
                    rows_to_drop.extend(bonus_indices)
            else:
                i += 1

    # Drop free spin rows
    df = df.drop(index=rows_to_drop).reset_index(drop=True)

    # Rename columns
    df.rename(columns={"account": "bet", "cus_account": "profit"}, inplace=True)
    df["adjusted_profit"] = df["profit"] + df["free_spin_total"]
    df.drop(["profit", "free_spin_total"], axis=1, inplace=True)

    # Sort again for context features
    df.sort_values(["loginname", "billtime"], inplace=True)

    # Create context features
    df["prev_bet"] = df.groupby("loginname")["bet"].shift(1).fillna(0)
    df["prev_basepoint"] = df.groupby("loginname")["basepoint"].shift(1).fillna(df["basepoint"])
    df["prev_profit"] = df.groupby("loginname")["adjusted_profit"].shift(1).fillna(0)
    df["total_profit"] = df.groupby("loginname")["adjusted_profit"].cumsum()
    df["original_currency"] = df["currency"]

    # Simple currency mapping instead of one-hot encoding
    unique_currencies = df["currency"].unique()
    currency_map = {curr: i for i, curr in enumerate(unique_currencies)}
    df["currency_onehot"] = df["currency"].map(currency_map).astype("int8")

    # Calculate bet indices
    if bet_dict:
        df["bet_idx"] = df.apply(
            lambda row: find_bet_index(row["bet"], row["original_currency"], bet_dict), axis=1
        ).astype("int16")
    else:
        df["bet_idx"] = 0

    # Process delta_t (keeping original for reference)
    df["delta_t_original"] = df["delta_t"]
    df["delta_t"] = np.where(df["delta_t"] <= 7, 0, np.where(df["delta_t"] <= 50, 1, 2)).astype("int8")

    # State columns (REMOVED 'delta_bet')
    state_cols = [
        "slottype",
        "adjusted_profit",
        "basepoint",
        "delta_t_original",
        "delta_profit",
        "delta_payout",
        "prev_bet",
        "prev_basepoint",
        "prev_profit",
        "total_profit",
        "streak",
        "currency_onehot",
        "bet_idx",
        "delta_t",
    ]

    # Drop rows with missing values
    initial_size = len(df)
    df = df.dropna(subset=state_cols)
    final_size = len(df)

    print(f"Dropped {initial_size - final_size} rows with missing values")
    print(f"Final size: {final_size} rows")

    return df, state_cols


def get_flattened_trajectories(df, normalize=True, clip_range=(-3, 3), testsize=0.2):
    """
    Create flattened transitions directly instead of creating Trajectory objects
    with optional normalization of observations and clipping to handle extreme values.
    Modified to exclude 'delta_bet' from feature vectors.

    Rewards will be computed separately by a reward network - this function only
    prepares the transitions data.
    """
    # Create a list to hold all transitions
    all_obs = []
    all_acts = []
    all_infos = []
    all_next_obs = []
    all_dones = []  # flagger for the last transition

    # Placeholders for reward - will be filled by reward network later
    all_rews = []

    # If normalization is enabled, compute stats for normalization
    normalizers = {}
    if normalize:
        # Calculate means and standard deviations for numerical features
        # REMOVED 'delta_bet' from this list
        numeric_cols = [
            "adjusted_profit",
            "basepoint",
            "delta_profit",
            "delta_payout",
            "streak",
            "prev_bet",
            "prev_basepoint",
            "prev_profit",
            "total_profit",
        ]

        for col in numeric_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    std = 1.0  # Avoid division by zero
                normalizers[col] = (mean, std)

    # Helper function to normalize and clip a value
    def normalize_and_clip(value, feature_name):
        if normalize and feature_name in normalizers:
            mean, std = normalizers[feature_name]
            normalized_value = (value - mean) / std
            return np.clip(normalized_value, clip_range[0], clip_range[1])
        return value

    counter = 0
    skip = 0

    # Group data by player
    for player, player_data in df.groupby("loginname"):
        # Sort by time
        player_data = player_data.sort_values("billtime").copy()

        # Identify session boundaries
        session_breaks = player_data["delta_t_original"] > 50
        player_data["session_id"] = session_breaks.cumsum()

        # Process each session
        for session_id, session_data in player_data.groupby("session_id"):
            if len(session_data) < 3:
                skip += 1
                continue

            counter += 1
            # Calculate correct session length
            correct_length = len(session_data)

            # Process rows one by one
            for i in range(len(session_data) - 1):  # all but the last one
                current_row = session_data.iloc[i]
                next_row = session_data.iloc[i + 1]

                # Create current observation vector with normalization and clipping
                # REMOVED delta_bet from feature vector
                curr_features = []

                # bet_idx is discrete, we don't normalize it
                curr_features.append(current_row["bet_idx"])

                # Normalize and clip numeric features
                curr_features.append(normalize_and_clip(current_row["adjusted_profit"], "adjusted_profit"))

                # slottype is categorical
                curr_features.append(current_row["slottype"])

                # Add remaining features with normalization and clipping
                curr_features.append(normalize_and_clip(current_row["basepoint"], "basepoint"))
                # REMOVED: curr_features.append(normalize_and_clip(current_row['delta_bet'], 'delta_bet'))
                curr_features.append(normalize_and_clip(current_row["delta_profit"], "delta_profit"))
                curr_features.append(normalize_and_clip(current_row["delta_payout"], "delta_payout"))
                curr_features.append(normalize_and_clip(current_row["streak"], "streak"))

                # Remaining features
                curr_features.append(normalize_and_clip(current_row["prev_bet"], "prev_bet"))
                curr_features.append(normalize_and_clip(current_row["prev_basepoint"], "prev_basepoint"))
                curr_features.append(normalize_and_clip(current_row["prev_profit"], "prev_profit"))
                curr_features.append(normalize_and_clip(current_row["total_profit"], "total_profit"))

                # Create next observation vector with the same approach
                # REMOVED delta_bet from next features as well
                next_features = []
                next_features.append(next_row["bet_idx"])
                next_features.append(normalize_and_clip(next_row["adjusted_profit"], "adjusted_profit"))
                next_features.append(next_row["slottype"])
                next_features.append(normalize_and_clip(next_row["basepoint"], "basepoint"))
                # REMOVED: next_features.append(normalize_and_clip(next_row['delta_bet'], 'delta_bet'))
                next_features.append(normalize_and_clip(next_row["delta_profit"], "delta_profit"))
                next_features.append(normalize_and_clip(next_row["delta_payout"], "delta_payout"))
                next_features.append(normalize_and_clip(next_row["streak"], "streak"))

                next_features.append(normalize_and_clip(next_row["prev_bet"], "prev_bet"))
                next_features.append(normalize_and_clip(next_row["prev_basepoint"], "prev_basepoint"))
                next_features.append(normalize_and_clip(next_row["prev_profit"], "prev_profit"))
                next_features.append(normalize_and_clip(next_row["total_profit"], "total_profit"))

                # Store observations
                all_obs.append(np.array(curr_features, dtype=np.float32))
                all_next_obs.append(np.array(next_features, dtype=np.float32))

                # Get action (bet_idx and delta_t)
                bet_idx = next_row["bet_idx"]
                delta_t = 0

                # Encode action with terminate flag
                action_tuple = (bet_idx, np.array([0]))

                # Create wrapper for encoding
                wrapper = TupleActionWrapper(None)
                encoded_action = wrapper.reverse_action(action_tuple)

                # Store action, done flag
                all_acts.append(encoded_action)
                all_dones.append(False)  # we append the last entry all at once.

                # Create info dictionary with termination information
                info = {
                    "early_terminate": False,
                    "late_terminate": False,
                    "position_in_session": i + 1,
                    "correct_length": correct_length,
                    "length_difference": None,
                    "expert_action": bet_idx,
                }

                all_infos.append(info)
                all_rews.append(0.0)  # Placeholder value SINCE GAIL DOESN'T NEED THIS REWARD

            # ---------------------------------------------
            # Add a terminal state at the end.
            final_index = len(session_data) - 1
            current_row = session_data.iloc[final_index]

            # Create current observation vector with normalization and clipping
            # REMOVED delta_bet from terminal state as well
            curr_features = []

            # bet_idx is discrete, we don't normalize it
            curr_features.append(current_row["bet_idx"])

            # Normalize and clip numeric features
            curr_features.append(normalize_and_clip(current_row["adjusted_profit"], "adjusted_profit"))

            # slottype is categorical
            curr_features.append(current_row["slottype"])

            # Add remaining features with normalization and clipping
            curr_features.append(normalize_and_clip(current_row["basepoint"], "basepoint"))
            # REMOVED: curr_features.append(normalize_and_clip(current_row['delta_bet'], 'delta_bet'))
            curr_features.append(normalize_and_clip(current_row["delta_profit"], "delta_profit"))
            curr_features.append(normalize_and_clip(current_row["delta_payout"], "delta_payout"))
            curr_features.append(normalize_and_clip(current_row["streak"], "streak"))

            # Remaining features
            curr_features.append(normalize_and_clip(current_row["prev_bet"], "prev_bet"))
            curr_features.append(normalize_and_clip(current_row["prev_basepoint"], "prev_basepoint"))
            curr_features.append(normalize_and_clip(current_row["prev_profit"], "prev_profit"))
            curr_features.append(normalize_and_clip(current_row["total_profit"], "total_profit"))

            all_obs.append(np.array(curr_features, dtype=np.float32))
            all_next_obs.append(np.zeros_like(curr_features, dtype=np.float32))

            # Store action, done flag
            all_acts.append(0)
            all_dones.append(True)

            # Create info dictionary with termination information
            info = {
                "early_terminate": False,
                "late_terminate": False,
                "position_in_session": final_index + 1,
                "correct_length": correct_length,
                "length_difference": 0,
                "expert_action": 0,
            }
            all_infos.append(info)
            all_rews.append(0.0)

    print(
        f"DEBUG!! - length obs = {len(all_obs)}, next_obs = {len(all_next_obs)}, feature = {len(curr_features)}, counter = {counter}, skipped = {skip}"
    )

    # Train test split here and construct the train and test transitions.
    train_length = int(len(all_obs) * (1 - testsize))
    train_obs = all_obs[:train_length]
    train_next_obs = all_next_obs[:train_length]
    train_acts = all_acts[:train_length]
    train_dones = all_dones[:train_length]
    train_infos = all_infos[:train_length]
    train_rews = all_rews[:train_length]

    test_obs = all_obs[train_length:]
    test_next_obs = all_next_obs[train_length:]
    test_acts = all_acts[train_length:]
    test_dones = all_dones[train_length:]
    test_infos = all_infos[train_length:]
    test_rews = all_rews[train_length:]

    train_obs_array = np.array(train_obs, dtype=np.float32)
    train_next_obs_array = np.array(train_next_obs, dtype=np.float32)
    train_acts_array = np.array(train_acts, dtype=np.float32)
    train_dones_array = np.array(train_dones, dtype=np.bool_)
    train_rews_array = np.array(train_rews, dtype=np.float32)

    test_obs_array = np.array(test_obs, dtype=np.float32)
    test_next_obs_array = np.array(test_next_obs, dtype=np.float32)
    test_acts_array = np.array(test_acts, dtype=np.float32)
    test_dones_array = np.array(test_dones, dtype=np.bool_)
    test_rews_array = np.array(test_rews, dtype=np.float32)

    # Create transitions object
    train_transitions = TransitionsWithRew(
        obs=train_obs_array,
        acts=train_acts_array,
        infos=train_infos,
        next_obs=train_next_obs_array,
        dones=train_dones_array,
        rews=train_rews_array,
    )

    test_transitions = TransitionsWithRew(
        obs=test_obs_array,
        acts=test_acts_array,
        infos=test_infos,
        next_obs=test_next_obs_array,
        dones=test_dones_array,
        rews=test_rews_array,
    )

    # Store normalization parameters for later use
    if normalize:
        return train_transitions, test_transitions, normalizers, clip_range
    else:
        return train_transitions, test_transitions, None, None


def pretrain_with_behavior_cloning(
    policy,
    expert_transitions,
    venv,
    learning_rate=1e-4,
    batch_size=64,
    epochs=10,
    weight_decay=1e-5,
    save_path="pretrained_policy.pt",
    collect_transitions=True,
    num_transitions_to_collect=1000,
):
    """
    Pretrain policy with behavior cloning and optionally collect transitions.

    Parameters:
    -----------
    policy : PPO policy
        The policy to be pretrained
    expert_transitions : TransitionsWithRew
        Expert demonstrations
    venv : VecEnv
        Vectorized environment for collecting transitions
    learning_rate : float
        Learning rate for the optimizer
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs to train
    weight_decay : float
        L2 regularization coefficient
    save_path : str
        Path to save the pretrained model
    collect_transitions : bool
        Whether to collect transitions after pretraining
    num_transitions_to_collect : int
        Number of transitions to collect if collect_transitions is True

    Returns:
    --------
    tuple
        (losses, collected_transitions)
    """
    print("Starting behavior cloning pretraining...")

    # Get the device that the policy is on
    device = next(policy.parameters()).device
    print(f"Policy is on device: {device}")

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Track losses
    losses = []

    # Access the expert data and move to the same device as the policy
    expert_obs = torch.tensor(expert_transitions.obs, dtype=torch.float32).to(device)
    expert_acts = torch.tensor(expert_transitions.acts, dtype=torch.long).to(device)

    # Get dataset size
    dataset_size = len(expert_obs)

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []

        # Generate random indices for batching
        indices = np.random.permutation(dataset_size)

        # Batch training
        for start_idx in range(0, dataset_size, batch_size):
            # Get batch indices
            batch_indices = indices[start_idx : start_idx + batch_size]

            # Get batch data (already on the correct device)
            obs_batch = expert_obs[batch_indices]
            act_batch = expert_acts[batch_indices]

            # Forward pass through policy
            policy.set_training_mode(True)

            try:
                # First, try getting the distribution and accessing logits
                action_dist = policy.get_distribution(obs_batch)
                # Check if action_dist has logits attribute
                if hasattr(action_dist, "logits"):
                    logits = action_dist.logits
                # If it has probs but not logits
                elif hasattr(action_dist, "distribution") and hasattr(action_dist.distribution, "probs"):
                    logits = torch.log(action_dist.distribution.probs)
                # Fall back to using the raw tensor if it's directly returned
                else:
                    logits = action_dist
            except Exception as e:
                print(f"Error trying to get action distribution: {e}")
                # Option 2: Try direct forward pass through policy
                try:
                    logits = policy.forward(obs_batch)
                except Exception as e2:
                    print(f"Error trying direct forward: {e2}")
                    # Last resort: use _predict but process the output
                    actions = policy._predict(obs_batch, deterministic=False)
                    if isinstance(actions, torch.Tensor):
                        # If it's a tensor, assume these are the direct actions
                        # Convert to one-hot and use as pseudo-logits
                        num_actions = policy.action_space.n
                        logits = F.one_hot(actions.long(), num_actions).float()
                    else:
                        # If it's a distribution object
                        logits = actions

            # Calculate cross-entropy loss
            # If logits are one-hot encoded, use MSE loss instead
            if logits.shape == act_batch.shape:
                loss = F.mse_loss(logits.float(), F.one_hot(act_batch, logits.shape[1]).float())
            else:
                loss = F.cross_entropy(logits, act_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_losses.append(loss.item())

        # Calculate average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on a few examples
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                sample_indices = np.random.choice(dataset_size, size=5)
                sample_obs = expert_obs[sample_indices]
                sample_acts = expert_acts[sample_indices]

                # Get predictions
                try:
                    actions, _ = policy.predict(sample_obs)
                except Exception as e:
                    print(f"Error with policy.predict: {e}")
                    # Fallback to forward pass and argmax
                    with torch.no_grad():
                        try:
                            logits = policy.forward(sample_obs)
                            actions = torch.argmax(logits, dim=1).cpu().numpy()
                        except Exception as e2:
                            print(f"Error with forward pass: {e2}")
                            output = policy._predict(sample_obs, deterministic=True)
                            if isinstance(output, torch.Tensor):
                                actions = output.cpu().numpy()
                            else:
                                try:
                                    actions = output.mode().cpu().numpy()
                                except:
                                    print("Could not extract actions from model output")
                                    continue  # Skip evaluation if we can't get actions

                print("Sample Predictions:")
                for i in range(len(sample_indices)):
                    try:
                        wrapper = TupleActionWrapper(None)
                        expert_bet_idx, expert_delta_t = wrapper.action(sample_acts[i].item())
                        pred_bet_idx, pred_delta_t = wrapper.action(actions[i])
                        print(f"  Expert: action={sample_acts[i].item()} (bet={expert_bet_idx}, dt={expert_delta_t})")
                        print(f"  Pred:   action={actions[i]} (bet={pred_bet_idx}, dt={pred_delta_t})")
                    except Exception as e:
                        print(f"  Error decoding action {i}: {e}")
                    print()

    # Save the pretrained model
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs": epochs,
            "final_loss": losses[-1],
        },
        save_path,
    )
    print(f"Pretrained model saved to {save_path}")

    # Collect transitions if requested
    collected_transitions = None
    if collect_transitions:
        print("Collecting transitions from behavior cloning model...")
        collected_transitions = collect_transitions_from_policy(
            policy=policy, venv=venv, expert_transitions=expert_transitions, num_transitions=num_transitions_to_collect
        )
        print(f"Collected {len(collected_transitions)} transitions")

    print("Behavior cloning pretraining completed!")
    return losses, collected_transitions


def collect_transitions_from_policy(policy, venv, expert_transitions, num_transitions=1000):
    """
    Collect transitions by running the policy in the environment.
    Also includes additional information about how the policy's actions
    compare to expert actions for prioritization purposes.

    Parameters:
    -----------
    policy : PPO policy
        The policy to collect transitions from
    venv : VecEnv
        Vectorized environment
    expert_transitions : TransitionsWithRew
        Expert demonstrations for comparison
    num_transitions : int
        Number of transitions to collect

    Returns:
    --------
    list
        of transitions
    """
    # Create action wrapper for decoding
    wrapper = TupleActionWrapper(None)

    # Create a dictionary mapping observations to expert actions for comparison
    expert_obs_to_act = {}
    for i in range(len(expert_transitions.obs)):
        # Use a string representation of the observation as key
        obs_key = str(expert_transitions.obs[i].tolist())
        expert_obs_to_act[obs_key] = expert_transitions.acts[i]

    # Storage for collected transitions
    collected_transitions = []
    transitions_per_env = num_transitions // venv.num_envs

    # Reset environment
    obs = venv.reset()

    for step in range(transitions_per_env):
        # Get actions from policy
        actions, _ = policy.predict(obs, deterministic=True)

        # Step the environment
        next_obs, rewards, dones, infos = venv.step(actions)

        # Store transitions with additional info
        for i in range(venv.num_envs):
            # Decode policy action
            bet_idx, delta_t = wrapper.action(actions[i])

            # Create info dictionary
            info = infos[i].copy() if isinstance(infos, list) else infos.copy()

            # Add policy action to info
            info["action"] = actions[i]

            # Check if this observation matches any expert observations
            obs_key = str(obs[i].tolist())
            if obs_key in expert_obs_to_act:
                # Found a match - add expert action to info
                expert_action = expert_obs_to_act[obs_key]
                info["expert_action"] = expert_action

                # Decode expert action
                expert_bet_idx, expert_delta_t = wrapper.action(expert_action)

                # Add comparison info
                info["action_match"] = actions[i] == expert_action
                info["bet_match"] = bet_idx == expert_bet_idx
                info["delta_t_match"] = np.isclose(delta_t, expert_delta_t, rtol=0.1, atol=0.1)

                # Add termination info
                info["early_terminate"] = bet_idx == 0 and expert_bet_idx != 0
                info["late_terminate"] = bet_idx != 0 and expert_bet_idx == 0

            # Create transition tuple
            transition = (obs[i].copy(), actions[i], next_obs[i].copy(), dones[i], info)
            collected_transitions.append(transition)

            # Print progress
            if len(collected_transitions) % 100 == 0:
                print(f"Collected {len(collected_transitions)}/{num_transitions} transitions")

            # Break if we have enough transitions
            if len(collected_transitions) >= num_transitions:
                return collected_transitions

        # Update observations
        obs = next_obs

    return collected_transitions


def load_pretrained_model(policy, path="pretrained_policy.pt", eval_mode=False):
    """
    Load a pretrained policy model.

    Parameters:
    -----------
    policy : PPO policy
        The policy to load the weights into
    path : str
        Path to the saved model
    eval_mode : bool
        Whether to set the policy to evaluation mode

    Returns:
    --------
    policy : PPO policy
        The loaded policy
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pretrained model found at {path}")

    device = next(policy.parameters()).device
    checkpoint = torch.load(path, map_location=device)

    policy.load_state_dict(checkpoint["policy_state_dict"])

    if eval_mode:
        policy.set_training_mode(False)

    print(f"Successfully loaded pretrained model from {path}")
    print(f"Training info - Epochs: {checkpoint['epochs']}, Final Loss: {checkpoint['final_loss']:.4f}")

    return policy


def save_gail_model(gail, path="gail_model.zip"):
    """
    Save the GAIL model with prioritized replay buffer and its components.

    Parameters:
    -----------
    gail : GAILWithPrioritizedTermination or GAIL
        The GAIL model to save (can be either basic GAIL or with prioritized replay)
    path : str
        Path to save the model (should have .zip extension)
    """
    # Create temporary directory for saving components
    temp_dir = tempfile.mkdtemp(prefix="gail_save_")

    try:
        # Check if we're dealing with a GAILWithPrioritizedTermination or basic GAIL

        base_gail = gail
        if hasattr(base_gail, "gail_agent"):
            base_gail = base_gail.gail_agent
            is_prioritized = True
        else:
            # This is a regular GAIL
            base_gail = gail
            is_prioritized = False

        # Save the PPO agent
        agent_path = os.path.join(temp_dir, "ppo_agent.zip")
        base_gail.gen_algo.save(agent_path)

        # Save the reward network (discriminator)
        reward_net_path = os.path.join(temp_dir, "reward_net.pt")

        # Check if reward_train is wrapped in a module
        if hasattr(base_gail.reward_train, "module"):
            reward_state_dict = base_gail.reward_train.module.state_dict()
        else:
            reward_state_dict = base_gail.reward_train.state_dict()

        torch.save(reward_state_dict, reward_net_path)

        # Save discriminator optimizer state
        # disc_optim_path = os.path.join(temp_dir, "disc_optim.pt")
        # torch.save(base_gail.disc_opt.state_dict(), disc_optim_path)

        # Determine the reward network class
        reward_net_class_name = base_gail.reward_train.__class__.__name__

        # Save GAIL configuration and training state
        config = {
            "demo_batch_size": base_gail.demo_batch_size,
            "n_disc_updates_per_round": base_gail.n_disc_updates_per_round,
            "n_gen_updates_per_round": getattr(base_gail, "n_gen_updates_per_round", 1),
            "n_updates": getattr(base_gail, "n_updates", 0),
            "allow_variable_horizon": base_gail.allow_variable_horizon,
            "is_prioritized_replay": is_prioritized,
            # Store reward network configuration
            "reward_net_config": {
                "class_name": reward_net_class_name,
                "r1_gamma": getattr(base_gail.reward_train, "r1_gamma", None),
                "early_termination_penalty": getattr(base_gail.reward_train, "early_termination_penalty", 5.0),
                "late_termination_penalty": getattr(base_gail.reward_train, "late_termination_penalty", 3.0),
                "bet_correct_bonus": getattr(base_gail.reward_train, "bet_correct_bonus", 1.0),
                "bet_wrong_penalty": getattr(base_gail.reward_train, "bet_wrong_penalty", 1.0),
                "bet_diversity_factor": getattr(base_gail.reward_train, "bet_diversity_factor", 0.1),
            },
        }

        # If using prioritized replay, save those parameters too
        if is_prioritized:
            # Handle direct attributes from GAILWithPrioritizedTermination
            prioritized_attrs = [
                "buffer_capacity",
                "alpha",
                "beta",
                "term_priority_factor",
                "bet_priority_factor",
                "initial_gen_update_freq",
                "final_gen_update_freq",
                "gen_freq_annealing_steps",
                "current_step",
            ]

            config["prioritized_replay_config"] = {}
            for attr in prioritized_attrs:
                if hasattr(gail, attr):
                    config["prioritized_replay_config"][attr] = getattr(gail, attr)

            # Save replay buffer state
            buffer_path = os.path.join(temp_dir, "replay_buffer.pkl")
            buffer_state = {
                "experiences": gail.experiences if hasattr(gail, "experiences") else [],
                "priorities": gail.priorities if hasattr(gail, "priorities") else [],
                "pos": gail.pos if hasattr(gail, "pos") else 0,
                "full": gail.full if hasattr(gail, "full") else False,
            }
            with open(buffer_path, "wb") as f:
                pickle.dump(buffer_state, f)

        # Save the configuration
        config_path = os.path.join(temp_dir, "gail_config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

        # Create a zip file containing all components
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        print(f"GAIL model successfully saved to {path}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def load_gail_model(path, venv=None, expert_transitions=None):
    """
    Load a complete GAIL model, including the PPO agent, reward network, and prioritized replay if available.

    Parameters:
    -----------
    path : str
        Path to the saved GAIL model (should be a .zip file)
    venv : VecEnv or None
        Vectorized environment (if None, will be loaded from the saved model)
    expert_transitions : TransitionsWithRew or None
        Expert demonstrations (if None and required, will raise an error)

    Returns:
    --------
    tuple
        (gail, agent) - The GAIL model (either GAIL or GAILWithPrioritizedTermination) and the PPO agent
    """

    # Create temporary directory for extracting components
    temp_dir = tempfile.mkdtemp(prefix="gail_load_")

    try:
        # Extract all files from the zip
        with zipfile.ZipFile(path, "r") as zipf:
            zipf.extractall(temp_dir)

        # Load GAIL configuration
        config_path = os.path.join(temp_dir, "gail_config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        # Load the PPO agent
        agent_path = os.path.join(temp_dir, "ppo_agent.zip")
        agent = PPO.load(agent_path, env=venv)

        # If venv is None, use the environment from the loaded model
        if venv is None:
            venv = agent.get_env()

        # Get the reward network configuration
        reward_net_config = config.get("reward_net_config", {})
        reward_class_name = reward_net_config.get("class_name", "")

        # Determine which reward network to create based on the class name
        if reward_class_name == "RewardWithR1":
            reward_net = RewardWithR1(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                venv=venv,
                r1_gamma=reward_net_config.get("r1_gamma", 10.0),
            )
        else:
            # Fallback to a default reward network or raise an error
            raise ValueError(f"Unsupported reward network class: {reward_class_name}")

        # Load reward network state
        reward_net_path = os.path.join(temp_dir, "reward_net.pt")
        reward_state_dict = torch.load(
            reward_net_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if hasattr(reward_net, "module"):
            reward_net.module.load_state_dict(reward_state_dict)
        else:
            reward_net.load_state_dict(reward_state_dict)

        # Create base GAIL instance
        base_gail = GAIL(
            demonstrations=expert_transitions,
            demo_batch_size=config.get("demo_batch_size", 128),
            gen_algo=agent,
            reward_net=reward_net,
            venv=venv,
            n_disc_updates_per_round=config.get("n_disc_updates_per_round", 4),
            allow_variable_horizon=config.get("allow_variable_horizon", True),
        )

        # Set the number of updates (training progress)
        if "n_updates" in config:
            base_gail.n_updates = config["n_updates"]

        # Check if this is a prioritized replay GAIL
        is_prioritized = config.get("is_prioritized_replay", False)

        if is_prioritized:
            # Create GAILWithPrioritizedTermination using parameters from SlotPlayer_Simulator
            buffer_capacity = 50000  # Default from SlotPlayer_Simulator
            alpha = 0.6  # Default alpha from SlotPlayer_Simulator
            beta = 0.4  # Default beta from SlotPlayer_Simulator
            term_priority_factor = 5  # Default from SlotPlayer_Simulator
            bet_priority_factor = 2  # Default from SlotPlayer_Simulator
            initial_gen_update_freq = 1  # Default from SlotPlayer_Simulator
            final_gen_update_freq = 1  # Default from SlotPlayer_Simulator
            gen_freq_annealing_steps = 100  # Default from SlotPlayer_Simulator

            # Override with saved values if available
            prioritized_config = config.get("prioritized_replay_config", {})
            if prioritized_config:
                buffer_capacity = prioritized_config.get("buffer_capacity", buffer_capacity)
                alpha = prioritized_config.get("alpha", alpha)
                beta = prioritized_config.get("beta", beta)
                term_priority_factor = prioritized_config.get("term_priority_factor", term_priority_factor)
                bet_priority_factor = prioritized_config.get("bet_priority_factor", bet_priority_factor)
                initial_gen_update_freq = prioritized_config.get("initial_gen_update_freq", initial_gen_update_freq)
                final_gen_update_freq = prioritized_config.get("final_gen_update_freq", final_gen_update_freq)
                gen_freq_annealing_steps = prioritized_config.get("gen_freq_annealing_steps", gen_freq_annealing_steps)

            # Create GAILWithPrioritizedTermination
            gail = GAILWithPrioritizedTermination(
                gail_agent=base_gail,
                buffer_capacity=buffer_capacity,
                alpha=alpha,
                beta_start=beta,
                initial_transitions=None,  # Will be populated with buffer state if available
                term_priority_factor=term_priority_factor,
                bet_priority_factor=bet_priority_factor,
                initial_gen_update_freq=initial_gen_update_freq,
                final_gen_update_freq=final_gen_update_freq,
                gen_freq_annealing_steps=gen_freq_annealing_steps,
                expert_transitions=expert_transitions,
            )

            # Set the current step (if available)
            if "current_step" in prioritized_config:
                gail.current_step = prioritized_config["current_step"]

            # Load replay buffer state if available
            buffer_path = os.path.join(temp_dir, "replay_buffer.pkl")
            if os.path.exists(buffer_path):
                with open(buffer_path, "rb") as f:
                    buffer_state = pickle.load(f)

                # Restore buffer state
                gail.experiences = buffer_state.get("experiences", [])
                gail.priorities = buffer_state.get("priorities", [])
                gail.pos = buffer_state.get("pos", 0)
                gail.full = buffer_state.get("full", False)

            print(f"Successfully loaded GAIL model with prioritized replay from {path}")

            return gail, agent
        else:
            print(f"Successfully loaded basic GAIL model from {path}")
            return base_gail, agent

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def verify(
    agent, test_expert_transitions, output_csv_path="model_verification_results.csv", num_samples=500000, verbose=1
):
    """
    Verify model performance against test expert transitions and save results to CSV.

    Parameters:
    -----------
    agent : PPO
        The trained PPO agent to evaluate
    test_expert_transitions : TransitionsWithRew
        Test expert transitions to compare against
    output_csv_path : str
        Path to save the CSV results
    num_samples : int
        Number of samples to evaluate (default: 500000)
    verbose : int
        Verbosity level

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the verification results
    """

    if verbose > 0:
        print("\n===== FINAL MODEL VERIFICATION =====")

    # Prepare results storage
    results = []

    # Sample indices from test set
    if len(test_expert_transitions.obs) > num_samples:
        sample_indices = np.random.choice(len(test_expert_transitions.obs), size=num_samples, replace=False)
    else:
        # If we have fewer samples than requested, use all of them
        sample_indices = np.arange(len(test_expert_transitions.obs))
        if verbose > 0:
            print(f"Using all {len(sample_indices)} available test samples (fewer than requested {num_samples})")

    # Define delta_t ranges for display
    delta_t_ranges = ["0-7s", "7-50s", "50+s"]

    # Create TupleActionWrapper for decoding actions
    wrapper = TupleActionWrapper(None)

    # Track accuracy metrics
    correct_bet_count = 0
    correct_delta_t_count = 0
    total_count = 0

    # Add tracking for termination accuracy (bet = 0)
    termination_correct_count = 0
    expert_termination_count = 0

    # Process each sample
    for i, idx in enumerate(sample_indices):
        obs = test_expert_transitions.obs[idx]
        expert_action = test_expert_transitions.acts[idx]

        # Get model prediction
        model_action, _ = agent.predict(obs[np.newaxis, :], deterministic=True)

        # For expert action
        expert_action_int = int(expert_action)
        expert_bet_idx, expert_delta_t_cat = wrapper.action(expert_action_int)
        expert_delta_t = expert_delta_t_cat[0]

        # For model action
        model_action_int = int(model_action[0])
        model_bet_idx, model_delta_t_cat = wrapper.action(model_action_int)
        model_delta_t = model_delta_t_cat[0]

        # Map category back to original delta_t range for display
        expert_delta_t_range = delta_t_ranges[int(expert_delta_t)]
        model_delta_t_range = delta_t_ranges[int(model_delta_t)]

        # Check if predictions match
        bet_correct = expert_bet_idx == model_bet_idx
        delta_t_correct = expert_delta_t == model_delta_t

        # Track termination accuracy (bet = 0)
        is_expert_termination = expert_bet_idx == 0
        if is_expert_termination:
            expert_termination_count += 1
            if bet_correct:  # Model correctly predicted termination
                termination_correct_count += 1

        # Update accuracy counters
        if bet_correct:
            correct_bet_count += 1
        if delta_t_correct:
            correct_delta_t_count += 1
        total_count += 1

        # Store results
        results.append(
            {
                "sample_idx": idx,
                "expert_bet_idx": expert_bet_idx,
                "model_bet_idx": model_bet_idx,
                "expert_delta_t": expert_delta_t,
                "model_delta_t": model_delta_t,
                "expert_delta_t_range": expert_delta_t_range,
                "model_delta_t_range": model_delta_t_range,
                "bet_correct": bet_correct,
                "delta_t_correct": delta_t_correct,
                "is_termination": is_expert_termination,
            }
        )

        # Print results if verbose
        if verbose > 1 and i < 10:  # Print only first 10 samples in detail to avoid clutter
            print(f"Verification sample {i + 1}:")
            print(f"Expert: Bet idx = {expert_bet_idx}, Delta_t category = {expert_delta_t} ({expert_delta_t_range})")
            print(f"Model:  Bet idx = {model_bet_idx}, Delta_t category = {model_delta_t} ({model_delta_t_range})")
            print(f"Correct: Bet = {bet_correct}, Delta_t = {delta_t_correct}")
            if is_expert_termination:
                print(
                    f"This is a termination case (bet = 0): {'Correctly predicted' if bet_correct else 'Incorrectly predicted'}"
                )
            print()

    # Calculate overall accuracy
    bet_accuracy = correct_bet_count / total_count if total_count > 0 else 0
    delta_t_accuracy = correct_delta_t_count / total_count if total_count > 0 else 0
    combined_accuracy = (
        sum(1 for r in results if r["bet_correct"] and r["delta_t_correct"]) / total_count if total_count > 0 else 0
    )

    # Calculate termination accuracy
    termination_accuracy = termination_correct_count / expert_termination_count if expert_termination_count > 0 else 0

    # Print summary
    if verbose > 0:
        print("\n===== VERIFICATION SUMMARY =====")
        print(f"Total samples evaluated: {total_count}")
        print(f"Bet accuracy: {bet_accuracy:.4f} ({correct_bet_count}/{total_count})")
        print(f"Delta_t accuracy: {delta_t_accuracy:.4f} ({correct_delta_t_count}/{total_count})")
        print(f"Combined accuracy (both correct): {combined_accuracy:.4f}")
        print(
            f"Termination accuracy (bet = 0): {termination_accuracy:.4f} ({termination_correct_count}/{expert_termination_count})"
        )
        print(
            f"Proportion of termination cases: {expert_termination_count / total_count:.4f} ({expert_termination_count}/{total_count})"
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    if verbose > 0:
        print(f"Verification results saved to {output_csv_path}")

    # Add summary row
    summary = pd.DataFrame(
        [
            {
                "sample_idx": "SUMMARY",
                "expert_bet_idx": None,
                "model_bet_idx": None,
                "expert_delta_t": None,
                "model_delta_t": None,
                "expert_delta_t_range": None,
                "model_delta_t_range": None,
                "bet_correct": bet_accuracy,
                "delta_t_correct": delta_t_accuracy,
                "termination_accuracy": termination_accuracy,
                "termination_count": expert_termination_count,
                "termination_proportion": expert_termination_count / total_count if total_count > 0 else 0,
            }
        ]
    )

    # Save summary CSV
    summary_path = os.path.splitext(output_csv_path)[0] + "_summary.csv"
    summary.to_csv(summary_path, index=False)
    if verbose > 0:
        print(f"Summary results saved to {summary_path}")

    return results_df


class GAILWithPrioritizedTermination:
    """
    A wrapper around GAIL that adds termination-aware prioritized replay
    by leveraging GAIL's existing train_disc and train_gen methods.
    """

    def __init__(
        self,
        gail_agent,
        buffer_capacity=10000,
        alpha=0.6,
        beta_start=0.4,
        initial_transitions=None,
        initial_gen_update_freq=0.2,
        final_gen_update_freq=1.0,
        gen_freq_annealing_steps=5000,
        term_priority_factor=5,
        bet_priority_factor=2,
        expert_transitions=None,
    ):
        """
        Initialize the wrapper.

        Parameters:
        -----------
        gail_agent : GAIL
            The original GAIL agent from imitation library
        buffer_capacity : int
            Maximum size of the prioritized buffer
        alpha : float
            How much prioritization to use
        beta_start : float
            Starting value for importance sampling correction
        initial_transitions : list
            List of transitions from behavior cloning
        initial_gen_update_freq : float
            Initial frequency of generator updates
        final_gen_update_freq : float
            Final frequency of generator updates
        gen_freq_annealing_steps : int
            Number of steps to anneal frequency
        term_priority_factor : float
            Priority multiplier for termination issues
        bet_priority_factor : float
            Priority multiplier for incorrect bets
        expert_transitions : TransitionsWithRew
            Expert demonstrations
        """
        self.gail_agent = gail_agent

        # Create prioritized buffer
        self.buffer = PrioritizedBuffer(
            capacity=buffer_capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_steps=buffer_capacity,
            term_priority_factor=term_priority_factor,
            bet_priority_factor=bet_priority_factor,
        )

        # Store basic parameters
        self.demo_batch_size = gail_agent.demo_batch_size

        # Get components from GAIL
        self.gen_algo = gail_agent.gen_algo
        self.reward_net = gail_agent.reward_train
        self.venv = gail_agent.venv

        # Store expert transitions
        self.demonstrations = expert_transitions

        # Generator update frequency parameters
        self.initial_gen_update_freq = initial_gen_update_freq
        self.final_gen_update_freq = final_gen_update_freq
        self.gen_freq_annealing_steps = gen_freq_annealing_steps
        self.current_gen_update_freq = initial_gen_update_freq
        self.freq_increment = (final_gen_update_freq - initial_gen_update_freq) / gen_freq_annealing_steps
        self.steps_taken = 0

        # Initialize buffer with expert demonstrations if provided
        if self.demonstrations is not None:
            self._initialize_buffer_with_demos()

        # Handle initial transitions from behavior cloning
        if initial_transitions is not None:
            print(f"Adding {len(initial_transitions)} transitions from behavior cloning")
            for transition in initial_transitions:
                # Expect (obs, act, next_obs, done, info) format
                if len(transition) == 5:
                    obs, act, next_obs, done, info = transition

                    # Make sure action is in info for prioritization
                    if "action" not in info:
                        info["action"] = act

                    # Add to buffer
                    self.buffer.add((obs, act, next_obs, done), info)
                else:
                    print(f"WARNING: Unexpected BC transition format with {len(transition)} elements")
                    # Try to add anyway
                    self.buffer.add(transition)

    def _initialize_buffer_with_demos(self):
        """
        Add expert demonstrations to the buffer with high priority.
        """
        if self.demonstrations is not None:
            print(f"Adding {len(self.demonstrations.obs)} expert demonstrations to buffer")

            # Process expert demonstrations
            for i in range(len(self.demonstrations.obs)):
                # Create transition components
                obs = self.demonstrations.obs[i]
                act = self.demonstrations.acts[i]
                next_obs = self.demonstrations.next_obs[i] if hasattr(self.demonstrations, "next_obs") else None
                done = self.demonstrations.dones[i] if hasattr(self.demonstrations, "dones") else False

                # Create info dictionary
                info = {}
                if hasattr(self.demonstrations, "infos") and self.demonstrations.infos:
                    info = self.demonstrations.infos[i]

                # Add expert action to info for prioritization
                info["expert_action"] = act

                # Add to buffer
                self.buffer.add((obs, act, next_obs, done), info)
        else:
            print("WARNING: No expert demonstrations available for buffer initialization")

    def train_step(self):
        """
        Perform one training step with prioritized replay.
        Uses GAIL's built-in train_disc and train_gen methods.
        """
        # 1. Collect transitions from generator policy
        obs_list, acts_list, next_obs_list, dones_list, infos_list = self.collect_transitions()

        # 2. Add transitions to prioritized buffer
        for i in range(len(obs_list)):
            transition = (obs_list[i], acts_list[i], next_obs_list[i], dones_list[i], infos_list[i])
            self.buffer.add(transition, infos_list[i])

        # 3. Sample from buffer for discriminator training
        for _ in range(self.gail_agent.n_disc_updates_per_round):
            # Sample from buffer
            gen_samples, indices, weights = self.buffer.sample(self.demo_batch_size)

            # Extract components
            gen_obs = np.stack([t[0] for t in gen_samples])
            gen_acts = np.stack([t[1] for t in gen_samples])

            gen_next_obs = np.stack([t[2] if t[2] is not None else np.zeros_like(t[0]) for t in gen_samples])
            gen_dones = np.stack([t[3] for t in gen_samples])

            # Create gen_samples dictionary for train_disc
            gen_samples_dict = {"obs": gen_obs, "acts": gen_acts, "next_obs": gen_next_obs, "dones": gen_dones}

            # Use GAIL's train_disc method
            disc_stats = self.gail_agent.train_disc(gen_samples=gen_samples_dict)

            # Get loss for each sample to update priorities
            device = next(self.reward_net.parameters()).device
            gen_obs_tensor = torch.as_tensor(gen_obs, device=device)
            gen_acts_tensor = torch.as_tensor(gen_acts, device=device)
            gen_next_obs_tensor = torch.as_tensor(gen_next_obs, device=device)
            gen_dones_tensor = torch.as_tensor(gen_dones, device=device)

            # Get logit to compute per-sample losses
            with torch.no_grad():
                gen_logits = self.reward_net(gen_obs_tensor, gen_acts_tensor, gen_next_obs_tensor, gen_dones_tensor)
                gen_labels = torch.zeros_like(gen_logits)
                per_sample_losses = (
                    F.binary_cross_entropy_with_logits(gen_logits, gen_labels, reduction="none").cpu().numpy()
                )

            # Update buffer priorities
            self.buffer.update_priorities(indices, per_sample_losses)

        # 4. Update generator
        should_update_generator = np.random.random() < self.current_gen_update_freq
        if should_update_generator:
            # Use GAIL's train_gen method
            self.gail_agent.train_gen(total_timesteps=self.gen_algo.n_steps)

        self.steps_taken += 1
        if self.steps_taken < self.gen_freq_annealing_steps:
            self.current_gen_update_freq = min(
                self.final_gen_update_freq, self.initial_gen_update_freq + self.freq_increment * self.steps_taken
            )

    def collect_transitions(self):
        """
        Collect transitions using the generator policy.
        """
        # Reference the environment
        venv = self.venv
        gen_policy = self.gen_algo.policy

        # Reset environment
        obs = venv.reset()
        dones = np.zeros(venv.num_envs, dtype=bool)

        # Storage
        collected_obs = []
        collected_acts = []
        collected_next_obs = []
        collected_dones = []
        collected_infos = []

        # Collect transitions
        for _ in range(self.gen_algo.n_steps):
            # Get actions
            acts, _ = gen_policy.predict(obs, deterministic=False)

            # Step the environment
            next_obs, _, dones, infos = venv.step(acts)

            # Store transitions
            for i in range(venv.num_envs):
                collected_obs.append(obs[i].copy())
                collected_acts.append(acts[i].copy())
                collected_next_obs.append(next_obs[i].copy())
                collected_dones.append(dones[i])

                # Create info with additional data
                info = infos[i].copy() if isinstance(infos, list) else infos.copy()
                info["action"] = acts[i]
                collected_infos.append(info)

            # Update observation
            obs = next_obs

        return collected_obs, collected_acts, collected_next_obs, collected_dones, collected_infos

    def train(self, total_steps, callback=None):
        """
        Train the agent for a specified number of steps.
        """
        steps_taken = 0

        # Training loop
        while steps_taken < total_steps:
            # Perform one training step
            self.train_step()

            # Update steps count
            steps_taken += self.gen_algo.n_steps

            # Call callback if provided
            if callback is not None:
                callback(steps_taken)

            # Print progress occasionally
            if steps_taken % (total_steps // 10) == 0:
                print(f"Training progress: {steps_taken}/{total_steps} steps")
                print(f"Current generator update frequency: {self.current_gen_update_freq:.3f}")

        return self.gen_algo
