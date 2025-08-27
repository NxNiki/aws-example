import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from imitation.data.types import TransitionsWithRew
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def load_and_process_data(csv_path, bet_dict=bet_dictionary):
    """
    Load and process slot machine data from CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV data file
    bet_dict : dict
        Dictionary of bet amounts by currency

    Returns:
    --------
    df : DataFrame
        Processed DataFrame with all required features
    state_cols : list
        List of columns used for the state representation
    """
    # Load and sort data
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["loginname", "billtime"]).reset_index(drop=True)

    # --- Handle free spins ---

    def assign_and_remove_free_spins(df):
        df = df.copy()
        df["free_spin_total"] = 0
        rows_to_drop = []

        for player, group in df.groupby("loginname"):
            group = group.sort_values("billtime").reset_index()
            i = 0
            while i < len(group):
                if group.loc[i, "slottype"] == 2:
                    # Start of a free spin block
                    start = i
                    while i < len(group) and group.loc[i, "slottype"] == 2:
                        i += 1
                    end = i  # Exclusive

                    # Look back for the triggering spin
                    trigger_idx = start - 1
                    while trigger_idx >= 0 and group.loc[trigger_idx, "slottype"] != 1:
                        trigger_idx -= 1

                    if trigger_idx >= 0:
                        bonus_total = group.loc[start:end, "cus_account"].sum()
                        trigger_row_idx = group.loc[trigger_idx, "index"]
                        df.at[trigger_row_idx, "free_spin_total"] = bonus_total
                        rows_to_drop.extend(group.loc[start : end - 1, "index"].tolist())
                else:
                    i += 1

        df = df.drop(index=rows_to_drop).reset_index(drop=True)
        return df

    df = assign_and_remove_free_spins(df)

    # Rename columns
    df = df.rename(columns={"account": "bet", "cus_account": "profit"})

    # Adjust profit by adding free spin total
    df["adjusted_profit"] = df["profit"] + df["free_spin_total"]

    # Calculate context features
    df["prev_bet"] = df.groupby("loginname")["bet"].shift(1).fillna(0)
    df["prev_basepoint"] = df.groupby("loginname")["basepoint"].shift(1).fillna(df["basepoint"])
    df["prev_profit"] = df.groupby("loginname")["adjusted_profit"].shift(1).fillna(0)
    df["total_profit"] = df.groupby("loginname")["adjusted_profit"].cumsum()

    df["win_streak"] = np.where(df["win_streak"] > 0, df["win_streak"], -df["lose_streak"])
    df = df.drop("lose_streak", axis=1)
    # Store original currency
    df["original_currency"] = df["currency"].copy()

    # One-hot encode currency
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    currency_ohe = enc.fit_transform(df[["currency"]])

    # Store one-hot encoded values as lists in a single column
    df["currency_onehot"] = [row.tolist() for row in currency_ohe]

    # Calculate bet indices (action discrete index) for each row
    # df['bet'] = 1  # can be hard-coded to 1 since in this part of the expert data there is no "termination"

    # Store original delta_t values
    df["delta_t_original"] = df["delta_t"].copy()

    # State columns for modeling right now the handling of bet amount and idx is a little weird...
    # consider making delta bet difference between index too.
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
        "bet",
        "win_streak",
    ]

    # Drop rows with missing state values
    df = df.dropna(subset=state_cols)
    return df, state_cols


def get_flattened_trajectories(df, testsize=0.2):
    """
    Create flattened transitions directly instead of creating Trajectory objects
    with normalization applied to the observation fields.

    Observation fields (flattened):
    - current_profit (1,) -> adjusted_profit
    - total_profit (1,)
    - streak (1,)
    - win_streak (1,) -> already fixed in preprocessing (positive wins, negative losses)
    - prev_bet (1,)
    - basepoint (1,)
    - prev_profit (1,)

    Total flattened observation dimension: 7 (removed current_bet)

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

                # Create current observation vector (flattened, no normalization)
                curr_features = []

                # current_profit (adjusted_profit)
                curr_features.append(float(current_row["adjusted_profit"]))

                # total_profit
                curr_features.append(float(current_row["total_profit"]))

                # REMOVED: current_bet (bet amount)
                curr_features.append(float(current_row["bet"]))

                # streak
                curr_features.append(float(current_row["streak"]))

                # win_streak (already fixed in preprocessing - single column with pos/neg values)
                curr_features.append(float(current_row["win_streak"]))

                # prev_bet
                curr_features.append(float(current_row["prev_bet"]))

                # basepoint (current balance)
                curr_features.append(float(current_row["basepoint"]))

                # prev_profit
                curr_features.append(float(current_row["prev_profit"]))

                # Create next observation vector with the same approach
                next_features = []

                # current_profit (adjusted_profit)
                next_features.append(float(next_row["adjusted_profit"]))

                # total_profit
                next_features.append(float(next_row["total_profit"]))

                # REMOVED: current_bet (bet amount)
                next_features.append(float(next_row["bet"]))

                # streak
                next_features.append(float(next_row["streak"]))

                # win_streak (already fixed in preprocessing - single column)
                next_features.append(float(next_row["win_streak"]))

                # prev_bet
                next_features.append(float(next_row["prev_bet"]))

                # basepoint (current balance)
                next_features.append(float(next_row["basepoint"]))

                # prev_profit
                next_features.append(float(next_row["prev_profit"]))

                # Store observations
                all_obs.append(np.array(curr_features, dtype=np.float32))
                all_next_obs.append(np.array(next_features, dtype=np.float32))

                # Get action (bet_idx and delta_t)
                bet_idx = next_row["bet"]

                # Store action, done flag
                all_acts.append(1)
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
                all_rews.append(0.0)  # Placeholder value since DQN will compute its own rewards

            # ---------------------------------------------
            # Add a terminal state at the end (keep original termination handling)
            final_index = len(session_data) - 1
            current_row = session_data.iloc[final_index]

            # Create current observation vector (same fields as above)
            curr_features = []

            # current_profit (adjusted_profit)
            curr_features.append(float(current_row["adjusted_profit"]))

            # total_profit
            curr_features.append(float(current_row["total_profit"]))

            # REMOVED: current_bet (bet amount) - NO MORE ARTIFICIAL 0!
            curr_features.append(float(current_row["bet"]))  # TODO: double check in the previous script.

            # streak
            curr_features.append(float(current_row["streak"]))

            # win_streak (already fixed in preprocessing)
            curr_features.append(float(current_row["win_streak"]))

            # prev_bet
            curr_features.append(float(current_row["prev_bet"]))

            # basepoint (current balance)
            curr_features.append(float(current_row["basepoint"]))

            # prev_profit
            curr_features.append(float(current_row["prev_profit"]))

            all_obs.append(np.array(curr_features, dtype=np.float32))
            all_next_obs.append(np.zeros_like(curr_features, dtype=np.float32))

            # Store action, done flag (termination)
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
        f"DEBUG!! - length obs = {len(all_obs)}, next_obs = {len(all_next_obs)}, "
        f"feature_dim = {len(curr_features)}, sessions = {counter}, skipped = {skip}"
    )

    # Train test split
    train_length = int(len(all_obs) * (1 - testsize))

    # Training data
    train_obs = all_obs[:train_length]
    train_next_obs = all_next_obs[:train_length]
    train_acts = all_acts[:train_length]
    train_dones = all_dones[:train_length]
    train_infos = all_infos[:train_length]
    train_rews = all_rews[:train_length]

    # Test data
    test_obs = all_obs[train_length:]
    test_next_obs = all_next_obs[train_length:]
    test_acts = all_acts[train_length:]
    test_dones = all_dones[train_length:]
    test_infos = all_infos[train_length:]
    test_rews = all_rews[train_length:]

    # Convert to numpy arrays
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

    # NORMALIZATION
    scaler = StandardScaler()

    # Normalize training data
    train_obs_normalized = scaler.fit_transform(train_obs_array)
    train_next_obs_normalized = scaler.transform(train_next_obs_array)

    # Normalize test data using training scaler
    test_obs_normalized = scaler.transform(test_obs_array)
    test_next_obs_normalized = scaler.transform(test_next_obs_array)

    print(f"Normalization applied - Mean: {scaler.mean_}, Std: {scaler.scale_}")

    # Create transitions object with normalized observations
    train_transitions = TransitionsWithRew(
        obs=train_obs_normalized.astype(np.float32),
        acts=train_acts_array,
        infos=train_infos,
        next_obs=train_next_obs_normalized.astype(np.float32),
        dones=train_dones_array,
        rews=train_rews_array,
    )

    test_transitions = TransitionsWithRew(
        obs=test_obs_normalized.astype(np.float32),
        acts=test_acts_array,
        infos=test_infos,
        next_obs=test_next_obs_normalized.astype(np.float32),
        dones=test_dones_array,
        rews=test_rews_array,
    )

    # Return transitions with normalization scaler
    return train_transitions, test_transitions, scaler


class BasicDQN(nn.Module):
    """DQN network for binary action selection (continue=1, terminate=0)"""

    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(BasicDQN, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        # Output layer for 2 actions: [terminate=0, continue=1]
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TerminationDataProcessor:
    """Process data and maintain termination proportion"""

    def __init__(self, target_termination_ratio: float = 0.3):
        self.target_termination_ratio = target_termination_ratio

    def balance_training_transitions(self, train_transitions) -> Dict:
        """Balance termination and continuation transitions from existing training set"""

        # Extract data from TransitionsWithRew object
        obs = train_transitions.obs
        acts = train_transitions.acts
        next_obs = train_transitions.next_obs
        dones = train_transitions.dones
        infos = train_transitions.infos

        # Separate termination vs continue transitions
        termination_indices = []
        continue_indices = []

        for i, action in enumerate(acts):
            if action == 0:  # Termination action
                termination_indices.append(i)
            else:  # Continue action (action == 1)
                continue_indices.append(i)

        n_term = len(termination_indices)
        n_cont = len(continue_indices)

        print(f"Original training set: {n_term} termination, {n_cont} continuation transitions")

        # Calculate target sizes
        total_target = min(n_term + n_cont, 3000000)  # Cap at 1M for memory
        n_term_target = int(total_target * self.target_termination_ratio)
        n_cont_target = total_target - n_term_target

        # Sample to target sizes
        if n_term > n_term_target:
            term_sample_indices = random.sample(termination_indices, n_term_target)
        else:
            # Oversample if we need more termination cases
            term_sample_indices = termination_indices * (n_term_target // n_term + 1)
            term_sample_indices = term_sample_indices[:n_term_target]

        if n_cont > n_cont_target:
            cont_sample_indices = random.sample(continue_indices, n_cont_target)
        else:
            # Oversample if we need more continuation cases
            cont_sample_indices = continue_indices * (n_cont_target // n_cont + 1)
            cont_sample_indices = cont_sample_indices[:n_cont_target]

        # Combine and sort the indices and make then interlace
        all_indices = term_sample_indices + cont_sample_indices
        all_indices.sort()

        balanced_obs = obs[all_indices]
        balanced_next_obs = next_obs[all_indices]
        balanced_actions = acts[all_indices]
        balanced_dones = dones[all_indices]
        balanced_infos = [infos[i] for i in all_indices]

        return {
            "obs": balanced_obs,
            "next_obs": balanced_next_obs,
            "actions": balanced_actions,
            "dones": balanced_dones,
            "infos": balanced_infos,
            "is_termination": balanced_actions == 0,
        }


# ------ The sequence processor.
class SessionTerminationDataProcessor:
    """Process data maintaining session locality and termination proportion with metadata voting"""

    def __init__(self, target_termination_ratio: float = 0.3, sequence_length: int = 5):
        self.target_termination_ratio = target_termination_ratio
        self.sequence_length = sequence_length

    def balance_training_transitions(self, train_transitions, method: str = "tda") -> Dict:
        """
        Balance termination and continuation sequences

        Args:
            train_transitions: Input transition data
            method: "bc" for regular resampling, "tda" for session-grouped sequences
        """
        # Extract data from TransitionsWithRew object
        obs = train_transitions.obs
        acts = train_transitions.acts
        next_obs = train_transitions.next_obs
        dones = train_transitions.dones
        infos = train_transitions.infos

        print(f"Processing {len(obs)} transitions for {method.upper()} method...")

        # ALWAYS add session information to all transitions first
        enhanced_infos = self._add_session_metadata(infos, acts)

        if method.lower() == "bc":
            # BC: Regular resampling without session constraints
            return self._balance_for_bc(obs, acts, next_obs, dones, enhanced_infos)
        elif method.lower() == "tda":
            # TDA: Session-grouped sequences maintaining locality
            return self._balance_for_tda(obs, acts, next_obs, dones, enhanced_infos)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bc' or 'tda'")

    def _add_session_metadata(self, infos: List[Dict], acts: np.ndarray) -> List[Dict]:
        """Add session number and position info to all transitions"""
        enhanced_infos = []
        current_session = 0
        position_in_session = 0

        for i, (info, action) in enumerate(zip(infos, acts)):
            # Copy original info and add session metadata
            enhanced_info = info.copy()
            enhanced_info["original_index"] = i
            enhanced_info["session_number"] = current_session
            enhanced_info["position_in_session"] = position_in_session

            enhanced_infos.append(enhanced_info)

            # Update position and session tracking
            position_in_session += 1

            # If this action is termination (0), next transition starts new session
            if action == 0:
                current_session += 1
                position_in_session = 0

        print(f"Added session metadata: {current_session + 1} sessions identified")
        return enhanced_infos

    def _balance_for_bc(
        self, obs: np.ndarray, acts: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, enhanced_infos: List[Dict]
    ) -> Dict:
        """BC method: Regular resampling without session constraints"""
        print("Using BC method: Regular resampling...")

        # Identify termination and continuation transitions
        is_termination = acts == 0
        termination_indices = np.where(is_termination)[0]
        continuation_indices = np.where(~is_termination)[0]

        n_term = len(termination_indices)
        n_cont = len(continuation_indices)

        print(f"Found {n_term} termination transitions, {n_cont} continuation transitions")

        # Calculate target counts based on ratio
        if self.target_termination_ratio > 0:
            total_target = min(5000000, n_term + n_cont)  # Cap for memory
            n_term_target = int(total_target * self.target_termination_ratio)
            n_cont_target = total_target - n_term_target
        else:
            n_term_target = 0
            n_cont_target = min(5000000, n_cont)

        # Sample transitions
        if n_term >= n_term_target:
            selected_term_indices = np.random.choice(termination_indices, n_term_target, replace=False)
        else:
            selected_term_indices = np.random.choice(termination_indices, n_term_target, replace=True)

        if n_cont >= n_cont_target:
            selected_cont_indices = np.random.choice(continuation_indices, n_cont_target, replace=False)
        else:
            selected_cont_indices = np.random.choice(continuation_indices, n_cont_target, replace=True)

        # Combine and shuffle
        all_selected_indices = np.concatenate([selected_term_indices, selected_cont_indices])
        np.random.shuffle(all_selected_indices)

        print(f"BC balanced: {len(selected_term_indices)} term, {len(selected_cont_indices)} cont")

        return {
            "obs": obs[all_selected_indices],
            "next_obs": next_obs[all_selected_indices],
            "actions": acts[all_selected_indices],
            "dones": dones[all_selected_indices],
            "infos": [enhanced_infos[i] for i in all_selected_indices],
            "is_termination": acts[all_selected_indices] == 0,
        }

    def _balance_for_tda(
        self, obs: np.ndarray, acts: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, enhanced_infos: List[Dict]
    ) -> Dict:
        """TDA method: Session-grouped sequences maintaining locality"""
        print("Using TDA method: Session-grouped sequences...")

        # Group transitions by session using enhanced info
        session_groups = self._group_by_sessions_from_metadata(obs, acts, next_obs, dones, enhanced_infos)

        # Extract valid sequences from sessions
        termination_sequences, continuation_sequences = self._extract_sequences(session_groups)

        print(
            f"Found {len(termination_sequences)} termination sequences, {len(continuation_sequences)} continuation sequences"
        )

        # Balance according to target ratio
        balanced_sequences = self._balance_sequences(termination_sequences, continuation_sequences)

        # Convert back to transition format
        return self._sequences_to_transitions(balanced_sequences)

    def _group_by_sessions_from_metadata(
        self, obs: np.ndarray, acts: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, enhanced_infos: List[Dict]
    ) -> List[List[Dict]]:
        """Group transitions by sessions using the pre-computed session metadata"""
        sessions = defaultdict(list)

        for i, info in enumerate(enhanced_infos):
            session_num = info["session_number"]
            sessions[session_num].append(
                {
                    "index": i,
                    "original_index": info["original_index"],
                    "obs": obs[i],
                    "act": acts[i],
                    "next_obs": next_obs[i],
                    "done": dones[i],
                    "info": info,
                }
            )

        # Convert to list and sort sessions by session number
        session_list = [sessions[i] for i in sorted(sessions.keys())]

        print(f"Grouped transitions into {len(session_list)} sessions")
        print(
            f"Session lengths: min={min(len(s) for s in session_list) if session_list else 0}, "
            f"max={max(len(s) for s in session_list) if session_list else 0}, "
            f"avg={np.mean([len(s) for s in session_list]) if session_list else 0:.1f}"
        )

        return session_list

    def _extract_sequences(self, sessions: List[List[Dict]]) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """Extract 5-consecutive-transition sequences from sessions"""
        termination_sequences = []
        continuation_sequences = []

        for session_idx, session in enumerate(sessions):
            if len(session) < self.sequence_length:
                continue  # Skip sessions too short for sequences

            # Extract overlapping sequences from this session
            for i in range(len(session) - self.sequence_length + 1):
                sequence = session[i : i + self.sequence_length]

                # A sequence is "termination" if and only if it ends with action=0
                # (Not based on done flag, only the actual termination action)
                last_action = sequence[-1]["act"]

                if last_action == 0:  # True termination sequence
                    termination_sequences.append(sequence)
                else:  # Continuation sequence
                    continuation_sequences.append(sequence)

        return termination_sequences, continuation_sequences

    def _balance_sequences(
        self, termination_sequences: List[List[Dict]], continuation_sequences: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Balance termination and continuation sequences according to target ratio"""

        n_term = len(termination_sequences)
        n_cont = len(continuation_sequences)

        print(f"Before balancing: {n_term} termination sequences, {n_cont} continuation sequences")

        # Calculate target counts
        if n_term == 0:
            print("Warning: No termination sequences found!")
            return continuation_sequences[:10000]  # Return some continuation sequences

        # Calculate total target based on available data
        max_term_sequences = min(n_term, 1000000)  # Cap for memory
        max_cont_sequences = min(n_cont, 3000000)  # Cap for memory

        # Calculate based on target ratio
        if self.target_termination_ratio > 0:
            # If we want X% termination sequences
            target_total = min(
                int(max_term_sequences / self.target_termination_ratio), max_term_sequences + max_cont_sequences
            )
            n_term_target = int(target_total * self.target_termination_ratio)
            n_cont_target = target_total - n_term_target
        else:
            n_term_target = 0
            n_cont_target = max_cont_sequences

        # Sample termination sequences
        if n_term >= n_term_target:
            selected_term_sequences = random.sample(termination_sequences, n_term_target)
        else:
            # Oversample if needed
            repetitions = (n_term_target // n_term) + 1
            oversampled = termination_sequences * repetitions
            selected_term_sequences = random.sample(oversampled, n_term_target)

        # Sample continuation sequences
        if n_cont >= n_cont_target:
            selected_cont_sequences = random.sample(continuation_sequences, n_cont_target)
        else:
            # Oversample if needed
            repetitions = (n_cont_target // n_cont) + 1
            oversampled = continuation_sequences * repetitions
            selected_cont_sequences = random.sample(oversampled, n_cont_target)

        # Combine and shuffle
        balanced_sequences = selected_term_sequences + selected_cont_sequences
        random.shuffle(balanced_sequences)

        print(
            f"After balancing: {len(selected_term_sequences)} termination sequences, "
            f"{len(selected_cont_sequences)} continuation sequences"
        )
        print(f"Total sequences: {len(balanced_sequences)}")

        return balanced_sequences

    def _sequences_to_transitions(self, sequences: List[List[Dict]]) -> Dict:
        """
        Convert sequences back to flat transition format while preserving ordering

        Output: Same format as original - flat lists of obs, actions, etc.
        Ordering: Groups of 5 consecutive transitions from same sequence,
                 but groups themselves are shuffled for class balance
        """
        all_obs = []
        all_next_obs = []
        all_actions = []
        all_dones = []
        all_infos = []

        # Add sequences one by one (each sequence = 5 consecutive transitions)
        for sequence in sequences:
            # Add all 5 transitions from this sequence consecutively
            for transition in sequence:
                all_obs.append(transition["obs"])
                all_next_obs.append(transition["next_obs"])
                all_actions.append(transition["act"])
                all_dones.append(transition["done"])

                # Keep original info but add sequence metadata
                enhanced_info = transition["info"].copy()
                enhanced_info["from_balanced_sequence"] = True
                enhanced_info["sequence_length"] = self.sequence_length
                all_infos.append(enhanced_info)

        # Convert to numpy arrays - same format as original
        balanced_obs = np.array(all_obs)
        balanced_next_obs = np.array(all_next_obs)
        balanced_actions = np.array(all_actions)
        balanced_dones = np.array(all_dones)

        print(f"Created balanced dataset with {len(all_obs)} transitions from {len(sequences)} sequences")
        print(f"Data structure: {len(sequences)} groups of {self.sequence_length} consecutive transitions")

        # Return same format as original TerminationDataProcessor
        return {
            "obs": balanced_obs,
            "next_obs": balanced_next_obs,
            "actions": balanced_actions,
            "dones": balanced_dones,
            "infos": all_infos,
            "is_termination": balanced_actions == 0,
        }

    def get_sequence_statistics(self, sequences: List[List[Dict]]) -> Dict:
        """Get statistics about the extracted sequences"""
        if not sequences:
            return {"num_sequences": 0}

        sequence_lengths = [len(seq) for seq in sequences]
        termination_counts = []

        for sequence in sequences:
            term_count = sum(1 for t in sequence if t["act"] == 0)
            termination_counts.append(term_count)

        return {
            "num_sequences": len(sequences),
            "avg_sequence_length": np.mean(sequence_lengths),
            "min_sequence_length": np.min(sequence_lengths),
            "max_sequence_length": np.max(sequence_lengths),
            "avg_terminations_per_sequence": np.mean(termination_counts),
            "sequences_with_termination": sum(1 for count in termination_counts if count > 0),
        }


# --- The Loss Functions ------
# A collection because the data sucks. 0.9% termination are you serious.
class WeightedCrossEntropy(nn.Module):
    """Cross-entropy with extreme class weighting for 0.9% minority class"""

    def __init__(self, minority_weight=80.0, majority_weight=1.0):
        super().__init__()
        self.class_weights = torch.tensor([minority_weight, majority_weight], dtype=torch.float32)

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights.to(inputs.device))


class CostSensitiveLoss(nn.Module):
    """Loss based on business costs - very high penalty for missing rare class"""

    def __init__(self, miss_cost=70.0, false_alarm_cost=5.0):
        super().__init__()
        self.miss_cost = miss_cost  # Cost of missing termination
        self.false_alarm_cost = false_alarm_cost  # Cost of false positive

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        pred_term_prob = probs[:, 0]
        pred_cont_prob = probs[:, 1]

        # Cost for each sample
        cost = torch.where(
            targets == 0,  # Actual termination
            self.miss_cost * pred_cont_prob,  # Penalty for missing termination
            self.false_alarm_cost * pred_term_prob,  # Penalty for false alarm
        )

        return cost.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=0.05, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BCTrainer:
    """Behavioral cloning trainer with metadata-based ensemble voting"""

    def __init__(
        self,
        state_dim: int,
        hidden_dims=[128, 64, 32],
        learning_rate: float = 1e-3,
        device: str = "auto",
        weight_decay=1e-5,
        use_isolation_forest=False,
        isolation_forest_params=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        self.model = BasicDQN(state_dim, hidden_dims=hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = FocalLoss()

        # Isolation Forest setup
        self.use_isolation_forest = use_isolation_forest
        if use_isolation_forest:
            # Default parameters
            default_params = {
                "contamination": 0.1,
                "random_state": 42,
                "n_estimators": 100,
                "max_samples": "auto",
                "bootstrap": True,
                "n_jobs": -1,
            }

            # Update with user parameters
            if isolation_forest_params:
                default_params.update(isolation_forest_params)

            self.isolation_forest = IsolationForest(**default_params)
            self.scaler = StandardScaler()
            print(f"Isolation Forest parameters: {default_params}")

        print(f"Using device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, transitions: Dict, epochs: int = 100, batch_size: int = 256, validation_split: float = 0.2) -> Dict:
        """Train the model using behavioral cloning"""

        # Split data
        n_samples = len(transitions["obs"])
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_idx, val_idx = indices[n_val:], indices[:n_val]

        # Convert to tensors
        train_obs = torch.FloatTensor(transitions["obs"][train_idx]).to(self.device)
        train_actions = torch.LongTensor(transitions["actions"][train_idx]).to(self.device)
        val_obs = torch.FloatTensor(transitions["obs"][val_idx]).to(self.device)
        val_actions = torch.LongTensor(transitions["actions"][val_idx]).to(self.device)

        print(f"Training on {len(train_obs)} samples, validating on {len(val_obs)} samples")

        # Training loop
        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(train_obs), batch_size):
                batch_obs = train_obs[i : i + batch_size]
                batch_actions = train_actions[i : i + batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(batch_obs)
                loss = self.criterion(outputs, batch_actions)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            # Validation
            if epoch % 10 == 0:
                val_loss, val_acc = self._validate(val_obs, val_actions)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                print(
                    f"Epoch {epoch:3d}: Train Loss {avg_train_loss:.4f}, "
                    f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}"
                )

        # Final evaluation
        final_val_loss, final_val_acc = self._validate(val_obs, val_actions)
        print(f"\nFinal Validation - Loss: {final_val_loss:.4f}, Accuracy: {final_val_acc:.4f}")

        # Detailed evaluation
        self._detailed_evaluation(val_obs, val_actions)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "final_accuracy": final_val_acc,
        }

    def _validate(self, val_obs: torch.Tensor, val_actions: torch.Tensor) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(val_obs)
            loss = self.criterion(outputs, val_actions).item()

            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == val_actions).float().mean().item()

        return loss, accuracy

    def _detailed_evaluation(self, val_obs: torch.Tensor, val_actions: torch.Tensor):
        """Detailed evaluation with classification report"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(val_obs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            true_actions = val_actions.cpu().numpy()

        print("\nDetailed Classification Report:")
        print(classification_report(true_actions, predictions, target_names=["Terminate", "Continue"]))

        # Termination-specific metrics
        term_mask = true_actions == 0
        if np.any(term_mask):
            term_accuracy = accuracy_score(true_actions[term_mask], predictions[term_mask])
            print(f"Termination-only accuracy: {term_accuracy:.4f}")

        cont_mask = true_actions == 1
        if np.any(cont_mask):
            cont_accuracy = accuracy_score(true_actions[cont_mask], predictions[cont_mask])
            print(f"Continue-only accuracy: {cont_accuracy:.4f}")

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict actions for given observations"""
        self.model.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            q_values = self.model(obs_tensor)

            if deterministic:
                actions = torch.argmax(q_values, dim=1)
            else:
                actions = torch.argmax(q_values, dim=1)

            return actions.cpu().numpy()

    def save(self, path: str):
        """Save the trained model"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {path}")

    # ==================== ENSEMBLE VOTING METHODS ====================

    def _get_bc_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Get BC prediction probabilities"""
        self.model.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            bc_outputs = self.model(obs_tensor)
            bc_probs = torch.softmax(bc_outputs, dim=1)[:, 0].cpu().numpy()  # Termination probabilities
        return bc_probs

    def _apply_ensemble_voting_simple(
        self, obs: np.ndarray, bc_predictions: np.ndarray, bc_probs: np.ndarray
    ) -> np.ndarray:
        """
        Apply ensemble voting using simple [-5:] lookback
        """
        final_predictions = bc_predictions.copy()
        lookback = 5
        changes_made = 0

        # Find all BC termination predictions
        termination_indices = np.where(bc_predictions == 0)[0]

        for term_idx in termination_indices:
            # Get lookback window
            start_idx = max(0, term_idx - lookback + 1)
            end_idx = term_idx + 1
            window_obs = obs[start_idx:end_idx]

            if len(window_obs) < 2:
                continue

            try:
                # Compute TDA features
                tda_features = self._compute_tda_features_window(window_obs)

                # Combine current state with TDA features
                current_obs = obs[term_idx]
                combined_features = np.concatenate([current_obs, tda_features])

                # Scale and get IF score
                combined_scaled = self.scaler.transform(combined_features.reshape(1, -1))
                isolation_score = self.isolation_forest.decision_function(combined_scaled)[0]

                # Ensemble decision
                bc_confidence = max(bc_probs[term_idx], 1 - bc_probs[term_idx])

                # Simple threshold-based voting
                if bc_confidence > 0.6 and isolation_score > -0.1:
                    final_predictions[term_idx] = 0  # Keep termination
                else:
                    final_predictions[term_idx] = 1  # Change to continue
                    changes_made += 1

            except Exception as e:
                print(f"Ensemble voting failed for index {term_idx}: {e}")
                continue

        print(f"Ensemble voting changed {changes_made} predictions")
        return final_predictions

    def _train_isolation_forest(self, obs: np.ndarray, bc_predictions: np.ndarray, infos: List[Dict]):
        """
        For all BC termination predictions, use [-5:] slice to get last 5 observations
        and compute TDA features. Data is already session-ordered from TDA processor.
        """
        print("Training Isolation Forest using BC predictions with [-5:] lookback...")

        tp_samples = []
        lookback = 5

        # Find all positions where BC predicted termination
        termination_indices = np.where(bc_predictions == 0)[0]

        print(f"Found {len(termination_indices)} BC termination predictions")

        for term_idx in termination_indices:
            # Simple lookback: get last 5 observations (or from start if less than 5)
            start_idx = max(0, term_idx - lookback + 1)
            end_idx = term_idx + 1  # Include current position

            # Extract lookback window
            window_obs = obs[start_idx:end_idx]

            # Skip if window is too small
            if len(window_obs) < 2:
                continue

            try:
                # Compute TDA features for this window
                tda_features = self._compute_tda_features_window(window_obs)

                # Combine current state with TDA features
                current_obs = obs[term_idx]
                combined_features = np.concatenate([current_obs, tda_features])

                tp_samples.append(combined_features)

            except Exception as e:
                print(f"TDA failed for index {term_idx}: {e}")
                continue

        tp_samples = np.array(tp_samples) if tp_samples else np.array([]).reshape(0, obs.shape[1] + 2)

        print(f"Collected {len(tp_samples)} TP samples from BC termination predictions")

        if len(tp_samples) < 5:
            print("Warning: Too few TP samples for reliable Isolation Forest training")
            return

        # Train Isolation Forest
        contamination = 0.1

        if not hasattr(self, "isolation_forest") or self.isolation_forest is None:
            self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        else:
            self.isolation_forest.set_params(contamination=contamination)

        if not hasattr(self, "scaler") or self.scaler is None:
            self.scaler = StandardScaler()

        # Scale and train
        tp_scaled = self.scaler.fit_transform(tp_samples)
        self.isolation_forest.fit(tp_scaled)

        print(f"Isolation Forest trained on {len(tp_samples)} TP samples")

    def _compute_tda_features_window(self, window_obs: np.ndarray) -> np.ndarray:
        """Compute TDA features for a window of observations"""
        try:
            window_size, n_features = window_obs.shape

            # Apply PCA if too many features
            if n_features > 4:
                pca = PCA(n_components=4)
                window_obs = pca.fit_transform(window_obs)

            # Reshape for gtda (point cloud format)
            point_cloud = window_obs.reshape(1, window_size, -1)

            # Compute Vietoris-Rips persistence
            VR = VietorisRipsPersistence(homology_dimensions=[0, 1], collapse_edges=True, n_jobs=1)
            barcodes = VR.fit_transform(point_cloud)

            # Compute persistence entropy
            PE = PersistenceEntropy()
            entropy = PE.fit_transform(barcodes)[0]  # Shape: (2,) for H0 and H1

            return entropy

        except Exception as e:
            print(f"TDA computation failed: {e}. Using zeros.")
            return np.zeros(2)

    def _save_isolation_forest_model(self, save_path=None, tp_count=0, fp_count=0, contamination=0.0):
        """Save the trained Isolation Forest model and scaler"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"weights/isolation_forest_model_{timestamp}.pkl"

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        model_data = {
            "isolation_forest": self.isolation_forest,
            "scaler": self.scaler,
            "training_stats": {
                "tp_count": tp_count,
                "fp_count": fp_count,
                "contamination": contamination,
                "training_date": datetime.now().isoformat(),
            },
            "model_params": self.isolation_forest.get_params(),
        }

        try:
            with open(save_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Isolation Forest model saved to: {save_path}")
            return save_path
        except Exception as e:
            print(f"Error saving Isolation Forest model: {e}")
            return None

    def load_isolation_forest_model(self, load_path):
        """Load a previously trained Isolation Forest model and scaler"""
        try:
            with open(load_path, "rb") as f:
                model_data = pickle.load(f)

            self.isolation_forest = model_data["isolation_forest"]
            self.scaler = model_data["scaler"]

            stats = model_data.get("training_stats", {})
            print(f"Loaded Isolation Forest model from: {load_path}")
            print(f"  Training date: {stats.get('training_date', 'Unknown')}")
            print(f"  TP count: {stats.get('tp_count', 'Unknown')}")
            print(f"  FP count: {stats.get('fp_count', 'Unknown')}")
            print(f"  Contamination: {stats.get('contamination', 'Unknown')}")

            return True
        except Exception as e:
            print(f"Error loading Isolation Forest model: {e}")
            return False


def training_pipeline(
    data_path: str,
    train_expert_transitions,
    test_expert_transitions,
    model_save_path: str = "termination_bc_model.pth",
    target_termination_ratio: float = 0.3,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dims=[128, 64, 32],
    weight_decay=1e-5,
    prediction_method: int = 0,
    isolation_forest_params=None,
):
    """
    Main training pipeline with metadata-based ensemble voting

    Args:
        prediction_method:
            0 - BC only (no Isolation Forest)
            1 - Isolation Forest only (TDA features, no BC)
            2 - Ensemble voting (BC + Isolation Forest with metadata)
    """

    print("=== Termination Behavioral Cloning Pipeline ===")
    print(f"Prediction method: {['BC only', 'Isolation Forest only', 'Ensemble voting'][prediction_method]}")

    # 1. Balance training transitions using appropriate method
    print("\n1. Balancing training transitions...")
    processor = SessionTerminationDataProcessor(target_termination_ratio, sequence_length=5)

    # for BC:
    bc_balanced_transitions = processor.balance_training_transitions(train_expert_transitions, method="bc")
    # for IF
    if_balanced_transitions = processor.balance_training_transitions(train_expert_transitions, method="tda")

    state_dim = bc_balanced_transitions["obs"].shape[1]
    print(f"State dimension: {state_dim}")

    # 2. Initialize trainer based on method
    use_isolation_forest = prediction_method > 0
    trainer = BCTrainer(
        state_dim=state_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_isolation_forest=use_isolation_forest,
        isolation_forest_params=isolation_forest_params,
    )

    # 3. Training based on method
    if prediction_method == 0:
        # BC only - standard training
        print("\n2. Training behavioral cloning model...")
        training_history = trainer.train(bc_balanced_transitions, epochs, batch_size)

        print("\n3. Evaluating on test set... (BC only)")
        test_predictions = trainer.predict(test_expert_transitions.obs)

    elif prediction_method == 1:
        # Isolation Forest only - skip BC training, train directly on TDA features
        print("\n2. Skipping BC training for Isolation Forest only method...")

        print("\n2.5. Training Isolation Forest on TDA features...")
        X = if_balanced_transitions["obs"]
        y_true = if_balanced_transitions["actions"]
        infos = if_balanced_transitions["infos"]

        # Use session metadata to group and analyze
        termination_samples = []
        continuation_samples = []

        # Group by session using metadata
        session_groups = defaultdict(list)
        for i, info in enumerate(infos):
            session_num = info.get("session_number", 0)
            session_groups[session_num].append(
                {"obs": X[i], "action": y_true[i], "position": info.get("position_in_session", 0)}
            )

        # Process each session to extract TDA features
        for session_num, session_data in session_groups.items():
            if len(session_data) < 5:
                continue

            # Sort by position within session
            session_data.sort(key=lambda x: x["position"])
            session_obs = np.array([item["obs"] for item in session_data])
            session_actions = np.array([item["action"] for item in session_data])

            try:
                # Compute TDA features for this session
                tda_features = trainer._compute_tda_features_session(session_obs)

                # Separate by action type
                for i, action in enumerate(session_actions):
                    if action == 0:  # Termination
                        termination_samples.append(tda_features[i])
                    else:  # Continuation
                        continuation_samples.append(tda_features[i])

            except Exception as e:
                print(f"TDA failed for session {session_num}: {e}")
                continue

        termination_samples = np.array(termination_samples) if termination_samples else np.array([]).reshape(0, 2)
        continuation_samples = np.array(continuation_samples) if continuation_samples else np.array([]).reshape(0, 2)

        print(f"Found {len(termination_samples)} termination samples, {len(continuation_samples)} continuation samples")

        if len(termination_samples) >= 5:
            # Train on termination samples as "normal" class
            contamination = len(continuation_samples) / (len(termination_samples) + len(continuation_samples))
            contamination = min(0.4, max(0.05, contamination))

            trainer.isolation_forest.set_params(contamination=contamination)
            term_scaled = trainer.scaler.fit_transform(termination_samples)
            trainer.isolation_forest.fit(term_scaled)

            print(f"Isolation Forest trained on {len(termination_samples)} termination samples")
            print(f"Contamination parameter: {contamination:.3f}")
        else:
            print("Warning: Too few termination samples for reliable training")

        training_history = {"message": "Isolation Forest only - no BC training"}

        print("\n3. Evaluating on test set... (Isolation Forest only)")
        test_predictions = trainer._predict_tda_only(test_expert_transitions.obs, test_expert_transitions.infos)

    elif prediction_method == 2:
        # FIXED: Ensemble voting - BC trained on regular resampling, IF trained using BC predictions on TDA data
        print("\n2. Training behavioral cloning model with regular resampling...")

        # Train BC with regular balanced data (no session constraints)
        training_history = trainer.train(bc_balanced_transitions, epochs, batch_size)

        print("\n2.5. Training Isolation Forest using BC predictions on TDA-balanced data...")

        # Get BC predictions on the TDA balanced data (which has proper session ordering)
        bc_predictions_on_tda = trainer.predict(if_balanced_transitions["obs"])

        # Train IF using BC predictions with [-5:] lookback on session-ordered TDA data
        trainer._train_isolation_forest(
            if_balanced_transitions["obs"], bc_predictions_on_tda, if_balanced_transitions["infos"]
        )

        print("\n2.6. Saving Isolation Forest model...")
        if_save_path = model_save_path.replace(".pth", "_isolation_forest.pkl")
        trainer._save_isolation_forest_model(
            save_path=if_save_path,
            tp_count=np.sum(bc_predictions_on_tda == 0),
            fp_count=len(bc_predictions_on_tda) - np.sum(bc_predictions_on_tda == 0),
            contamination=(
                trainer.isolation_forest.contamination if hasattr(trainer.isolation_forest, "contamination") else 0.1
            ),
        )
        print(f"Isolation Forest saved to: {if_save_path}")

        print("\n3. Evaluating on test set... (Ensemble voting)")
        # For test evaluation, use simple ensemble voting
        test_bc_predictions = trainer.predict(test_expert_transitions.obs)
        test_bc_probs = trainer._get_bc_probabilities(test_expert_transitions.obs)
        test_predictions = trainer._apply_ensemble_voting_simple(
            test_expert_transitions.obs, test_bc_predictions, test_bc_probs
        )

    else:
        raise ValueError(f"Invalid prediction_method: {prediction_method}. Must be 0, 1, or 2.")

    # 4. Ensure test predictions and actions have same length
    test_actions = test_expert_transitions.acts

    # Trim test data if needed to match prediction length
    if len(test_predictions) != len(test_actions):
        min_length = min(len(test_predictions), len(test_actions))
        test_predictions = test_predictions[:min_length]
        test_actions = test_actions[:min_length]
        print(f"Warning: Trimmed test data to {min_length} samples for evaluation")

    # 5. Evaluation metrics (same for all methods)
    True_Terminations = test_actions == 0
    True_Continues = test_actions == 1
    Predict_Terminations = test_predictions == 0
    Predict_Continues = test_predictions == 1
    Correct_Prediction = test_predictions == test_actions
    Correct_Terminations = True_Terminations & Correct_Prediction  # TP
    Correct_Continues = True_Continues & Correct_Prediction  # TN
    False_Terminations = True_Continues & Predict_Terminations  # FP
    False_Continues = True_Terminations & Predict_Continues  # FN

    # Confusion Matrix Elements
    TP = np.sum(Correct_Terminations)
    TN = np.sum(Correct_Continues)
    FP = np.sum(False_Terminations)
    FN = np.sum(False_Continues)

    # Confusion Matrix
    print("Confusion Matrix:")
    print("                 Predicted")
    print("              Term    Cont")
    print(f"Actual Term   {TP:4d}    {FN:4d}")
    print(f"       Cont   {FP:4d}    {TN:4d}")

    # Basic Accuracy Metrics
    Overall_Accuracy = np.sum(Correct_Prediction) / len(Correct_Prediction) if len(Correct_Prediction) > 0 else 0
    Termination_Acc = np.sum(Correct_Terminations) / np.sum(True_Terminations) if np.sum(True_Terminations) > 0 else 0
    Continues_Acc = np.sum(Correct_Continues) / np.sum(True_Continues) if np.sum(True_Continues) > 0 else 0

    # Precision, Recall, F1 for Each Class
    Precision_Term = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall_Term = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_Term = (
        2 * (Precision_Term * Recall_Term) / (Precision_Term + Recall_Term) if (Precision_Term + Recall_Term) > 0 else 0
    )

    Precision_Cont = TN / (TN + FN) if (TN + FN) > 0 else 0
    Recall_Cont = TN / (TN + FP) if (TN + FP) > 0 else 0
    F1_Cont = (
        2 * (Precision_Cont * Recall_Cont) / (Precision_Cont + Recall_Cont) if (Precision_Cont + Recall_Cont) > 0 else 0
    )

    # Macro Averages
    Macro_Precision = (Precision_Term + Precision_Cont) / 2
    Macro_Recall = (Recall_Term + Recall_Cont) / 2
    Macro_F1 = (F1_Term + F1_Cont) / 2

    # Weighted Averages
    Total_Term = np.sum(True_Terminations)
    Total_Cont = np.sum(True_Continues)
    Total_Samples = Total_Term + Total_Cont

    if Total_Samples > 0:
        Weight_Term = Total_Term / Total_Samples
        Weight_Cont = Total_Cont / Total_Samples
        Weighted_Precision = (Precision_Term * Weight_Term) + (Precision_Cont * Weight_Cont)
        Weighted_Recall = (Recall_Term * Weight_Term) + (Recall_Cont * Weight_Cont)
        Weighted_F1 = (F1_Term * Weight_Term) + (F1_Cont * Weight_Cont)
        Balanced_Accuracy = (Recall_Term + Recall_Cont) / 2
    else:
        Weighted_Precision = Weighted_Recall = Weighted_F1 = Balanced_Accuracy = 0

    # Print results
    print(f"\n=== METRICS SUMMARY ===")
    print(f"Overall Accuracy:     {Overall_Accuracy:.4f}")
    print(f"Balanced Accuracy:    {Balanced_Accuracy:.4f}")
    print(f"\n--- PER CLASS ---")
    print(f"Termination Precision: {Precision_Term:.4f}")
    print(f"Termination Recall:    {Recall_Term:.4f}")
    print(f"Termination F1:        {F1_Term:.4f}")
    print(f"\nContinuation Precision: {Precision_Cont:.4f}")
    print(f"Continuation Recall:    {Recall_Cont:.4f}")
    print(f"Continuation F1:        {F1_Cont:.4f}")
    print(f"\n--- MACRO AVERAGES ---")
    print(f"Macro Precision:      {Macro_Precision:.4f}")
    print(f"Macro Recall:         {Macro_Recall:.4f}")
    print(f"Macro F1:             {Macro_F1:.4f}")
    print(f"\n--- WEIGHTED AVERAGES ---")
    print(f"Weighted Precision:   {Weighted_Precision:.4f}")
    print(f"Weighted Recall:      {Weighted_Recall:.4f}")
    print(f"Weighted F1:          {Weighted_F1:.4f}")
    print(f"\n--- CLASS DISTRIBUTION ---")
    print(f"Termination samples:  {Total_Term} ({Weight_Term:.1%})" if Total_Samples > 0 else "No samples")
    print(f"Continuation samples: {Total_Cont} ({Weight_Cont:.1%})" if Total_Samples > 0 else "No samples")

    # 6. Save model (only for BC-based methods)
    if prediction_method in [0, 2]:
        print(f"\n4. Saving model to {model_save_path}")
        trainer.save(model_save_path)

    print("\n=== Training Complete ===")
    return trainer, training_history


def BinaryBC(
    data_path: str,
    expert_data_path: str = "expert_data_dqn.pkl",
    model_save_path: str = "termination_bc_model.pth",
    target_termination_ratio: float = 0.3,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dims=[128, 64, 32],
    weight_decay=1e-5,
    prediction_method: int = 0,
    isolation_forest_params=None,
):
    """
    Train behavioral cloning model with metadata-based ensemble voting

    Args:
        data_path: Path to raw CSV data
        expert_data_path: Path to save/load processed expert data
        model_save_path: Path to save trained model
        target_termination_ratio: Target ratio of termination transitions in balanced dataset
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        hidden_dims: Hidden layer dimensions for neural network
        weight_decay: L2 regularization weight
        prediction_method: 0=BC only, 1=Isolation Forest only, 2=Ensemble voting with metadata
        isolation_forest_params: Dict with Isolation Forest parameters
    """

    expert_data_loaded = False

    # Check if expert data file exists
    if os.path.exists(expert_data_path):
        print(f"Loading existing expert data from {expert_data_path}")
        try:
            with open(expert_data_path, "rb") as f:
                expert_data = pickle.load(f)

            train_expert_transitions = expert_data["train_expert_transitions"]
            test_expert_transitions = expert_data["test_expert_transitions"]
            scaler = expert_data.get("scaler", None)

            print(
                f"Loaded {len(train_expert_transitions.obs)} training transitions, "
                f"{len(test_expert_transitions.obs)} test transitions"
            )
            expert_data_loaded = True

        except Exception as e:
            print(f"Error loading expert data: {e}. Processing from raw data...")

    # Process raw data if loading failed or file doesn't exist
    if not expert_data_loaded:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        print(f"Processing raw data from {data_path}")
        df, state_cols = load_and_process_data(data_path)
        train_expert_transitions, test_expert_transitions, scaler = get_flattened_trajectories(df)

        # Save the processed data for future use
        try:
            expert_data_to_save = {
                "train_expert_transitions": train_expert_transitions,
                "test_expert_transitions": test_expert_transitions,
                "scaler": scaler,
                "processing_params": {
                    "data_path": data_path,
                    "processing_date": datetime.now().isoformat(),
                    "prediction_method": prediction_method,
                },
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(expert_data_path)) or ".", exist_ok=True)

            with open(expert_data_path, "wb") as f:
                pickle.dump(expert_data_to_save, f)
            print(f"Saved processed data to {expert_data_path}")

        except Exception as e:
            print(f"Warning: Could not save processed data: {e}")

    if len(train_expert_transitions.obs) == 0:
        raise ValueError("No training data available")
    if len(test_expert_transitions.obs) == 0:
        raise ValueError("No test data available")

    print(f"Data validation passed:")
    print(f"  Training samples: {len(train_expert_transitions.obs)}")
    print(f"  Test samples: {len(test_expert_transitions.obs)}")
    print(f"  State dimension: {train_expert_transitions.obs.shape[1]}")

    # Validate metadata presence for ensemble methods
    if prediction_method > 0:
        sample_info = train_expert_transitions.infos[0] if train_expert_transitions.infos else {}
        if "session_number" not in sample_info:
            print("Warning: Session metadata not found. Adding metadata...")
            # This should already be done by the processor, but double-check

    # Run training pipeline
    trainer, history = training_pipeline(
        data_path=data_path,
        train_expert_transitions=train_expert_transitions,
        test_expert_transitions=test_expert_transitions,
        model_save_path=model_save_path,
        target_termination_ratio=target_termination_ratio,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dims=hidden_dims,
        weight_decay=weight_decay,
        isolation_forest_params=isolation_forest_params,
        prediction_method=prediction_method,
    )

    return {
        "trainer": trainer,
        "training_history": history,
        "scaler": scaler,
        "data_info": {
            "train_samples": len(train_expert_transitions.obs),
            "test_samples": len(test_expert_transitions.obs),
            "state_dim": train_expert_transitions.obs.shape[1],
            "prediction_method": prediction_method,
        },
    }


def BinaryBC(
    data_path: str,
    expert_data_path: str = "expert_data_dqn.pkl",
    model_save_path: str = "termination_bc_model.pth",
    target_termination_ratio: float = 0.3,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dims=[128, 64, 32],
    weight_decay=1e-5,
    prediction_method: int = 0,
    isolation_forest_params=None,
):
    """
    Train behavioral cloning model with metadata-based ensemble voting

    Args:
        data_path: Path to raw CSV data
        expert_data_path: Path to save/load processed expert data
        model_save_path: Path to save trained model
        target_termination_ratio: Target ratio of termination transitions in balanced dataset
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        hidden_dims: Hidden layer dimensions for neural network
        weight_decay: L2 regularization weight
        prediction_method: 0=BC only, 1=Isolation Forest only, 2=Ensemble voting with metadata
        isolation_forest_params: Dict with Isolation Forest parameters

    Returns:
        dict: {
            'trainer': BCTrainer instance,
            'training_history': Training metrics and logs,
            'scaler': Data normalization scaler,
            'data_info': Information about processed data,
            'processor': SessionTerminationDataProcessor instance
        }
    """

    print("=== Binary BC with Metadata-Based Ensemble Voting ===")
    print(f"Prediction method: {['BC only', 'Isolation Forest only', 'Ensemble voting'][prediction_method]}")
    print(f"Data path: {data_path}")
    print(f"Expert data path: {expert_data_path}")

    expert_data_loaded = False

    # 1. Load or process expert data
    if os.path.exists(expert_data_path):
        print(f"\nLoading existing expert data from {expert_data_path}")
        try:
            with open(expert_data_path, "rb") as f:
                expert_data = pickle.load(f)

            train_expert_transitions = expert_data["train_expert_transitions"]
            test_expert_transitions = expert_data["test_expert_transitions"]
            scaler = expert_data.get("scaler", None)

            print(f"✓ Loaded {len(train_expert_transitions.obs)} training transitions")
            print(f"✓ Loaded {len(test_expert_transitions.obs)} test transitions")

            # Check if data has session metadata
            sample_info = train_expert_transitions.infos[0] if train_expert_transitions.infos else {}
            has_metadata = "session_number" in sample_info
            print(f"✓ Session metadata present: {has_metadata}")

            expert_data_loaded = True

        except Exception as e:
            print(f"✗ Error loading expert data: {e}")
            print("Processing from raw data...")

    # Process raw data if loading failed or file doesn't exist
    if not expert_data_loaded:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        print(f"\nProcessing raw data from {data_path}")
        df, state_cols = load_and_process_data(data_path)
        train_expert_transitions, test_expert_transitions, scaler = get_flattened_trajectories(df)

        # Save the processed data for future use
        try:
            expert_data_to_save = {
                "train_expert_transitions": train_expert_transitions,
                "test_expert_transitions": test_expert_transitions,
                "scaler": scaler,
                "processing_params": {
                    "data_path": data_path,
                    "processing_date": datetime.now().isoformat(),
                    "prediction_method": prediction_method,
                    "state_cols": state_cols,
                },
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(expert_data_path)) or ".", exist_ok=True)

            with open(expert_data_path, "wb") as f:
                pickle.dump(expert_data_to_save, f)
            print(f"✓ Saved processed data to {expert_data_path}")

        except Exception as e:
            print(f"⚠ Warning: Could not save processed data: {e}")

    # 2. Data validation
    if len(train_expert_transitions.obs) == 0:
        raise ValueError("No training data available")
    if len(test_expert_transitions.obs) == 0:
        raise ValueError("No test data available")

    print(f"\n=== Data Validation ===")
    print(f"✓ Training samples: {len(train_expert_transitions.obs):,}")
    print(f"✓ Test samples: {len(test_expert_transitions.obs):,}")
    print(f"✓ State dimension: {train_expert_transitions.obs.shape[1]}")

    # Check action distribution
    train_actions = train_expert_transitions.acts
    train_term_count = np.sum(train_actions == 0)
    train_cont_count = np.sum(train_actions == 1)
    train_term_ratio = train_term_count / len(train_actions) if len(train_actions) > 0 else 0

    print(f"✓ Training termination ratio: {train_term_ratio:.1%} ({train_term_count:,}/{len(train_actions):,})")

    test_actions = test_expert_transitions.acts
    test_term_count = np.sum(test_actions == 0)
    test_cont_count = np.sum(test_actions == 1)
    test_term_ratio = test_term_count / len(test_actions) if len(test_actions) > 0 else 0

    print(f"✓ Test termination ratio: {test_term_ratio:.1%} ({test_term_count:,}/{len(test_actions):,})")

    # 3. Validate metadata for ensemble methods
    if prediction_method > 0:
        sample_info = train_expert_transitions.infos[0] if train_expert_transitions.infos else {}
        if "session_number" not in sample_info:
            print("\n⚠ Warning: Session metadata not found for ensemble method!")
            print("Adding session metadata using processor...")

            # This should be handled by the processor, but let's ensure it's there
            processor = SessionTerminationDataProcessor(target_termination_ratio)
            enhanced_infos = processor._add_session_metadata(
                train_expert_transitions.infos, train_expert_transitions.acts
            )

            # Update transitions with enhanced infos
            train_expert_transitions = TransitionsWithRew(
                obs=train_expert_transitions.obs,
                acts=train_expert_transitions.acts,
                infos=enhanced_infos,
                next_obs=train_expert_transitions.next_obs,
                dones=train_expert_transitions.dones,
                rews=train_expert_transitions.rews,
            )

            # Do the same for test data
            test_enhanced_infos = processor._add_session_metadata(
                test_expert_transitions.infos, test_expert_transitions.acts
            )
            test_expert_transitions = TransitionsWithRew(
                obs=test_expert_transitions.obs,
                acts=test_expert_transitions.acts,
                infos=test_enhanced_infos,
                next_obs=test_expert_transitions.next_obs,
                dones=test_expert_transitions.dones,
                rews=test_expert_transitions.rews,
            )

            print("✓ Session metadata added")
        else:
            print("✓ Session metadata already present")

    # 4. Initialize processor and run training pipeline
    processor = SessionTerminationDataProcessor(target_termination_ratio, sequence_length=5)

    print(f"\n=== Starting Training Pipeline ===")
    print(f"Target termination ratio: {target_termination_ratio:.1%}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Hidden dims: {hidden_dims}")

    try:
        trainer, history = training_pipeline(
            data_path=data_path,
            train_expert_transitions=train_expert_transitions,
            test_expert_transitions=test_expert_transitions,
            model_save_path=model_save_path,
            target_termination_ratio=target_termination_ratio,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dims=hidden_dims,
            weight_decay=weight_decay,
            isolation_forest_params=isolation_forest_params,
            prediction_method=prediction_method,
        )

        print("\n✓ Training pipeline completed successfully")

    except Exception as e:
        print(f"\n✗ Training pipeline failed: {e}")
        raise

    # 5. Additional validation and statistics
    print(f"\n=== Final Model Statistics ===")

    # Model info
    if hasattr(trainer, "model"):
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"✓ Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Training history info
    if isinstance(history, dict) and "final_accuracy" in history:
        print(f"✓ Final validation accuracy: {history['final_accuracy']:.4f}")

    # Isolation Forest info (if applicable)
    if prediction_method > 0 and hasattr(trainer, "isolation_forest"):
        if hasattr(trainer.isolation_forest, "contamination"):
            print(f"✓ Isolation Forest contamination: {trainer.isolation_forest.contamination:.3f}")
        if hasattr(trainer, "scaler") and hasattr(trainer.scaler, "mean_"):
            print(f"✓ TDA feature scaler fitted: {len(trainer.scaler.mean_)} features")

    # 6. Save final metadata
    try:
        final_metadata = {
            "training_completed": datetime.now().isoformat(),
            "prediction_method": prediction_method,
            "target_termination_ratio": target_termination_ratio,
            "final_model_path": model_save_path,
            "data_info": {
                "train_samples": len(train_expert_transitions.obs),
                "test_samples": len(test_expert_transitions.obs),
                "state_dim": train_expert_transitions.obs.shape[1],
                "train_termination_ratio": train_term_ratio,
                "test_termination_ratio": test_term_ratio,
            },
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_dims": hidden_dims,
                "weight_decay": weight_decay,
            },
        }

        metadata_path = model_save_path.replace(".pth", "_metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(final_metadata, f, indent=2)
        print(f"✓ Saved training metadata to {metadata_path}")

    except Exception as e:
        print(f"⚠ Warning: Could not save metadata: {e}")

    print("\n=== BinaryBC Completed Successfully ===")

    return {
        "trainer": trainer,
        "training_history": history,
        "scaler": scaler,
        "processor": processor,
        "data_info": {
            "train_samples": len(train_expert_transitions.obs),
            "test_samples": len(test_expert_transitions.obs),
            "state_dim": train_expert_transitions.obs.shape[1],
            "prediction_method": prediction_method,
            "train_termination_ratio": train_term_ratio,
            "test_termination_ratio": test_term_ratio,
        },
        "final_metadata": final_metadata if "final_metadata" in locals() else None,
    }


# Helper function for quick model evaluation
def evaluate_trained_model(results_dict, test_data_path=None):
    """
    Quick evaluation helper for trained BinaryBC models

    Args:
        results_dict: Output from BinaryBC function
        test_data_path: Optional path to additional test data

    Returns:
        dict: Evaluation metrics
    """
    trainer = results_dict["trainer"]
    data_info = results_dict["data_info"]

    print("=== Model Evaluation ===")
    print(
        f"Prediction method: {['BC only', 'Isolation Forest only', 'Ensemble voting'][data_info['prediction_method']]}"
    )
    print(f"Training samples: {data_info['train_samples']:,}")
    print(f"Test samples: {data_info['test_samples']:,}")
    print(f"State dimension: {data_info['state_dim']}")

    # Model size
    if hasattr(trainer, "model"):
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"Model parameters: {total_params:,}")

    # Isolation Forest info
    if data_info["prediction_method"] > 0 and hasattr(trainer, "isolation_forest"):
        print(f"Isolation Forest: {trainer.isolation_forest.get_params()}")

    return {
        "model_info": {
            "prediction_method": data_info["prediction_method"],
            "total_parameters": total_params if "total_params" in locals() else 0,
            "state_dimension": data_info["state_dim"],
        },
        "data_info": data_info,
    }


# Helper function for making predictions on new data
def predict_with_trained_model(results_dict, new_obs, new_infos=None):
    """
    Make predictions using a trained BinaryBC model

    Args:
        results_dict: Output from BinaryBC function
        new_obs: New observation data (numpy array)
        new_infos: Info dictionaries with session metadata (for ensemble methods)

    Returns:
        numpy array: Predictions (0=terminate, 1=continue)
    """
    trainer = results_dict["trainer"]
    scaler = results_dict["scaler"]
    data_info = results_dict["data_info"]

    print(f"Making predictions on {len(new_obs)} samples...")

    # Normalize observations using the training scaler
    if scaler is not None:
        new_obs_normalized = scaler.transform(new_obs)
    else:
        new_obs_normalized = new_obs
        print("Warning: No scaler found, using raw observations")

    # Make predictions based on method
    if data_info["prediction_method"] == 0:
        # BC only
        predictions = trainer.predict(new_obs_normalized)
    elif data_info["prediction_method"] == 2:
        # Ensemble voting with metadata
        if new_infos is None:
            print("Warning: No metadata provided for ensemble method, falling back to BC only")
            predictions = trainer.predict(new_obs_normalized)
        else:
            predictions = trainer.predict_with_metadata_voting(new_obs_normalized, new_infos)
    elif data_info["prediction_method"] == 1:
        # Isolation Forest only
        if new_infos is None:
            print("Warning: No metadata provided for IF method, using default predictions")
            predictions = np.ones(len(new_obs))  # Default to continue
        else:
            predictions = trainer._predict_tda_only(new_obs_normalized, new_infos)
    else:
        raise ValueError(f"Unknown prediction method: {data_info['prediction_method']}")

    termination_count = np.sum(predictions == 0)
    termination_ratio = termination_count / len(predictions)

    print(f"Predicted {termination_count} terminations ({termination_ratio:.1%})")

    return predictions
