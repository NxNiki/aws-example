"""Load the DS feature export into model-ready (X_raw, lengths, feature_names)."""

import argparse

import numpy as np
import pandas as pd

# The 8 columns the loader outputs. Downstream (churn_helpers / io_hmm) drops the
# near-constant multiplier_change_count_ratio, so the model actually sees Golden 7.
FEATURE_NAMES = [
    "no_bet_streak_days",
    "bet_amount_ratio_today_vs_history",
    "avg_bet_one_time_today_log",
    "rtp_7_bet_days",
    "loss_streak_ratio_today",
    "current_balance_max_to_avg_bet_ratio",
    "target_selection_entropy",
    "multiplier_change_count_ratio",
]


def load_model_data(csv_path):
    """Read the DS export CSV and produce model-ready (X_raw, lengths, feature_names).

    - selects the Golden 8 columns (raises on missing columns)
    - drops each user's first bet-day row (the only source of NaN in history features)
    - sorts by [user_id, bet_date] and rebuilds lengths (the HMM's per-user sequence sizes)
    - asserts no residual NaN / sum(lengths)==rows / non-empty -- fail fast, never fillna

    X_raw is NOT standardized: the HMM path scales downstream, and decision trees
    need raw values so their thresholds stay in real business units.
    """
    df = pd.read_csv(csv_path, parse_dates=["bet_date"])

    missing = [c for c in ["user_id", "bet_date", *FEATURE_NAMES] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing} (from {csv_path})")

    df = df.sort_values(["user_id", "bet_date"], kind="stable")
    df = df[df.groupby("user_id").cumcount() > 0]
    if df.empty:
        raise ValueError("No data left after dropping first bet-days (all single-bet-day users?)")

    lengths = df.groupby("user_id", sort=False).size().tolist()
    X_raw = df[FEATURE_NAMES].to_numpy(dtype=float)

    if np.isnan(X_raw).any():
        bad = [FEATURE_NAMES[i] for i in np.unique(np.where(np.isnan(X_raw))[1])]
        raise ValueError(
            f"NaN remains after dropping first bet-days, columns: {bad} -- investigate before deciding a fill strategy"
        )
    if sum(lengths) != len(X_raw):
        raise ValueError(f"sum(lengths) {sum(lengths)} != row count {len(X_raw)}")

    return X_raw, lengths, FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Turn the DS HMM feature CSV into model-ready (X_raw, lengths).")
    parser.add_argument("csv_path", help="selected_hmm_features_*.csv exported by DS")
    args = parser.parse_args()

    X_raw, lengths, feature_names = load_model_data(args.csv_path)
    print(f"users: {len(lengths)}")
    print(f"rows: {len(X_raw)}")
    print(f"features: {len(feature_names)} -> {feature_names}")
    print(f"sequence lengths: min={min(lengths)} max={max(lengths)} sum={sum(lengths)}")


if __name__ == "__main__":
    main()
