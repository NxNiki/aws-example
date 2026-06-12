"""
EDA script to compute and visualize RTP time course for two A/B groups.

Features:
- Load a local CSV file
- Compute per-user, per-session RTP (sum of payout / sum of bet) partitioned by A/B group
- Plot time course of RTP for each group using the event time column (resampled)
"""

from __future__ import annotations

import os

import matplotlib

# Use a non-interactive backend suitable for servers/CLI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# Configuration - Update these paths and column names as needed
# =============================================================================

# File paths
CSV_FILE_PATH = "/path/to/your/data.csv"  # Update this path
OUTPUT_PLOT_PATH = "/Users/niuxin/Documents/aws-example/jobs/figures/rtp_time_course.png"
OUTPUT_SESSION_RTP_PATH = "/Users/niuxin/Documents/aws-example/jobs/output/session_rtp.csv"

# Column names in your CSV
PAYOUT_COL = "payout"
BET_COL = "bet_amount"
USER_COL = "user_id"
SESSION_COL = "session_id"
GROUP_COL = "ab_partition"
TIME_COL = "event_time"

# Time resampling frequency for plotting
TIME_FREQ = "15min"  # Options: 5min, 15min, 1H, 1D, etc.


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}. Available: {list(df.columns)}")


def compute_session_rtp(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to per user-session per group RTP.

    RTP = sum(payout) / sum(bet) within each (group, user, session)
    """
    grouped = (
        df.groupby([GROUP_COL, USER_COL, SESSION_COL], dropna=False)[[PAYOUT_COL, BET_COL]]
        .apply(lambda x: x.sort_values(TIME_COL).cumsum() if TIME_COL in x else x.sum(min_count=1))
        .reset_index()
    )
    grouped = grouped[grouped[BET_COL] != 0]
    grouped["rtp"] = grouped[PAYOUT_COL] / grouped[BET_COL]
    return grouped


def compute_timecourse_rtp(df: pd.DataFrame) -> pd.DataFrame:
    """Resample by time within each group and compute RTP = sum(payout) / sum(bet)."""
    # Keep only necessary columns
    ts_df = df[[TIME_COL, GROUP_COL, PAYOUT_COL, BET_COL]].copy()
    ts_df = ts_df.dropna(subset=[TIME_COL])
    ts_df[TIME_COL] = pd.to_datetime(ts_df[TIME_COL], errors="coerce")
    ts_df = ts_df.dropna(subset=[TIME_COL])
    ts_df = ts_df.sort_values(TIME_COL)

    # Resample within each group and compute sums, then RTP
    ts_df = ts_df.set_index(TIME_COL)
    pieces = []
    for group_value, gdf in ts_df.groupby(GROUP_COL):
        agg = gdf.resample(TIME_FREQ)[[PAYOUT_COL, BET_COL]].sum(min_count=1)
        agg = agg[agg[BET_COL] != 0]
        agg["rtp"] = agg[PAYOUT_COL] / agg[BET_COL]
        agg[GROUP_COL] = group_value
        pieces.append(agg.reset_index())
    if not pieces:
        return pd.DataFrame(columns=[TIME_COL, GROUP_COL, "rtp"])  # empty
    out = pd.concat(pieces, ignore_index=True)
    return out[[TIME_COL, GROUP_COL, "rtp"]]


def plot_timecourse(timecourse_df: pd.DataFrame) -> None:
    if timecourse_df.empty:
        raise ValueError("No data to plot after resampling. Check input columns and time-freq.")

    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH) or ".", exist_ok=True)
    plt.figure(figsize=(12, 6))
    for group_value, gdf in timecourse_df.groupby(GROUP_COL):
        plt.plot(gdf[TIME_COL], gdf["rtp"], label=str(group_value))
    plt.xlabel(TIME_COL)
    plt.ylabel("RTP (payout / bet)")
    plt.title("RTP Time Course by Group")
    plt.legend(title=GROUP_COL)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    # Load CSV
    df = pd.read_csv(CSV_FILE_PATH, parse_dates=[TIME_COL], infer_datetime_format=True)

    # Validate columns
    required_cols = [PAYOUT_COL, BET_COL, USER_COL, SESSION_COL, GROUP_COL, TIME_COL]
    validate_columns(df, required_cols)

    # Compute per-session RTP per group
    session_rtp = compute_session_rtp(df)

    # Save session RTP
    os.makedirs(os.path.dirname(OUTPUT_SESSION_RTP_PATH) or ".", exist_ok=True)
    session_rtp.to_csv(OUTPUT_SESSION_RTP_PATH, index=False)

    # Compute time-course RTP per group
    timecourse_rtp = compute_timecourse_rtp(df)

    # Plot
    plot_timecourse(timecourse_rtp)

    # Print brief summary to stdout
    print("=== Session RTP (head) ===")
    print(session_rtp.head(10).to_string(index=False))
    print()
    print("=== Time-course RTP (head) ===")
    print(timecourse_rtp.head(10).to_string(index=False))
    print()
    print(f"Saved time-course plot to: {OUTPUT_PLOT_PATH}")
    print(f"Saved session RTP to: {OUTPUT_SESSION_RTP_PATH}")


if __name__ == "__main__":
    main()
