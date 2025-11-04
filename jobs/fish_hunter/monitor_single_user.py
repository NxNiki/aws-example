from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style


def plot_user_series(
    csv_path,
    value_col="fish_value",
    x_col="event_timestamp",
    user_col="user_id",
    session_col="session_id",
    session_flag_col="is_session_start_flag",
    date_col="date",
    strategy_col="strategy_name",
    output_dir=None,
):
    """
    Plot a value column over event_time for each user in the dataset.
    Removes the first session if it does not start with session_start_flag == 1.
    Saves each user’s plot separately.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV file.
    value_col : str
        Column name to plot on the y-axis.
    x_col : str
        Column name to use for ordering (x-axis).
    user_col : str
        Column name for user ID.
    session_col : str
        Column name for session ID.
    session_flag_col : str
        Column name for session start flag (1 = start of session).
    date_col : str
        Column name for date (used as x-tick labels).
    strategy_col : str
        Column name for strategy name (displayed in title).
    output_dir : str or Path, optional
        Directory to save user plots. If None, plots are shown instead.
    """

    df = pd.read_csv(csv_path)
    df = df.sort_values(by=[user_col, x_col]).reset_index(drop=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for user_id, group in df.groupby(user_col):
        group = group.copy()

        # identify the first session
        first_session = group[session_col].iloc[0]
        first_session_rows = group[group[session_col] == first_session]

        # remove if first session does not start properly
        if first_session_rows[session_flag_col].iloc[0] != 1:
            group = group[group[session_col] != first_session]

        # skip empty groups
        if group.empty:
            print(f"Skipping user {user_id}: no valid rows after filtering.")
            continue

        # strategy name (should be identical)
        if "BOOST_POOL" in group[strategy_col].unique():
            strategy_name = "BOOST"
        elif "DYNAMIC_RTP" in group[strategy_col].unique():
            strategy_name = "DYNA_RTP"
        else:
            strategy_name = "DEFAULT"

        # strategy_name = group[strategy_col].mode()[0] if strategy_col in group else "N/A"

        # plot
        plt.figure(figsize=(10, 5))

        # gray connecting line
        plt.plot(
            group[x_col],
            group[value_col],
            color="gray",
            linewidth=1,
            linestyle="--",
            zorder=1,
        )

        # plot dots
        killed_mask = group["killed"] == 1
        alive_mask = ~killed_mask

        # normal (not killed) dots - blue
        plt.scatter(
            group.loc[alive_mask, x_col], group.loc[alive_mask, value_col], s=1, color="blue", label="Alive", zorder=2
        )

        # killed == 1 dots - red
        plt.scatter(
            group.loc[killed_mask, x_col],
            group.loc[killed_mask, value_col],
            s=1.2,
            color="red",
            label="Killed",
            zorder=3,
        )

        plt.title(f"User {user_id} - Strategy: {strategy_name}")
        plt.xlabel("Event Time")
        plt.ylabel(value_col)

        # use date for x-ticks (omit duplicates)
        xticks = []
        xticklabels = []
        last_date = None
        for idx, date in enumerate(group[date_col]):
            if date != last_date:
                xticks.append(idx)
                xticklabels.append(date)
                last_date = date

        plt.xticks(xticks, xticklabels, rotation=45, ha="right")
        plt.tight_layout()

        if output_dir:
            output_path = output_dir / f"user_{user_id}_{value_col}_{strategy_name}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            print(f"Saved plot for user {user_id} to: {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    csv_file = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/Result_21.csv"
    plot_user_series(
        csv_path=csv_file,
        value_col="fish_value",
        output_dir="/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/plots",
    )
