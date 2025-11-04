import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_lines_metric_color(
    csv_path,
    metric_cols,
    x_col="bet_index",
    group_col="strategy_name",
    session_col="session_start_date",
    output_dir="plots",
):
    """
    Generate line plots for each strategy_name separately:
    - Color encodes metric (mean vs sum)
    - Line style is solid
    - One figure per group per session
    - Figure filename includes group name
    """
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Colors for metrics
    colors_hex = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3"]  # red, blue, green, orange, purple
    color_cycle = itertools.cycle(colors_hex)

    sessions = df[session_col].unique()

    for session in sessions:
        df_session = df[df[session_col] == session]
        strategies = df_session[group_col].unique()

        for strategy in strategies:
            df_group = df_session[df_session[group_col] == strategy]
            plt.figure(figsize=(14, 6))

            for i, y_col in enumerate(metric_cols):
                color = next(color_cycle)  # color encodes metric
                plt.plot(
                    df_group[x_col],
                    df_group[y_col],
                    label=y_col,
                    color=color,
                    linestyle="-",  # solid line
                    linewidth=1.2,
                    alpha=0.2,
                )

            plt.title(f"{strategy} - {', '.join(metric_cols)} over {x_col} (Session: {session})")
            plt.xlabel(x_col)
            plt.ylabel("Value")
            plt.yscale("symlog", linthresh=100)
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.legend()
            plt.tight_layout()

            safe_strategy = strategy.replace(" ", "_").replace("/", "_")
            filename = f"{'_'.join(metric_cols)}_{safe_strategy}_{session}.png"
            plt.savefig(output_path / filename)
            plt.close()
            print(f"Saved plot: {output_path / filename}")


# Example usage
csv_file = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/result_betindex_avg.csv"
output_dir = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/result_betindex_avg"

# Fish value per group
plot_lines_metric_color(csv_file, metric_cols=["fish_value_mean", "fish_value_sum"], output_dir=output_dir)

# Payout per group
plot_lines_metric_color(csv_file, metric_cols=["payout_mean", "payout_sum"], output_dir=output_dir)
