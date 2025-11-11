import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_file = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/result_daily_stats.csv"
output_dir = os.path.dirname(data_file)

# Color palette and markers for strategies
strategy_colors = {"DEFAULT_FALLBACK": "blue", "BOOST_POOL": "green", "DYNAMIC_RTP": "orange"}
strategy_markers = {"DEFAULT_FALLBACK": "o", "BOOST_POOL": "s", "DYNAMIC_RTP": "^"}


def plot_two_metrics_all_strategies(
    df,
    x_col,
    y1_col,
    y2_col,
    title,
    xlabel,
    ylabel1,
    ylabel2,
    twin_axis=True,
    log_y=False,
):
    """
    Plot two columns with all strategies in one figure.
    Uses twin axes if values differ significantly in scale.
    Optionally allows y-axis (or both y-axes) to be log scale.

    Args:
        df: DataFrame containing the data
        x_col: Name of the x-column
        y1_col: Name of the first y-column (left axis or axis)
        y2_col: Name of the second y-column (right axis or same axis)
        title: Title for the plot
        xlabel: X-axis label
        ylabel1: Left or primary y-axis label
        ylabel2: Right or secondary y-axis label
        twin_axis: Whether to use a twin y-axis (if scale difference is large)
        log_y: Whether to use log scale for the y-axis/axes (default: False)
    """
    # Check if we need twin axis by comparing scales across all data
    y1_max = abs(df[y1_col]).max()
    y2_max = abs(df[y2_col]).max()
    scale_ratio = max(y1_max, y2_max) / min(y1_max, y2_max) if min(y1_max, y2_max) > 0 else 1

    fig, ax1 = plt.subplots(figsize=(14, 7))

    if scale_ratio > 10 and twin_axis:
        # Use twin axis
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) > 0:
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[y1_col],
                    marker=strategy_markers.get(strategy, "o"),
                    color=strategy_colors.get(strategy, "black"),
                    linestyle="-",
                    label=f"{strategy} - {ylabel1}",
                )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel1, color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        if log_y:
            ax1.set_yscale("symlog", linthresh=10)

        ax2 = ax1.twinx()
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) > 0:
                ax2.plot(
                    strategy_df[x_col],
                    strategy_df[y2_col],
                    marker=strategy_markers.get(strategy, "o"),
                    color=strategy_colors.get(strategy, "black"),
                    linestyle="--",
                    label=f"{strategy} - {ylabel2}",
                )

        ax2.set_ylabel(ylabel2, color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        if log_y:
            ax2.set_yscale("symlog", linthresh=10)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    else:
        # Plot on same axis
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) > 0:
                color = strategy_colors.get(strategy, "black")
                marker = strategy_markers.get(strategy, "o")
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[y1_col],
                    color=color,
                    marker=marker,
                    linestyle="-",
                    label=f"{strategy} - {ylabel1}",
                )
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[y2_col],
                    color=color,
                    marker=marker,
                    linestyle="--",
                    label=f"{strategy} - {ylabel2}",
                )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Value")
        if log_y:
            ax1.set_yscale("symlog", linthresh=10)
        ax1.legend(loc="best", fontsize=8)

    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_three_metrics_all_strategies(df, x_col, y1_col, y2_col, y3_col, title, xlabel, ylabel1, ylabel2, ylabel3):
    """
    Plot three columns with all strategies in one figure.
    Uses pairwise comparison to determine optimal grouping for twin axes.
    Groups the two metrics with the most similar scales on one axis.
    """
    # Calculate max values for each metric
    y1_max = abs(df[y1_col]).max()
    y2_max = abs(df[y2_col]).max()
    y3_max = abs(df[y3_col]).max()

    # Calculate pairwise scale ratios
    def calc_scale_ratio(max1, max2):
        """Calculate scale ratio between two values"""
        if max1 == 0 or max2 == 0:
            return float("inf")
        return max(max1, max2) / min(max1, max2)

    ratio_12 = calc_scale_ratio(y1_max, y2_max)
    ratio_13 = calc_scale_ratio(y1_max, y3_max)
    ratio_23 = calc_scale_ratio(y2_max, y3_max)

    # Find the pair with the smallest scale ratio (most similar)
    min_ratio = min(ratio_12, ratio_13, ratio_23)

    # Determine which metrics to group together
    # Store (metric_index, left_metrics, right_metrics, left_labels, right_label)
    if min_ratio == ratio_12:
        # y1 and y2 are most similar - group them, y3 goes to right axis
        left_cols = [(y1_col, ylabel1, "-"), (y2_col, ylabel2, "--")]
        right_col = (y3_col, ylabel3, ":")
    elif min_ratio == ratio_13:
        # y1 and y3 are most similar - group them, y2 goes to right axis
        left_cols = [(y1_col, ylabel1, "-"), (y3_col, ylabel3, ":")]
        right_col = (y2_col, ylabel2, "--")
    else:  # ratio_23 is smallest
        # y2 and y3 are most similar - group them, y1 goes to right axis
        left_cols = [(y2_col, ylabel2, "--"), (y3_col, ylabel3, ":")]
        right_col = (y1_col, ylabel1, "-")

    # Check if we need twin axis (threshold = 10x difference)
    overall_ratio = max(y1_max, y2_max, y3_max) / min(y1_max, y2_max, y3_max)
    use_twin_axis = overall_ratio > 10

    fig, ax1 = plt.subplots(figsize=(14, 7))

    if use_twin_axis:
        # Plot left axis metrics
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) == 0:
                continue
            color = strategy_colors.get(strategy, "black")
            marker = strategy_markers.get(strategy, "o")

            for col, label, linestyle in left_cols:
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[col],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2 if linestyle == "-" else 1,
                    label=f"{strategy} - {label}",
                    markersize=6 if linestyle == "-" else 4,
                )

        # Create right axis labels from left metrics
        left_labels = " / ".join([label for _, label, _ in left_cols])
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(left_labels, color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        # Plot right axis metric
        ax2 = ax1.twinx()
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) == 0:
                continue
            color = strategy_colors.get(strategy, "black")
            marker = strategy_markers.get(strategy, "o")

            ax2.plot(
                strategy_df[x_col],
                strategy_df[right_col[0]],
                color=color,
                marker=marker,
                linestyle=right_col[2],
                linewidth=2,
                label=f"{strategy} - {right_col[1]}",
                markersize=6,
            )

        ax2.set_ylabel(right_col[1], color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    else:
        # Plot all on same axis
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) == 0:
                continue
            color = strategy_colors.get(strategy, "black")
            marker = strategy_markers.get(strategy, "o")

            # Plot all three metrics
            for col, label, linestyle in left_cols:
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[col],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2 if linestyle == "-" else 1,
                    label=f"{strategy} - {label}",
                    markersize=6 if linestyle == "-" else 4,
                )
            ax1.plot(
                strategy_df[x_col],
                strategy_df[right_col[0]],
                color=color,
                marker=marker,
                linestyle=right_col[2],
                linewidth=2,
                label=f"{strategy} - {right_col[1]}",
                markersize=6,
            )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Value")
        ax1.legend(loc="best", fontsize=8)

    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def main():
    # Read the data
    df = pd.read_csv(data_file)
    print(df[df["date"] > "2025-10-22"].to_markdown())
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["strategy_name", "date"])

    df = df[df["date"] <= "2025-11-05"]

    df["retention_ratio_day1"] = df["num_users_day1"] / df["num_users"]
    df["retention_ratio_day3"] = df["num_users_day3"] / df["num_users"]
    df["total_bet_per_user"] = df["daily_total_bet"] / df["num_users"]

    # Strategy colors and markers
    strategy_colors = {"DEFAULT_FALLBACK": "blue", "BOOST_POOL": "green", "DYNAMIC_RTP": "orange"}
    strategy_markers = {"DEFAULT_FALLBACK": "o", "BOOST_POOL": "s", "DYNAMIC_RTP": "^"}

    # Figure 1: avg_killed_fish_value and avg_fish_value - All strategies together
    fig1 = plot_two_metrics_all_strategies(
        df,
        "date",
        "avg_killed_fish_value",
        "avg_fish_value",
        "Fish Values - All Strategies",
        "Date",
        "Avg Killed Fish Value",
        "Avg Fish Value",
    )
    plt.savefig(f"{output_dir}/fish_values_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("Generated fish_values_all.png")

    # Figure 2: avg_killed_profit and avg_profit - All strategies together
    fig2 = plot_two_metrics_all_strategies(
        df,
        "date",
        "avg_killed_profit",
        "avg_profit",
        "Profit Metrics - All Strategies",
        "Date",
        "Avg Killed Profit",
        "Avg Profit",
    )
    plt.savefig(f"{output_dir}/profit_metrics_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("Generated profit_metrics_all.png")

    # Figure 3: rtp and daily_total_bet - All strategies together
    fig3 = plot_two_metrics_all_strategies(
        df,
        "date",
        "rtp",
        "total_bet_per_user",
        "RTP and Daily Total Bet - All Strategies",
        "Date",
        "RTP",
        "Total Bet Per User",
    )
    plt.savefig(f"{output_dir}/rtp_and_bet_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("Generated rtp_and_bet_all.png")

    # Figure 4: num_users, num_users_day1, num_users_day3 - All strategies together
    fig4 = plot_two_metrics_all_strategies(
        df,
        "date",
        "num_users",
        "num_users_day1",
        "User Metrics - All Strategies",
        "Date",
        "Num Users",
        "Num Users Day 1",
        twin_axis=False,
        log_y=True,
    )
    plt.savefig(f"{output_dir}/user_number_day1_retention.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print("Generated user_number_day1_retention.png")

    # Figure 4: num_users, num_users_day1, num_users_day3 - All strategies together
    fig4 = plot_two_metrics_all_strategies(
        df,
        "date",
        "num_users",
        "num_users_day3",
        "User Metrics - All Strategies",
        "Date",
        "Num Users",
        "Num Users Day 3",
        twin_axis=False,
        log_y=True,
    )
    plt.savefig(f"{output_dir}/user_number_day3_retention.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print("Generated user_number_day3_retention.png")

    # Figure 5: Retention ratios - All strategies together
    fig5 = plot_two_metrics_all_strategies(
        df,
        "date",
        "retention_ratio_day1",
        "retention_ratio_day3",
        "User Count and Retention Ratios - All Strategies",
        "Date",
        "Retention Ratio Day 1",
        "Retention Ratio Day 3",
    )
    plt.savefig(f"{output_dir}/retention_ratios_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print("Generated retention_ratios_all.png")

    # Figure 6: bullet hit ratio - All strategies together
    fig5 = plot_two_metrics_all_strategies(
        df,
        "date",
        "num_bullets_per_user",
        "bullets_kill_ratio",
        "User Count and Retention Ratios - All Strategies",
        "Date",
        "Num Bullets Per User",
        "Bullets Kill Ratio",
    )
    plt.savefig(f"{output_dir}/bullets_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print("Generated bullets_all.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
