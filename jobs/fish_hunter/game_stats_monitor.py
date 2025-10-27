import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_file = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/Result_13.csv"
output_dir = os.path.dirname(data_file)


def plot_two_metrics_all_strategies(df, x_col, y1_col, y2_col, title, xlabel, ylabel1, ylabel2):
    """
    Plot two columns with all strategies in one figure.
    Uses twin axes if values differ significantly in scale.
    """
    # Check if we need twin axis by comparing scales across all data
    y1_max = abs(df[y1_col]).max()
    y2_max = abs(df[y2_col]).max()
    scale_ratio = max(y1_max, y2_max) / min(y1_max, y2_max) if min(y1_max, y2_max) > 0 else 1

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Color palette and markers for strategies
    strategy_colors = {"DEFAULT_FALLBACK": "blue", "BOOST_POOL": "green", "DYNAMIC_RTP": "orange"}
    strategy_markers = {"DEFAULT_FALLBACK": "o", "BOOST_POOL": "s", "DYNAMIC_RTP": "^"}

    if scale_ratio > 10:
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
        ax1.legend(loc="best", fontsize=8)

    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_three_metrics_all_strategies(df, x_col, y1_col, y2_col, y3_col, title, xlabel, ylabel1, ylabel2, ylabel3):
    """
    Plot three columns with all strategies in one figure.
    Uses twin axes if values differ significantly in scale.
    """
    # Check if we need twin axis by comparing scales across all data
    y1_max = abs(df[y1_col]).max()
    y2_max = abs(df[y2_col]).max()
    y3_max = abs(df[y3_col]).max()

    max_values = [y1_max, y2_max, y3_max]
    max_max = max(max_values)
    min_max = min(max_values)
    scale_ratio = max_max / min_max if min_max > 0 else 1

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Color palette and markers for strategies
    strategy_colors = {"DEFAULT_FALLBACK": "blue", "BOOST_POOL": "green", "DYNAMIC_RTP": "orange"}
    strategy_markers = {"DEFAULT_FALLBACK": "o", "BOOST_POOL": "s", "DYNAMIC_RTP": "^"}

    if scale_ratio > 10:
        # Use twin axis - group y1 with either y2 or y3 based on which is closer
        # Plot y1 on left axis
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) > 0:
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[y1_col],
                    marker=strategy_markers.get(strategy, "o"),
                    color=strategy_colors.get(strategy, "black"),
                    linestyle="-",
                    linewidth=2,
                    label=f"{strategy} - {ylabel1}",
                    markersize=6,
                )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel1, color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        # Plot y2 and y3 on right axis
        ax2 = ax1.twinx()
        for strategy in df["strategy_name"].unique():
            strategy_df = df[df["strategy_name"] == strategy].sort_values(x_col)
            if len(strategy_df) > 0:
                color = strategy_colors.get(strategy, "black")
                marker = strategy_markers.get(strategy, "o")
                ax2.plot(
                    strategy_df[x_col],
                    strategy_df[y2_col],
                    color=color,
                    marker=marker,
                    linestyle="--",
                    linewidth=1,
                    label=f"{strategy} - {ylabel2}",
                    markersize=4,
                )
                ax2.plot(
                    strategy_df[x_col],
                    strategy_df[y3_col],
                    color=color,
                    marker=marker,
                    linestyle=":",
                    linewidth=1,
                    label=f"{strategy} - {ylabel3}",
                    markersize=4,
                )

        ax2.set_ylabel(f"{ylabel2} / {ylabel3}", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    else:
        # Plot all on same axis
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
                    linewidth=2,
                    label=f"{strategy} - {ylabel1}",
                    markersize=6,
                )
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[y2_col],
                    color=color,
                    marker=marker,
                    linestyle="--",
                    linewidth=1,
                    label=f"{strategy} - {ylabel2}",
                    markersize=4,
                )
                ax1.plot(
                    strategy_df[x_col],
                    strategy_df[y3_col],
                    color=color,
                    marker=marker,
                    linestyle=":",
                    linewidth=1,
                    label=f"{strategy} - {ylabel3}",
                    markersize=4,
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
    fig4 = plot_three_metrics_all_strategies(
        df,
        "date",
        "num_users",
        "num_users_day1",
        "num_users_day3",
        "User Metrics - All Strategies",
        "Date",
        "Num Users",
        "Num Users Day 1",
        "Num Users Day 3",
    )
    plt.savefig(f"{output_dir}/user_metrics_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print("Generated user_metrics_all.png")

    # Figure 5: Retention ratios - All strategies together
    fig5 = plot_three_metrics_all_strategies(
        df,
        "date",
        "num_users",
        "retention_ratio_day1",
        "retention_ratio_day3",
        "User Count and Retention Ratios - All Strategies",
        "Date",
        "Num Users",
        "Retention Ratio Day 1",
        "Retention Ratio Day 3",
    )
    plt.savefig(f"{output_dir}/retention_ratios_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print("Generated retention_ratios_all.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
