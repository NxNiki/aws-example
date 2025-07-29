"""
In the previous steps, we trained knn cluster model to identify 3 groups of bet behaviors for the data of 2024.
The cluster model was applied to data in month 1, 2, 3 of 2025.
GAI model was trained with data in 2024 and applied to data in 2025 to simulate gaming behaviors.
The GAI model is deployed to generate simulation data.

This script gets simulation data from the GAI model (slot machine) and compares game metrics such as bet, profit, etc.
between gamers from different clusters and using different math tables.
"""

from typing import List, Union

import pandas as pd

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import Anova, DataProfiler, split_column_by_threshold
from bituslabs_ds.s3_utils import list_s3_files, read_files

setup_logging(".", "analysis_gai_simulation_result.log")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


def load_process_data(reload: bool = False) -> pd.DataFrame:
    local_output = "./output/v1_all_sessions_summary_fixed.csv"
    s3_files: List[str] = []
    if reload:
        s3_files = list_s3_files(
            S3_BUCKET, prefix="gail_simulator_data_raw/results_fixed/", pattern="sim_20250713_.*_sessions_summary.csv"
        )
    data = read_files(s3_files, local_cache_path=local_output, reload=reload)
    data.drop(columns=["session_id", "balance_change"], inplace=True)
    data["cluster_index"] = data["player_id"].str.extract(r"(cluster\d+)_", expand=False).astype(str)
    print(data.shape)
    # data = data[data["total_spins"]>40]
    print(data.shape)

    data = split_column_by_threshold(
        data, columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"], threshold=[0, 0, 1, 10]
    )

    return data


if __name__ == "__main__":

    data = load_process_data(reload=False)
    data_profiler = DataProfiler(data)
    data_profiler.plot_distribution(figure_name="./figures/gai_simulation_distribution.png", log=True, add_kde=True)
    data_profiler.transform_skewed_columns(pos_suffix="_log", neg_suffix="_exp")
    data_profiler.plot_distribution(
        figure_name="./figures/gai_simulation_distribution_log.png", log=False, add_kde=True
    )
    data_profiler.augment_columns(
        columns=["total_bet_log", "total_profit_log"], col_name="compound_metric", method=["mean", "pca"]
    )
    data_profiler.show_group_stats(
        group_cols=["machine_id", "cluster_index"],
        var_columns=["total_spins", "total_bet", "total_profit", "total_profit_log"],
        transpose=True,
    )
    data_profiler.plot_correlation_heatmap(title=f"correlation for: all group")

    # two-way ANOVA:
    remove_cols = {
        "active",
        "bonus_triggered",
        "base_game_win_log",
        "big_win_count_log",
        "free_game_win_log",
        "win_count_log",
        "duration_log",
        "sim_duration_log",
        "final_balance_log",
        "base_game_win_below_0",
        "free_game_win_below_0",
    }
    anova_dv = [col for col in data_profiler.processed_numerical_columns if col not in remove_cols] + [
        "compound_metric_mean",
        "compound_metric_pca1",
        "compound_metric_pca2",
    ]

    anova = Anova(data_profiler.df, between_vars=["machine_id", "cluster_index"], var_columns=anova_dv)
    anova.run_anova()
    anova.run_post_hoc_analysis(anova_dv, effects="main")
    anova.run_post_hoc_analysis(anova_dv, effects="interaction", group_var="cluster_index")
    anova.show_box_plot("cluster_index", "machine_id")
