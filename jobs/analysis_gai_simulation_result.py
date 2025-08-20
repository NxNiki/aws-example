"""
In the previous steps, we trained knn cluster model to identify 3 groups of bet behaviors for the data of 2024.
The cluster model was applied to data in month 1, 2, 3 of 2025.
GAI model was trained with data in 2024 and applied to data in 2025 to simulate gaming behaviors.
The GAI model is deployed to generate simulation data.

This script gets simulation data from the GAI model (slot machine) and compares game metrics such as bet, profit, etc.
between gamers from different clusters and using different math tables.
"""

import os
from typing import List, Union

import pandas as pd

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import Anova, DataProfiler, DataVisualizer, split_column_by_threshold
from bituslabs_ds.s3_utils import list_s3_files, read_files

setup_logging(".", "analysis_gai_simulation_result.log")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_process_data(reload: bool = False) -> pd.DataFrame:
    local_output = f"{SCRIPT_DIR}/output/v1_all_sessions_summary_20250806.csv"
    s3_files: List[str] = []
    if reload or not os.path.exists(local_output):
        s3_files += list_s3_files(
            S3_BUCKET,
            prefix="gail_simulator_data_raw/results_v20250725/sim_20250806/",
            pattern=".*_sessions_summary.csv",
        )
        # s3_files += list_s3_files(
        #     S3_BUCKET,
        #     prefix="gail_simulator_data_raw/results_v20250725/sim_20250728_1706/",
        #     pattern=".*_sessions_summary.csv",
        # )
        # s3_files += list_s3_files(
        #     S3_BUCKET,
        #     prefix="gail_simulator_data_raw/results_v20250725/sim_20250725_1947/",
        #     pattern=".*_sessions_summary.csv",
        # )
    data = read_files(s3_files, local_cache_path=local_output, reload=reload)
    data.drop(columns=["active", "session_id", "balance_change"], inplace=True)
    data["cluster_index"] = data["player_id"].str.extract(r"(cluster\d+)_", expand=False).astype(str)
    data = data[data["cluster_index"] == "cluster2"]
    print(data.shape)
    # data = data[data["total_spins"]>40]
    print(data.shape)

    data = split_column_by_threshold(
        data, columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"], threshold=[0, 0, 0, 9]
    )

    return data


if __name__ == "__main__":

    data = load_process_data(reload=False)
    data_profiler = DataProfiler(data, skewness_threshold=1.5)

    viz = DataVisualizer(data_profiler)
    # Plot distributions with distribution statistics
    viz.create_figure(
        layout_cols=data_profiler.processed_numerical_columns, group_col="cluster_index", n_cols=5, fig_size=(7, 3.5)
    )
    viz.add_histogram(show_distribution_stats=True, kde=True)
    viz.figure.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
    viz.display()
    viz.save(f"{SCRIPT_DIR}/figures/gai_simulation_distribution.png")

    # Transform skewed columns
    viz.data_profiler.transform_skewed_columns(pos_suffix="_log", neg_suffix="_exp")
    viz.create_figure(
        layout_cols=data_profiler.processed_numerical_columns, group_col="cluster_index", n_cols=5, fig_size=(7, 3.5)
    )
    viz.add_histogram(show_distribution_stats=True, kde=True)
    viz.figure.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
    viz.display()
    viz.save(f"{SCRIPT_DIR}/figures/gai_simulation_distribution_transformed.png")

    # Plot correlation heatmap
    viz.create_figure(fig_title="Correlation for: cluster 2", fig_size=(12, 7))
    viz.add_correlation_heatmap()
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    viz.display()
    viz.save(f"{SCRIPT_DIR}/figures/gai_simulation_correlation.png")

    # Plot correlation heatmap for machine_id
    viz.create_figure(layout_cols=["machine_id"], fig_title="Correlation for: cluster 2", fig_size=(12, 7))
    viz.add_correlation_heatmap()
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97, wspace=0.25, hspace=0.35)
    viz.display()
    viz.save(f"{SCRIPT_DIR}/figures/gai_simulation_correlation_machine_id.png")

    # two-way ANOVA:
    # anova_between_vars = ["machine_id", "cluster_index"]
    anova_between_vars = ["machine_id"]

    data_profiler.augment_columns(
        columns=data_profiler.processed_numerical_columns, col_name="compound_metric", method=["pca"]
    )

    remove_cols: set = set([])
    anova_dv = [col for col in data_profiler.processed_numerical_columns if col not in remove_cols] + [
        "compound_metric_pca1",
        "compound_metric_pca2",
        "compound_metric_pca3",
    ]

    data_profiler.show_group_stats(
        group_cols=["machine_id", "cluster_index"],
        var_columns=["total_spins", "total_bet", "total_profit"],
        transpose=False,
    )

    data_profiler.show_group_stats(
        group_cols=["machine_id", "cluster_index"],
        var_columns=anova_dv,
        transpose=False,
    )

    anova = Anova(data_profiler.df, between_vars=anova_between_vars, var_columns=anova_dv)
    anova.run_anova()
    anova.run_post_hoc_analysis(anova_dv, effects="main", p_thresh=0.05)
    if len(anova_between_vars) > 1:
        anova.run_post_hoc_analysis(anova_dv, effects="interaction", group_var="cluster_index", p_thresh=0.05)
        anova.show_box_plot(
            x_col="cluster_index",
            group_col="machine_id",
            output_path=f"{SCRIPT_DIR}/figures",
            fig_title="boxplot_two_factors",
        )
    else:
        anova.show_box_plot(x_col="machine_id", output_path=f"{SCRIPT_DIR}/figures", fig_title="boxplot_machine_id")
