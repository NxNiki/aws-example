"""
In the previous steps, we trained knn cluster model to identify 3 groups of bet behaviors for the data of 2024.
The cluster model was applied to data in month 1, 2, 3 of 2025.
GAI model was trained with data in 2024 and applied to data in 2025 to simulate gaming behaviors.
The GAI model is deployed to generate simulation data.

This script gets simulation data from the GAI model (slot machine) and compares game metrics such as bet, profit, etc.
between gamers from different clusters and using different math tables.
"""

from collections import defaultdict
from typing import List

import pandas as pd
import pingouin as pg

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import plot_correlation, plot_multiple_box_swarm, split_column_by_threshold
from bituslabs_ds.s3_utils import read_files
from bituslabs_ds.utils import group_iterator, log_transform

setup_logging(".", "analysis_gai_simulation_result.log")


def run_two_way_anova(var_columns: List[str]) -> pd.DataFrame:

    report = defaultdict(list)

    for col in var_columns:
        anova_res = pg.anova(dv=col, between=["machine_id", "cluster_index"], data=data, detailed=True)
        with pd.option_context("display.max_columns", None):
            print(anova_res)

        report["variable"].append(col)
        report["machine_id-p_value"].append(anova_res.loc[0, "p-unc"])
        report["machine_id-eta2"].append(anova_res.loc[0, "np2"])
        report["cluster_index-p_value"].append(anova_res.loc[1, "p-unc"])
        report["cluster_index-eta2"].append(anova_res.loc[1, "np2"])
        report["interaction-p_value"].append(anova_res.loc[2, "p-unc"])
        report["interaction-eta2"].append(anova_res.loc[2, "np2"])

    res = pd.DataFrame(report)
    with pd.option_context("display.max_columns", None):
        print(res)
    return res


if __name__ == "__main__":

    s3_files = [
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_carousels_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_carousels_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_carousels_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_dropTower_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_dropTower_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_dropTower_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_fireworkShow_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_fireworkShow_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_fireworkShow_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_giftShop_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_giftShop_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_giftShop_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_newBee_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_newBee_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_newBee_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_rollerCoaster_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_rollerCoaster_sessions_summary.csv",
        f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_rollerCoaster_sessions_summary.csv",
    ]

    local_output = "./output/v1_all_sessions_summary.csv"

    data = read_files(s3_files, local_cache_path=local_output)

    data["cluster_index"] = data["player_id"].str.extract(r"cluster(\d+)_", expand=False).astype(int)
    data.sort_values("cluster_index", inplace=True)
    data = split_column_by_threshold(
        data, columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"], threshold=[0, 0, 1, 10]
    )

    print(data.head())
    print(data["duration"].describe())
    print(data["batch_count"].describe())

    data = log_transform(
        data, col_names=["free_spins_count", "free_spins_count_above_10", "big_win_count_above_1"], suffix="_log"
    )

    run_two_way_anova(["free_spins_count", "free_spins_count_above_10", "big_win_count_above_1"])

    # plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["balance_change", "base_game_win", "big_win_count_above_1", "free_spins_count_above_10"], n_cols = 1, group_col="cluster_index", log_scale=True)
    # plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["end_balance", "free_game_win_above_0", "return_to_player", "sim_duration"], n_cols = 1, group_col="cluster_index", log_scale=True)
    #  plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["start_balance", "total_bet", "total_profit", "total_spins"], n_cols = 1, group_col="cluster_index", log_scale=True)
    # plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["total_win", "win_count", "win_rate"], n_cols = 1, group_col="cluster_index", log_scale=True)
    #
    # for data_group, group, _ in group_iterator(data, group_col="machine_id"):
    #     # plot correlation between selected columns:
    #     plot_correlation(
    #         data_group[
    #             [
    #                 "balance_change",
    #                 "end_balance",
    #                 "free_game_win",
    #                 "win_rate",
    #                 "return_to_player",
    #                 "start_balance",
    #                 "total_profit",
    #                 "base_game_win",
    #                 "total_win",
    #                 "total_bet",
    #                 "sim_duration",
    #                 "big_win_count",
    #                 "free_spins_count",
    #                 "total_spins",
    #                 "win_count",
    #             ]
    #         ],
    #         title=f"correlation for: {group}",
    #     )
