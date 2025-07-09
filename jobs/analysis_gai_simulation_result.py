"""
In the previous steps, we trained knn cluster model to identify 3 groups of bet behaviors for the data of 2024.
The cluster model was applied to data in month 1, 2, 3 of 2025.
GAI model was trained with data in 2024 and applied to data in 2025 to simulate gaming behaviors.
The GAI model is deployed to generate simulation data.

This script gets simulation data from the GAI model (slot machine) and compares game metrics such as bet, profit, etc.
between gamers from different clusters and using different math tables.
"""

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import plot_correlation, plot_multiple_box_swarm, split_column_by_threshold
from bituslabs_ds.s3_utils import read_files

setup_logging(".", "analysis_gai_simulation_result.log")

s3_files = [
    f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_carousels_sessions_summary.csv",
    f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_carousels_sessions_summary.csv",
    f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_carousels_sessions_summary.csv",
]

local_output = "./output/v1_carousels_sessions_summary.csv"


data = read_files(s3_files, local_cache_path=local_output)

data["cluster_index"] = data["player_id"].str.extract(r"cluster(\d+)_", expand=False).astype(int)
data.sort_values("cluster_index", inplace=True)
data = split_column_by_threshold(
    data, columns=["free_game_win", "big_win_count", "free_spins_count"], threshold=[0, 1, 10]
)

print(data.head())
print(data["duration"].describe())
print(data["batch_count"].describe())

# plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["balance_change", "base_game_win", "big_win_count_above_1", "free_spins_count_above_10"], n_cols = 4, group_col="cluster_index", log_scale=True)
# plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["end_balance", "free_game_win_above_0", "return_to_player", "sim_duration"], n_cols = 4, group_col="cluster_index", log_scale=True)
# plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["start_balance", "total_bet", "total_profit", "total_spins"], n_cols = 4, group_col="cluster_index", log_scale=True)
# plot_multiple_box_swarm(data, x_cols=["machine_id"], y_cols=["total_win", "win_count", "win_rate"], n_cols = 3, group_col="cluster_index", log_scale=True)


# plot correlation between selected columns:
plot_correlation(
    data[
        [
            "balance_change",
            "end_balance",
            "free_game_win",
            "win_rate",
            "return_to_player",
            "start_balance",
            "total_profit",
            "base_game_win",
            "total_win",
            "total_bet",
            "sim_duration",
            "big_win_count",
            "free_spins_count",
            "total_spins",
            "win_count",
        ]
    ]
)
