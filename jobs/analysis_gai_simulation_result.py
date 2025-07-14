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
from bituslabs_ds.s3_utils import list_s3_files, read_files
from bituslabs_ds.utils import group_iterator, log_transform

setup_logging(".", "analysis_gai_simulation_result.log")

# Set max columns and display width
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Set a wide enough console width
pd.set_option("display.expand_frame_repr", False)  # Don't wrap columns


def run_two_way_anova(
    data: pd.DataFrame, var_columns: List[str], transpose_report: bool = False, p_thresh: float = 0.05
) -> pd.DataFrame:

    report = defaultdict(list)

    for col in var_columns:
        anova_output = pg.anova(dv=col, between=["machine_id", "cluster_index"], data=data, detailed=True)
        print(anova_output)

        if all(anova_output["p-unc"] > p_thresh):
            print(f"skipping {col}")
            continue

        report["variable"].append(col)
        report["machine_id-p_value"].append(anova_output.loc[0, "p-unc"])
        report["machine_id-eta2"].append(anova_output.loc[0, "np2"])
        report["cluster_index-p_value"].append(anova_output.loc[1, "p-unc"])
        report["cluster_index-eta2"].append(anova_output.loc[1, "np2"])
        report["interaction-p_value"].append(anova_output.loc[2, "p-unc"])
        report["interaction-eta2"].append(anova_output.loc[2, "np2"])

    if transpose_report:
        res = pd.DataFrame(report).set_index("variable").transpose()
    else:
        res = pd.DataFrame(report)

    print(res.to_markdown(index=False))
    return res


def _perform_and_report_post_hoc(
    dv_col: str, between_factor: str, method: str, data_df: pd.DataFrame, report_dict: defaultdict, p_thresh: float
):
    """
    Helper function to perform a post-hoc test and append results to the report dictionary.
    """
    print(f"  Performing {method} for {between_factor} on {dv_col}")

    post_hoc_res = pd.DataFrame()  # Initialize to an empty DataFrame

    if method == "Sidak":
        post_hoc_res = pg.pairwise_ttests(
            dv=dv_col, between=between_factor, data=data_df, padjust="sidak", effsize="hedges"
        )
        p_col = "p-sidak"
        # Filter for significant results for printing and reporting
        post_hoc_res.rename(columns={"p-corr": p_col}, inplace=True)
        sig_res = post_hoc_res[post_hoc_res[p_col] < p_thresh]
        print(sig_res[["A", "B", "p-unc", p_col, "hedges", "BF10"]].to_markdown(index=False))

    elif method == "Games-Howell":
        # Games-Howell is only meaningful for factors with > 2 levels.
        if len(data_df[between_factor].unique()) <= 2:
            print(f"  Skipping explicit Games-Howell for {between_factor} on {dv_col} (only 2 levels).")
            return
        post_hoc_res = pg.pairwise_gameshowell(dv=dv_col, between=between_factor, data=data_df, effsize="hedges")
        p_col = "p-gameshowell"
        post_hoc_res.rename(columns={"pval": p_col}, inplace=True)
        # Filter for significant results for printing and reporting
        sig_res = post_hoc_res[post_hoc_res[p_col] < p_thresh]
        print(sig_res[["A", "B", p_col, "hedges"]].to_markdown(index=False))

    elif method == "Tukey":
        # Tukey's HSD typically assumes equal variances (homoscedasticity) and balanced groups,
        # though pingouin's implementation (pairwise_tukey) can handle unequal N.
        # If Levene's test for homogeneity of variance (which pingouin runs in anova output if detailed=True)
        # indicates heterogeneity (p < .05), Games-Howell is generally preferred.
        if len(data_df[between_factor].unique()) <= 2:
            print(f"  Skipping Tukey's HSD for {between_factor} on {dv_col} (only 2 levels).")
            return
        post_hoc_res = pg.pairwise_tukey(dv=dv_col, between=between_factor, data=data_df, effsize="hedges")
        p_col = "p-tukey"
        post_hoc_res.rename(columns={"p-corr": p_col}, inplace=True)
        # Filter for significant results for printing and reporting
        sig_res = post_hoc_res[post_hoc_res[p_col] < p_thresh]
        print(sig_res[["A", "B", p_col, "hedges"]].to_markdown(index=False))

    else:
        raise ValueError(f"Unknown post-hoc method: {method}")

    for _, row in sig_res.iterrows():
        report_dict["variable"].append(dv_col)
        report_dict["factor"].append(between_factor)
        report_dict["comparison"].append(f"{row['A']} vs {row['B']}")
        report_dict["method"].append(method)
        report_dict["p_value"].append(row[p_col])
        report_dict["effect_size"].append(row["hedges"])


def run_post_hoc_analysis(
    data: pd.DataFrame,
    anova_results: pd.DataFrame,
    var_columns: List[str],
    p_thresh: float = 0.05,
    method: str = "Tukey",
) -> pd.DataFrame:

    post_hoc_report: defaultdict = defaultdict(list)

    print("\n--- Running Post-Hoc Analysis ---")

    for col in var_columns:
        # Check if the variable was processed by ANOVA and if any main effect or interaction was significant
        if col not in anova_results["variable"].values:
            print(
                f"Skipping post-hoc for {col} as it was not included in the ANOVA summary (likely no significant effects)."
            )
            continue

        anova_row = anova_results[anova_results["variable"] == col].iloc[0]

        # Post-hoc for machine_id if significant
        if anova_row["machine_id-p_value"] < p_thresh:
            print(f"\nSignificance detected for machine_id on {col} (p={anova_row['machine_id-p_value']:.4f})")
            _perform_and_report_post_hoc(col, "machine_id", method, data, post_hoc_report, p_thresh)

        # Post-hoc for cluster_index if significant
        if anova_row["cluster_index-p_value"] < p_thresh:
            print(f"\nSignificance detected for cluster_index on {col} (p={anova_row['cluster_index-p_value']:.4f})")
            _perform_and_report_post_hoc(col, "cluster_index", method, data, post_hoc_report, p_thresh)

        # Post-hoc for interaction if significant
        if anova_row["interaction-p_value"] < p_thresh:
            print(
                f"\nSignificance detected for interaction (machine_id * cluster_index) on {col} (p={anova_row['interaction-p_value']:.4f})"
            )
            # Create a combined interaction group temporarily
            data["interaction_group"] = data["machine_id"].astype(str) + "_" + data["cluster_index"].astype(str)
            _perform_and_report_post_hoc(col, "interaction_group", method, data, post_hoc_report, p_thresh)
            data.drop(columns=["interaction_group"], inplace=True)

    res_post_hoc = pd.DataFrame(post_hoc_report)
    if not res_post_hoc.empty:
        print("\n--- Summary Post-Hoc Report (Significant Comparisons Only) ---")
        print(res_post_hoc.to_markdown(index=False))
    else:
        print("\nNo significant main effects or interactions found in ANOVA, so no post-hoc tests were performed.")
    return res_post_hoc


if __name__ == "__main__":

    # s3_files = [
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_carousels_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_carousels_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_carousels_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_dropTower_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_dropTower_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_dropTower_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_fireworkShow_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_fireworkShow_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_fireworkShow_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_giftShop_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_giftShop_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_giftShop_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_newBee_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_newBee_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_newBee_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster0_player_rollerCoaster_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster1_player_rollerCoaster_sessions_summary.csv",
    #     f"s3://{S3_BUCKET}/gail_simulator_data_raw/temp/test_run_20250702_1e5/summary/v1_cluster2_player_rollerCoaster_sessions_summary.csv",
    # ]

    # new files with simulation issue fixed:
    # s3_files = list_s3_files(S3_BUCKET, prefix="gail_simulator_data_raw/results_fixed/", pattern="sim_20250713_.*_sessions_summary.csv")
    s3_files: List[str] = []
    local_output = "./output/v1_all_sessions_summary.csv"
    data = read_files(s3_files, local_cache_path=local_output, reload=False)

    data["cluster_index"] = data["player_id"].str.extract(r"cluster(\d+)_", expand=False).astype(int)
    data.sort_values("cluster_index", inplace=True)
    data = split_column_by_threshold(
        data, columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"], threshold=[0, 0, 1, 10]
    )

    print(data.head())
    print(data["duration"].describe())
    # print(data["batch_count"].describe())

    log_columns = [
        "balance_change",
        "base_game_win",
        "free_spins_count",
        "free_spins_count_above_10",
        "big_win_count_above_1",
        "end_balance",
        "free_game_win_above_0",
        "return_to_player",  # "sim_duration",
        "start_balance",
        "total_bet",
        "total_profit",
        "total_spins",
        "total_win",
        "win_count",
        "win_rate",
    ]
    data = log_transform(data, col_names=log_columns, suffix="_log")

    anova_iv = [f"{col}_log" for col in log_columns]
    anova_res = run_two_way_anova(data, anova_iv)
    run_post_hoc_analysis(data, anova_res, [f"{col}_log" for col in log_columns])

    # plot_correlation(
    #     data[
    #         [
    #             "balance_change",
    #             "end_balance",
    #             "free_game_win",
    #             "win_rate",
    #             "return_to_player",
    #             "start_balance",
    #             "total_profit",
    #             "base_game_win",
    #             "total_win",
    #             "total_bet",
    #             "sim_duration",
    #             "big_win_count",
    #             "free_spins_count",
    #             "total_spins",
    #             "win_count",
    #         ]
    #     ],
    #     title=f"correlation for: all group",
    # )

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

    plot_multiple_box_swarm(
        data,
        x_cols=["machine_id"],
        y_cols=[
            "free_spins_count_above_10",
            "free_game_win_above_0",
            "base_game_win_above_0",
            "balance_change",
            "start_balance",
        ],
        n_cols=1,
        group_col="cluster_index",
        log_scale=True,
    )
