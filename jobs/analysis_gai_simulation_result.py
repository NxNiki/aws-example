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
from bituslabs_ds.eda import plot_correlation, plot_df_distribution, plot_multiple_box_swarm, split_column_by_threshold
from bituslabs_ds.s3_utils import list_s3_files, read_files
from bituslabs_ds.utils import batch_iterator, group_iterator, log_transform

setup_logging(".", "analysis_gai_simulation_result.log")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


GROUP_VAR_SEP = "|"


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
        if isinstance(row["A"], str) and "_" in row["A"]:
            report_dict["comparison"].append(
                (tuple(row["A"].rsplit(GROUP_VAR_SEP, 1)), tuple(row["B"].rsplit(GROUP_VAR_SEP, 1)))
            )
        else:
            report_dict["comparison"].append((row["A"], row["B"]))
        report_dict["method"].append(method)
        report_dict["p_value"].append(row[p_col])
        report_dict["effect_size"].append(row["hedges"])


def run_post_hoc_analysis(
    data: pd.DataFrame,
    anova_results: pd.DataFrame,
    var_columns: List[str],
    p_thresh: float = 0.05,
    method: str = "Tukey",
    effects: str = "main",
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

        if effects == "main":
            # Post-hoc for machine_id if significant
            if anova_row["machine_id-p_value"] < p_thresh:
                print(f"\nSignificance detected for machine_id on {col} (p={anova_row['machine_id-p_value']:.4f})")
                _perform_and_report_post_hoc(col, "machine_id", method, data, post_hoc_report, p_thresh)

            # Post-hoc for cluster_index if significant
            if anova_row["cluster_index-p_value"] < p_thresh:
                print(
                    f"\nSignificance detected for cluster_index on {col} (p={anova_row['cluster_index-p_value']:.4f})"
                )
                _perform_and_report_post_hoc(col, "cluster_index", method, data, post_hoc_report, p_thresh)

        elif effects == "interaction":
            # Post-hoc for interaction if significant
            if anova_row["interaction-p_value"] < p_thresh:
                print(
                    f"\nSignificance detected for interaction (machine_id * cluster_index) on {col} (p={anova_row['interaction-p_value']:.4f})"
                )

                # run post-hoc for each cluster group separately:
                for data_interaction, cluster_index, _ in group_iterator(data, group_col="cluster_index"):
                    # Create a combined interaction group temporarily
                    data_interaction["interaction_group"] = (
                        data_interaction["cluster_index"].astype(str)
                        + GROUP_VAR_SEP
                        + data_interaction["machine_id"].astype(str)
                    )
                    _perform_and_report_post_hoc(
                        col, "interaction_group", method, data_interaction, post_hoc_report, p_thresh
                    )
        else:
            raise ValueError(f"Unknown effect: {effects}")

    res_post_hoc = pd.DataFrame(post_hoc_report)
    if not res_post_hoc.empty:
        print("\n--- Summary Post-Hoc Report (Significant Comparisons Only) ---")
        print(res_post_hoc.to_markdown(index=False))
    else:
        print("\nNo significant main effects or interactions found in ANOVA, so no post-hoc tests were performed.")
    return res_post_hoc


if __name__ == "__main__":

    reload = False
    local_output = "./output/v1_all_sessions_summary_fixed.csv"
    s3_files: List[str] = []
    if reload:
        s3_files = list_s3_files(
            S3_BUCKET, prefix="gail_simulator_data_raw/results_fixed/", pattern="sim_20250713_.*_sessions_summary.csv"
        )
    data = read_files(s3_files, local_cache_path=local_output, reload=reload)
    data.drop(columns=["balance_change"], inplace=True)

    # plot_df_distribution(data.drop(columns=["session_id"]), figure_name="./figures/gai_simulation_distribution.png")

    data["cluster_index"] = data["player_id"].str.extract(r"cluster(\d+)_", expand=False).astype(str)
    data.sort_values("cluster_index", inplace=True)
    data = split_column_by_threshold(
        data, columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"], threshold=[0, 0, 1, 10]
    )

    log_columns = [
        "base_game_win",
        "big_win_count",
        "big_win_count_above_1",
        "duration",
        "final_balance",
        "first_bet",
        "free_game_win",
        "free_game_win_above_0",
        "free_spins_count",
        "free_spins_count_above_10",
        "initial_balance",
        "return_to_player",
        "sim_duration",
        "total_bet",
        "total_profit",
        "total_spins",
        "total_win",
        "win_count",
        "win_rate",
    ]
    data = log_transform(data, col_names=log_columns, suffix="_log")

    corr_cols = [
        f"{col}_log"
        for col in [
            "base_game_win",
            "big_win_count",
            "duration",
            "final_balance",
            "first_bet",
            "free_game_win",
            "free_spins_count",
            "initial_balance",
            "return_to_player",
            "sim_duration",
            "total_bet",
            "total_profit",
            "total_spins",
            "total_win",
            "win_count",
            "win_rate",
        ]
    ]
    # plot_correlation(data[corr_cols], title=f"correlation for: all group")

    remove_cols = {
        "base_game_win",
        "big_win_count",
        "free_game_win",
        "win_count",
        "duration",
        "sim_duration",
        "final_balance",
    }
    anova_iv = [f"{col}_log" for col in log_columns if col not in remove_cols]
    anova_res = run_two_way_anova(data, anova_iv)
    main_effects = run_post_hoc_analysis(data, anova_res, anova_iv, effects="main")
    interaction_effects = run_post_hoc_analysis(data, anova_res, anova_iv, effects="interaction")

    for y_cols, i in batch_iterator(anova_res["variable"], 5):
        plot_multiple_box_swarm(
            data,
            x_cols=["cluster_index"],
            y_cols=list(y_cols),
            n_cols=1,
            group_col="machine_id",
            log_scale=True,
            fig_title=f"variables with sig anova: {i}/{(len(anova_res)-1)//5+1}",
            post_hoc_table=interaction_effects,
        )
