"""
In the previous steps, we trained knn cluster model to identify 3 groups of bet behaviors for the data of 2024.
The cluster model was applied to data in month 1, 2, 3 of 2025.
GAI model was trained with data in 2024 and applied to data in 2025 to simulate gaming behaviors.
The GAI model is deployed to generate simulation data.

This script gets simulation data from the GAI model (slot machine) and compares game metrics such as bet, profit, etc.
between gamers from different clusters and using different math tables.
"""

from collections import defaultdict
from typing import List, Union

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import plot_correlation, plot_df_distribution, plot_multiple_box_swarm, split_column_by_threshold
from bituslabs_ds.s3_utils import list_s3_files, read_files
from bituslabs_ds.utils import batch_iterator, group_iterator, log_transform

setup_logging(".", "analysis_gai_simulation_result.log")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


def run_two_way_anova(
    data: pd.DataFrame, var_columns: List[str], transpose_report: bool = False, p_thresh: float = 0.05
) -> pd.DataFrame:

    report = defaultdict(list)
    between_vars = ["machine_id", "cluster_index"]
    for col in var_columns:
        if (not pd.api.types.is_numeric_dtype(data[col])) or pd.api.types.is_bool_dtype(data[col]):
            print(f"skip non numeric columns: {col}.")
            continue

        anova_output = pg.anova(dv=col, between=between_vars, data=data, detailed=True)
        print(anova_output)

        if "p-unc" not in anova_output or all(anova_output["p-unc"] > p_thresh):
            print(f"skip non-significant column: {col}")
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


def show_group_stats(
    data: pd.DataFrame, group_cols: List[str], var_columns: List[str], transpose: bool = False
) -> None:
    """
    show mean and median for the combination of each level for the between variables:
    :param data:
    :param group_cols:
    :param var_columns:
    :param transpose: transpose the output for each group
    :return:
    """

    for col in var_columns:
        report = (
            data[group_cols + [col]]
            .groupby(group_cols)
            .agg(["mean", "median"])
            .sort_values([(col, "median"), (col, "mean")], ascending=False)
        )
        if transpose:
            print(report.transpose().to_markdown(index=True))
        else:
            print(report.to_markdown(index=True))


def _perform_and_report_post_hoc(
    dv_col: str, between_factor: str, method: str, data_df: pd.DataFrame, report_dict: defaultdict, p_thresh: float
):
    """
    Helper function to perform a post-hoc test and append results to the report dictionary.
    """

    def tuple_to_string(element):
        if isinstance(element, tuple):
            return ",".join(map(str, element))
        else:
            return element

    def string_to_tuple(element):
        return tuple(element.rsplit(",", 1))

    data_df = data_df.copy()
    data_df[between_factor] = data_df[between_factor].apply(tuple_to_string)

    print(f"  Performing {method} for {between_factor} on {dv_col}")
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
        report_dict["comparison"].append((string_to_tuple(row["A"]), string_to_tuple(row["B"])))
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
                    data_interaction["interaction_group"] = list(
                        zip(data_interaction["cluster_index"].astype(str), data_interaction["machine_id"].astype(str))
                    )
                    _perform_and_report_post_hoc(
                        col, "interaction_group", method, data_interaction, post_hoc_report, p_thresh
                    )
        else:
            raise ValueError(f"Unknown effect: {effects}")

    res_post_hoc = pd.DataFrame(post_hoc_report)
    if not res_post_hoc.empty:
        print("\n--- Summary Post-Hoc Report (Significant Comparisons Only) ---")
        for res_post_hoc_group, _, _ in group_iterator(res_post_hoc, group_col="variable"):
            print(res_post_hoc_group.sort_values(by="comparison").to_markdown(index=False))
    else:
        print("\nNo significant main effects or interactions found in ANOVA, so no post-hoc tests were performed.")
    return res_post_hoc


def augment_data(
    data: pd.DataFrame, columns: List[str], col_name: str, method: Union[List[str], str] = "pca"
) -> pd.DataFrame:

    if isinstance(method, str):
        method = [method]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data[columns])

    if "mean" in method:
        data[f"{col_name}_mean"] = df_scaled.mean(axis=1)
    if "pca" in method:
        pca = PCA(n_components=None)
        principal_components = pca.fit_transform(df_scaled)
        explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
        print(f"Cumulative Explained Variance for '{col_name}' PCA components:")
        for i, cum_var in enumerate(explained_variance_ratio_cumsum):
            print(f"  PC{i + 1}: {cum_var:.4f}")

        for i in range(principal_components.shape[1]):
            data[f"{col_name}_pca{i + 1}"] = principal_components[:, i]

    return data


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
    data["cluster_index"] = data["player_id"].str.extract(r"(cluster\d+)_", expand=False).astype(str)
    print(data.shape)
    # data = data[data["total_spins"]>40]
    print(data.shape)

    data = split_column_by_threshold(
        data, columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"], threshold=[0, 0, 1, 10]
    )

    # plot_df_distribution(data.drop(columns=["session_id"]), figure_name="./figures/gai_simulation_distribution.png", log=False)
    numeric_cols, feature_skewness, feature_unimodality_p = plot_df_distribution(
        data.drop(columns=["session_id"]),
        figure_name="./figures/gai_simulation_distribution_log.png",
        log=True,
        add_kde=True,
    )
    positive_skew_columns = [f for f, s in zip(numeric_cols, feature_skewness) if s > 1.5]
    print(f"skewed columns: \n {positive_skew_columns}")
    data = log_transform(data, col_names=positive_skew_columns, suffix="_log")
    print(data.columns)

    data = augment_data(
        data, columns=["total_bet_log", "total_profit_log"], col_name="compound_metric", method=["mean", "pca"]
    )
    show_group_stats(
        data,
        group_cols=["machine_id", "cluster_index"],
        var_columns=["total_spins", "total_bet", "total_profit", "total_profit_log"],
        transpose=True,
    )
    corr_cols = [f"{f}_log" if s > 1.5 else f for f, s in zip(numeric_cols, feature_skewness)]
    plot_correlation(data[corr_cols], title=f"correlation for: all group")

    remove_cols = {
        "base_game_win_log",
        "big_win_count_log",
        "free_game_win_log",
        "win_count_log",
        "duration_log",
        "sim_duration_log",
        "final_balance_log",
    }
    anova_iv = [col for col in corr_cols if col not in remove_cols] + [
        "compound_metric_mean",
        "compound_metric_pca1",
        "compound_metric_pca2",
    ]
    anova_res = run_two_way_anova(data, anova_iv)
    # main_effects = run_post_hoc_analysis(data, anova_res, anova_iv, effects="main")
    interaction_effects = run_post_hoc_analysis(data, anova_res, anova_iv, effects="interaction")

    for y_cols, i, num_chunks in batch_iterator(interaction_effects["variable"].drop_duplicates(), 5):
        plot_multiple_box_swarm(
            data,
            x_cols=["cluster_index"],
            y_cols=list(y_cols),
            n_cols=1,
            group_col="machine_id",
            log_scale=True,
            fig_title=f"variables with sig anova: {i}/{num_chunks}",
            post_hoc_table=interaction_effects,
        )
