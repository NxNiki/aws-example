"""
In the previous steps, we trained knn cluster model to identify 3 groups of bet behaviors for the data of 2024.
The cluster model was applied to data in month 1, 2, 3 of 2025.
GAI model was trained with data in 2024 and applied to data in 2025 to simulate gaming behaviors.
The GAI model is deployed to generate simulation data.

This script gets simulation data from the GAI model (slot machine) and compares game metrics such as bet, profit, etc.
between gamers from different clusters and using different math tables.
"""

import argparse
import logging
import os
from datetime import datetime
from pprint import pformat
from typing import Dict, List, Literal, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager, rcParams
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.eda import Anova, DataProfiler, DataVisualizer, split_column_by_threshold
from bituslabs_ds.s3_utils import list_s3_files, read_files, write_df_to_s3

logger = logging.getLogger(__name__)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)

fontP = font_manager.FontProperties()
fontP.set_family("Heiti TC")
fontP.set_size(12)
rcParams["font.family"] = ["Heiti TC"]


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_file}:")
    logger.info(f"config: \n {pformat(config)} \n")
    return config


def load_data(config, reload: bool = False) -> pd.DataFrame:

    local_output = f"{LOCAL_ROOT}/jobs/{config['work_dir']}/{config['data_loader']['local_cache']}"
    s3_files: List[str] = []
    if reload or not os.path.exists(local_output):
        s3_files += list_s3_files(
            S3_BUCKET,
            prefix=config["data_loader"]["prefix"],
            pattern=config["data_loader"]["pattern"],
        )

    data = read_files(
        s3_files, local_cache_path=local_output, reload=reload, columns=config["data_loader"]["columns_to_read"]
    )
    data["cluster_index"] = data["player_id"].str.extract(r"(cluster\d+)_", expand=False).astype(str)
    data["cluster_index"].replace(
        {"cluster0": "user group: 0", "cluster1": "user group: 1", "cluster2": "user group: 2"}, inplace=True
    )

    print(data.shape)
    # data = data[data["total_spins"]>40]
    # print(data.shape)

    data = split_column_by_threshold(
        data,
        columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"],
        threshold=[0, 0, 0, 9],
    )

    return data


def process_data(data, config, output_path) -> pd.DataFrame:
    data = data.copy()
    transformer = PowerTransformer()

    metrics = config["radar_plot_columns"]
    data[metrics] = transformer.fit_transform(data[metrics])

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(transformer, f"{output_path}/power_transformer.joblib")

    if config["data_loader"]["scaler"] == "standard":
        scaler = StandardScaler()
    elif config["data_loader"]["scaler"] == "minMax":
        scaler = MinMaxScaler()

    data[metrics] = scaler.fit_transform(data[metrics])
    # Save scaler mean and std to a CSV file
    if config["data_loader"]["scaler"] == "standard":
        scaler_stats = pd.DataFrame({"metric": metrics, "mean": scaler.mean_, "std": scaler.scale_})
        scaler_stats.to_csv(f"{output_path}/scaler_stats.csv", index=False)
    # Also save the scaler model for later use
    scaler_filename = f"{output_path}/scaler_model.joblib"
    joblib.dump(scaler, scaler_filename)

    return data


def make_radar_plot(
    data: pd.DataFrame,
    group: str,
    config: Dict,
    agg_method: Literal["mean", "median"],
    title: str,
    output_path: str,
) -> None:
    """
    Generate and save a radar plot for specified metrics, aggregated over groups.

    Args:
        data (pd.DataFrame): Input dataframe containing the data to plot.
        group (str): Categorical column name in `data` to group by, curves will be plotted for each unique value.
        metrics (List[str]): List of columns containing the numeric metrics to plot on the radar chart's axes.
        agg_method (Literal["mean", "median"]): Method to aggregate `metrics` within each group ('mean' or 'median').
        title (str): Title for the radar plot. If empty or None, a default title is constructed.
        output_path (str): Path to save the resulting radar plot figure (e.g., "output.png").
    """

    metrics = config["radar_plot_columns"]
    metrics_rename = config["radar_plot_rename"]

    # Compute mean values per group for the metrics
    if agg_method == "mean":
        grouped = data.groupby(group)[metrics].mean().reset_index()
    elif agg_method == "median":
        grouped = data.groupby(group)[metrics].median().reset_index()
    else:
        raise ValueError(f"unsupported agg_method: {agg_method}")

    # Radar plot setup
    labels = metrics_rename
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in grouped.iterrows():
        values = row[metrics].tolist()
        values += values[:1]  # close the polygon
        ax.plot(angles, values, label=row[group])
        ax.fill(angles, values, alpha=0.25)

    # Add labels and aesthetics
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    for i, label in enumerate(ax.get_xticklabels()):
        if (i + len(labels) // 4) // (len(labels) // 2) % 2 == 0:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")

        if i // (len(labels) // 2) == 0:
            label.set_verticalalignment("bottom")
        else:
            label.set_verticalalignment("top")

        label.set_y(label.get_position()[1] + 0.05)

    # --- Set y-axis (radial axis) scale for the radar plot. ---
    if config["data_loader"]["scaler"] == "minMax":
        ax.set_ylim(0, 1)
        num_rgrids = 5
        grid_values = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_rgrids)
        ax.set_yticks(grid_values)
        ax.set_yticklabels([f"{v:.2f}" for v in grid_values], fontsize=10)

    if title is None or len(title) == 0:
        title = f"Radar plot by {group}"
    ax.set_title(title, fontsize=14, pad=15, fontproperties=fontP)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main(config_file):

    project_name = os.path.basename(config_file).replace(".yaml", "")
    time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(LOCAL_ROOT / "jobs/log", f"simulation_analysis_{project_name}_{time_tag}.log")

    config = load_config(config_file)
    output_path = f"{LOCAL_ROOT}/jobs/{config['work_dir']}/radar_plot_{config['data_loader']['scaler']}"

    data = load_data(config, reload=False)
    data_profiler = DataProfiler(data, skewness_threshold=1.5)

    # # Radar plot for each math table (3 clusters in one plot)
    data = process_data(data, config, output_path=output_path)
    for math_table in data["machine_id"].unique():
        logger.info(f"make radar plot for {math_table}")
        for agg_method in ["mean", "median"]:
            make_radar_plot(
                data[data["machine_id"] == math_table],
                group="cluster_index",
                config=config,
                agg_method=agg_method,
                title=f"数学表: {math_table}",
                output_path=f"{output_path}/radar_plot_{math_table}_{agg_method}.png",
            )

    data_profiler.transform_skewed_columns(pos_suffix="", neg_suffix="")
    output_path = f"{LOCAL_ROOT}/jobs/{config['work_dir']}/anova_result"
    os.makedirs(output_path, exist_ok=True)

    # viz = DataVisualizer(data_profiler)

    # # Plot distributions with distribution statistics
    # viz.create_figure(
    #     layout_cols=data_profiler.processed_numerical_columns, group_col="cluster_index", n_cols=5, fig_size=(7, 3.5)
    # )
    # viz.add_histogram(show_distribution_stats=True, kde=True)
    # viz.figure.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.97, wspace=0.2, hspace=0.25)
    # viz.display()
    # viz.save(f"{output_path}/gai_simulation_distribution_transformed.png")

    # # Plot correlation heatmap
    # viz.create_figure(fig_title="Correlation for: cluster 2", fig_size=(12, 7))
    # viz.add_correlation_heatmap(cbar="Red")
    # viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    # viz.display()
    # viz.save(f"{output_path}/gai_simulation_correlation.png")

    # # Plot correlation heatmap for machine_id
    # viz.create_figure(layout_cols=["machine_id"], fig_title="Correlation for: cluster 2", fig_size=(12, 7))
    # viz.add_correlation_heatmap(cbar="Red")
    # viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97, wspace=0.25, hspace=0.35)
    # viz.display()
    # viz.save(f"{output_path}/gai_simulation_correlation_machine_id.png")

    # ANOVA:
    anova_between_vars = ["machine_id", "cluster_index"]
    # anova_between_vars = ["machine_id"]

    data_profiler.augment_columns(columns=config["pca_columns"], col_name="compound_metric", method=["pca"])

    anova_dv = [
        "total_bet",
        "total_spins",
        "total_profit",
        "compound_metric_pca1",
        "compound_metric_pca2",
        "compound_metric_pca3",
    ]

    res = data_profiler.show_group_stats(
        group_cols=["machine_id", "cluster_index"],
        var_columns=anova_dv,
        transpose=False,
        agg_stats=["mean", "median", "std"],
    )
    res.to_csv(f"{output_path}/report_selected_features.csv")

    res = data_profiler.show_group_stats(
        group_cols=["machine_id", "cluster_index"],
        var_columns=anova_dv,
        transpose=False,
    )
    res.to_csv(f"{output_path}/report_all_features.csv")

    anova = Anova(data_profiler.df, between_vars=anova_between_vars, var_columns=anova_dv)
    anova_res = anova.run_anova()
    write_df_to_s3(anova_res, S3_BUCKET, key="ds-data-kmeans/wucaishen_simulation/anova_result.csv")

    anova.run_post_hoc_analysis(anova_dv, effects="main", p_thresh=0.05)

    if len(anova_between_vars) > 1:
        anova.run_post_hoc_analysis(anova_dv, effects="interaction", group_var="cluster_index", p_thresh=0.05)
        anova.show_box_plot(
            x_col="cluster_index",
            group_col="machine_id",
            output_path=output_path,
            fig_title="boxplot_two_factors",
            stripplot_kws={"size": 2},
        )
    else:
        anova.show_box_plot(x_col="machine_id", output_path=output_path, fig_title="boxplot_machine_id")


if __name__ == "__main__":

    project = "deepdive"
    # project = "wucaishen"

    current_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description="simulation analysis pipeline")
    parser.add_argument(
        "--config_file", default=f"{current_path}/simulation_analysis_config-{project}.yaml", help="Project to analyze"
    )
    args = parser.parse_args()

    main(args.config_file)
