"""
This script uses multiple feature selection algorithms and elbow method to help determine the features to feed into
cluster analysis and the optimal number of clusters.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from balanced_kmeans import kmeans_equal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.ml import calculate_inertia, calculate_silhouette_score
from bituslabs_ds.s3_utils import list_s3_files, read_dataset, read_files
from bituslabs_ds.utils import column_iterator, df_power_transform, keep_numeric_columns, remove_outliers, save_list

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import; silent if no config


def elbow_method(
    data: pd.DataFrame, features: List[str], n_features: Optional[Union[List[int], int]] = None, output_path: str = "."
) -> Dict[int, Dict[int, int]]:
    """
    run elbow method to determine the number of clusters
    :param data:
    :param features: label of columns of data that order by feature importance.
    :param n_features: select top n features
    :param output_path:
    :return: the cluster index for each feature set.
    """

    k_range = range(2, 10)
    scaler = StandardScaler()

    cluster_indices = {}
    for x, n in column_iterator(data, features, n_features):
        x, filter_index = remove_outliers(x)
        x = scaler.fit_transform(x)
        cluster_indices_by_k = {}
        inertia = []
        silhouette_scores = []
        cluster_sizes = []
        for k in k_range:

            # model = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=100) # run faster for testing
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(x)
            labels = model.labels_
            inertia.append(model.inertia_)

            # Convert NumPy array to PyTorch tensor and add batch dimension
            # x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # Add batch dimension: (1, num_samples, num_features)
            # labels_tensor, centroids_tensor = kmeans_equal(x_tensor, num_clusters=k, cluster_size=int(len(x) / k))
            # labels = labels_tensor.squeeze(0).numpy()  # Remove batch dimension: (num_samples,)
            # inertia.append(calculate_inertia(x, labels))

            cluster_index = np.full(len(filter_index), np.nan)
            cluster_index[filter_index] = labels
            cluster_indices_by_k[k] = cluster_index
            silhouette_scores.append(calculate_silhouette_score(x, labels))
            cluster_counts = np.bincount(labels)
            cluster_sizes.append(cluster_counts)

        cluster_indices[n] = cluster_indices_by_k

        title = f"Elbow Method for Optimal k n_features({n})"
        fig, ax1 = plt.subplots(figsize=(12, 9))
        (line1,) = ax1.plot(k_range, inertia, marker="o", linestyle="-", label="Inertia")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia")
        ax1.set_title(title, fontsize=16)

        ax2 = ax1.twinx()
        (line2,) = ax2.plot(
            k_range, silhouette_scores, marker="s", linestyle="-", color="red", label="Silhouette Score"
        )
        ax2.set_ylabel("Silhouette Score")

        lines = [line1, line2]
        labels = [str(line.get_label()) for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        # Add table with cluster sizes
        # Convert and pad cluster counts to string rows of equal length
        max_clusters = max(len(sizes) for sizes in cluster_sizes)
        cluster_sizes_str = []
        for sizes in cluster_sizes:
            row = [
                f"{count} ({count/sum(sizes):.3f})".replace("(0.", "(.") for count in sizes
            ]  # convert counts to strings
            row += [""] * (max_clusters - len(row))  # pad with empty strings
            cluster_sizes_str.append(row)

        # Transpose to get clusters as rows
        cluster_sizes_table = list(map(list, zip(*cluster_sizes_str)))

        # Create column labels like C1, C2, ..., Cn
        row_labels = [f"C{i + 1}" for i in range(max_clusters)]

        # Add table below plot
        plt.table(
            cellText=cluster_sizes_table,
            rowLabels=row_labels,
            colLabels=[f"k={k}" for k in k_range],
            cellLoc="center",
            loc="bottom",
            bbox=[0.0, -0.5, 1, 0.3],
        )  # [left, bottom, width, height]

        plt.subplots_adjust(left=0.1, bottom=0.3)

        plt.savefig(f"{output_path}/figures/{title}.png")
        # plt.show()

    return cluster_indices


def get_feature_names() -> Tuple[List[str], List[str], List[str]]:

    non_features = ["group_id", "loginname", "start_time"]

    base_features = [
        "bet",
        "basepoint",
        "payout",
        "profit",
        "delta_t",
        "delta_bet",
        "delta_profit",
        "streak",
        "win_streak",
        "lose_streak",
        "deposit",
    ]
    suffixes = ["min", "max", "mean", "p25", "median", "p75"]
    features = [f"{bf}_{suf}" for bf in base_features for suf in suffixes]
    skewed_features = ["rtp_mean"] + features

    normal_features = [
        # "group_num",
        "slottype_2_count",
        "payout_rate",
        "profit_rate",
        "profit_stddev",
        "account_stddev",
        "morning_count",
        "afternoon_count",
        "night_count",
        "midnight_count",
        "weekend_count",
        "duration_seconds",
        "avg_time_per_bet",
        # currently we combine all currencies as different currency users may have different purchase power.
        # "currency_label",
    ]

    return non_features, normal_features, skewed_features


def load_data(output_file: str, columns: Optional[List[str]] = None, pattern: str = ".*") -> pd.DataFrame:

    files = list_s3_files(S3_BUCKET, "wucaishen_processed_data", pattern)
    data = read_files(
        files,
        local_cache_path=output_file,
        columns=columns,
        reload=False,
    )

    # dataset = read_dataset(f"{S3_BUCKET}/wucaishen_process_grouped", REGION)
    # # table = dataset.to_table(filter=(ds.field("month") == "01"))
    # table = dataset.to_table(columns=columns, use_threads=True)
    # data = table.to_pandas(use_threads=True)

    return data


def main(output_path: str):

    os.makedirs(f"{output_path}/output", exist_ok=True)
    os.makedirs(f"{output_path}/features", exist_ok=True)
    os.makedirs(f"{output_path}/figures", exist_ok=True)

    non_features, normal_features, skewed_features = get_feature_names()
    wucaishen_data = load_data(
        f"{output_path}/output/wucaishen_grouped_stat_output_24.csv",
        columns=[*non_features, *normal_features, *skewed_features],
        pattern=r"wucaishen_grouped_stat_output_24.*\.csv$",
    )

    DataProfiler.count_df_missing_columns(wucaishen_data)
    wucaishen_data.fillna(0, inplace=True)
    wucaishen_data = df_power_transform(wucaishen_data, skewed_features)
    save_list(normal_features + skewed_features, f"{output_path}/features/log_transform_features.json")

    viz = DataVisualizer(wucaishen_data[normal_features + skewed_features])
    viz.create_figure(fig_title="Correlation of features: wucaishen", fig_size=(20, 17))
    viz.add_correlation_heatmap(annot=False, cmap="coolwarm")
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    viz.display()
    viz.save(f"{output_path}/figures/wucaishen_correlation.png")

    # remove highly correlated features:
    _, kept_features = smart_feature_selection(wucaishen_data[normal_features + skewed_features], threshold=0.9)
    _, kept_features = feature_selection_by_variance(wucaishen_data[kept_features], threshold=0.01)
    data_select = wucaishen_data[kept_features + non_features]

    important_features = feature_selection_by_pca(data_select, output_path)
    elbow_method(wucaishen_data, important_features, [15, 20, 25], output_path)


if __name__ == "__main__":

    default_output_path = Path(__file__).parent / "output"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=False, default=default_output_path)
    args = parser.parse_args()

    setup_logging(args.output_path, "analysis_cluster_01_elbow_method.log")

    main(args.output_path)
