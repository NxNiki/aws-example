"""
This script uses multiple feature selection algorithms and elbow method to help determine the features to feed into
cluster analysis and the optimal number of clusters.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, Tuple, Union

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as fs
import seaborn as sns
from joblib import parallel_backend
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.config import REGION, S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, parse_s3_path, read_dataset, read_files, upload_file_to_s3
from bituslabs_ds.utils import (
    column_iterator,
    count_missing_columns,
    keep_numeric_columns,
    log_transform,
    remove_outliers,
    save_list,
)

S3_OUTPUT_PATH = "wucaishen_analysis_kmeans"
s3 = boto3.client("s3")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import; silent if no config


def smart_feature_selection(
    data: pd.DataFrame, threshold: float = 0.9, prefer_keywords: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    remove features (one of two) that are highly correlated with each other
    data: DataFrame，完整数据集
    threshold: float，相关性阈值，比如 0.9
    prefer_keywords: list，优先保留的关键词，比如 'mean', 'median'

    return:
        to_drop: list，需要删除的特征
        kept_features: list，保留的特征
    """

    if prefer_keywords is None:
        prefer_keywords = ["mean", "median"]

    data = keep_numeric_columns(data)
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    kept = set()
    for column in upper.columns:
        # 找到当前列与其他列高度相关的
        high_corr = upper[column][upper[column] > threshold].index.tolist()
        if high_corr:
            # 包括自己和高度相关的列
            high_corr = [column] + high_corr
            # 保留这一组的第一个
            keep_feat = high_corr[0]
            # 看这组里面有没有带 prefer_keywords 的特征
            for feat in high_corr:
                if any(key in feat.lower() for key in prefer_keywords):
                    # 如果有偏好的，保留偏好的第一个，删除其他
                    keep_feat = feat
                    break
            kept.add(keep_feat)
            high_corr.remove(keep_feat)  # 删掉自己
            to_drop.update(high_corr)
        else:
            # 如果没有高度相关的，可以直接保留
            kept.add(column)

    logger.info(f"kept: {len(kept)} features: \n({kept}")
    logger.info(f"remove: {len(to_drop)} features: \n({to_drop}")
    return list(to_drop), list(kept)


def feature_selection_by_variance(data: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    remove features with variance < threshold.
    :param data:
    :param threshold:
    :return:
    """
    selector = VarianceThreshold(threshold=threshold)
    data = keep_numeric_columns(data)
    data_filtered = selector.fit_transform(data)
    # 获取保留的列名
    selected_features = data.columns[selector.get_support()].tolist()
    logging.info(f"方差筛选后保留的变量：{list(selected_features)}")
    return data_filtered, selected_features


def feature_selection_by_pca(data: pd.DataFrame, output_path: str = ".") -> List[str]:
    """
    remove features with variance < threshold.
    :param data:
    :param output_path:
    :return:
    """

    data = keep_numeric_columns(data)
    pca = PCA(n_components=data.shape[1])  # 保留所有主成分
    pca.fit(data)

    # 计算每个特征在所有主成分上的贡献度（取绝对值求和）
    importance = np.abs(pca.components_).sum(axis=0).tolist()

    features = data.columns.tolist()
    feature_importance = pd.DataFrame({"Feature": features, "Importance": importance})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(18, 10))
    colors = sns.color_palette("viridis", len(feature_importance))
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=feature_importance, palette=colors, legend=False)
    title = "Feature Importance Rank"
    plt.title(title, fontsize=16)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.savefig(f"{output_path}/figures/{title}.png")
    plt.show()

    features = [features[i] for i in np.argsort(importance)[::-1]]
    save_list(features, f"{output_path}/features/important_features.json")

    return features


def plot_correlation(data: pd.DataFrame, output_path: str = ".") -> None:
    data = keep_numeric_columns(data)
    corr_matrix = data.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    title = "Feature Correlation Heatmap"
    plt.title(title, fontsize=16)
    plt.savefig(f"{output_path}/figures/{title}.png")
    plt.show()


def elbow_method(
    data: pd.DataFrame, features: List[str], n_features: Optional[Union[List[int], int]] = None, output_path: str = "."
):
    """
    run elbow method to determine number of clusters
    :param data:
    :param features: label of columns of data that order by feature importance.
    :param n_features: select top n features
    :param output_path:
    :return:
    """

    k_range = range(2, 10)
    scaler = StandardScaler()

    for x, n in column_iterator(data, features, n_features):
        x, _ = remove_outliers(x)
        x = scaler.fit_transform(x)
        inertia = []
        silhouette_scores = []
        cluster_sizes = []
        for k in k_range:
            # kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=100) # run faster for testing
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(x)
            inertia.append(kmeans.inertia_)

            silhouette_scores.append(calculate_silhouette_score(x, kmeans))
            cluster_counts = np.bincount(kmeans.labels_)
            cluster_sizes.append(cluster_counts)

        title = f"Elbow Method for Optimal k n_features({n})"
        fig, ax1 = plt.subplots(figsize=(10, 9))
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
        plt.show()


def calculate_silhouette_score(x: Union[np.ndarray, pd.DataFrame], cluster_obj: ClusterMixin) -> float:

    if len(np.unique(cluster_obj.labels_)) < 2:
        return float("nan")

    try:
        with parallel_backend("loky"):
            # a small sample size may lead to small cluster totally omitted!
            score = silhouette_score(x, cluster_obj.labels_, sample_size=min(5000, x.shape[0]), random_state=42)
    except ValueError:
        score = float("nan")
    return score


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


def main(output_path: str):

    os.makedirs(f"{output_path}/.log", exist_ok=True)
    os.makedirs(f"{output_path}/output", exist_ok=True)
    os.makedirs(f"{output_path}/features", exist_ok=True)
    os.makedirs(f"{output_path}/figures", exist_ok=True)

    non_features, normal_features, skewed_features = get_feature_names()

    # wucaishen_files = list_s3_files(S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_24.*\.csv$")
    # wucaishen_data = read_files(
    #     wucaishen_files, local_cache_path="./output/wucaishen_grouped_stat_output_24.csv", reload=False
    # )

    dataset = read_dataset(f"{S3_BUCKET}/wucaishen_process_grouped", REGION)
    # table = dataset.to_table(filter=(ds.field("month") == "01"))
    table = dataset.to_table(columns=non_features + normal_features + skewed_features, use_threads=True)
    wucaishen_data = table.to_pandas(use_threads=True)

    count_missing_columns(wucaishen_data)
    wucaishen_data.fillna(0, inplace=True)
    wucaishen_data = log_transform(wucaishen_data, skewed_features)
    save_list(normal_features + skewed_features, "./features/log_transform_features.json")
    upload_file_to_s3(
        f"{output_path}/features/log_transform_features.json",
        S3_BUCKET,
        f"{S3_OUTPUT_PATH}/features/log_transform_features.json",
    )
    plot_correlation(wucaishen_data[normal_features + skewed_features])

    # remove highly correlated features:
    _, kept_features = smart_feature_selection(wucaishen_data[normal_features + skewed_features], threshold=0.9)
    _, kept_features = feature_selection_by_variance(wucaishen_data[kept_features], threshold=0.01)
    data_select = wucaishen_data[kept_features + non_features]

    important_features = feature_selection_by_pca(
        data_select, f"s3://{S3_BUCKET}/{S3_OUTPUT_PATH}/features/important_features.json"
    )
    elbow_method(wucaishen_data, important_features, [15, 20, 25, 30, 35, 40], output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, default=f"s3://bituslabs-team-ai/wucaishen_processed_data/")
    parser.add_argument("--output_path", required=True, default=".")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f"{args.output_path}/.log/analysis_cluster_01_elbow_method.log"),
            logging.StreamHandler(),
        ],
    )

    main(args.output_path)
