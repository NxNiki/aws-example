"""
This script use multiple feature selection algorithms and elbow method to help determine the features to feed into
cluster analysis and the optimal number of clusters.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple, Union

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import parallel_backend
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, parse_s3_path, read_files, upload_file_to_s3
from bituslabs_ds.utils import (
    column_iterator,
    count_missing_columns,
    keep_numeric_columns,
    log_transform,
    remove_outliers,
    save_list,
)

OUTPUT_PATH = "wucaishen_analysis_kmeans"
s3 = boto3.client("s3")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import; silent if no config


def smart_feature_selection(
    data: pd.DataFrame, threshold: float = 0.9, prefer_keywords: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
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
            group = [column] + high_corr
            # 看这组里面有没有带 prefer_keywords 的特征
            preferred = []
            for feat in group:
                if any(key in feat.lower() for key in prefer_keywords):
                    preferred.append(feat)
            if preferred:
                # 如果有偏好的，保留偏好的第一个，删除其他
                keep_feat = preferred[0]
            else:
                # 否则，保留这一组的第一个
                keep_feat = group[0]
            kept.add(keep_feat)
            group.remove(keep_feat)  # 删掉自己
            to_drop.update(group)
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


def feature_selection_by_pca(data: pd.DataFrame, s3_path: Optional[str] = None) -> List[str]:
    """
    remove features with variance < threshold.
    :param data:
    :param s3_path:
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

    plt.savefig(f"./figures/{title}.png")
    plt.show()
    # upload_file_to_s3(f"./figures/{title}.png", S3_BUCKET, f"{OUTPUT_PATH}/{title}.png")

    features = [features[i] for i in np.argsort(importance)[::-1]]
    save_list(features, f"./features/important_features.json")

    if s3_path:
        logger.info(f"upload important_features.json to: {s3_path}")
        bucket, key = parse_s3_path(s3_path)
        upload_file_to_s3(
            f"./features/important_features.json",
            bucket,
            key,
        )

    return features


def plot_correlation(data: pd.DataFrame):
    data = keep_numeric_columns(data)
    corr_matrix = data.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    title = "Feature Correlation Heatmap"
    plt.title(title, fontsize=16)
    plt.savefig(f"./figures/{title}.png")
    plt.show()
    # upload_file_to_s3(f"./figures/{title}.png", S3_BUCKET, f"{OUTPUT_PATH}/{title}.png")


def elbow_method(data: pd.DataFrame, features: List[str], n_features: Optional[Union[List[int], int]] = None):
    """
    run elbow method to determine number of clusters
    :param data:
    :param features: label of columns of data that order by feature importance.
    :param n_features: select top n features
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
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=100)
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

        plt.savefig(f"./figures/{title}.png")
        plt.show()
        # upload_file_to_s3(f"./figures/{title}.png", S3_BUCKET, f"{OUTPUT_PATH}/{title}.png")


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


if __name__ == "__main__":

    os.makedirs("./.log", exist_ok=True)
    os.makedirs(".output", exist_ok=True)
    os.makedirs("./features", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler("./.log/analysis_cluster_01_elbow_method.log"), logging.StreamHandler()],
    )

    non_feature_col = ["group_id", "loginname", "start_time"]
    feature_col = [
        # "group_num",
        "rtp_mean",
        "bet_min",
        "bet_max",
        "bet_mean",
        "bet_p25",
        "bet_median",
        "bet_p75",
        "basepoint_min",
        "basepoint_max",
        "basepoint_mean",
        "basepoint_p25",
        "basepoint_median",
        "basepoint_p75",
        "payout_min",
        "payout_max",
        "payout_mean",
        "payout_p25",
        "payout_median",
        "payout_p75",
        "profit_min",
        "profit_max",
        "profit_mean",
        "profit_p25",
        "profit_median",
        "profit_p75",
        "delta_t_min",
        "delta_t_max",
        "delta_t_mean",
        "delta_t_p25",
        "delta_t_median",
        "delta_t_p75",
        "delta_bet_min",
        "delta_bet_max",
        "delta_bet_mean",
        "delta_bet_p25",
        "delta_bet_median",
        "delta_bet_p75",
        "delta_profit_min",
        "delta_profit_max",
        "delta_profit_mean",
        "delta_profit_p25",
        "delta_profit_median",
        "delta_profit_p75",
        "streak_min",
        "streak_max",
        "streak_mean",
        "streak_p25",
        "streak_median",
        "streak_p75",
        "win_streak_min",
        "win_streak_max",
        "win_streak_mean",
        "win_streak_p25",
        "win_streak_median",
        "win_streak_p75",
        "lose_streak_min",
        "lose_streak_max",
        "lose_streak_mean",
        "lose_streak_p25",
        "lose_streak_median",
        "lose_streak_p75",
        "deposit_min",
        "deposit_max",
        "deposit_mean",
        "deposit_p25",
        "deposit_median",
        "deposit_p75",
        "withdrawal_min",
        "withdrawal_max",
        "withdrawal_mean",
        "withdrawal_p25",
        "withdrawal_median",
        "withdrawal_p75",
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
        "currency_label",
    ]

    # features to apply log transform, this should be decided with EDA:
    feature_col_log = [
        "rtp_mean",
        "bet_min",
        "bet_max",
        "bet_mean",
        "bet_p25",
        "bet_median",
        "bet_p75",
        "basepoint_min",
        "basepoint_max",
        "basepoint_mean",
        "basepoint_p25",
        "basepoint_median",
        "basepoint_p75",
        "payout_min",
        "payout_max",
        "payout_mean",
        "payout_p25",
        "payout_median",
        "payout_p75",
        "profit_min",
        "profit_max",
        "profit_mean",
        "profit_p25",
        "profit_median",
        "profit_p75",
        "delta_t_min",
        "delta_t_max",
        "delta_t_mean",
        "delta_t_p25",
        "delta_t_median",
        "delta_t_p75",
        "delta_bet_min",
        "delta_bet_max",
        "delta_bet_mean",
        "delta_bet_p25",
        "delta_bet_median",
        "delta_bet_p75",
        "delta_profit_min",
        "delta_profit_max",
        "delta_profit_mean",
        "delta_profit_p25",
        "delta_profit_median",
        "delta_profit_p75",
        "streak_min",
        "streak_max",
        "streak_mean",
        "streak_p25",
        "streak_median",
        "streak_p75",
        "win_streak_min",
        "win_streak_max",
        "win_streak_mean",
        "win_streak_p25",
        "win_streak_median",
        "win_streak_p75",
        "lose_streak_min",
        "lose_streak_max",
        "lose_streak_mean",
        "lose_streak_p25",
        "lose_streak_median",
        "lose_streak_p75",
        "deposit_min",
        "deposit_max",
        "deposit_mean",
        "deposit_p25",
        "deposit_median",
        "deposit_p75",
        "withdrawal_min",
        "withdrawal_max",
        "withdrawal_mean",
        "withdrawal_p25",
        "withdrawal_median",
        "withdrawal_p75",
    ]

    wucaishen_files = list_s3_files(S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_24.*\.csv$")
    wucaishen_data = read_files(
        wucaishen_files, local_cache_path="./output/wucaishen_grouped_stat_output_24.csv", reload=False
    )

    count_missing_columns(wucaishen_data)

    wucaishen_data = log_transform(wucaishen_data, feature_col_log)
    save_list(feature_col_log, "./features/log_transform_features.json")
    upload_file_to_s3(
        "./features/log_transform_features.json",
        S3_BUCKET,
        f"{OUTPUT_PATH}/features/log_transform_features.json",
    )
    plot_correlation(wucaishen_data[feature_col])

    # remove highly correlated features:
    _, kept_features = smart_feature_selection(wucaishen_data[feature_col], threshold=0.9)
    _, kept_features = feature_selection_by_variance(wucaishen_data[kept_features], threshold=0.01)
    data_select = wucaishen_data[kept_features + non_feature_col]

    important_features = feature_selection_by_pca(
        data_select,
        f"s3://{S3_BUCKET}/{OUTPUT_PATH}/features/important_features.json",
    )
    elbow_method(wucaishen_data, important_features, [10, 15, 20, 25])
