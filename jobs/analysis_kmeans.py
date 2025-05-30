import logging
import os
from typing import List, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from schema import Optional
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_files, upload_file_to_s3

os.makedirs("./.log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("../.log/s3_utils.log"), logging.StreamHandler()],
)

s3 = boto3.client("s3")

OUT_PATH = "wucaishen_analysis_kmeans"


def count_missing_columns(df: pd.DataFrame, verbose: bool = True) -> int:
    """
    Count the number of columns in a DataFrame that contain missing (NaN) values.

    :param df: Input DataFrame
    :param verbose: If True, print columns with their missing counts
    :return: Number of columns with missing values
    """
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if verbose:
        logging.info(f"Columns with missing values: \n{cols_with_missing}")

    return len(cols_with_missing)


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

    logging.info(f"kept: {len(kept)} features: \n({kept}")
    logging.info(f"remove: {len(to_drop)} features: \n({to_drop}")

    return list(to_drop), list(kept)


def feature_selection_by_variance(data: pd.DataFrame, threshold: float = 0.01) -> List[str]:
    """
    remove features with variance < threshold.
    :param data:
    :param threshold:
    :return:
    """
    selector = VarianceThreshold(threshold=threshold)
    # 获取保留的列名
    selected_features = data.columns[selector.get_support()]
    print(f"方差筛选后保留的变量：{list(selected_features)}")
    return selected_features


# DISCUSSION: should we consider features with high contribution to other components?
def feature_selection_by_pca(data: pd.DataFrame):
    pca = PCA(n_components=data.shape[1])  # 保留所有主成分
    pca.fit(data)

    # 计算每个特征在所有主成分上的贡献度（取绝对值求和）
    importance = np.abs(pca.components_).sum(axis=0).tolist()

    features = data.columns.tolist()
    feature_importance = pd.DataFrame({"Feature": features, "Importance": importance})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(18, 10))
    colors = sns.color_palette("viridis", len(feature_importance))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette=colors)
    title = "Feature Importance Rank"
    plt.title(title, fontsize=16)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.show()

    plt.savefig(f"./images/{title}.png")
    upload_file_to_s3(f"./images/{title}.png", S3_BUCKET, f"{OUT_PATH}/{title}.png")

    important_features = [features[i] for i in np.argsort(importance)]
    logging.info(f"Top Important Features: \n{important_features}")

    return important_features


def plot_correlation(data: pd.DataFrame):
    corr_matrix = data.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    title = "Feature Correlation Heatmap"
    plt.title(title, fontsize=16)
    plt.savefig(f"./images/{title}.png")
    plt.show()
    upload_file_to_s3(f"./images/{title}.png", S3_BUCKET, f"{OUT_PATH}/{title}.png")


if __name__ == "__main__":
    non_feature_col = ["group_id", "loginname", "start_time"]
    feature_col = [
        "group_num",
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
        "start_time",
        "end_time",
        "morning_count",
        "afternoon_count",
        "night_count",
        "midnight_count",
        "weekend_count",
        "duration_seconds",
        "avg_time_per_bet",
        "currency_label",
    ]

    wucaishen_files = list_s3_files(
        S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_2401.*\.csv$"
    )
    wucaishen_data = read_files(S3_BUCKET, wucaishen_files)
    count_missing_columns(wucaishen_data)

    # remove highly correlated features:
    _, kept_features = smart_feature_selection(wucaishen_data[feature_col], threshold=0.9)
    kept_features = feature_selection_by_variance(wucaishen_data[kept_features], threshold=0.01)
    data_select = wucaishen_data[kept_features + non_feature_col]
