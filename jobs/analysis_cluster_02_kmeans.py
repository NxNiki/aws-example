import json
import logging
import os
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_files, upload_file_to_s3
from bituslabs_ds.utils import remove_outliers
from jobs.analysis_cluster_01_elbow_method import OUTPUT_PATH

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import; silent if no config


def scale_features(data: pd.DataFrame, output_dir: str, output_file_name: str) -> pd.DataFrame:
    """
    Normalize the columns of df to have zero mean and unit variance.
    :param data:
    :param output_dir:
    :param output_file_name:
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(f"{output_dir}/{output_file_name}.csv"):
        logger.info(f"read existing output file {output_file_name}...")
        df_scaled = pd.read_csv(f"{output_dir}/{output_file_name}.csv")
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(data)
        df_scaled = pd.DataFrame(x_scaled, columns=data.columns)

        df_scaled.to_csv(f"{output_dir}/{output_file_name}.csv", index=False)
        df_scaled.to_json(f"{output_dir}/{output_file_name}.json", orient="records", indent=2)

        mean_std_df = pd.DataFrame({"Mean": scaler.mean_, "Std": scaler.scale_}, index=data.columns)
        mean_std_df.to_csv(f"{output_dir}/{output_file_name}_parameters.csv")
        print("\n 每个特征的标准化参数（均值与标准差）：")
        print(mean_std_df)

        logger.info(f"update standardized_features.csv to s3: {OUTPUT_PATH}")
        upload_file_to_s3(f"{output_dir}/{output_file_name}.csv", S3_BUCKET, f"{OUTPUT_PATH}/{output_file_name}.csv")
        upload_file_to_s3(f"{output_dir}/{output_file_name}.json", S3_BUCKET, f"{OUTPUT_PATH}/{output_file_name}.json")
        upload_file_to_s3(
            f"{output_dir}/{output_file_name}_parameters.csv",
            S3_BUCKET,
            f"{OUTPUT_PATH}/{output_file_name}_parameters.csv",
        )

    return df_scaled


def run_cluster_analysis(data: pd.DataFrame, n_clusters: int, output_dir: str = "models") -> np.ndarray:
    """
    run cluster analysis and return the cluster index.
    :param data:
    :param n_clusters:
    :param output_dir:
    :return:
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data_cluster = kmeans.fit_predict(data)

    # ===== 每个聚类中心在标准化空间的特征值 =====
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    logger.info(f"\n 各聚类中心的标准化特征值：\n {centroids_df}")
    centroids_df.to_csv(f"./{output_dir}/cluster_centers_standardized.csv", index=False)

    # ===== 模型保存 =====
    joblib.dump(kmeans, f"./{output_dir}/kmeans_model.pkl")
    n_features = data.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(kmeans, initial_types=initial_type)
    with open(f"./{output_dir}/kmeans_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    upload_file_to_s3(f"./{output_dir}/kmeans_model.pkl", S3_BUCKET, f"{OUTPUT_PATH}/{output_dir}/kmeans_model.pkl")
    upload_file_to_s3(f"./{output_dir}/kmeans_model.onnx", S3_BUCKET, f"{OUTPUT_PATH}/{output_dir}/kmeans_model.onnx")
    logger.info(f"save cluster model to s3：{OUTPUT_PATH}/models/")
    return data_cluster


def plot_pca_2(
    data: pd.DataFrame, data_cluster: np.ndarray, output_dir: str = "./figures", output_file_name: str = "PCA_Clusters"
) -> None:
    """
    run pca and plot the first 2 components
    :param data:
    :param data_cluster:
    :param output_dir:
    :param output_file_name:
    :return:
    """
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(data)

    plt.figure(figsize=(20, 16))
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=data_cluster, palette="Set1", alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Visualization of KMeans Clusters")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{output_file_name}.png")
    plt.show()

    unique_values, counts = np.unique(data_cluster, return_counts=True)
    cluster_counts = dict(zip(unique_values, counts))
    print(f"\n 每个聚类的样本数量：{cluster_counts}")


def plot_radar_chart(data: pd.DataFrame, data_cluster: np.ndarray) -> None:
    """

    :param data:
    :param data_cluster:
    :return:
    """
    data["_cluster"] = data_cluster
    cluster_means = data.groupby("_cluster").mean().T
    categories = cluster_means.index
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(20, 16), subplot_kw=dict(polar=True))
    for cluster, values in cluster_means.items():
        vals = values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=f"Cluster {cluster}", linewidth=2)
        ax.fill(angles, vals, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    plt.title("Cluster Feature Means (Standardized) - Radar Chart")
    plt.legend(loc="upper right")
    plt.subplots_adjust(left=0.1, bottom=0.1)
    plt.savefig("./figures/Radar_Clusters.png")
    plt.show()


def save_cluster_data(
    data_original: pd.DataFrame,
    data_cluster: pd.DataFrame,
    merge_columns: List[str],
    cluster_column: str = "Cluster",
    feature_columns: Optional[List[str]] = None,
) -> None:
    """
    merge cluster index with original data and save data for each cluster.
    :param data_original:
    :param data_cluster: data with cluster index and reference columns to merge to the original data
    :param merge_columns:
    :param cluster_column:
    :param feature_columns: show stats of columns used in cluster analysis.
    :return:
    """

    data_merged = data_original.merge(data_cluster, on=merge_columns, how="inner")
    for cluster, group_df in data_merged.groupby(cluster_column):
        file_name = f"original_data_cluster_{cluster}.csv"
        group_df.drop(columns=[cluster_column]).to_csv(f"./output/{file_name}", index=False)
        upload_file_to_s3(f"./output/{file_name}", S3_BUCKET, f"{OUTPUT_PATH}/{file_name}")

        if feature_columns is not None:
            stats = group_df[feature_columns].describe().T  # include: count, mean, std, min, 25%, 50%, 75%, max
            print(f"\n 原始特征分布 - cluster {cluster}:")
            print(stats[["mean", "std", "min", "25%", "50%", "75%", "max"]])


if __name__ == "__main__":

    os.makedirs("./.log", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler("./.log/analysis_cluster_02_kmeans.log"), logging.StreamHandler()],
    )

    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    n_clusters = 3

    non_feature_col = ["group_id", "loginname", "start_time"]
    features = [
        "streak_max",
        "morning_count",
        "delta_bet_p25",
        "delta_bet_mean",
        "lose_streak_mean",
        "delta_bet_p75",
        "basepoint_min",
        "lose_streak_p25",
        "delta_profit_min",
        "midnight_count",
        "delta_profit_mean",
        "basepoint_median",
        "lose_streak_max",
        "delta_profit_max",
        "delta_bet_max",
        "lose_streak_median",
        "delta_profit_median",
        "delta_profit_p25",
        "payout_min",
        "payout_p25",
        "delta_bet_min",
        "bet_median",
        "account_stddev",
        "delta_profit_p75",
        "payout_median",
        "payout_p75",
        "profit_p75",
        "payout_max",
        "bet_mean",
        "profit_max",
        "profit_min",
        "payout_mean",
        "profit_median",
        "profit_mean",
        "bet_max",
        "bet_min",
    ]

    wucaishen_files = list_s3_files(S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_24.*\.csv$")
    wucaishen_data = read_files(S3_BUCKET, wucaishen_files, "./output/wucaishen_grouped_stat_output_24.csv")

    data = scale_features(wucaishen_data[features], "./result", "standardized_features")
    data, row_index = remove_outliers(data)
    data_reference = wucaishen_data.loc[row_index, non_feature_col]

    cluster_index = run_cluster_analysis(data, 3)

    plot_pca_2(data, cluster_index)
    plot_radar_chart(data, cluster_index)

    data_reference["Cluster"] = cluster_index
    save_cluster_data(wucaishen_data, data_reference, non_feature_col, "Cluster")
