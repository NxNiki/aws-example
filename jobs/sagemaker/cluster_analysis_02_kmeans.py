import argparse
import json
import logging
import os
from typing import List, Optional, Tuple

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

from bituslabs_ds.config import setup_logging
from bituslabs_ds.utils import log_transform, remove_outliers

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def scale_features(
    data: pd.DataFrame, output_dir: str, output_file_name: str, load_cache: bool = False
) -> pd.DataFrame:
    """
    Normalize the columns of df to have zero mean and unit variance.
    :param data:
    :param output_dir:
    :param output_file_name:
    :param load_cache:
    :return:
    """
    output_dir = output_dir.rstrip("/")

    if os.path.exists(f"{output_dir}/features/{output_file_name}.csv") and load_cache:
        logger.info(f"read existing output file {output_file_name}...")
        df_scaled = pd.read_csv(f"{output_dir}/features/{output_file_name}.csv")
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(data)
        df_scaled = pd.DataFrame(x_scaled, columns=data.columns)

        df_scaled.to_csv(f"{output_dir}/features/{output_file_name}.csv", index=False)
        df_scaled.to_json(f"{output_dir}/features/{output_file_name}.json", orient="records", indent=2)

        mean_std_df = pd.DataFrame({"Mean": scaler.mean_, "Std": scaler.scale_}, index=data.columns)
        mean_std_df.to_csv(f"{output_dir}/features/{output_file_name}_parameters.csv")
        print("\n 每个特征的标准化参数（均值与标准差）：")
        print(mean_std_df)

    return df_scaled


def load_features(feature_path: str, top_features: int) -> Tuple[List[str], List[str]]:

    important_features = json.load(open(f"{feature_path}/important_features.json", "r"))
    important_features = important_features[:top_features]
    print(important_features)

    features_log = json.load(open(f"{feature_path}/log_transform_features.json", "r"))
    features_log = [f for f in features_log if f in important_features]
    print(features_log)

    return important_features, features_log


def run_cluster_analysis(data: pd.DataFrame, n_clusters: int, output_dir: str = ".") -> np.ndarray:
    """
    run cluster analysis and return the cluster index.
    :param data:
    :param n_clusters:
    :param output_dir:
    :return:
    """
    output_dir = output_dir.rstrip("/")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data_cluster = kmeans.fit_predict(data)

    # ===== 每个聚类中心在标准化空间的特征值 =====
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
    logger.info(f"\n 各聚类中心的标准化特征值：\n {centroids_df}")
    centroids_df.to_csv(f"{output_dir}/models/cluster_centers_standardized_{len(data.columns)}.csv", index=False)

    # ===== 模型保存 =====
    joblib.dump(kmeans, f"{output_dir}/models/kmeans_model.pkl")
    n_features = data.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(kmeans, initial_types=initial_type)
    with open(f"{output_dir}/models/kmeans_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"save cluster model to s3：{output_dir}/models")
    return data_cluster


def plot_pca_2(
    data: pd.DataFrame, data_cluster: np.ndarray, output_dir: str = ".", output_file_name: str = "PCA_Clusters"
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
    plt.savefig(f"{output_dir}/figures/{output_file_name}.png")
    plt.show()

    unique_values, counts = np.unique(data_cluster, return_counts=True)
    cluster_counts = dict(zip(unique_values, counts))
    print(f"\n 每个聚类的样本数量：{cluster_counts}")


def plot_radar_chart(data: pd.DataFrame, data_cluster: np.ndarray, output_dir: str = ".") -> None:
    """

    :param data:
    :param data_cluster:
    :param output_dir:
    :return:
    """
    output_dir = output_dir.rstrip("/")
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
    plt.savefig(f"{output_dir}/figures/Radar_Clusters.png")
    plt.show()


def save_cluster_data(
    data_original: pd.DataFrame,
    data_cluster: pd.DataFrame,
    merge_columns: List[str],
    cluster_column: str = "Cluster",
    feature_columns: Optional[List[str]] = None,
    output_dir: str = ".",
) -> None:
    """
    merge cluster index with original data and save data for each cluster.
    :param data_original:
    :param data_cluster: data with cluster index and reference columns to merge to the original data
    :param merge_columns:
    :param cluster_column:
    :param feature_columns: show stats of columns used in cluster analysis.
    :param output_dir:
    :return:
    """

    output_dir.rstrip("/")
    data_merged = data_original.merge(data_cluster, on=merge_columns, how="inner")
    for cluster, group_df in data_merged.groupby(cluster_column):
        file_name = f"original_data_cluster_{cluster}.csv"
        group_df.drop(columns=[cluster_column]).to_csv(f"{output_dir}/output/{file_name}", index=False)
        if feature_columns is not None:
            stats = group_df[feature_columns].describe().T  # include: count, mean, std, min, 25%, 50%, 75%, max
            print(f"\n 原始特征分布 - cluster {cluster}:")
            print(stats[["mean", "std", "min", "25%", "50%", "75%", "max"]])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--top_features", type=int, default=25)
    parser.add_argument("--input_path_data", type=str, default=".input")
    parser.add_argument("--input_path_features", type=str, default=".input")
    parser.add_argument("--output_path", type=str, default=".output")
    args = parser.parse_args()

    output_path = args.output_path
    os.makedirs(f"{args.output_path}/figures", exist_ok=True)
    os.makedirs(f"{args.output_path}/models", exist_ok=True)
    os.makedirs(f"{args.output_path}/features", exist_ok=True)
    os.makedirs(f"{args.output_path}/output", exist_ok=True)

    setup_logging(output_path, "analysis_cluster_02_kmeans.log")

    non_features = ["group_id", "loginname", "start_time"]
    important_features, features_log = load_features(args.input_path_features, args.top_features)

    wucaishen_data = pd.read_csv(
        f"{args.input_path_data}/wucaishen_grouped_stat_output_24.csv",
        usecols=[*non_features, *important_features],
    )

    wucaishen_data = log_transform(wucaishen_data, col_names=features_log)

    data = scale_features(
        wucaishen_data[important_features], output_path, f"standardized_features_top_{args.top_features}"
    )
    data, row_index = remove_outliers(data)
    data_reference = wucaishen_data.loc[row_index, non_features]

    cluster_index = run_cluster_analysis(data, args.n_clusters, output_path)

    plot_pca_2(data, cluster_index, output_path)
    plot_radar_chart(data, cluster_index, output_path)

    data_reference["Cluster"] = cluster_index
    save_cluster_data(wucaishen_data, data_reference, non_features, "Cluster", output_dir=output_path)
