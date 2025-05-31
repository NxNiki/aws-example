import os
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import onnx
import pandas as pd
import seaborn as sns
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import upload_file_to_s3
from jobs.analysis_cluster_01_elbow_method import OUTPUT_PATH


def run_kmeans_pipeline(data_original, df, features, n_clusters=3, z_thresh=3, random_state=42):
    os.makedirs("./images", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    # ===== 1. 特征标准化 =====
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    df_scaled = pd.DataFrame(X_scaled, columns=features)
    df_scaled.to_csv("./output/standardized_features.csv", index=False)
    df_scaled.to_json("./output/standardized_features.json", orient="records", indent=2)

    mean_std_df = pd.DataFrame({"Mean": scaler.mean_, "Std": scaler.scale_}, index=features)
    mean_std_df.to_csv("./output/feature_standardization_parameters.csv")
    print("\n 每个特征的标准化参数（均值与标准差）：")
    print(mean_std_df)

    # ===== 2. 异常值剔除（z-score） =====
    z_scores = np.abs(zscore(X_scaled))
    filtered_indices = (z_scores < z_thresh).all(axis=1)
    X_filtered = X_scaled[filtered_indices]
    df_filtered = df[filtered_indices].copy()

    # ===== 3. KMeans 聚类 =====
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_filtered["Cluster"] = kmeans.fit_predict(X_filtered)

    # ===== 4. Elbow Curve =====
    inertias = []
    k_range = range(2, min(11, len(X_filtered)))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_filtered)
        inertias.append(km.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker="o")
    plt.title("Elbow Method - Inertia vs. Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.savefig("./images/Elbow_Curve.png")
    plt.show()

    # ===== 5. PCA 可视化 =====
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_filtered)

    plt.figure(figsize=(20, 16))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_filtered["Cluster"], palette="Set1", alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Visualization of KMeans Clusters")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.savefig("./images/PCA_Clusters.png")
    plt.show()

    # ===== 6. 雷达图 =====
    def plot_radar_chart(X_std, cluster_labels):
        df_radar = pd.DataFrame(X_std, columns=features)
        df_radar["Cluster"] = cluster_labels
        cluster_means = df_radar.groupby("Cluster").mean().T
        categories = cluster_means.index
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(20, 16), subplot_kw=dict(polar=True))
        for cluster, values in cluster_means.iteritems():
            vals = values.tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=f"Cluster {cluster}", linewidth=2)
            ax.fill(angles, vals, alpha=0.2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        plt.title("Cluster Feature Means (Standardized) - Radar Chart")
        plt.legend(loc="upper right")
        plt.savefig("./images/Radar_Clusters.png")
        plt.show()

    plot_radar_chart(X_filtered, df_filtered["Cluster"])

    # ===== 7. 每个聚类的样本数 =====
    cluster_counts = df_filtered["Cluster"].value_counts().sort_index()
    print("\n 每个聚类的样本数量：")
    print(cluster_counts)

    # ===== 8. 每个聚类中心在标准化空间的特征值 =====
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\n 各聚类中心的标准化特征值：")
    print(centroids_df)
    centroids_df.to_csv("./output/cluster_centers_standardized.csv", index=False)

    # ===== 9. 模型保存 =====
    joblib.dump(kmeans, "./models/kmeans_model.pkl")
    n_features = X_scaled.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(kmeans, initial_types=initial_type)
    with open("./models/kmeans_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # ===== 10. 将聚类结果合并回原始数据（保留全部字段） =====
    # 1. 准备用于 merge 的 key
    merge_keys = ["group_id", "loginname", "start_time"]

    # 2. 提取聚类结果对应的索引行，并加上聚类标签
    df_cluster_mapping = df_filtered[merge_keys].copy()
    df_cluster_mapping["Cluster"] = df_filtered["Cluster"].values

    # 3. merge 到原始数据
    df_with_cluster = data_original.copy()
    df_with_cluster = df_with_cluster.merge(df_cluster_mapping, on=merge_keys, how="left")
    df_with_cluster["Cluster"] = df_with_cluster["Cluster"].fillna(-1).astype(int)

    for i in range(n_clusters):
        group_df = df_with_cluster[df_with_cluster["Cluster"] == i].drop(columns=["Cluster"])
        group_df.to_csv(f"./output/wucaishen_202503_group_{i + 1}.csv", index=False)
    # 创建字典：key 为 (loginname, group_id)，value 为 Cluster
    # 将 (loginname, group_id) 转为字符串 key
    cluster_dict = {f"{row['group_id']}": int(row["Cluster"]) for _, row in df_with_cluster.iterrows()}
    # 保存为 JSON 文件
    with open("./output/cluster_mapping_dict.json", "w") as f:
        json.dump(cluster_dict, f, indent=2)

    print("\n 已保存 loginname + group_id 到 Cluster 的映射字典：cluster_mapping_dict.json")

    print("\n 模型保存成功：kmeans_model.pkl 和 kmeans_model.onnx")
    print("📄 已输出：标准化特征、聚类中心、标准化参数、分组原始数据。")
    # ===== 11. 每个聚类的原始特征值范围（标准化前） =====

    for i in range(n_clusters):
        group_raw = df[filtered_indices].copy()
        group_raw["Cluster"] = df_filtered["Cluster"].values
        group_i = group_raw[group_raw["Cluster"] == i][features]

        stats = group_i.describe().T  # include: count, mean, std, min, 25%, 50%, 75%, max
        print(f"\n 原始特征分布 - Group {i + 1}:")
        print(stats[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    return df_filtered


if __name__ == "__main__":
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
