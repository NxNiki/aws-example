"""
apply kmeans model to 2025 data and make predictions
"""

import numpy as np
import pandas as pd
from cluster_analysis_01_elbow_method import get_feature_names, load_data
from joblib import load

from bituslabs_ds.config import setup_logging
from bituslabs_ds.utils import log_transform
from jobs.sagemaker.cluster_analysis_02_kmeans import load_features


def apply_scale_features(df_features: pd.DataFrame, df_scale: pd.DataFrame) -> pd.DataFrame:
    df_scaled = df_features.copy()

    for col in df_scale.columns:
        print(f"scale feature: {col}")
        mean = df_scale.loc["Mean", col]
        std = df_scale.loc["Std", col]
        if std != 0:
            df_scaled[col] = (df_features[col] - mean) / std

    return df_scaled


if __name__ == "__main__":

    output_path = "."
    top_features = 25

    setup_logging(output_path, "analysis_cluster_03_fit_kmeans.log")

    non_features, _, _ = get_feature_names()
    important_features, features_log = load_features(f"{output_path}/features", top_features)

    wucaishen_data = load_data(
        f"{output_path}/output/wucaishen_grouped_stat_output_2025.csv",
        columns=[*non_features, *important_features],
        pattern=r"wucaishen_grouped_stat_output_25.*\.csv$",
    )

    wucaishen_data = log_transform(wucaishen_data, col_names=features_log)

    scale_params = pd.read_csv(
        f"{output_path}/features/standardized_features_top_{top_features}_parameters.csv", index_col=0
    )
    wucaishen_data = apply_scale_features(wucaishen_data, scale_params.T)
    kmeans = load(f"{output_path}/models/kmeans_model.pkl")
    cluster_labels = kmeans.predict(wucaishen_data[important_features])

    wucaishen_data["cluster"] = cluster_labels
    print(wucaishen_data.head(5))

    for cluster in np.unique(cluster_labels):
        print(f"save cluster: {cluster}")
        wucaishen_data.loc[wucaishen_data["cluster"] == cluster, :].drop(columns="cluster").to_csv(
            f"{output_path}/output/grouped_data_2025_cluster_{cluster}.csv", index=False
        )
