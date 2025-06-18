"""
apply kmeans model to 2025 data and make predictions
"""

import json

import numpy as np
import pandas as pd
from joblib import load

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_files
from bituslabs_ds.utils import log_transform


def scale_features(df_features: pd.DataFrame, df_scale: pd.DataFrame) -> pd.DataFrame:
    df_scaled = df_features.copy()

    for col in df_scale.columns:
        print(f"scale feature: {col}")
        mean = df_scale.loc["Mean", col]
        std = df_scale.loc["Std", col]
        if std != 0:
            df_scaled[col] = (df_features[col] - mean) / std

    return df_scaled


output_path = "."

wucaishen_files = list_s3_files(S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_25.*\.csv$")
wucaishen_data = read_files(
    wucaishen_files, local_cache_path=f"{output_path}/output/wucaishen_grouped_stat_output_25.csv", reload=False
)

non_features = ["group_id", "loginname", "start_time"]
important_features = json.load(open(f"{output_path}/features/important_features.json", "r"))
important_features = important_features[:25]
print(important_features)

features_log = json.load(open(f"{output_path}/features/log_transform_features.json", "r"))
features_log = [f for f in features_log if f in important_features]
print(features_log)

wucaishen_data = log_transform(wucaishen_data, features_log)

scale_params = pd.read_csv(f"{output_path}/features/standardized_features_top_25_parameters.csv", index_col=0)

wucaishen_data = scale_features(wucaishen_data, scale_params.T)

kmeans = load(f"{output_path}/models/kmeans_model.pkl")
cluster_labels = kmeans.predict(wucaishen_data[important_features])

# 4. Attach cluster labels to new data
wucaishen_data["cluster"] = cluster_labels

print(wucaishen_data.head(5))

for cluster in np.unique(cluster_labels):
    print(f"save cluster: {cluster}")
    wucaishen_data.loc[wucaishen_data["cluster"] == cluster, :].drop(columns="cluster").to_csv(
        f"{output_path}/output/grouped_data_2025_cluster_{cluster}.csv", index=False
    )
