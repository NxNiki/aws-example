import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cluster_analysis_01_elbow_method_deepdive import load_data
from cluster_analysis_02_kmeans import (
    load_features,
    plot_pca_2,
    plot_radar_chart,
    run_cluster_analysis,
    save_cluster_data,
    scale_features,
)
from cluster_config import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_TOP_FEATURES,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    OUTPUT_PATH,
    USE_CNY,
    WORK_DIR,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.config import setup_logging
from bituslabs_ds.utils import df_power_transform, remove_outliers

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--top_features", type=int, default=DEFAULT_TOP_FEATURES)
    parser.add_argument("--input_path_data", type=str, default=WORK_DIR)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    output_path = args.output_path
    os.makedirs(f"{args.output_path}/figures", exist_ok=True)
    os.makedirs(f"{args.output_path}/models", exist_ok=True)
    os.makedirs(f"{args.output_path}/features", exist_ok=True)
    os.makedirs(f"{args.output_path}/output", exist_ok=True)

    setup_logging(output_path, "analysis_cluster_02_kmeans.log")

    non_features = ["group_id", "loginname", "start_time"]
    important_features, features_log = load_features(f"{args.output_path}/features", args.top_features)

    player_data = pd.read_csv(
        f"{args.input_path_data}/deepdive_grouped_stat_output_25.csv",
        usecols=[*non_features, *important_features],
    )

    player_data = df_power_transform(player_data, col_names=features_log)

    data = scale_features(
        player_data[important_features], output_path, f"standardized_features_top_{args.top_features}"
    )
    data, row_index = remove_outliers(data)
    data_reference = player_data.loc[row_index, non_features]

    cluster_index = run_cluster_analysis(data, args.n_clusters, output_path)

    plot_pca_2(data, cluster_index, output_path)
    plot_radar_chart(data, cluster_index, output_path)

    data_reference["Cluster"] = cluster_index
    save_cluster_data(player_data, data_reference, non_features, "Cluster", output_dir=output_path)
