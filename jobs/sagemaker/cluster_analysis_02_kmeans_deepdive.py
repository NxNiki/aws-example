import argparse
import logging
import os
from datetime import datetime

import pandas as pd
from cluster_analysis_02_kmeans import (
    load_features,
    plot_pca_2,
    plot_radar_chart,
    run_cluster_analysis,
    save_cluster_data,
)
from cluster_config import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_TOP_FEATURES,
    OUTPUT_PATH,
    WORK_DIR,
)

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.s3_utils import upload_folder_to_s3

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--top_features", type=int, default=DEFAULT_TOP_FEATURES)
    parser.add_argument("--input_path_data", type=str, default=WORK_DIR)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--upload_result_to_s3", type=bool, default=False)
    args = parser.parse_args()

    output_path = args.output_path
    os.makedirs(f"{args.output_path}/figures", exist_ok=True)
    os.makedirs(f"{args.output_path}/models", exist_ok=True)
    os.makedirs(f"{args.output_path}/features", exist_ok=True)
    os.makedirs(f"{args.output_path}/output", exist_ok=True)

    setup_logging(f"{output_path}/log", "analysis_cluster_02_kmeans.log")

    non_features = ["group_id", "loginname", "start_time"]
    important_features, features_log = load_features(f"{args.output_path}", args.top_features)

    player_data = pd.read_csv(
        f"{args.input_path_data}/deepdive_grouped_stat_output_25_outliers_removed.csv",
        usecols=[*non_features, *important_features],
    )
    cluster_index, data_trasformed = run_cluster_analysis(
        player_data[important_features], args.n_clusters, features_log, output_path
    )

    plot_pca_2(data_trasformed, cluster_index, output_path)
    plot_radar_chart(data_trasformed, cluster_index, output_path)

    data_reference = player_data[non_features]
    data_reference["Cluster"] = cluster_index
    save_cluster_data(player_data, data_reference, non_features, "Cluster", output_dir=output_path)

    if args.upload_result_to_s3:
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_prefix = f"deepdive_analysis_kmeans/{time_tag}"
        upload_folder_to_s3(output_path, S3_BUCKET, s3_prefix)
