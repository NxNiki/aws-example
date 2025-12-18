"""
Refactored elbow method clustering analysis using OOP structure and pipeline method.
This script uses the new ClusterAnalysisPipline class and create_clustering_pipeline method
for streamlined clustering analysis with automatic power transformation and scaling.
"""

import argparse
import gc
import logging
import os
import time
from datetime import datetime

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.ml import ClusterAnalysisPipeline
from bituslabs_ds.s3_utils import upload_folder_to_s3

logger = logging.getLogger(__name__)


# This script now uses the create_clustering_pipeline method which automatically:
# 1. Applies power transformation to skewed features
# 2. Scales features using RobustScaler
# 3. Performs K-means clustering
# All in a single, reusable pipeline method for elbow method analysis

RELOAD_CLUSTER_DATA = False
RELOAD_ATTACH_DATA = False


def feature_selection(data, cluster_pipeline):
    data_transformed = cluster_pipeline.preprocess_data(data)

    # Create correlation heatmap
    features = cluster_pipeline.normal_features + cluster_pipeline.skewed_features
    viz = DataVisualizer(data_transformed[features])
    viz.create_figure(fig_title=f"Correlation of features: {cluster_pipeline.project_name}", fig_size=(20, 17))
    viz.add_correlation_heatmap(annot=False, cmap="coolwarm")
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    # viz.display()
    viz.save(str(cluster_pipeline.output_path / "figures" / f"{cluster_pipeline.project_name}_correlation.png"))

    del viz
    gc.collect()

    # Feature selection
    features = cluster_pipeline.normal_features + cluster_pipeline.skewed_features
    _, kept_features = cluster_pipeline.smart_feature_selection(data_transformed[features])
    _, kept_features = cluster_pipeline.feature_selection_by_variance(data_transformed[kept_features])
    important_features = cluster_pipeline.feature_selection_by_pca(data_transformed[kept_features])

    return important_features


def main(config_path: str):
    start_time = time.time()

    project_name = os.path.basename(config_path).replace(".yaml", "")
    time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(LOCAL_ROOT / "jobs/log", f"cluster_analysis_{project_name}_{time_tag}.log")

    cluster_pipeline = ClusterAnalysisPipeline(config_path)
    data = cluster_pipeline.load_cluster_data(reload=RELOAD_CLUSTER_DATA)

    # Data profiling and cleaning
    DataProfiler.count_df_missing_columns(data)
    data.dropna(how="any", inplace=True)

    important_features = feature_selection(data, cluster_pipeline)

    # feed original data (without power transform which is packed in the clustering pipeline)
    if cluster_pipeline.run_elbow_method:
        cluster_pipeline.elbow_method(
            data=data,
            features_ordered_by_importance=important_features,
        )

    if cluster_pipeline.run_cluster_analysis:
        cluster_pipeline.cluster_analysis(
            data=data,
            features_ordered_by_importance=important_features,
        )

    del data
    gc.collect()

    if cluster_pipeline.run_fit_cluster_model:
        pass

    if cluster_pipeline.run_attach_cluster_label:
        cluster_pipeline.attach_cluster_label(reload=RELOAD_ATTACH_DATA)

    if cluster_pipeline.run_get_cluster_stats:
        cluster_pipeline.get_cluster_stats()

    if cluster_pipeline.run_upload_result_to_s3:
        upload_folder_to_s3(cluster_pipeline.output_path, S3_BUCKET, f"{cluster_pipeline.s3_prefix}_{time_tag}")

    elapsed_time = time.time() - start_time
    logger.info(f"Total running time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":

    project = "deepdive"
    # project = "wucaishen"

    current_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description="cluster analysis pipeline")
    parser.add_argument(
        "--config_file", default=f"{current_path}/cluster_config-{project}.yaml", help="Project to analyze"
    )
    args = parser.parse_args()

    main(args.config_file)
