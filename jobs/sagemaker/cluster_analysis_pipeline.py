"""
Refactored elbow method clustering analysis using OOP structure and pipeline method.
This script uses the new ClusterAnalysisPipline class and create_clustering_pipeline method
for streamlined clustering analysis with automatic power transformation and scaling.
"""

import argparse
import logging
import os

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.ml import ClusterAnalysisPipeline

logger = logging.getLogger(__name__)


# This script now uses the create_clustering_pipeline method which automatically:
# 1. Applies power transformation to skewed features
# 2. Scales features using RobustScaler
# 3. Performs K-means clustering
# All in a single, reusable pipeline method for elbow method analysis


def main(config_path: str):

    project_name = os.path.basename(config_path).replace(".yaml", "")
    setup_logging(LOCAL_ROOT / "jobs/log", f"cluster_analysis_pipeline_{project_name}.log")

    cluster_pipeline = ClusterAnalysisPipeline(config_path)
    data = cluster_pipeline.load_data(data_label="cluster_data", reload=False)

    # Data profiling and cleaning
    DataProfiler.count_df_missing_columns(data)
    data.fillna(0, inplace=True)

    data_transformed = cluster_pipeline.power_transform(data)

    # Create correlation heatmap
    features = cluster_pipeline.normal_features + cluster_pipeline.skewed_features
    viz = DataVisualizer(data_transformed[features])
    viz.create_figure(fig_title=f"Correlation of features: {cluster_pipeline.project_name}", fig_size=(20, 17))
    viz.add_correlation_heatmap(annot=False, cmap="coolwarm")
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    viz.display()
    viz.save(str(cluster_pipeline.output_path / "figures" / f"{cluster_pipeline.project_name}_correlation.png"))

    # Feature selection
    _, kept_features = cluster_pipeline.smart_feature_selection(data_transformed[features])
    _, kept_features = cluster_pipeline.feature_selection_by_variance(data_transformed[kept_features])

    # Select features for clustering
    data_select = data_transformed[kept_features]
    important_features = cluster_pipeline.feature_selection_by_pca(data_select)

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

    if cluster_pipeline.run_fit_cluster_model:
        pass

    if cluster_pipeline.run_attach_cluster_label:
        cluster_pipeline.attach_cluster_label()


if __name__ == "__main__":

    # project = "deepdive"
    project = "wucaishen"

    current_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description="cluster analysis pipeline")
    parser.add_argument(
        "--config_file", default=f"{current_path}/cluster_config-{project}.yaml", help="Project to analyze"
    )
    args = parser.parse_args()

    # Setup logging
    main(args.config_file)
