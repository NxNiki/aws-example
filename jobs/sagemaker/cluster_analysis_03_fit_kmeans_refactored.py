"""
Refactored K-means model application using OOP structure.
This script applies a trained K-means model to new data using the ClusteringAnalysis class.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.ml import ClusterAnalysis
from bituslabs_ds.utils import df_power_transform

logger = logging.getLogger(__name__)


def main(project_name: str, top_features: int, output_path: str):
    """Main function for applying trained K-means model."""
    # Initialize clustering analysis
    clustering = ClusterAnalysis(project_name)

    # Get configuration
    config = clustering.read_cluster_config()
    logger.info(f"Using project: {project_name}")
    logger.info(f"Output path: {clustering.output_path}")

    # Get feature names from configuration
    key_features, normal_features, skewed_features = clustering.get_feature_names()

    # Load features
    important_features, features_log = clustering.load_features(top_features)
    logger.info(f"Important features: {important_features}")
    logger.info(f"Log transform features: {features_log}")

    # Load new data for prediction using configuration
    data_file = f"{output_path}/output/{project_name}_grouped_stat_output_2025.csv"
    data = clustering.load_grouped_data(data_file)

    # Apply power transformation
    data = df_power_transform(data, col_names=features_log)

    # Load scaling parameters and apply scaling
    scale_params = pd.read_csv(
        clustering.output_path / "features" / f"standardized_features_top_{top_features}_parameters.csv", index_col=0
    )
    data_scaled = clustering.apply_scale_features(data, scale_params.T)

    # Predict clusters using trained model
    cluster_labels = clustering.predict_clusters(data_scaled, important_features)

    # Add cluster labels to data
    data["cluster"] = cluster_labels
    logger.info(f"Predicted clusters: {data['cluster'].value_counts().sort_index()}")

    # Save data for each cluster
    for cluster in sorted(data["cluster"].unique()):
        cluster_data = data[data["cluster"] == cluster].drop(columns="cluster")
        output_file = clustering.output_path / "output" / f"grouped_data_2025_cluster_{cluster}.csv"
        cluster_data.to_csv(output_file, index=False)
        logger.info(f"Saved cluster {cluster} data: {output_file}")

    logger.info(f"K-means model application completed for {project_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactored K-means Model Application")
    parser.add_argument("--project", choices=["wucaishen", "deepdive"], default="wucaishen", help="Project to analyze")
    parser.add_argument("--top_features", type=int, default=25, help="Number of top features")
    parser.add_argument("--output_path", type=str, default=".", help="Output path")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_path, f"analysis_cluster_03_fit_kmeans_{args.project}.log")

    main(args.project, args.top_features, args.output_path)
