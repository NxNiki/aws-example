"""
Refactored K-means clustering analysis using OOP structure.
This script uses the new ClusteringAnalysis class for K-means clustering.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.ml import ClusterAnalysis
from bituslabs_ds.utils import df_power_transform, remove_outliers

logger = logging.getLogger(__name__)


# Feature names are now loaded from configuration files via clustering.get_feature_names()


def main(
    project_name: str,
    n_clusters: int,
    top_features: int,
    input_path_data: str,
    input_path_features: str,
    output_path: str,
):
    """Main function for K-means clustering analysis."""
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

    # Load data
    data_file = f"{input_path_data}/{project_name}_grouped_stat_output_24.csv"
    data = pd.read_csv(data_file, usecols=[*key_features, *important_features])

    # Apply power transformation
    data = df_power_transform(data, col_names=features_log)

    # Scale features
    scaled_data = clustering.scale_features(
        data[important_features], f"standardized_features_top_{top_features}", load_cache=False
    )

    # Remove outliers
    scaled_data, row_index = remove_outliers(scaled_data)
    data_reference = data.loc[row_index, key_features]

    # Run clustering analysis
    cluster_index = clustering.run_cluster_analysis(scaled_data, n_clusters)

    # Create visualizations
    clustering.plot_pca_2(scaled_data, cluster_index)
    clustering.plot_radar_chart(scaled_data, cluster_index)

    # Save cluster data
    data_reference["Cluster"] = cluster_index
    clustering.save_cluster_data(data, data_reference, key_features, "Cluster", feature_columns=important_features)

    logger.info(f"K-means clustering analysis completed for {project_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactored K-means Clustering Analysis")
    parser.add_argument("--project", choices=["wucaishen", "deepdive"], default="wucaishen", help="Project to analyze")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--top_features", type=int, default=25, help="Number of top features")
    parser.add_argument("--input_path_data", type=str, default=".input", help="Input data path")
    parser.add_argument("--input_path_features", type=str, default=".input", help="Input features path")
    parser.add_argument("--output_path", type=str, default=".output", help="Output path")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_path, f"analysis_cluster_02_kmeans_{args.project}.log")

    main(
        args.project,
        args.n_clusters,
        args.top_features,
        args.input_path_data,
        args.input_path_features,
        args.output_path,
    )
