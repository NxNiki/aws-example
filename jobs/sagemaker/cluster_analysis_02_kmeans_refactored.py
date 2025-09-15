"""
Refactored K-means clustering analysis using OOP structure and pipeline method.
This script uses the new ClusterAnalysis class and create_clustering_pipeline method
for streamlined clustering analysis with automatic power transformation and scaling.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.ml import ClusterAnalysis
from bituslabs_ds.utils import remove_outliers

logger = logging.getLogger(__name__)


# This script now uses the create_clustering_pipeline method which automatically:
# 1. Applies power transformation to skewed features
# 2. Scales features using RobustScaler
# 3. Performs K-means clustering
# All in a single, reusable pipeline method


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
    config = clustering.config
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

    # Prepare data for clustering (exclude key features)
    clustering_data = data[important_features]

    # Remove outliers before clustering
    clustering_data, row_index = remove_outliers(clustering_data)
    data_reference = data.loc[row_index, key_features]

    # Use the new pipeline method for clustering
    logger.info("Running clustering analysis with pipeline method...")
    pipeline, cluster_index = clustering.create_clustering_pipeline(
        data=clustering_data,
        transform_columns=features_log,  # Apply power transformation to skewed features
        n_clusters=n_clusters,
    )

    # Get transformed data for visualization
    transformed_data = pipeline[:-1].transform(clustering_data)
    scaled_data = pd.DataFrame(transformed_data, columns=important_features)

    # Save the complete pipeline model
    clustering.save_pipeline_model(pipeline, f"{clustering.output_path}/models/kmeans_pipeline_model.pkl")

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
