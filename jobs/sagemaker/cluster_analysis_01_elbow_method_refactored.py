"""
Refactored elbow method clustering analysis using OOP structure and pipeline method.
This script uses the new ClusterAnalysis class and create_clustering_pipeline method
for streamlined clustering analysis with automatic power transformation and scaling.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.ml import ClusterAnalysis, calculate_silhouette_score
from bituslabs_ds.utils import save_list

logger = logging.getLogger(__name__)


# This script now uses the create_clustering_pipeline method which automatically:
# 1. Applies power transformation to skewed features
# 2. Scales features using RobustScaler
# 3. Performs K-means clustering
# All in a single, reusable pipeline method for elbow method analysis


def main(project_name: str, output_path: str):
    """Main function for elbow method analysis."""
    # Initialize clustering analysis
    clustering = ClusterAnalysis(project_name)

    # Get configuration
    config = clustering.config
    logger.info(f"Using project: {project_name}")
    logger.info(f"Output path: {clustering.output_path}")

    # Get feature names from configuration
    key_features, normal_features, skewed_features = clustering.get_feature_names()

    # Load data using configuration
    data_file = f"{output_path}/output/{project_name}_grouped_stat_output_24.csv"
    # Get all features needed for analysis
    all_features = key_features + normal_features + skewed_features
    data = clustering.load_grouped_data(data_file, columns=all_features)

    # Data profiling and cleaning
    DataProfiler.count_df_missing_columns(data)
    data.fillna(0, inplace=True)

    # Save log transform features for reference
    save_list(skewed_features, str(clustering.output_path / "features" / "log_transform_features.json"))

    # Create correlation heatmap
    viz = DataVisualizer(data[normal_features + skewed_features])
    viz.create_figure(fig_title=f"Correlation of features: {project_name}", fig_size=(20, 17))
    viz.add_correlation_heatmap(annot=False, cmap="coolwarm")
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    viz.display()
    viz.save(str(clustering.output_path / "figures" / f"{project_name}_correlation.png"))

    # Feature selection
    _, kept_features = clustering.smart_feature_selection(
        data[normal_features + skewed_features], threshold=config["feature_selection"]["correlation_threshold"]
    )
    _, kept_features = clustering.feature_selection_by_variance(
        data[kept_features], threshold=config["feature_selection"]["variance_threshold"]
    )

    # Select features for clustering
    data_select = data[kept_features + key_features]
    important_features = clustering.feature_selection_by_pca(data_select)

    # Prepare data for clustering (exclude key features)
    clustering_data = data[important_features]

    # Create a flexible clustering pipeline that allows changing n_clusters
    logger.info("Creating flexible clustering pipeline for elbow method analysis...")
    flexible_pipeline = clustering.create_flexible_clustering_pipeline(
        data=clustering_data,
        transform_columns=skewed_features,  # Apply power transformation to skewed features
        n_clusters=config["elbow_method"]["k_range"][0],  # Start with first k value
    )

    # Save the flexible pipeline
    clustering.save_pipeline_model(flexible_pipeline, f"{clustering.output_path}/models/elbow_flexible_pipeline.pkl")

    # Run elbow method using the flexible pipeline approach
    logger.info("Running elbow method analysis with flexible pipeline...")
    n_features_list = config["elbow_method"]["k_range"][:3]  # Use first 3 values
    k_range = config["elbow_method"]["k_range"]

    # Use the flexible pipeline for elbow method evaluation
    cluster_indices = {}

    for n_features in n_features_list:
        logger.info(f"Running elbow method for {n_features} features...")
        cluster_indices_by_k = {}
        inertia = []
        silhouette_scores = []
        cluster_sizes = []

        # Get the data subset for this number of features
        data_subset = clustering_data.iloc[:, :n_features]

        for k in k_range:
            # Use the flexible pipeline with different k values
            cluster_labels = clustering.fit_predict_with_n_clusters(flexible_pipeline, data_subset, k)

            # Calculate metrics
            inertia.append(flexible_pipeline.named_steps["kmeans"].inertia_)
            silhouette_scores.append(calculate_silhouette_score(data_subset, cluster_labels))
            cluster_counts = np.bincount(cluster_labels)
            cluster_sizes.append(cluster_counts)

            cluster_indices_by_k[k] = cluster_labels

        cluster_indices[n_features] = cluster_indices_by_k

        # Plot elbow method for this feature set
        clustering._plot_elbow_method(k_range, inertia, silhouette_scores, cluster_sizes, n_features)

    logger.info(f"Elbow method analysis completed for {project_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactored Elbow Method Clustering Analysis")
    parser.add_argument("--project", choices=["wucaishen", "deepdive"], default="wucaishen", help="Project to analyze")
    parser.add_argument("--output_path", type=str, default="./output", help="Output path for results")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_path, f"analysis_cluster_01_elbow_method_{args.project}.log")

    main(args.project, args.output_path)
