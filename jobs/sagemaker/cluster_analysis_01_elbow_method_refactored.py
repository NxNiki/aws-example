"""
Refactored elbow method clustering analysis using OOP structure.
This script uses the new ClusteringAnalysis class for feature selection and elbow method.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.ml import ClusterAnalysis
from bituslabs_ds.utils import df_power_transform, save_list

logger = logging.getLogger(__name__)


def main(project_name: str, output_path: str):
    """Main function for elbow method analysis."""
    # Initialize clustering analysis
    clustering = ClusterAnalysis(project_name)

    # Get configuration
    config = clustering.read_cluster_config()
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
    data = df_power_transform(data, skewed_features)
    save_list(
        normal_features + skewed_features, str(clustering.output_path / "features" / "log_transform_features.json")
    )

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

    # Run elbow method
    n_features_list = config["elbow_method"]["k_range"][:3]  # Use first 3 values
    cluster_indices = clustering.elbow_method(data, important_features, n_features_list)

    logger.info(f"Elbow method analysis completed for {project_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactored Elbow Method Clustering Analysis")
    parser.add_argument("--project", choices=["wucaishen", "deepdive"], default="wucaishen", help="Project to analyze")
    parser.add_argument("--output_path", type=str, default="./output", help="Output path for results")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_path, f"analysis_cluster_01_elbow_method_{args.project}.log")

    main(args.project, args.output_path)
