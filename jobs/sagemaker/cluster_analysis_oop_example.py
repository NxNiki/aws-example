"""
Example usage of the new OOP clustering analysis structure.
This demonstrates how to use the ClusteringAnalysis class for both wucaishen and deepdive projects.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.ml import ClusterAnalysis
from bituslabs_ds.utils import df_power_transform

logger = logging.getLogger(__name__)


# Feature names are now loaded from configuration files via clustering.get_feature_names()


def run_wucaishen_analysis():
    """Run clustering analysis for wucaishen project."""
    logger.info("Starting wucaishen clustering analysis...")

    # Initialize clustering analysis for wucaishen
    clustering = ClusterAnalysis("wucaishen")

    # Get configuration
    config = clustering.read_cluster_config()
    logger.info(f"Using config: {config}")

    # Load data (this would be replaced with actual data loading)
    # Get feature names from configuration
    key_features, normal_features, skewed_features = clustering.get_feature_names()

    # Example data loading - replace with actual implementation
    # data = load_wucaishen_data()

    # Apply power transformation
    # data = df_power_transform(data, skewed_features)

    # Feature selection
    # _, kept_features = clustering.smart_feature_selection(data[normal_features + skewed_features])
    # _, kept_features = clustering.feature_selection_by_variance(data[kept_features])

    # PCA feature selection
    # important_features = clustering.feature_selection_by_pca(data[kept_features + non_features])

    # Elbow method
    # cluster_indices = clustering.elbow_method(data, important_features, [15, 20, 25])

    # Scale features
    # scaled_data = clustering.scale_features(data[important_features], "standardized_features_top_25")

    # Run clustering
    # cluster_labels = clustering.run_cluster_analysis(scaled_data, config["analysis"]["default_n_clusters"])

    # Visualizations
    # clustering.plot_pca_2(scaled_data, cluster_labels)
    # clustering.plot_radar_chart(scaled_data, cluster_labels)

    logger.info("Wucaishen analysis completed!")


def run_deepdive_analysis():
    """Run clustering analysis for deepdive project."""
    logger.info("Starting deepdive clustering analysis...")

    # Initialize clustering analysis for deepdive
    clustering = ClusterAnalysis("deepdive")

    # Get configuration
    config = clustering.read_cluster_config()
    logger.info(f"Using config: {config}")

    # Similar analysis steps as wucaishen but with deepdive-specific configuration
    # The clustering class will automatically use the correct config file

    logger.info("Deepdive analysis completed!")


def run_prediction_example():
    """Example of using trained model for prediction."""
    logger.info("Running prediction example...")

    # Initialize clustering analysis
    clustering = ClusterAnalysis("wucaishen")

    # Load features
    important_features, features_log = clustering.load_features(25)

    # Load new data for prediction
    # new_data = load_new_data()
    # new_data = df_power_transform(new_data, features_log)

    # Apply scaling
    # scale_params = pd.read_csv(clustering.output_path / "features" / "standardized_features_top_25_parameters.csv", index_col=0)
    # new_data_scaled = clustering.apply_scale_features(new_data, scale_params.T)

    # Predict clusters
    # cluster_predictions = clustering.predict_clusters(new_data_scaled, important_features)

    logger.info("Prediction example completed!")


def main():
    """Main function to run clustering analysis examples."""
    parser = argparse.ArgumentParser(description="OOP Clustering Analysis Example")
    parser.add_argument("--project", choices=["wucaishen", "deepdive"], default="wucaishen", help="Project to analyze")
    parser.add_argument("--output_path", type=str, default="./output", help="Output path for logs")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_path, f"clustering_analysis_{args.project}.log")

    if args.project == "wucaishen":
        run_wucaishen_analysis()
    elif args.project == "deepdive":
        run_deepdive_analysis()

    # Run prediction example
    run_prediction_example()


if __name__ == "__main__":
    main()
