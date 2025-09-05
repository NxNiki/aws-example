"""
Comprehensive clustering analysis pipeline.
This script consolidates all 4 clustering steps into a single configurable pipeline.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.ml import ClusterAnalysis
from bituslabs_ds.utils import df_power_transform, remove_outliers, save_list

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """Comprehensive clustering analysis pipeline."""

    def __init__(self, project_name: str):
        """Initialize the clustering pipeline for a specific project."""
        self.project_name = project_name
        self.clustering = ClusterAnalysis(project_name)
        self.config = self.clustering.read_cluster_config()
        self.pipeline_config = self.config["pipeline"]

        logger.info(f"Initialized clustering pipeline for project: {project_name}")
        logger.info(f"Pipeline steps enabled: {[k for k, v in self.pipeline_config.items() if v]}")

    def run_pipeline(self, output_path: Optional[str] = None):
        """Run the complete clustering pipeline based on configuration."""
        if output_path is None:
            output_path = str(self.clustering.output_path)

        start_time = time.time()
        logger.info(f"Starting clustering pipeline for {self.project_name}")

        # Step 1: Elbow Method Analysis
        if self.pipeline_config.get("elbow_method", False):
            logger.info("=" * 50)
            logger.info("STEP 1: Elbow Method Analysis")
            logger.info("=" * 50)
            self.run_elbow_method(output_path)

        # Step 2: K-means Clustering
        if self.pipeline_config.get("kmeans", False):
            logger.info("=" * 50)
            logger.info("STEP 2: K-means Clustering")
            logger.info("=" * 50)
            self.run_kmeans_clustering(output_path)

        # Step 3: Apply Trained Model
        if self.pipeline_config.get("fit_kmeans", False):
            logger.info("=" * 50)
            logger.info("STEP 3: Apply Trained Model")
            logger.info("=" * 50)
            self.run_fit_kmeans(output_path)

        # Step 4: Attach Cluster Index
        if self.pipeline_config.get("attach_cluster_index", False):
            logger.info("=" * 50)
            logger.info("STEP 4: Attach Cluster Index")
            logger.info("=" * 50)
            self.run_attach_cluster_index(output_path)

        elapsed_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        logger.info("=" * 50)

    def run_elbow_method(self, output_path: str):
        """Run elbow method analysis."""
        logger.info("Running elbow method analysis...")

        # Get feature names from configuration
        key_features, normal_features, skewed_features = self.clustering.get_feature_names()

        # Load data using configuration
        data_file = f"{output_path}/output/{self.project_name}_grouped_stat_output_24.csv"
        data = self.clustering.load_grouped_data(data_file)

        # Data profiling and cleaning
        DataProfiler.count_df_missing_columns(data)
        data.fillna(0, inplace=True)
        data = df_power_transform(data, skewed_features)
        save_list(
            normal_features + skewed_features,
            str(self.clustering.output_path / "features" / "log_transform_features.json"),
        )

        # Create correlation heatmap
        viz = DataVisualizer(data[normal_features + skewed_features])
        viz.create_figure(fig_title=f"Correlation of features: {self.project_name}", fig_size=(20, 17))
        viz.add_correlation_heatmap(annot=False, cmap="coolwarm")
        viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
        viz.display()
        viz.save(str(self.clustering.output_path / "figures" / f"{self.project_name}_correlation.png"))

        # Feature selection
        _, kept_features = self.clustering.smart_feature_selection(
            data[normal_features + skewed_features], threshold=self.config["feature_selection"]["correlation_threshold"]
        )
        _, kept_features = self.clustering.feature_selection_by_variance(
            data[kept_features], threshold=self.config["feature_selection"]["variance_threshold"]
        )

        # Select features for clustering
        data_select = data[kept_features + key_features]
        important_features = self.clustering.feature_selection_by_pca(data_select)

        # Run elbow method
        n_features_list = self.config["elbow_method"]["k_range"][:3]  # Use first 3 values
        cluster_indices = self.clustering.elbow_method(data, important_features, n_features_list)

        logger.info("Elbow method analysis completed")

    def run_kmeans_clustering(self, output_path: str):
        """Run K-means clustering analysis."""
        logger.info("Running K-means clustering analysis...")

        # Get feature names from configuration
        key_features, normal_features, skewed_features = self.clustering.get_feature_names()

        # Load features
        top_features = self.config["analysis"]["default_top_features"]
        important_features, features_log = self.clustering.load_features(top_features)
        logger.info(f"Important features: {important_features}")
        logger.info(f"Log transform features: {features_log}")

        # Load data
        data_file = f"{output_path}/output/{self.project_name}_grouped_stat_output_24.csv"
        data = self.clustering.load_grouped_data(data_file)

        # Apply power transformation
        data = df_power_transform(data, col_names=features_log)

        # Scale features
        scaled_data = self.clustering.scale_features(
            data[important_features], f"standardized_features_top_{top_features}", load_cache=False
        )

        # Remove outliers
        scaled_data, row_index = remove_outliers(scaled_data)
        data_reference = data.loc[row_index, key_features]

        # Run clustering analysis
        n_clusters = self.config["analysis"]["default_n_clusters"]
        cluster_index = self.clustering.run_cluster_analysis(scaled_data, n_clusters)

        # Create visualizations
        self.clustering.plot_pca_2(scaled_data, cluster_index)
        self.clustering.plot_radar_chart(scaled_data, cluster_index)

        # Save cluster data
        data_reference["Cluster"] = cluster_index
        self.clustering.save_cluster_data(
            data, data_reference, key_features, "Cluster", feature_columns=important_features
        )

        logger.info("K-means clustering analysis completed")

    def run_fit_kmeans(self, output_path: str):
        """Apply trained K-means model to new data."""
        logger.info("Applying trained K-means model to new data...")

        # Get feature names from configuration
        key_features, normal_features, skewed_features = self.clustering.get_feature_names()

        # Load features
        top_features = self.config["analysis"]["default_top_features"]
        important_features, features_log = self.clustering.load_features(top_features)
        logger.info(f"Important features: {important_features}")
        logger.info(f"Log transform features: {features_log}")

        # Load new data for prediction using configuration
        data_file = f"{output_path}/output/{self.project_name}_grouped_stat_output_2025.csv"
        data = self.clustering.load_grouped_data(data_file)

        # Apply power transformation
        data = df_power_transform(data, col_names=features_log)

        # Load scaling parameters and apply scaling
        scale_params = pd.read_csv(
            self.clustering.output_path / "features" / f"standardized_features_top_{top_features}_parameters.csv",
            index_col=0,
        )
        data_scaled = self.clustering.apply_scale_features(data, scale_params.T)

        # Predict clusters using trained model
        cluster_labels = self.clustering.predict_clusters(data_scaled, important_features)

        # Add cluster labels to data
        data["cluster"] = cluster_labels
        logger.info(f"Predicted clusters: {data['cluster'].value_counts().sort_index()}")

        # Save data for each cluster
        for cluster in sorted(data["cluster"].unique()):
            cluster_data = data[data["cluster"] == cluster].drop(columns="cluster")
            output_file = self.clustering.output_path / "output" / f"grouped_data_2025_cluster_{cluster}.csv"
            cluster_data.to_csv(output_file, index=False)
            logger.info(f"Saved cluster {cluster} data: {output_file}")

        logger.info("K-means model application completed")

    def run_attach_cluster_index(self, output_path: str):
        """Attach cluster indices to original data."""
        logger.info("Attaching cluster indices to original data...")

        # Load enriched data using configuration
        data_file = f"{output_path}/{self.project_name}_enriched_data.csv"
        data = self.clustering.load_enriched_data(data_file)

        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Data preview:\n{data.head(10)}")

        # Load cluster data and merge
        for i in range(3):  # Assuming 3 clusters
            cluster_file = f"{output_path}/kmeans_output/grouped_data_2025_cluster_{i}.csv"

            if os.path.exists(cluster_file):
                cluster_data = pd.read_csv(cluster_file, usecols=["group_id"])
                cluster_data["cluster"] = i

                # Merge with original data
                merged_data = pd.merge(data, cluster_data, how="inner", on="group_id")

                # Save merged data
                output_file = f"{self.project_name}_with_cluster_2025_{i}.csv"
                merged_data.to_csv(f"{output_path}/{output_file}", index=False)
                logger.info(f"Saved cluster {i} data: {output_file} ({len(merged_data)} records)")
            else:
                logger.warning(f"Cluster file not found: {cluster_file}")

        logger.info("Cluster index attachment completed")


def main():
    """Main function to run the clustering pipeline."""
    parser = argparse.ArgumentParser(description="Comprehensive Clustering Analysis Pipeline")
    parser.add_argument("--project", choices=["wucaishen", "deepdive"], default="wucaishen", help="Project to analyze")
    parser.add_argument(
        "--output_path", type=str, default=None, help="Output path for results (default: project-specific)"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["elbow_method", "kmeans", "fit_kmeans", "attach_cluster_index"],
        help="Specific steps to run (overrides config)",
    )
    args = parser.parse_args()

    # Setup logging
    log_path = args.output_path or f"./output_{args.project}"
    setup_logging(log_path, f"clustering_pipeline_{args.project}.log")

    # Initialize pipeline
    pipeline = ClusteringPipeline(args.project)

    # Override pipeline steps if specified
    if args.steps:
        for step in pipeline.pipeline_config:
            pipeline.pipeline_config[step] = step in args.steps
        logger.info(f"Overriding pipeline steps: {args.steps}")

    # Run pipeline
    pipeline.run_pipeline(args.output_path)


if __name__ == "__main__":
    main()
