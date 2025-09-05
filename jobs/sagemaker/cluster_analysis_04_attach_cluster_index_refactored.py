"""
Refactored cluster index attachment using OOP structure.
This script attaches cluster indices to original data using the ClusteringAnalysis class.
"""

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.ml import ClusterAnalysis

logger = logging.getLogger(__name__)


def main(project_name: str, input_dir: str, output_dir: str):
    """Main function for attaching cluster indices to original data."""
    start_time = time.time()

    # Initialize clustering analysis
    clustering = ClusterAnalysis(project_name)

    # Get configuration
    config = clustering.read_cluster_config()
    logger.info(f"Using project: {project_name}")
    logger.info(f"Output path: {clustering.output_path}")

    # Load enriched data using configuration
    data_file = f"{input_dir}/{project_name}_enriched_data.csv"
    data = clustering.load_enriched_data(data_file)

    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"Data preview:\n{data.head(10)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load cluster data and merge
    for i in range(3):  # Assuming 3 clusters
        cluster_file = f"{input_dir}/kmeans_output/grouped_data_2025_cluster_{i}.csv"

        if os.path.exists(cluster_file):
            cluster_data = pd.read_csv(cluster_file, usecols=["group_id"])
            cluster_data["cluster"] = i

            # Merge with original data
            merged_data = pd.merge(data, cluster_data, how="inner", on="group_id")

            # Save merged data
            output_file = f"{project_name}_with_cluster_2025_{i}.csv"
            merged_data.to_csv(f"{output_dir}/{output_file}", index=False)
            logger.info(f"Saved cluster {i} data: {output_file} ({len(merged_data)} records)")
        else:
            logger.warning(f"Cluster file not found: {cluster_file}")

    elapsed_time = time.time() - start_time
    logger.info(f"Cluster index attachment completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactored Cluster Index Attachment")
    parser.add_argument("--project", choices=["wucaishen", "deepdive"], default="wucaishen", help="Project to analyze")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, default="./output", help="Output directory")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output, f"analysis_cluster_04_attach_cluster_index_{args.project}.log")

    main(args.project, args.input, args.output)
