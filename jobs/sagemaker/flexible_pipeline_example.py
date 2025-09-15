#!/usr/bin/env python3
"""
Example demonstrating the flexible clustering pipeline functionality.
This shows how to create a pipeline and change the number of clusters after creation.
"""

import logging

import numpy as np
import pandas as pd

from bituslabs_ds.ml import ClusterAnalysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_flexible_pipeline():
    """Demonstrate the flexible clustering pipeline functionality."""

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    data = pd.DataFrame(
        {
            "normal_feature_1": np.random.normal(0, 1, n_samples),
            "normal_feature_2": np.random.normal(0, 1, n_samples),
            "skewed_feature_1": np.random.exponential(1, n_samples),
            "skewed_feature_2": np.random.gamma(2, 1, n_samples),
            "normal_feature_3": np.random.normal(0, 1, n_samples),
        }
    )

    transform_columns = ["skewed_feature_1", "skewed_feature_2"]

    print(f"Sample data shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    print(f"Transform columns: {transform_columns}")

    # Initialize clustering analysis
    clustering = ClusterAnalysis("wucaishen")

    print("\n=== Creating Flexible Clustering Pipeline ===")

    # Create flexible pipeline with initial n_clusters=3
    flexible_pipeline = clustering.create_flexible_clustering_pipeline(
        data=data, transform_columns=transform_columns, n_clusters=3  # Initial number of clusters
    )

    print(f"Pipeline steps: {[step[0] for step in flexible_pipeline.steps]}")
    print(f"Initial n_clusters: {flexible_pipeline.named_steps['kmeans'].n_clusters}")

    print("\n=== Testing Different Numbers of Clusters ===")

    # Test different numbers of clusters
    k_values = [2, 3, 4, 5, 6]

    for k in k_values:
        print(f"\n--- Testing k={k} ---")

        # Method 1: Use set_n_clusters and fit_predict
        clustering.set_n_clusters(flexible_pipeline, k)
        cluster_labels_1 = flexible_pipeline.fit_predict(data)

        # Method 2: Use the convenience method
        cluster_labels_2 = clustering.fit_predict_with_n_clusters(flexible_pipeline, data, k)

        # Verify both methods give the same result
        labels_match = np.array_equal(cluster_labels_1, cluster_labels_2)
        print(f"Both methods give same result: {labels_match}")

        # Show cluster distribution
        unique_labels, counts = np.unique(cluster_labels_1, return_counts=True)
        cluster_dist = dict(zip(unique_labels, counts))
        print(f"Cluster distribution: {cluster_dist}")

        # Show inertia
        inertia = flexible_pipeline.named_steps["kmeans"].inertia_
        print(f"Inertia: {inertia:.2f}")

    print("\n=== Benefits of Flexible Pipeline ===")
    print("✅ Same preprocessing pipeline for all k values")
    print("✅ Consistent power transformation and scaling")
    print("✅ Easy to test different numbers of clusters")
    print("✅ Perfect for elbow method analysis")
    print("✅ Reusable pipeline object")

    print("\n=== Usage in Elbow Method ===")
    print(
        """
# Create flexible pipeline once
flexible_pipeline = clustering.create_flexible_clustering_pipeline(
    data=data, transform_columns=skewed_features, n_clusters=2
)

# Test different k values
for k in k_range:
    cluster_labels = clustering.fit_predict_with_n_clusters(
        flexible_pipeline, data, k
    )
    # Calculate metrics (inertia, silhouette score, etc.)
    """
    )

    print("\nFlexible pipeline demonstration completed!")


if __name__ == "__main__":
    demonstrate_flexible_pipeline()
