"""
Example demonstrating the new clustering pipeline approach.
This shows how all 4 clustering steps are now consolidated into a single configurable pipeline,
and includes a comparison between manual pipeline construction and the new pipeline method.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler

from bituslabs_ds.ml import ClusterAnalysis

logger = logging.getLogger(__name__)


def demonstrate_pipeline_configuration():
    """Demonstrate how pipeline configuration works."""

    # Initialize clustering analysis for wucaishen
    clustering_wucaishen = ClusterAnalysis("wucaishen")
    config_wucaishen = clustering_wucaishen.read_cluster_config()

    # Initialize clustering analysis for deepdive
    clustering_deepdive = ClusterAnalysis("deepdive")
    config_deepdive = clustering_deepdive.read_cluster_config()

    print("=== Wucaishen Pipeline Configuration ===")
    pipeline_config = config_wucaishen["pipeline"]
    print(f"Elbow method: {pipeline_config['elbow_method']}")
    print(f"K-means: {pipeline_config['kmeans']}")
    print(f"Fit K-means: {pipeline_config['fit_kmeans']}")
    print(f"Attach cluster index: {pipeline_config['attach_cluster_index']}")

    print("\n=== Deepdive Pipeline Configuration ===")
    pipeline_config = config_deepdive["pipeline"]
    print(f"Elbow method: {pipeline_config['elbow_method']}")
    print(f"K-means: {pipeline_config['kmeans']}")
    print(f"Fit K-means: {pipeline_config['fit_kmeans']}")
    print(f"Attach cluster index: {pipeline_config['attach_cluster_index']}")


def demonstrate_pipeline_usage():
    """Demonstrate how to use the clustering pipeline."""

    print("\n=== Pipeline Usage Examples ===")

    print("1. Run complete pipeline (all steps):")
    print("   python cluster_analysis_pipeline.py --project wucaishen")
    print("   python cluster_analysis_pipeline.py --project deepdive")

    print("\n2. Run specific steps only:")
    print("   python cluster_analysis_pipeline.py --project wucaishen --steps elbow_method kmeans")
    print("   python cluster_analysis_pipeline.py --project deepdive --steps fit_kmeans attach_cluster_index")

    print("\n3. Run with custom output path:")
    print("   python cluster_analysis_pipeline.py --project wucaishen --output_path /custom/path")

    print("\n4. Disable steps in configuration:")
    print("   Edit wucaishen_cluster_config.yaml:")
    print("   pipeline:")
    print("     elbow_method: false  # Skip this step")
    print("     kmeans: true")
    print("     fit_kmeans: true")
    print("     attach_cluster_index: false  # Skip this step")


def demonstrate_pipeline_benefits():
    """Demonstrate the benefits of the consolidated pipeline approach."""

    print("\n=== Pipeline Benefits ===")

    print("1. Single Script Execution:")
    print("   - One command runs the entire clustering pipeline")
    print("   - No need to run 4 separate scripts")
    print("   - Consistent logging and error handling")

    print("\n2. Configurable Steps:")
    print("   - Enable/disable steps via YAML configuration")
    print("   - Override steps via command line arguments")
    print("   - Project-specific pipeline configurations")

    print("\n3. Better Organization:")
    print("   - All clustering logic in one place")
    print("   - Clear step separation and logging")
    print("   - Consistent data loading across steps")

    print("\n4. Improved Maintainability:")
    print("   - Single point of entry for clustering analysis")
    print("   - Easier to debug and modify")
    print("   - Consistent configuration usage")

    print("\n5. Flexible Execution:")
    print("   - Run complete pipeline or specific steps")
    print("   - Project-specific configurations")
    print("   - Easy to integrate into larger workflows")


def demonstrate_configuration_control():
    """Demonstrate how to control pipeline execution via configuration."""

    print("\n=== Configuration Control Examples ===")

    print("1. Enable only specific steps:")
    print("   pipeline:")
    print("     elbow_method: true")
    print("     kmeans: true")
    print("     fit_kmeans: false")
    print("     attach_cluster_index: false")

    print("\n2. Run analysis only (no prediction):")
    print("   pipeline:")
    print("     elbow_method: true")
    print("     kmeans: true")
    print("     fit_kmeans: false")
    print("     attach_cluster_index: false")

    print("\n3. Run prediction only (skip analysis):")
    print("   pipeline:")
    print("     elbow_method: false")
    print("     kmeans: false")
    print("     fit_kmeans: true")
    print("     attach_cluster_index: true")

    print("\n4. Different configurations per project:")
    print("   - wucaishen: Full pipeline")
    print("   - deepdive: Analysis only")
    print("   - new_project: Custom step combination")


def demonstrate_pipeline_method_comparison():
    """Demonstrate the difference between manual pipeline construction and the new method."""

    print("\n=== Pipeline Method Comparison ===")

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
    n_clusters = 3

    print(f"Sample data shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    print(f"Transform columns: {transform_columns}")
    print(f"Number of clusters: {n_clusters}")

    print("\n--- OLD APPROACH: Manual Pipeline Construction ---")
    print("Code required (~15 lines):")
    print(
        """
pipeline_steps = []
if transform_columns is not None and len(transform_columns) > 1:
    power_columns = [i for i, f in enumerate(data.columns) if f in transform_columns]
    preprocessor = ColumnTransformer(
        transformers=[("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True), power_columns)],
        remainder="passthrough",
    )
    pipeline_steps.append(("power_transform", preprocessor))

pipeline_steps.extend([
    ("scaler", RobustScaler()), 
    ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
])

pipeline = Pipeline(pipeline_steps)
data_cluster = pipeline.fit_predict(data)
    """
    )

    print("\n--- NEW APPROACH: Pipeline Method ---")
    print("Code required (1 line):")
    print(
        """
pipeline, data_cluster = clustering.create_clustering_pipeline(
    data=data, transform_columns=transform_columns, n_clusters=n_clusters
)
    """
    )

    print("\n--- BENEFITS ---")
    print("✅ Code reduction: ~93% (15 lines → 1 line)")
    print("✅ Eliminates code duplication across scripts")
    print("✅ Ensures consistent pipeline configuration")
    print("✅ Centralized maintenance and updates")
    print("✅ Less error-prone and more readable")


def demonstrate_pipeline_method_usage():
    """Demonstrate how to use the new pipeline method."""

    print("\n=== Pipeline Method Usage Examples ===")

    print("1. Basic clustering without power transformation:")
    print(
        """
clustering = ClusterAnalysis("wucaishen")
pipeline, cluster_labels = clustering.create_clustering_pipeline(
    data=data[feature_columns],
    transform_columns=None,  # No power transformation
    n_clusters=3
)
    """
    )

    print("\n2. Clustering with power transformation for skewed features:")
    print(
        """
key_features, normal_features, skewed_features = clustering.get_feature_names()
pipeline, cluster_labels = clustering.create_clustering_pipeline(
    data=data[normal_features + skewed_features],
    transform_columns=skewed_features,  # Apply power transformation
    n_clusters=3
)
    """
    )

    print("\n3. Using pipeline for prediction on new data:")
    print(
        """
new_cluster_labels = pipeline.predict(new_data)
    """
    )

    print("\n4. Saving and loading pipeline models:")
    print(
        """
# Save complete pipeline
clustering.save_pipeline_model(pipeline, "models/my_pipeline.pkl")

# Load and use for prediction
loaded_pipeline = joblib.load("models/my_pipeline.pkl")
predictions = loaded_pipeline.predict(new_data)
    """
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Clustering Pipeline Example")
    print("=" * 50)

    demonstrate_pipeline_configuration()
    demonstrate_pipeline_usage()
    demonstrate_pipeline_benefits()
    demonstrate_configuration_control()
    demonstrate_pipeline_method_comparison()
    demonstrate_pipeline_method_usage()

    print("\n" + "=" * 50)
    print("This demonstrates the new consolidated pipeline approach!")
    print("Benefits:")
    print("- Single script for all clustering steps")
    print("- Configurable pipeline execution")
    print("- Better organization and maintainability")
    print("- Consistent data loading and processing")
    print("- Easy to integrate into larger workflows")
    print("- New pipeline method eliminates code duplication")
    print("- 93% code reduction for pipeline construction")
