"""
Example demonstrating the new feature configuration approach.
This shows how feature names are now loaded from YAML configuration files.
"""

import logging

from bituslabs_ds.ml import ClusterAnalysis

logger = logging.getLogger(__name__)


def demonstrate_feature_config():
    """Demonstrate how feature configuration works."""

    # Initialize clustering analysis for wucaishen
    clustering_wucaishen = ClusterAnalysis("wucaishen")

    # Initialize clustering analysis for deepdive
    clustering_deepdive = ClusterAnalysis("deepdive")

    print("=== Wucaishen Project Features ===")
    key_features, normal_features, skewed_features = clustering_wucaishen.get_feature_names()

    print(f"Key features (for joining tables): {len(key_features)}")
    print(f"  {key_features}")

    print(f"\nNormal features (no transformation): {len(normal_features)}")
    print(f"  {normal_features}")

    print(f"\nSkewed features (need log transformation): {len(skewed_features)}")
    print(f"  First 10: {skewed_features[:10]}")
    print(f"  ... and {len(skewed_features) - 10} more")

    print("\n=== Deepdive Project Features ===")
    key_features, normal_features, skewed_features = clustering_deepdive.get_feature_names()

    print(f"Key features (for joining tables): {len(key_features)}")
    print(f"  {key_features}")

    print(f"\nNormal features (no transformation): {len(normal_features)}")
    print(f"  {normal_features}")

    print(f"\nSkewed features (need log transformation): {len(skewed_features)}")
    print(f"  First 10: {skewed_features[:10]}")
    print(f"  ... and {len(skewed_features) - 10} more")

    print("\n=== Configuration Comparison ===")
    config_wucaishen = clustering_wucaishen.read_cluster_config()
    config_deepdive = clustering_deepdive.read_cluster_config()

    print(f"Wucaishen default top features: {config_wucaishen['analysis']['default_top_features']}")
    print(f"Deepdive default top features: {config_deepdive['analysis']['default_top_features']}")

    print(f"Wucaishen k_range: {config_wucaishen['elbow_method']['k_range']}")
    print(f"Deepdive k_range: {config_deepdive['elbow_method']['k_range']}")


def demonstrate_feature_structure():
    """Demonstrate the simplified feature structure."""

    clustering = ClusterAnalysis("wucaishen")
    config = clustering.read_cluster_config()

    print("\n=== Simplified Feature Structure ===")

    key_features = config["features"]["key_features"]
    normal_features = config["features"]["normal_features"]
    skewed_features = config["features"]["skewed_features"]

    print(f"Key features (for table joins): {len(key_features)}")
    print(f"  {key_features}")

    print(f"\nNormal features (no transformation): {len(normal_features)}")
    print(f"  {normal_features}")

    print(f"\nSkewed features (log transformation): {len(skewed_features)}")
    print(f"  First 10: {skewed_features[:10]}")
    print(f"  Last 10: {skewed_features[-10:]}")

    print(f"\nTotal features for analysis: {len(normal_features) + len(skewed_features)}")
    print(f"  Normal: {len(normal_features)}")
    print(f"  Skewed: {len(skewed_features)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Feature Configuration Example")
    print("=" * 50)

    demonstrate_feature_config()
    demonstrate_feature_structure()

    print("\n" + "=" * 50)
    print("This demonstrates how feature names are now managed through configuration files!")
    print("Benefits:")
    print("- Project-specific feature sets")
    print("- Easy to modify without code changes")
    print("- Consistent feature generation")
    print("- Better maintainability")
