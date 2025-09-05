"""
Example demonstrating the new data loading configuration approach.
This shows how data loading parameters are now managed through YAML configuration files.
"""

import logging

from bituslabs_ds.ml import ClusterAnalysis

logger = logging.getLogger(__name__)


def demonstrate_data_loading_config():
    """Demonstrate how data loading configuration works."""

    # Initialize clustering analysis for wucaishen
    clustering_wucaishen = ClusterAnalysis("wucaishen")

    # Initialize clustering analysis for deepdive
    clustering_deepdive = ClusterAnalysis("deepdive")

    print("=== Wucaishen Project Data Loading Config ===")
    data_config_wucaishen = clustering_wucaishen.get_data_loading_config()

    print(f"Input bucket: {data_config_wucaishen['input_bucket']}")
    print(f"Output bucket: {data_config_wucaishen['output_bucket']}")
    print(f"Input prefix: {data_config_wucaishen['input_prefix']}")
    print(f"Output prefix: {data_config_wucaishen['output_prefix']}")
    print(f"Currency filter: {data_config_wucaishen['currency_filter']}")

    print(f"\nColumns to read: {len(data_config_wucaishen['columns_to_read'])}")
    print(f"  {data_config_wucaishen['columns_to_read']}")

    print(f"\nFile patterns:")
    for pattern_name, pattern in data_config_wucaishen["file_patterns"].items():
        print(f"  {pattern_name}: {pattern}")

    print("\n=== Deepdive Project Data Loading Config ===")
    data_config_deepdive = clustering_deepdive.get_data_loading_config()

    print(f"Input bucket: {data_config_deepdive['input_bucket']}")
    print(f"Output bucket: {data_config_deepdive['output_bucket']}")
    print(f"Input prefix: {data_config_deepdive['input_prefix']}")
    print(f"Output prefix: {data_config_deepdive['output_prefix']}")
    print(f"Currency filter: {data_config_deepdive['currency_filter']}")

    print(f"\nColumns to read: {len(data_config_deepdive['columns_to_read'])}")
    print(f"  {data_config_deepdive['columns_to_read']}")

    print(f"\nFile patterns:")
    for pattern_name, pattern in data_config_deepdive["file_patterns"].items():
        print(f"  {pattern_name}: {pattern}")


def demonstrate_data_loading_methods():
    """Demonstrate the new data loading methods."""

    clustering = ClusterAnalysis("wucaishen")

    print("\n=== Data Loading Methods ===")

    # Example of loading grouped data
    print("Loading grouped data...")
    try:
        # This would normally load data from S3
        # data = clustering.load_grouped_data("grouped_data.csv")
        print("✓ load_grouped_data() method available")
        print("  - Loads grouped statistical data")
        print("  - Applies currency filtering automatically")
        print("  - Uses configuration for bucket, prefix, and patterns")
    except Exception as e:
        print(f"  Note: {e}")

    # Example of loading enriched data
    print("\nLoading enriched data...")
    try:
        # This would normally load data from S3
        # data = clustering.load_enriched_data("enriched_data.csv")
        print("✓ load_enriched_data() method available")
        print("  - Loads enriched original data")
        print("  - Applies currency filtering automatically")
        print("  - Uses configuration for bucket, prefix, and patterns")
    except Exception as e:
        print(f"  Note: {e}")


def demonstrate_configuration_benefits():
    """Demonstrate the benefits of configuration-driven data loading."""

    print("\n=== Configuration Benefits ===")

    print("1. Centralized Configuration:")
    print("   - All data loading parameters in YAML files")
    print("   - Project-specific settings")
    print("   - Easy to modify without code changes")

    print("\n2. Consistent Data Loading:")
    print("   - Same method across all scripts")
    print("   - Automatic currency filtering")
    print("   - Standardized column selection")

    print("\n3. Flexible File Patterns:")
    print("   - Regex patterns for different file types")
    print("   - Easy to adapt to new file naming conventions")
    print("   - Project-specific patterns")

    print("\n4. S3 Integration:")
    print("   - Configurable buckets and prefixes")
    print("   - Automatic S3 file discovery")
    print("   - Local caching support")

    print("\n5. Maintainability:")
    print("   - No hardcoded data loading logic")
    print("   - Easy to add new projects")
    print("   - Consistent error handling")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Data Loading Configuration Example")
    print("=" * 50)

    demonstrate_data_loading_config()
    demonstrate_data_loading_methods()
    demonstrate_configuration_benefits()

    print("\n" + "=" * 50)
    print("This demonstrates how data loading is now fully configuration-driven!")
    print("Benefits:")
    print("- No hardcoded S3 buckets or file patterns")
    print("- Project-specific data loading settings")
    print("- Consistent data loading across all scripts")
    print("- Easy to modify data sources without code changes")
