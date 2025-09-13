# Clustering Analysis OOP Refactoring

This document describes the refactoring of the clustering analysis code to use Object-Oriented Programming (OOP) principles, consolidating all clustering functionality into a single `ClusteringAnalysis` class.

## Overview

The refactoring addresses the following goals:
- **Consolidate functionality**: All clustering operations are now in one class
- **Support multiple projects**: Separate configuration files for wucaishen and deepdive projects
- **Eliminate code duplication**: Shared functionality is centralized
- **Improve maintainability**: Cleaner, more organized code structure

## New Structure

### Core Class: `ClusterAnalysis`

Located in `src/bituslabs_ds/ml.py`, this class provides all clustering functionality:

```python
from bituslabs_ds.ml import ClusterAnalysis

# Initialize for wucaishen project
clustering = ClusterAnalysis("wucaishen")

# Initialize for deepdive project  
clustering = ClusterAnalysis("deepdive")
```

### Configuration Files

Each project now has its own configuration file with complete feature definitions:
- `wucaishen_cluster_config.yaml` - Configuration for wucaishen project
- `deepdive_cluster_config.yaml` - Configuration for deepdive project
- `cluster_config.yaml` - Default/fallback configuration

**New Features**: 
- Feature names are now defined in the configuration files, eliminating hardcoded feature lists in the code
- Data loading parameters (S3 buckets, file patterns, columns) are now configuration-driven
- **Consolidated Pipeline**: All 4 clustering steps are now in a single configurable pipeline script
- Pipeline steps can be enabled/disabled via YAML configuration or command line arguments
- All refactored scripts use `clustering.get_feature_names()` and `clustering.load_grouped_data()` methods

### Key Methods

#### Configuration Management
- `read_cluster_config()` - Read project-specific configuration
- `get_feature_names()` - Get feature names from configuration
- `get_data_loading_config()` - Get data loading configuration
- `_load_config()` - Load YAML configuration file
- `_setup_directories()` - Create necessary output directories

#### Data Loading
- `load_grouped_data()` - Load grouped statistical data from S3
- `load_enriched_data()` - Load enriched original data from S3

#### Feature Selection
- `smart_feature_selection()` - Remove highly correlated features
- `feature_selection_by_variance()` - Remove low-variance features
- `feature_selection_by_pca()` - Select features using PCA importance

#### Clustering Analysis
- `elbow_method()` - Determine optimal number of clusters
- `run_cluster_analysis()` - Perform K-means clustering
- `predict_clusters()` - Apply trained model to new data

#### Visualization
- `plot_pca_2()` - PCA visualization of clusters
- `plot_radar_chart()` - Radar chart of cluster features
- `_plot_elbow_method()` - Elbow method plots

#### Data Processing
- `scale_features()` - Normalize features
- `apply_scale_features()` - Apply scaling to new data
- `load_features()` - Load feature lists
- `save_cluster_data()` - Save clustered data

## Usage Examples

### Basic Usage

```python
from bituslabs_ds.ml import ClusterAnalysis

# Initialize for wucaishen project
clustering = ClusterAnalysis("wucaishen")

# Get configuration and feature names
config = clustering.read_cluster_config()
key_features, normal_features, skewed_features = clustering.get_feature_names()

# Load and process data
data = load_your_data()
data = df_power_transform(data, skewed_features)

# Feature selection
_, kept_features = clustering.smart_feature_selection(data)
important_features = clustering.feature_selection_by_pca(data[kept_features])

# Run elbow method
cluster_indices = clustering.elbow_method(data, important_features)

# Scale features
scaled_data = clustering.scale_features(data[important_features], "scaled_features")

# Run clustering
cluster_labels = clustering.run_cluster_analysis(scaled_data, n_clusters=3)

# Visualizations
clustering.plot_pca_2(scaled_data, cluster_labels)
clustering.plot_radar_chart(scaled_data, cluster_labels)
```

### Project-Specific Configuration

Each project can have different configurations including feature definitions:

```yaml
# wucaishen_cluster_config.yaml
analysis:
  default_n_clusters: 3
  default_top_features: 25
elbow_method:
  k_range: [15, 20, 25, 30, 35, 40]
data_loading:
  input_bucket: "hyber-slot"
  output_bucket: "hyber-slot"
  input_prefix: "wucaishen_processed_data"
  columns_to_read: ["loginname", "billno", "group_id", ...]
  file_patterns:
    grouped_data: "wucaishen_grouped_stat_output_24.*\\.csv$"
    enriched_data: "25\\d{2}/wucaishen_enriched_output/part-.*\\.csv"
  currency_filter: "CNY"
features:
  key_features: ["group_id", "loginname", "start_time"]  # For table joins
  skewed_features: ["rtp_mean", "bet_min", "bet_max", ...]  # Need log transform
  normal_features: ["slottype_2_count", "payout_rate", ...]  # No transform

# deepdive_cluster_config.yaml  
analysis:
  default_n_clusters: 3
  default_top_features: 30
elbow_method:
  k_range: [15, 20, 25, 30, 35, 40, 45, 50]
data_loading:
  # Same structure but potentially different buckets/prefixes/patterns
features:
  # Same structure but potentially different features
```

## Refactored Files

### New Files
- `cluster_analysis_pipeline.py` - **Main pipeline script** consolidating all 4 clustering steps
- `cluster_analysis_oop_example.py` - Example usage of the new OOP structure
- `cluster_analysis_01_elbow_method_refactored.py` - Refactored elbow method (legacy)
- `cluster_analysis_02_kmeans_refactored.py` - Refactored K-means clustering (legacy)
- `cluster_analysis_03_fit_kmeans_refactored.py` - Refactored model application (legacy)
- `cluster_analysis_04_attach_cluster_index_refactored.py` - Refactored cluster attachment (legacy)
- `wucaishen_cluster_config.yaml` - Wucaishen project configuration with features, data loading, and pipeline
- `deepdive_cluster_config.yaml` - Deepdive project configuration with features, data loading, and pipeline
- `feature_config_example.py` - Example demonstrating feature configuration
- `data_loading_example.py` - Example demonstrating data loading configuration
- `pipeline_example.py` - Example demonstrating pipeline configuration

### Original Files (Preserved)
- `cluster_analysis_01_elbow_method.py` - Original elbow method
- `cluster_analysis_02_kmeans.py` - Original K-means clustering
- `cluster_analysis_03_fit_kmeans.py` - Original model application
- `cluster_analysis_04_attach_cluster_index.py` - Original cluster attachment
- `cluster_config.py` - Original configuration loader
- `cluster_config.yaml` - Original configuration file

## Migration Guide

### From Original Scripts to OOP

**Before (Original):**
```python
# Multiple separate scripts with duplicated code
from cluster_analysis_01_elbow_method import elbow_method, feature_selection_by_pca
from cluster_analysis_02_kmeans import run_cluster_analysis, scale_features
from cluster_config import DEFAULT_N_CLUSTERS, KMEANS_RANDOM_STATE

# Manual configuration management
# Duplicated feature selection code
# Separate visualization functions
```

**After (OOP):**
```python
# Single class with all functionality
from bituslabs_ds.ml import ClusteringAnalysis

clustering = ClusteringAnalysis("wucaishen")
config = clustering.read_cluster_config()

# All methods available in one object
clustering.smart_feature_selection(data)
clustering.elbow_method(data, features)
clustering.run_cluster_analysis(data, config["analysis"]["default_n_clusters"])
```

### Configuration Migration

**Before:**
```python
from cluster_config import DEFAULT_N_CLUSTERS, KMEANS_RANDOM_STATE
# Hard-coded values, single config file
```

**After:**
```python
clustering = ClusteringAnalysis("wucaishen")
config = clustering.read_cluster_config()
n_clusters = config["analysis"]["default_n_clusters"]
# Project-specific configuration, flexible values
```

## Benefits

1. **Code Reusability**: Single class handles all clustering operations
2. **Project Isolation**: Separate configurations prevent conflicts
3. **Maintainability**: Centralized functionality is easier to maintain
4. **Extensibility**: Easy to add new projects or modify existing ones
5. **Consistency**: Standardized interface across all clustering operations
6. **Documentation**: Clear method documentation and type hints
7. **Configuration-Driven**: Feature names, data loading, and parameters in YAML files
8. **No Hardcoded Features**: All feature definitions in configuration
9. **No Hardcoded Data Loading**: All S3 buckets, patterns, and columns in configuration
10. **Clean Code**: Removed hardcoded functions from refactored scripts

## Pipeline Configuration

The pipeline steps can be controlled via YAML configuration:

```yaml
pipeline:
  elbow_method: true  # Run elbow method analysis
  kmeans: true  # Run K-means clustering
  fit_kmeans: true  # Apply trained model to new data
  attach_cluster_index: true  # Attach cluster indices to original data
```

**Pipeline Control Options:**
- **YAML Configuration**: Enable/disable steps in project config files
- **Command Line Override**: Use `--steps` to override configuration
- **Project-Specific**: Different pipeline configurations per project

**Example Configurations:**
- **Full Pipeline**: All steps enabled (default)
- **Analysis Only**: `elbow_method: true, kmeans: true, fit_kmeans: false, attach_cluster_index: false`
- **Prediction Only**: `elbow_method: false, kmeans: false, fit_kmeans: true, attach_cluster_index: true`

## Running the Clustering Pipeline

### Main Pipeline Script (Recommended)
```bash
# Run complete pipeline for wucaishen
python cluster_analysis_pipeline.py --project wucaishen

# Run complete pipeline for deepdive
python cluster_analysis_pipeline.py --project deepdive

# Run specific steps only
python cluster_analysis_pipeline.py --project wucaishen --steps elbow_method kmeans

# Run with custom output path
python cluster_analysis_pipeline.py --project wucaishen --output_path /custom/path
```

### Individual Scripts (Legacy)
```bash
# Individual refactored scripts (still available)
python cluster_analysis_01_elbow_method_refactored.py --project wucaishen --output_path ./output
python cluster_analysis_02_kmeans_refactored.py --project wucaishen --n_clusters 3 --top_features 25
python cluster_analysis_03_fit_kmeans_refactored.py --project wucaishen --top_features 25
python cluster_analysis_04_attach_cluster_index_refactored.py --project wucaishen --input ./data --output ./results
```

## Future Enhancements

1. **Additional Projects**: Easy to add new projects by creating new config files
2. **New Algorithms**: Can extend the class to support other clustering algorithms
3. **Advanced Visualization**: Add more visualization methods
4. **Model Persistence**: Enhanced model saving/loading capabilities
5. **Parallel Processing**: Add support for parallel clustering operations

## Troubleshooting

### Common Issues

1. **Config File Not Found**: Ensure the project-specific config file exists
2. **Directory Creation**: The class automatically creates necessary directories
3. **Feature Loading**: Ensure feature files exist from previous steps
4. **Model Loading**: Ensure trained models exist before prediction

### Debug Mode

Enable debug logging to see detailed information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
