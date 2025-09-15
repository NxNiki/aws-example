"""
Machine Learning utilities and clustering analysis classes.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from joblib import parallel_backend
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from bituslabs_ds.s3_utils import list_s3_files, read_files
from bituslabs_ds.utils import column_iterator, keep_numeric_columns, remove_outliers, save_list

logger = logging.getLogger(__name__)


class ClusterAnalysis:
    """
    Comprehensive clustering analysis class that consolidates all clustering functionality.
    Supports multiple projects with separate configuration files.
    """

    def __init__(self, project_name: str, config_path: Optional[str] = None):
        """
        Initialize clustering analysis for a specific project.

        Args:
            project_name: Name of the project (e.g., 'wucaishen', 'deepdive')
            config_path: Optional path to config file. If None, uses default location.
        """
        self.project_name = project_name
        self._config_path = config_path
        self._config = self._load_config(config_path)
        self._setup_directories()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def reload_config(self) -> None:
        self._config = self._load_config(self._config_path)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path based on project
            config_path = str(
                Path(__file__).parent.parent.parent / "jobs" / "sagemaker" / f"cluster_config-{self.project_name}.yaml"
            )

        if not os.path.exists(config_path):
            # Fallback to default config
            config_path = str(Path(__file__).parent.parent.parent / "jobs" / "sagemaker" / "cluster_config.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config for project '{self.project_name}' from {config_path}")
        return config

    def _setup_directories(self):
        """Create necessary directories for the project."""
        base_path = Path(__file__).parent.parent.parent / "jobs" / "sagemaker" / self.project_name
        self.output_path = base_path / self.config["directories"]["output_path"]
        self.work_dir = base_path / self.config["directories"]["work_dir"]

        # Create directories
        for dir_name in ["output", "features", "figures", "models", "log"]:
            os.makedirs(self.output_path / dir_name, exist_ok=True)

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get feature names from configuration.

        Returns:
            Tuple of (key_features, normal_features, skewed_features)
        """
        config = self.config

        # Get features from configuration
        key_features = config["features"]["key_features"]
        normal_features = config["features"]["normal_features"]
        skewed_features = config["features"]["skewed_features"]

        return key_features, normal_features, skewed_features

    def get_data_loading_config(self) -> Dict[str, Any]:
        """Get data loading configuration."""
        config = self.config
        return config["data_loading"]

    def load_grouped_data(
        self, output_file: str, pattern: Optional[str] = None, columns: Optional[list] = None
    ) -> pd.DataFrame:
        """Load grouped data from S3 using configuration."""
        data_config = self.get_data_loading_config()

        if pattern is None:
            pattern = data_config["file_patterns"]["grouped_data"]

        if columns is None:
            columns = data_config["columns_to_read"]

        return self._load_s3_data(data_config, pattern, output_file, columns)

    def load_enriched_data(
        self, output_file: str, pattern: Optional[str] = None, columns: Optional[list] = None
    ) -> pd.DataFrame:
        """Load enriched data from S3 using configuration."""
        data_config = self.get_data_loading_config()

        if pattern is None:
            pattern = data_config["file_patterns"]["enriched_data"]

        if columns is None:
            columns = data_config["columns_to_read"]

        return self._load_s3_data(data_config, pattern, output_file, columns)

    def _load_s3_data(
        self,
        data_config: Dict[str, Any],
        pattern: str,
        output_file: str,
        columns: Optional[list],
    ) -> pd.DataFrame:
        """List, read, and row-filter S3 dataset based on data_config."""
        files = list_s3_files(data_config["input_bucket"], data_config["input_prefix"], pattern)

        data = read_files(
            files,
            local_cache_path=output_file,
            columns=columns,
            reload=False,
        )

        # Apply general row filters if specified
        row_filters = data_config.get("row_filters")
        if row_filters:
            for column_name, allowed in row_filters.items():
                if column_name not in data.columns:
                    logger.warning(f"Row filter column '{column_name}' not in data; skipping this filter")
                    continue
                if isinstance(allowed, (list, set, tuple)):
                    data = data[data[column_name].isin(list(allowed))]
                    logger.info(
                        f"Applied filter on '{column_name}' with {len(list(allowed))} allowed values; remaining {len(data)} rows"
                    )
                else:
                    data = data[data[column_name] == allowed]
                    logger.info(f"Applied filter on '{column_name}' == {allowed!r}; remaining {len(data)} rows")

        return data

    def smart_feature_selection(
        self, data: pd.DataFrame, threshold: Optional[float] = None, prefer_keywords: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Remove features that are highly correlated with each other.

        Args:
            data: DataFrame with complete dataset
            threshold: Correlation threshold (default from config)
            prefer_keywords: Keywords to prefer when selecting features

        Returns:
            Tuple of (features_to_drop, features_to_keep)
        """
        if threshold is None:
            threshold = self.config["feature_selection"]["correlation_threshold"]
        if prefer_keywords is None:
            prefer_keywords = ["mean", "median"]

        data = keep_numeric_columns(data)
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        kept = set()

        for column in upper.columns:
            high_corr = upper[column][upper[column] > threshold].index.tolist()
            if high_corr:
                high_corr = [column] + high_corr
                keep_feat = high_corr[0]

                # Check for preferred keywords
                for feat in high_corr:
                    if any(key in feat.lower() for key in prefer_keywords):
                        keep_feat = feat
                        break

                kept.add(keep_feat)
                high_corr.remove(keep_feat)
                to_drop.update(high_corr)
            else:
                kept.add(column)

        logger.info(f"Kept: {len(kept)} features")
        logger.info(f"Removed: {len(to_drop)} features")
        return list(to_drop), list(kept)

    def feature_selection_by_variance(
        self, data: pd.DataFrame, threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with variance below threshold."""
        if threshold is None:
            threshold = self.config["feature_selection"]["variance_threshold"]

        data = keep_numeric_columns(data)
        selector = VarianceThreshold(threshold=threshold)
        data_filtered = selector.fit_transform(data)
        selected_features = data.columns[selector.get_support()].tolist()

        logger.info(f"Variance selection kept: {len(selected_features)} features")
        return data_filtered, selected_features

    def feature_selection_by_pca(self, data: pd.DataFrame) -> List[str]:
        """Select features using PCA importance."""

        data = keep_numeric_columns(data)
        pca = PCA(n_components=data.shape[1])
        pca.fit(data)

        # Calculate feature importance
        importance = np.abs(pca.components_).sum(axis=0).tolist()
        features = data.columns.tolist()
        feature_importance = pd.DataFrame({"Feature": features, "Importance": importance})
        feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

        # Plot feature importance
        plt.figure(figsize=(18, 10))
        colors = sns.color_palette("viridis", len(feature_importance))
        sns.barplot(x="Importance", y="Feature", hue="Feature", data=feature_importance, palette=colors, legend=False)
        plt.title("Feature Importance Rank", fontsize=16)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.savefig(self.output_path / "figures" / "Feature Importance Rank.png")
        plt.close()

        # Save feature importance
        features_ordered = [features[i] for i in np.argsort(importance)[::-1]]
        save_list(features_ordered, str(self.output_path / "features" / "important_features.json"))

        return features_ordered

    def elbow_method(
        self,
        data: pd.DataFrame,
        features: List[str],
        n_features: Optional[Union[List[int], int]] = None,
        pipeline: Optional[Pipeline] = None,
    ) -> Dict[Any, Dict[Any, Any]]:
        """Run elbow method to determine optimal number of clusters."""

        k_range = self.config["elbow_method"]["k_range"]
        cluster_indices = {}

        for x, n in column_iterator(data, features, n_features):
            cluster_indices_by_k = {}
            inertia = []
            silhouette_scores = []
            cluster_sizes = []

            # Use provided pipeline or create a basic scaler
            if pipeline is not None:
                # Use the provided pipeline for preprocessing
                x_transformed = pipeline[:-1].transform(x)  # Apply all steps except the final K-means
            else:
                # Fallback to basic scaling
                scaler = StandardScaler()
                x_transformed = scaler.fit_transform(x)

            for k in k_range:
                model = KMeans(
                    n_clusters=k,
                    random_state=self.config["kmeans"]["random_state"],
                    n_init=self.config["kmeans"]["n_init"],
                )
                model.fit(x_transformed)
                labels = model.labels_
                inertia.append(model.inertia_)

                cluster_indices_by_k[k] = labels
                silhouette_scores.append(calculate_silhouette_score(x_transformed, labels))
                cluster_counts = np.bincount(labels)
                cluster_sizes.append(cluster_counts)

            cluster_indices[n] = cluster_indices_by_k

            # Plot elbow method
            self._plot_elbow_method(k_range, inertia, silhouette_scores, cluster_sizes, n)

        return cluster_indices

    def _plot_elbow_method(self, k_range, inertia, silhouette_scores, cluster_sizes, n_features):
        """Plot elbow method results."""
        title = f"Elbow Method for Optimal k n_features({n_features})"
        fig, ax1 = plt.subplots(figsize=(12, 9))

        (line1,) = ax1.plot(k_range, inertia, marker="o", linestyle="-", label="Inertia")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia")
        ax1.set_title(title, fontsize=16)

        ax2 = ax1.twinx()
        (line2,) = ax2.plot(
            k_range, silhouette_scores, marker="s", linestyle="-", color="red", label="Silhouette Score"
        )
        ax2.set_ylabel("Silhouette Score")

        lines = [line1, line2]
        labels = [str(line.get_label()) for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        # Add cluster sizes table
        self._add_cluster_sizes_table(plt, k_range, cluster_sizes)

        plt.subplots_adjust(left=0.1, bottom=0.3)
        plt.savefig(self.output_path / "figures" / f"{title}.png")
        plt.close()

    def _add_cluster_sizes_table(self, plt, k_range, cluster_sizes):
        """Add cluster sizes table to elbow method plot."""
        max_clusters = max(len(sizes) for sizes in cluster_sizes)
        cluster_sizes_str = []

        for sizes in cluster_sizes:
            row = [f"{count} ({count/sum(sizes):.3f})".replace("(0.", "(.") for count in sizes]
            row += [""] * (max_clusters - len(row))
            cluster_sizes_str.append(row)

        cluster_sizes_table = list(map(list, zip(*cluster_sizes_str)))
        row_labels = [f"C{i + 1}" for i in range(max_clusters)]

        plt.table(
            cellText=cluster_sizes_table,
            rowLabels=row_labels,
            colLabels=[f"k={k}" for k in k_range],
            cellLoc="center",
            loc="bottom",
            bbox=[0.0, -0.5, 1, 0.3],
        )

    def scale_features(self, data: pd.DataFrame, output_file_name: str, load_cache: bool = False) -> pd.DataFrame:
        """Normalize features to zero mean and unit variance.
        This is obsolete and will be removed as we pack scaler into the pipline model (.onnx/.pickle file).
        """
        output_file = self.output_path / "features" / f"{output_file_name}.csv"

        if os.path.exists(output_file) and load_cache:
            logger.info(f"Loading cached scaled features: {output_file_name}")
            df_scaled = pd.read_csv(output_file)
        else:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(data)
            df_scaled = pd.DataFrame(x_scaled, columns=data.columns)

            # Save scaled data
            df_scaled.to_csv(output_file, index=False)
            df_scaled.to_json(self.output_path / "features" / f"{output_file_name}.json", orient="records", indent=2)

            # Save scaling parameters
            mean_std_df = pd.DataFrame({"Mean": scaler.mean_, "Std": scaler.scale_}, index=data.columns)
            mean_std_df.to_csv(self.output_path / "features" / f"{output_file_name}_parameters.csv")

            logger.info(f"Scaled features saved: {output_file_name}")

        return df_scaled

    def load_features(self, top_features: int) -> Tuple[List[str], List[str]]:
        """Load important features and log transform features."""
        important_features = json.load(open(self.output_path / "features" / "important_features.json", "r"))[
            :top_features
        ]

        features_log = json.load(open(self.output_path / "features" / "log_transform_features.json", "r"))
        features_log = [f for f in features_log if f in important_features]

        return important_features, features_log

    def run_cluster_analysis(
        self, data: pd.DataFrame, n_clusters: int, transform_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Pipeline]:
        """Run K-means clustering analysis using pipeline approach."""
        # Create clustering pipeline
        pipeline, data_cluster = self.create_clustering_pipeline(
            data=data, transform_columns=transform_columns, n_clusters=n_clusters
        )

        # Save cluster centers
        centroids_df = pd.DataFrame(pipeline.named_steps["kmeans"].cluster_centers_, columns=data.columns)
        logger.info(f"Cluster centers:\n{centroids_df}")
        centroids_df.to_csv(
            self.output_path / "models" / f"cluster_centers_standardized_{len(data.columns)}.csv", index=False
        )

        # Save pipeline model
        n_features = data.shape[1]
        model_name = f"kmeans_model_top{n_features}_features"
        joblib.dump(pipeline, self.output_path / "models" / f"{model_name}.pkl")

        # Save ONNX model with version from config
        onnx_opset_version = self.config.get("onnx", {}).get("opset_version", 19)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        logger.info(f"Using ONNX opset version: {onnx_opset_version}")
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=onnx_opset_version)
        onnx_filename = f"{model_name}_opset_{onnx_opset_version}.onnx"
        with open(self.output_path / "models" / onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"K-means pipeline model saved to: {self.output_path / 'models'}")
        logger.info(f"ONNX model saved as: {onnx_filename}")
        return data_cluster, pipeline

    def plot_pca_2(self, data: pd.DataFrame, data_cluster: np.ndarray, output_file_name: str = "PCA_Clusters") -> None:
        """Plot PCA visualization of clusters."""
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(data)

        plt.figure(figsize=(20, 16))
        sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=data_cluster, palette="Set1", alpha=0.7)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("PCA Visualization of KMeans Clusters")
        plt.legend(title="Cluster")
        plt.grid(True)
        plt.savefig(self.output_path / "figures" / f"{output_file_name}.png")
        plt.close()

        unique_values, counts = np.unique(data_cluster, return_counts=True)
        cluster_counts = dict(zip(unique_values, counts))
        logger.info(f"Cluster sizes: {cluster_counts}")

    def plot_radar_chart(self, data: pd.DataFrame, data_cluster: np.ndarray) -> None:
        """Plot radar chart of cluster feature means."""
        data_temp = data.copy()
        data_temp["_cluster"] = data_cluster
        cluster_means = data_temp.groupby("_cluster").mean().T
        categories = cluster_means.index
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(20, 16), subplot_kw=dict(polar=True))
        for cluster, values in cluster_means.items():
            vals = values.tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=f"Cluster {cluster}", linewidth=2)
            ax.fill(angles, vals, alpha=0.2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        plt.title("Cluster Feature Means (Standardized) - Radar Chart")
        plt.legend(loc="upper right")
        plt.subplots_adjust(left=0.1, bottom=0.1)
        plt.savefig(self.output_path / "figures" / "Radar_Clusters.png")
        plt.close()

    def save_cluster_data(
        self,
        data_original: pd.DataFrame,
        data_cluster: pd.DataFrame,
        merge_columns: List[str],
        cluster_column: str = "Cluster",
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        """Merge cluster index with original data and save data for each cluster."""
        data_merged = data_original.merge(data_cluster, on=merge_columns, how="inner")

        for cluster, group_df in data_merged.groupby(cluster_column):
            file_name = f"original_data_cluster_{cluster}.csv"
            group_df.drop(columns=[cluster_column]).to_csv(self.output_path / "output" / file_name, index=False)

            if feature_columns is not None:
                stats = group_df[feature_columns].describe().T
                logger.info(f"Cluster {cluster} feature statistics:")
                logger.info(stats[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    def apply_scale_features(self, df_features: pd.DataFrame, df_scale: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to features using saved scaling parameters."""
        df_scaled = df_features.copy()

        for col in df_scale.columns:
            mean = df_scale.loc["Mean", col]
            std = df_scale.loc["Std", col]
            if std != 0:
                df_scaled[col] = (df_features[col] - mean) / std

        return df_scaled

    def load_trained_model(self) -> KMeans:
        """Load the trained K-means model."""
        model_path = self.output_path / "models" / "kmeans_model.pkl"
        return joblib.load(model_path)

    def predict_clusters(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Predict clusters for new data using trained model."""
        model = self.load_trained_model()
        return model.predict(data[features])

    def save_pipeline_model(self, pipeline: Pipeline, file_path: str) -> None:
        """Save a complete pipeline model to disk."""
        # Save as pickle
        joblib.dump(pipeline, file_path)

        # Save as ONNX
        onnx_path = file_path.replace(".pkl", ".onnx")
        n_features = pipeline.named_steps["kmeans"].n_features_in_
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Pipeline model saved to: {file_path} and {onnx_path}")

    def create_preprocessing_pipeline(
        self, data: pd.DataFrame, transform_columns: Optional[List[str]] = None
    ) -> Pipeline:
        """
        Create a preprocessing pipeline with optional power transformation (no clustering).

        Args:
            data: DataFrame with features for preprocessing
            transform_columns: List of column names to apply power transformation to

        Returns:
            Fitted preprocessing pipeline
        """
        pipeline_steps = []

        if transform_columns is not None and len(transform_columns) > 1:
            power_columns = [i for i, f in enumerate(data.columns) if f in transform_columns]
            logger.info(f"Adding power transformation for columns: {transform_columns}")
            logger.info(f"Data columns: {data.columns.to_list()}")
            logger.info(f"Transform column indices: {power_columns}")

            preprocessor = ColumnTransformer(
                transformers=[("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True), power_columns)],
                remainder="passthrough",
            )
            pipeline_steps.append(("power_transform", preprocessor))

        pipeline_steps.append(("scaler", RobustScaler()))

        pipeline = Pipeline(pipeline_steps)
        pipeline.fit(data)

        logger.info(f"Created preprocessing pipeline with {len(pipeline_steps)} steps")
        logger.info(f"Pipeline steps: {[step[0] for step in pipeline_steps]}")

        return pipeline

    def create_flexible_clustering_pipeline(
        self, data: pd.DataFrame, transform_columns: Optional[List[str]] = None, n_clusters: int = 5
    ) -> Pipeline:
        """
        Create a flexible clustering pipeline where n_clusters can be changed after creation.

        Args:
            data: DataFrame with features for clustering
            transform_columns: List of column names to apply power transformation to
            n_clusters: Initial number of clusters (can be changed later)

        Returns:
            Fitted flexible clustering pipeline
        """
        pipeline_steps = []

        if transform_columns is not None and len(transform_columns) > 1:
            power_columns = [i for i, f in enumerate(data.columns) if f in transform_columns]
            logger.info(f"Adding power transformation for columns: {transform_columns}")
            logger.info(f"Data columns: {data.columns.to_list()}")
            logger.info(f"Transform column indices: {power_columns}")

            preprocessor = ColumnTransformer(
                transformers=[("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True), power_columns)],
                remainder="passthrough",
            )
            pipeline_steps.append(("power_transform", preprocessor))

        pipeline_steps.extend([("scaler", RobustScaler()), ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))])

        pipeline = Pipeline(pipeline_steps)
        pipeline.fit(data)

        logger.info(f"Created flexible clustering pipeline with {len(pipeline_steps)} steps")
        logger.info(f"Pipeline steps: {[step[0] for step in pipeline_steps]}")
        logger.info(f"Initial n_clusters: {n_clusters}")

        return pipeline

    def set_n_clusters(self, pipeline: Pipeline, n_clusters: int) -> Pipeline:
        """
        Set the number of clusters for a flexible clustering pipeline.

        Args:
            pipeline: The flexible clustering pipeline
            n_clusters: New number of clusters

        Returns:
            Updated pipeline with new n_clusters
        """
        if "kmeans" not in pipeline.named_steps:
            raise ValueError("Pipeline does not contain a K-means step")

        # Update the n_clusters parameter
        pipeline.named_steps["kmeans"].set_params(n_clusters=n_clusters)

        logger.info(f"Updated pipeline n_clusters to: {n_clusters}")
        return pipeline

    def fit_predict_with_n_clusters(self, pipeline: Pipeline, data: pd.DataFrame, n_clusters: int) -> np.ndarray:
        """
        Fit and predict with a specific number of clusters using a flexible pipeline.

        Args:
            pipeline: The flexible clustering pipeline
            data: Data to fit and predict on
            n_clusters: Number of clusters to use

        Returns:
            Cluster labels
        """
        # Set the number of clusters
        pipeline = self.set_n_clusters(pipeline, n_clusters)

        # Fit and predict
        cluster_labels = pipeline.fit_predict(data)

        logger.info(f"Fitted and predicted with n_clusters={n_clusters}")
        return cluster_labels

    def create_clustering_pipeline(
        self, data: pd.DataFrame, transform_columns: Optional[List[str]] = None, n_clusters: int = 5
    ) -> Tuple[Pipeline, np.ndarray]:
        """
        Create and fit a clustering pipeline with optional power transformation.

        Args:
            data: DataFrame with features for clustering
            transform_columns: List of column names to apply power transformation to
            n_clusters: Number of clusters for K-means

        Returns:
            Tuple of (fitted_pipeline, cluster_labels)
        """
        pipeline_steps = []

        if transform_columns is not None and len(transform_columns) > 1:
            power_columns = [i for i, f in enumerate(data.columns) if f in transform_columns]
            logger.info(f"Adding power transformation for columns: {transform_columns}")
            logger.info(f"Data columns: {data.columns.to_list()}")
            logger.info(f"Transform column indices: {power_columns}")

            preprocessor = ColumnTransformer(
                transformers=[("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True), power_columns)],
                remainder="passthrough",
            )
            pipeline_steps.append(("power_transform", preprocessor))

        pipeline_steps.extend([("scaler", RobustScaler()), ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))])

        pipeline = Pipeline(pipeline_steps)
        data_cluster = pipeline.fit_predict(data)

        logger.info(f"Created clustering pipeline with {len(pipeline_steps)} steps")
        logger.info(f"Pipeline steps: {[step[0] for step in pipeline_steps]}")

        return pipeline, data_cluster


def calculate_inertia(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the inertia (within-cluster sum of squared distances) for a given clustering.
    This is useful when we use some pacakge that does not report inertia directly (i.e. kmean_equal)

    Args:
        X (np.ndarray): The feature data, with shape (n_samples, n_features).
        y (np.ndarray): The cluster labels for each data point, with shape (n_samples,).

    Returns:
        float: The total inertia score.
    """

    unique_clusters = np.unique(y)
    total_inertia = 0.0

    for cluster_id in unique_clusters:
        cluster_points = x[y == cluster_id]

        if len(cluster_points) > 0:
            cluster_centroid = np.mean(cluster_points, axis=0)
            squared_distances = np.sum((cluster_points - cluster_centroid) ** 2, axis=1)
            total_inertia += np.sum(squared_distances)

    return total_inertia


def calculate_silhouette_score(x: Union[np.ndarray, pd.DataFrame], labels: np.ndarray) -> float:

    if len(np.unique(labels)) < 2:
        return float("nan")

    try:
        with parallel_backend("loky"):
            # a small sample size may lead to small cluster totally omitted!
            score = silhouette_score(x, labels, sample_size=min(5000, x.shape[0]), random_state=42)
    except ValueError:
        score = float("nan")
    return score
