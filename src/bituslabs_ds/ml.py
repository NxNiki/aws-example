"""
Machine Learning utilities and clustering analysis classes.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

from bituslabs_ds.config import LOCAL_ROOT
from bituslabs_ds.s3_utils import list_s3_files, read_files
from bituslabs_ds.utils import column_iterator, keep_numeric_columns, remove_outliers, save_list

logger = logging.getLogger(__name__)


class ClusterAnalysisPipeline:
    """
    Comprehensive clustering analysis class that consolidates all clustering functionality.
    Supports multiple projects with separate configuration files.
    """

    def __init__(self, config_file: str):
        """
        Initialize clustering analysis for a specific project.

        Args:
            project_name: Name of the project (e.g., 'wucaishen', 'deepdive')
            config_path: Optional path to config file. If None, uses default location.
        """

        self._config_file = config_file
        self._config = self._load_config(config_file)
        self.project_name = self._config["project_name"]
        self._setup_directories()

        cluster_data_config = self._config["data_loader"]["cluster_data"]
        self._cluster_data_files = list_s3_files(
            cluster_data_config["bucket"], cluster_data_config["prefix"], cluster_data_config["pattern"]
        )

        attach_data_config = self._config["data_loader"]["cluster_data"]
        self._attach_data_files = list_s3_files(
            attach_data_config["bucket"], attach_data_config["prefix"], attach_data_config["pattern"]
        )

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {config_file}:")
        logger.info(f"config: \n {config}")
        return config

    def _setup_directories(self) -> None:
        """Create necessary directories for the project."""
        self.work_dir = LOCAL_ROOT / "jobs" / "sagemaker" / self._config["work_dir"]
        self.output_path = self.work_dir / self._config["project_name"]

        # Create directories
        for dir_name in ["output", "features", "figures", "models"]:
            os.makedirs(self.output_path / dir_name, exist_ok=True)

    @property
    def key_features(self):
        return self._config["features"]["key_features"]

    @property
    def normal_features(self):
        return self._config["features"]["normal_features"]

    @property
    def skewed_features(self):
        return self._config["features"]["skewed_features"]

    @property
    def outlier_threshold(self):
        val = self.config["data_loader"]["cluster_data"].get("outlier_threshold", np.nan)
        return val

    @property
    def run_elbow_method(self):
        return self._config["pipeline"]["elbow_method"]

    @property
    def run_cluster_analysis(self):
        return self._config["pipeline"]["cluster_analysis"]

    @property
    def run_fit_cluster_model(self):
        return self._config["pipeline"]["fit_cluster_model"]

    @property
    def run_attach_cluster_label(self):
        return self._config["pipeline"]["attach_cluster_label"]

    @property
    def onnx_opset_version(self):
        return self._config.get("output", {}).get("onnx", {}).get("onnx_opset_version", 19)

    def get_transform_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[int]]:

        transform_columns = [f for i, f in enumerate(data.columns) if f in self.skewed_features]
        transform_columns_index = [i for i, f in enumerate(data.columns) if f in self.skewed_features]
        logger.info(f"Adding power transformation for columns: {transform_columns}")
        logger.info(f"Data columns: {data.columns.to_list()}")
        logger.info(f"Transform column indices: {transform_columns_index}")

        return transform_columns, transform_columns_index

    def load_data(
        self,
        data_label: str,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Load data from S3 using configuration.

        Args:
            data_label: Type of data to load ('grouped_data', 'enriched_data', etc.)
            output_file: Local file path for caching
            pattern: Optional regex pattern to override config
            columns: Optional columns to read (overrides config)

        Returns:
            Loaded and filtered DataFrame
        """

        if data_label == "cluster_data":
            files = self._cluster_data_files
            output_file = self._config["data_loader"]["cluster_data"]["local_cache"]
            columns = self.key_features + self.normal_features + self.skewed_features
            row_filters = self._config["data_loader"]["cluster_data"]["row_filters"]
        elif data_label == "attach_data":
            files = self._attach_data_files
            output_file = self._config["data_loader"]["attach_data"]["local_cache"]
            columns = self._config["data_loader"]["attach_data"]["columns_to_read"]
            row_filters = self._config["data_loader"]["attach_data"]["row_filters"]

        data = read_files(
            files,
            local_cache_path=f"{self.work_dir}/{output_file}",
            columns=columns,
            reload=reload,
        )

        if data_label == "cluster_data":
            data = remove_outliers(data)

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

    def df_smart_feature_selection(
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
            threshold = self._config["feature_selection"]["correlation_threshold"]
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
            threshold = self._config["feature_selection"]["variance_threshold"]

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

        # Save feature importance
        features_ordered_by_importance = [features[i] for i in np.argsort(importance)[::-1]]
        save_list(features_ordered_by_importance, str(self.output_path / "features" / "important_features.json"))
        self._plot_feature_importance(feature_importance)

        return features_ordered_by_importance

    def _plot_feature_importance(self, feature_importance: pd.DataFrame):

        plt.figure(figsize=(18, 10))
        colors = sns.color_palette("viridis", len(feature_importance))
        sns.barplot(x="Importance", y="Feature", hue="Feature", data=feature_importance, palette=colors, legend=False)
        plt.title("Feature Importance Rank", fontsize=16)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.savefig(self.output_path / "figures" / "Feature Importance Rank.png")
        plt.close()

    def elbow_method(
        self,
        data: pd.DataFrame,
        features_ordered_by_importance: List[str],
    ) -> Dict[Any, Dict[Any, Any]]:
        """Run elbow method to determine optimal number of clusters."""

        k_range = self._config["elbow_method"]["k_range"]
        n_features = self._config["elbow_method"]["top_features"]
        cluster_indices = {}

        for df_x, n in column_iterator(data, features_ordered_by_importance, n_features):
            cluster_indices_by_k = {}
            inertia = []
            silhouette_scores = []
            cluster_sizes = []

            for k in k_range:
                _, transform_columns_index = self.get_transform_columns(df_x)
                cluster_pipeline = self.create_clustering_pipeline(
                    n_clusters=k, transform_columns_index=transform_columns_index
                )
                labels = cluster_pipeline.fit_predict(df_x)
                inertia.append(cluster_pipeline.named_steps["cluster"].inertia_)
                cluster_indices_by_k[k] = labels
                x_transformed = cluster_pipeline[:-1].transform(df_x)
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

    def cluster_analysis(
        self, data: pd.DataFrame, features_ordered_by_importance: List[str]
    ) -> Tuple[np.ndarray, Pipeline]:
        """Run K-means clustering analysis using pipeline approach."""

        n_clusters = self._config["cluster_analysis"]["n_clusters"]
        top_features = self._config["cluster_analysis"]["top_features"]
        data = data[features_ordered_by_importance[:top_features]].copy()

        _, transform_columns_index = self.get_transform_columns(data)
        pipeline = self.create_clustering_pipeline(
            n_clusters=n_clusters,
            transform_columns_index=transform_columns_index,
        )
        cluster_label = pipeline.fit_predict(data)

        # Save cluster centers
        centroids_df = pd.DataFrame(pipeline.named_steps["cluster"].cluster_centers_, columns=data.columns)
        logger.info(f"Cluster centers:\n{centroids_df}")
        centroids_df.to_csv(
            self.output_path / "models" / f"cluster_centers_standardized_{len(data.columns)}.csv", index=False
        )
        # Save pipeline model
        n_features = data.shape[1]
        model_name = f"kmeans_model_top{n_features}_features"
        joblib.dump(pipeline, self.output_path / "models" / f"{model_name}.pkl")

        # Save ONNX model with version from config
        onnx_opset_version = self.onnx_opset_version
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        logger.info(f"Using ONNX opset version: {onnx_opset_version}")
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=onnx_opset_version)
        onnx_filename = f"{model_name}_opset_{onnx_opset_version}.onnx"
        with open(self.output_path / "models" / onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"K-means pipeline model saved to: {self.output_path / 'models'}")
        logger.info(f"ONNX model saved as: {onnx_filename}")
        return cluster_label, pipeline

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
        data = data.copy()
        data["_cluster"] = data_cluster
        cluster_means = data.groupby("_cluster").mean().T
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
        self,
        n_clusters: int = 5,
        transform_columns_index: Optional[List[int]] = None,
    ) -> Pipeline:
        """
        Create and fit a clustering pipeline with optional power transformation.

        Args:
            transform_columns: List of column names to apply power transformation to
            n_clusters: Number of clusters for K-means

        Returns:
            Tuple of (fitted_pipeline, cluster_labels)
        """
        pipeline_steps = []

        if transform_columns_index is not None and len(transform_columns_index) > 1:

            preprocessor = ColumnTransformer(
                transformers=[
                    ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True), transform_columns_index)
                ],
                remainder="passthrough",
            )
            pipeline_steps.append(("power_transform", preprocessor))

        pipeline_steps.extend([("scaler", RobustScaler()), ("cluster", KMeans(n_clusters=n_clusters, random_state=42))])
        pipeline = Pipeline(pipeline_steps)

        logger.info(f"Created clustering pipeline with {len(pipeline_steps)} steps")
        logger.info(f"Pipeline steps: {[step[0] for step in pipeline_steps]}")
        logger.info(f"n_clusters: {n_clusters}")

        return pipeline


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
