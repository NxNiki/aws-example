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
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
        self.config = self._load_config(config_path)
        self._setup_directories()

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

    def read_cluster_config(self) -> Dict[str, Any]:
        """Read and return the cluster configuration."""
        return self.config

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get feature names from configuration.

        Returns:
            Tuple of (key_features, normal_features, skewed_features)
        """
        config = self.read_cluster_config()

        # Get features from configuration
        key_features = config["features"]["key_features"]
        normal_features = config["features"]["normal_features"]
        skewed_features = config["features"]["skewed_features"]

        return key_features, normal_features, skewed_features

    def get_data_loading_config(self) -> Dict[str, Any]:
        """Get data loading configuration."""
        config = self.read_cluster_config()
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

        files = list_s3_files(data_config["input_bucket"], data_config["input_prefix"], pattern)

        data = read_files(
            files,
            local_cache_path=output_file,
            columns=columns,
            reload=False,
        )

        # Apply currency filter if specified
        if data_config.get("currency_filter"):
            data = data[data["currency"] == data_config["currency_filter"]]
            logger.info(f"Filtered data to {data_config['currency_filter']} currency only")

        return data

    def load_enriched_data(
        self, output_file: str, pattern: Optional[str] = None, columns: Optional[list] = None
    ) -> pd.DataFrame:
        """Load enriched data from S3 using configuration."""

        data_config = self.get_data_loading_config()

        if pattern is None:
            pattern = data_config["file_patterns"]["enriched_data"]

        if columns is None:
            columns = data_config["columns_to_read"]

        files = list_s3_files(data_config["input_bucket"], data_config["input_prefix"], pattern)

        data = read_files(
            files,
            local_cache_path=output_file,
            columns=columns,
            reload=False,
        )

        # Apply currency filter if specified
        if data_config.get("currency_filter"):
            data = data[data["currency"] == data_config["currency_filter"]]
            logger.info(f"Filtered data to {data_config['currency_filter']} currency only")

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
        self, data: pd.DataFrame, features: List[str], n_features: Optional[Union[List[int], int]] = None
    ) -> Dict[Any, Dict[Any, Any]]:
        """Run elbow method to determine optimal number of clusters."""

        k_range = self.config["elbow_method"]["k_range"]
        scaler = StandardScaler()
        cluster_indices = {}

        for x, n in column_iterator(data, features, n_features):
            x, filter_index = remove_outliers(x)
            x = scaler.fit_transform(x)
            cluster_indices_by_k = {}
            inertia = []
            silhouette_scores = []
            cluster_sizes = []

            for k in k_range:
                model = KMeans(
                    n_clusters=k,
                    random_state=self.config["kmeans"]["random_state"],
                    n_init=self.config["kmeans"]["n_init"],
                )
                model.fit(x)
                labels = model.labels_
                inertia.append(model.inertia_)

                cluster_index = np.full(len(filter_index), np.nan)
                cluster_index[filter_index] = labels
                cluster_indices_by_k[k] = cluster_index
                silhouette_scores.append(calculate_silhouette_score(x, labels))
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
        """Normalize features to zero mean and unit variance."""
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

    def run_cluster_analysis(self, data: pd.DataFrame, n_clusters: int) -> np.ndarray:
        """Run K-means clustering analysis."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config["kmeans"]["random_state"],
            n_init=self.config["kmeans"]["n_init"],
        )
        data_cluster = kmeans.fit_predict(data)

        # Save cluster centers
        centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
        logger.info(f"Cluster centers:\n{centroids_df}")
        centroids_df.to_csv(
            self.output_path / "models" / f"cluster_centers_standardized_{len(data.columns)}.csv", index=False
        )

        joblib.dump(kmeans, self.output_path / "models" / "kmeans_model.pkl")
        initial_type = [("float_input", FloatTensorType([None, data.shape[1]]))]
        onnx_model = convert_sklearn(kmeans, initial_types=initial_type)
        with open(self.output_path / "models" / "kmeans_model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"K-means model saved to: {self.output_path / 'models'}")
        return data_cluster

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
