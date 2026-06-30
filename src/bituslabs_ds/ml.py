"""
Machine Learning utilities and clustering analysis classes.
"""

import copy
import gc
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from botocore.exceptions import ClientError
from joblib import parallel_backend
from scipy.stats import skew
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances_argmin, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

from bituslabs_ds.config import DEFAULT_MAX_JOBS, LOCAL_ROOT
from bituslabs_ds.s3_utils import (
    apply_row_filters,
    get_s3_client,
    list_s3_files,
    parse_s3_path,
    read_files,
    read_local_cache,
    save_local_cache,
)
from bituslabs_ds.utils import (
    clip_outliers,
    column_iterator,
    convert_numpy_types,
    df_power_transform,
    keep_numeric_columns,
    load_config,
    remove_outliers,
    save_list,
)

logger = logging.getLogger(__name__)


class SubsampledAgglomerativeClustering(ClusterMixin):
    """Memory-bounded hierarchical clustering via random subsampling.

    sklearn's :class:`AgglomerativeClustering` allocates an O(n^2) linkage matrix; at
    n=70k that's ~40 GB. This wrapper fits the underlying model on at most
    ``sample_size`` random rows, computes per-cluster centroids (means) of the sample
    in the input feature space, then assigns every input row to its nearest centroid.
    Exposes ``predict`` so it slots into a sklearn ``Pipeline`` for downstream label
    attachment on new data.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        linkage: str = "ward",
        sample_size: int = 10000,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs: Any):
        self.fit_predict(X, y=y, **kwargs)
        return self

    def fit_predict(self, X, y=None, **kwargs: Any):
        X_arr = np.asarray(X)
        n = X_arr.shape[0]
        if n > self.sample_size:
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(n, self.sample_size, replace=False)
            X_sample = X_arr[sample_idx]
            logger.info(f"hierarchical: subsampling {self.sample_size} of {n} rows for AgglomerativeClustering fit")
        else:
            X_sample = X_arr

        agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        sample_labels = agg.fit_predict(X_sample)

        self.cluster_centers_ = np.vstack([X_sample[sample_labels == k].mean(axis=0) for k in range(self.n_clusters)])
        self.labels_ = pairwise_distances_argmin(X_arr, self.cluster_centers_)
        return self.labels_

    def predict(self, X):
        return pairwise_distances_argmin(np.asarray(X), self.cluster_centers_)


class ClusterAnalysisPipeline:
    """
    Comprehensive clustering analysis class that consolidates all clustering functionality.
    Supports multiple projects with separate configuration files.
    """

    def __init__(self, config_file: str, active_group: Optional[str] = None, run_id: Optional[str] = None):
        """
        Initialize clustering analysis for a specific project.

        Args:
            config_file: Path to the project YAML config.
            active_group: When the config declares a ``groups:`` mapping (one data
                source per ai_group, e.g. a ``train`` group to fit and an
                ``inference`` group to score),
                this selects which group's data_loader / pipeline / output_tag is
                active for this run. Ignored by legacy flat configs. Falls back to
                the config's ``active_group`` field when not passed on the CLI.
            run_id: Name of the per-run output folder under ``work_dir`` (holds
                ``<cluster_model>/{figures,models,output}`` and the shared
                ``cluster_labels.parquet``). Defaults to a timestamped, bin-size
                tagged id (e.g. ``result_2026-06-17_10-18_binsize50``) so successive
                runs don't overwrite each other and the bin size is visible. The
                two-phase ``inference`` run MUST reuse the ``train`` run's run_id
                (pass ``--run-id``) to find its model / feature_order / clip_bounds.
        """

        self._config_file = config_file
        self._config = load_config(config_file)
        self.project_name = self._config["project_name"]
        self.valid_sample_file = ""

        self._active_group = active_group or self._config.get("active_group")
        self._data_loader = self._resolve_data_loader()
        self._pipeline_flags = self._resolve_pipeline_flags()
        self._output_tag = self._resolve_output_tag()

        # Default run_id is timestamped and bin-size-tagged (so runs don't overwrite and
        # the bin size is visible in the folder); needs bin_size, hence after data_loader.
        self.run_id = (
            run_id or self._config.get("run_id") or f"result_{datetime.now():%Y-%m-%d_%H-%M}_binsize{self.bin_size}"
        )

        self._setup_directories()

        cluster_data_config = self._data_loader["cluster_data"]
        self._cluster_data_files = list_s3_files(
            cluster_data_config["bucket"], cluster_data_config["prefix"], cluster_data_config["pattern"]
        )

        attach_data_config = self._data_loader["attach_data"]
        self._attach_data_files = list_s3_files(
            attach_data_config["bucket"], attach_data_config["prefix"], attach_data_config["pattern"]
        )

    def _resolve_data_loader(self) -> Dict[str, Any]:
        """Resolve the active data_loader, supporting per-group overrides.

        Layout A (legacy, flat): ``data_loader`` is used as-is; ``groups`` absent.
        Layout B (single-config two-phase): ``data_loader`` holds the shared schema
        (bucket, pattern, data_types, columns_to_read, bin_size, merge_on, outlier
        thresholds, ...) and ``groups.<name>`` overrides only the per-group keys
        (prefix, row_filters, local_cache) inside ``cluster_data`` / ``attach_data``.
        The selected group is deep-merged over the shared block so the rest of the
        pipeline reads one flat data_loader regardless of layout.
        """
        base = self._config["data_loader"]
        groups = self._config.get("groups")
        if not groups:
            return base
        if self._active_group is None:
            raise ValueError(
                f"config declares groups {sorted(groups)} but no active_group was "
                "selected (pass --group on the CLI or set active_group in the config)."
            )
        if self._active_group not in groups:
            raise ValueError(f"active_group {self._active_group!r} not in groups {sorted(groups)}.")
        merged = copy.deepcopy(base)
        override = groups[self._active_group]
        for section in ("cluster_data", "attach_data"):
            if section in override:
                merged.setdefault(section, {}).update(override[section])
        return merged

    def _resolve_pipeline_flags(self) -> Dict[str, Any]:
        """Pipeline switches, optionally keyed by group (``pipeline.<group>``)."""
        pipeline = self._config["pipeline"]
        if self._active_group is not None and self._active_group in pipeline:
            return pipeline[self._active_group]
        return pipeline

    def _resolve_output_tag(self) -> str:
        """Filename tag that keeps per-group attach output from colliding.

        Sourced from ``groups.<name>.output_tag``; empty for legacy flat configs so
        existing single-group filenames (``enriched_data_cluster_{k}.parquet``) are
        preserved.
        """
        groups = self._config.get("groups")
        if groups and self._active_group in groups:
            return str(groups[self._active_group].get("output_tag", ""))
        return ""

    def _setup_directories(self) -> None:
        """Create necessary directories for the run (under work_dir/<run_id>/<model>)."""
        self.work_dir = LOCAL_ROOT / "jobs" / self._config["work_dir"]
        self.output_path = self.work_dir / self.run_id / self.cluster_model
        logger.info(f"run_id={self.run_id!r}; output_path={self.output_path}")

        # Create directories
        for dir_name in ["output", "features", "figures", "models"]:
            os.makedirs(self.output_path / dir_name, exist_ok=True)

    @property
    def cluster_model(self):
        return self._config["cluster_analysis"].get("model", "kmeans")

    @property
    def cluster_label_column(self):
        # DBSCAN: n_clusters is meaningless (clusters emerge from density); encode eps/min_samples instead.
        if self.cluster_model == "dbscan":
            cfg = self._config["cluster_analysis"]
            eps = cfg.get("dbscan_eps", 0.5)
            min_samples = cfg.get("dbscan_min_samples", 5)
            return f"dbscan-n_features_{self.n_top_features}-eps_{eps}-ms_{min_samples}"
        return f"{self.cluster_model}-n_features_{self.n_top_features}-k_{self.n_clusters}"

    @property
    def cluster_labels_file_path(self):
        """Shared parquet across all (model, n_features, k) runs in this run_id; one column per run."""
        return self.work_dir / self.run_id / "cluster_labels.parquet"

    @property
    def key_features(self):
        return self._config["features"]["key_features"]

    @property
    def merge_features(self):
        return self._data_loader["merge_on"]

    @property
    def normal_features(self):
        return self._config["features"]["normal_features"]

    @property
    def skewed_features(self):
        return self._config["features"]["skewed_features"]

    @property
    def n_top_features(self):
        return self._config["cluster_analysis"]["top_features"]

    @property
    def bin_size(self):
        return self._data_loader["attach_data"]["bin_size"]

    @property
    def n_clusters(self):
        return self._config["cluster_analysis"]["n_clusters"]

    @property
    def outlier_threshold(self):
        val = self._data_loader["cluster_data"].get("outlier_threshold", np.nan)
        return val

    @property
    def clip_threshold(self):
        val = self._data_loader["cluster_data"].get("clip_threshold", np.nan)
        return val

    @property
    def remove_short_sessions(self) -> bool:
        return bool(self._data_loader["cluster_data"].get("remove_short_sessions", False))

    @property
    def session_count_column(self) -> str:
        return self._data_loader["cluster_data"].get("session_count_column", "bet_rounds")

    @property
    def cluster_stats_columns(self):
        return self._data_loader["attach_data"]["cluster_stats_columns"]

    @property
    def output_tag(self) -> str:
        return self._output_tag

    @property
    def run_elbow_method(self):
        return self._pipeline_flags.get("elbow_method", False)

    @property
    def run_cluster_analysis(self):
        return self._pipeline_flags.get("cluster_analysis", False)

    @property
    def pretrained_model_dir(self) -> Optional[str]:
        """S3 URI of a prior run folder to apply (no retraining).

        When set (``cluster_analysis.pretrained_model_dir``), an apply-only run fetches
        the trained pipeline + its ``feature_order.json`` / ``clip_bounds.json`` /
        importance list from ``<dir>/models`` and ``<dir>/features`` into this run's
        folder, so scoring reuses the exact training-time preprocessing and the new
        run folder is self-contained / reproducible.
        """
        return self._config["cluster_analysis"].get("pretrained_model_dir")

    @property
    def pickle_model_name(self):
        return f"kmeans_model_top{self.n_top_features}_features_k_{self.n_clusters}.pkl"

    @property
    def onnx_model_name(self):
        return (
            f"kmeans_model_top{self.n_top_features}_features_k_{self.n_clusters}_opset_{self.onnx_opset_version}.onnx"
        )

    @property
    def run_apply_model(self):
        return self._pipeline_flags.get("apply_model", False)

    @property
    def run_attach_cluster_label(self):
        return self._pipeline_flags.get("attach_cluster_label", False)

    @property
    def run_get_cluster_stats(self):
        return self._pipeline_flags.get("get_cluster_stats", False)

    @property
    def run_upload_result_to_s3(self):
        return self._pipeline_flags.get("upload_result_to_s3", False)

    @property
    def onnx_opset_version(self):
        return self._config.get("output", {}).get("onnx", {}).get("onnx_opset_version", 19)

    @property
    def s3_prefix(self):
        return self._config.get("output", {}).get("prefix", "cluster_analysis_result")

    def get_transform_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[int]]:

        transform_columns = [f for f in data.columns if f in self.skewed_features]
        transform_columns_index = [i for i, f in enumerate(data.columns) if f in self.skewed_features]

        logger.info(
            f"Power transform columns: \n{transform_columns}\nData columns: \n{data.columns.to_list()}\nTransform column indices: \n{transform_columns_index}"
        )

        return transform_columns, transform_columns_index

    @staticmethod
    def get_scaler():
        return StandardScaler()

    def get_data_types(self, data_source: str) -> Dict[str, str]:
        return self._data_loader[data_source].get("data_types", {})

    def _processed_cache_path(self, base_name: str) -> str:
        """work_dir path for a processed cache, tagged with the active bin size.

        Processed caches (grouped/enriched after complete-bin filtering) have contents
        that depend on bin_size, so the name carries ``_binsize{N}``. The raw cache
        (unfiltered S3 pull, bin-independent) keeps the plain base name.
        """
        for ext in (".parquet", ".csv"):
            if base_name.endswith(ext):
                return f"{self.work_dir}/{base_name[: -len(ext)]}_binsize{self.bin_size}{ext}"
        raise ValueError(f"unsupported cache file format: {base_name}")

    def load_raw_data(self, data_label, reload: bool = False) -> pd.DataFrame:
        """
        load preprocessed data from s3.
        """

        if data_label == "attach_data":
            files = self._attach_data_files
            output_file = self._data_loader["attach_data"]["local_cache"]
            columns = self._data_loader["attach_data"]["columns_to_read"]
            row_filters = self._data_loader["attach_data"]["row_filters"]
            data_types = self.get_data_types("attach_data")
        elif data_label == "cluster_data":
            files = self._cluster_data_files
            output_file = self._data_loader["cluster_data"]["local_cache"]
            columns = self.key_features + self.normal_features + self.skewed_features
            # The session-count column (e.g. bet_rounds) is not a feature; read it so
            # load_cluster_data can drop incomplete bins.
            if self.remove_short_sessions and self.session_count_column not in columns:
                columns = columns + [self.session_count_column]
            row_filters = self._data_loader["cluster_data"]["row_filters"]
            data_types = self.get_data_types("cluster_data")
        else:
            raise ValueError(f"unsupported data_label: {data_label!r}")

        if output_file.endswith(".csv"):
            raw_output_file = output_file.replace(".csv", "_raw.csv")
        elif output_file.endswith(".parquet"):
            raw_output_file = output_file.replace(".parquet", "_raw.parquet")
        else:
            raise ValueError(f"unsupported output file format: {output_file}")

        data = read_files(
            files,
            local_cache_path=f"{self.work_dir}/{raw_output_file}",
            columns=columns,
            row_filters=row_filters,
            data_types=data_types,
            reload=reload,
            lazy_load=False,
            return_as_list=False,
        )

        return cast(pd.DataFrame, data)

    def load_attach_data(self, reload: bool = False) -> pd.DataFrame:
        """Read attach data (enriched data) and remove samples with short sessions.
        Save the result and valid samples (used to select cluster data) to a parquet file.
        """

        output_file = self._data_loader["attach_data"]["local_cache"]
        local_cache_path = self._processed_cache_path(output_file)
        row_filters = self._data_loader["attach_data"].get("row_filters")

        if not reload and os.path.exists(local_cache_path):
            logger.info(f"read local attach data: {local_cache_path}")
            data = read_local_cache(
                local_cache_path,
                data_types=self.get_data_types("attach_data"),
                lazy_load=False,
            )
        else:
            data = self.load_raw_data(data_label="attach_data")
            merge_cols: List[str] = (
                [self.merge_features] if isinstance(self.merge_features, str) else list(self.merge_features)
            )
            # Enriched is now bin-independent: it carries `session_bet_index` (1-based
            # row within session) instead of `agg_group`. Derive the per-bin agg_group
            # the same way the grouped SQL does (ROUND == integer division here), so the
            # merge_on join to cluster labels (which key on agg_group) lines up.
            if "agg_group" in merge_cols and "agg_group" not in data.columns:
                data["agg_group"] = (data["session_bet_index"] - 1) // self.bin_size
            # Legacy schema (wucaishen/deepdive) keys on a per-day column derived from `billtime`.
            # ss03's merge_on doesn't include merge_date and its enriched data has no `billtime`
            # column, so only build it when it's actually a merge key.
            if "merge_date" in merge_cols:
                data["merge_date"] = pd.to_datetime(data["billtime"]).dt.strftime("%Y_%m_%d")
            group_counts = data[merge_cols].value_counts(sort=False).reset_index(name="count")
            valid_groups = group_counts.loc[group_counts["count"] == self.bin_size, merge_cols]
            data = data.merge(valid_groups, on=merge_cols, how="inner")
            save_local_cache(data, local_cache_path)
        data = apply_row_filters(cast(pd.DataFrame, data), row_filters)
        return cast(pd.DataFrame, data)

    def load_cluster_data(
        self,
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

        output_file = self._data_loader["cluster_data"]["local_cache"]
        local_cache_path = self._processed_cache_path(output_file)
        row_filters = self._data_loader["cluster_data"].get("row_filters")

        if not reload and os.path.exists(local_cache_path):
            logger.info(f"read local attach data: {local_cache_path}")
            data = read_local_cache(
                local_cache_path,
                data_types=self.get_data_types("cluster_data"),
                lazy_load=False,
            )
        else:
            data = self.load_raw_data(data_label="cluster_data")
            # Drop incomplete bins: keep only the longest sessions (count == its max);
            # shorter ones are partial sessions and would skew the per-group stats.
            if self.remove_short_sessions:
                col = self.session_count_column
                max_count = data[col].max()
                before = len(data)
                data = data[data[col] >= max_count]
                logger.info(
                    f"remove_short_sessions: dropped {before - len(data)} of {before} rows "
                    f"with {col} < {max_count}; {len(data)} rows remain"
                )
            # Drop duplicate samples: the merge_on tuple is the unique grouping grain,
            # so the same group re-emitted across overlapping partition files is a dup.
            data = cast(pd.DataFrame, data)
            merge_cols = [self.merge_features] if isinstance(self.merge_features, str) else list(self.merge_features)
            duplicated = data.duplicated(subset=merge_cols)
            if duplicated.any():
                logger.warning(f"duplicated samples found in cluster data: {duplicated.sum()} / {len(data)}")
                data = data.loc[~duplicated]
            data, _ = remove_outliers(data, self.outlier_threshold)
            save_local_cache(cast(pd.DataFrame, data), local_cache_path)

        data = apply_row_filters(cast(pd.DataFrame, data), row_filters)
        return cast(pd.DataFrame, data)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        apply preprocess steps same as in the clustering pipeline, which includes:
            power transform
            scaler
        """
        data = data.copy()
        data = df_power_transform(data, self.skewed_features)
        numeric_columns = data.select_dtypes(include="number").columns.tolist()
        scaler = self.get_scaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

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
            threshold = self._config["feature_selection"]["correlation_threshold"]
        if prefer_keywords is None:
            prefer_keywords = ["mean", "median"]

        data = keep_numeric_columns(data)
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        kept = set()

        for column in upper.columns:
            col_series = upper.loc[:, column]
            high_corr = col_series[col_series > threshold].index.tolist()
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
        if threshold is None:
            raise ValueError("variance_threshold must be set in config or passed as an argument")

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

        plt.figure(figsize=(9, 12))
        colors = sns.color_palette("viridis", len(feature_importance))
        sns.barplot(x="Importance", y="Feature", hue="Feature", data=feature_importance, palette=colors, legend=False)
        plt.title("Feature Importance Rank", fontsize=16)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout(rect=(0.1, 0, 1, 1))  # Increase left margin to show all y-tick labels
        plt.savefig(self.output_path / "figures" / "Feature Importance Rank.png")
        plt.close()

    def create_cluster_model(self, n_clusters: int, eps: Optional[float] = None) -> ClusterMixin:
        if self.cluster_model == "kmeans":
            return KMeans(
                n_clusters=n_clusters,
                random_state=self._config["cluster_analysis"]["random_state"],
                n_init=self._config["cluster_analysis"].get("n_init", 10),
            )
        elif self.cluster_model == "hierarchical":
            # AgglomerativeClustering has O(n^2) memory; we subsample to keep it bounded.
            return SubsampledAgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self._config["cluster_analysis"].get("linkage", "ward"),
                sample_size=self._config["cluster_analysis"].get("hierarchical_sample_size", 10000),
                random_state=self._config["cluster_analysis"]["random_state"],
            )
        elif self.cluster_model == "dbscan":
            # DBSCAN ignores n_clusters: cluster count emerges from density. Returns -1 for noise.
            # sklearn DBSCAN has no random_state and no predict (refit-only).
            # `eps` override is used by elbow_method when sweeping eps_range; otherwise fall back
            # to the config value.
            return DBSCAN(
                eps=eps if eps is not None else self._config["cluster_analysis"].get("dbscan_eps", 0.5),
                min_samples=self._config["cluster_analysis"].get("dbscan_min_samples", 5),
                n_jobs=-1,
            )
        else:
            logger.critical(f"{self.cluster_model} not supported, fallback to kmeans")
            return KMeans(
                n_clusters=n_clusters,
                random_state=self._config["cluster_analysis"]["random_state"],
                n_init=self._config["cluster_analysis"].get("n_init", 10),
            )

    def create_clustering_pipeline(
        self,
        n_clusters: int = 5,
        transform_columns_index: Optional[List[int]] = None,
        eps: Optional[float] = None,
    ) -> Pipeline:
        """
        Create and fit a clustering pipeline with optional power transformation.

        Args:
            transform_columns: List of column names to apply power transformation to
            n_clusters: Number of clusters for K-means / hierarchical (ignored by DBSCAN)
            eps: DBSCAN neighborhood radius override (only used when model='dbscan',
                e.g. when sweeping eps_range in elbow_method).

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

        # avoid robust scaler as it gives werid data pattern and destroys clustering analyis.
        cluster_model = self.create_cluster_model(n_clusters, eps=eps)
        pipeline_steps.extend([("scaler", self.get_scaler()), ("cluster", cluster_model)])
        pipeline = Pipeline(pipeline_steps)

        logger.info(f"Created clustering pipeline with {len(pipeline_steps)} steps")
        logger.info(f"Pipeline steps: {[step[0] for step in pipeline_steps]}")
        logger.info(f"clustering algorithm: {self.cluster_model}")
        if self.cluster_model == "dbscan":
            logger.info(
                f"dbscan eps: {eps if eps is not None else self._config['cluster_analysis'].get('dbscan_eps', 0.5)}"
            )
        else:
            logger.info(f"n_clusters: {n_clusters}")

        return pipeline

    def elbow_method(
        self,
        data: pd.DataFrame,
        features_ordered_by_importance: List[str],
    ) -> Tuple[Dict[Any, Dict[Any, Any]], List[float], List[float]]:
        """Sweep the model's primary parameter and report fit quality per value.

        For kmeans / hierarchical the swept parameter is ``n_clusters`` taken from
        ``elbow_method.k_range``. For DBSCAN it is ``eps`` taken from
        ``elbow_method.eps_range`` (``n_clusters`` is meaningless for DBSCAN, so
        sweeping it is wasted work).
        """

        n_features = self._config["elbow_method"]["top_features"]
        if self.cluster_model == "dbscan":
            param_range = self._config["elbow_method"].get("eps_range")
            if not param_range:
                raise ValueError("elbow_method.eps_range must be set in the config when model='dbscan'")
            param_label = "eps"
        else:
            param_range = self._config["elbow_method"]["k_range"]
            param_label = "k"

        cluster_indices = {}
        clipped = clip_outliers(
            cast(pd.DataFrame, data[features_ordered_by_importance]),
            self.clip_threshold[0],
            self.clip_threshold[1],
        )
        data = cast(pd.DataFrame, clipped[0])
        upper_bounds, lower_bounds = clipped[1], clipped[2]

        for df_x, n in column_iterator(data, features_ordered_by_importance, n_features):
            cluster_indices_by_param = {}
            inertia = []
            silhouette_scores = []
            cluster_sizes = []

            def fit_one_param(param):
                _, transform_columns_index = self.get_transform_columns(df_x)
                if self.cluster_model == "dbscan":
                    cluster_pipeline = self.create_clustering_pipeline(
                        n_clusters=self.n_clusters,
                        transform_columns_index=transform_columns_index,
                        eps=param,
                    )
                else:
                    cluster_pipeline = self.create_clustering_pipeline(
                        n_clusters=param, transform_columns_index=transform_columns_index
                    )
                labels = cluster_pipeline.fit_predict(df_x)
                if self.cluster_model == "kmeans":
                    inertia_val = cluster_pipeline.named_steps["cluster"].inertia_
                else:
                    inertia_val = np.nan
                x_transformed = cluster_pipeline[:-1].transform(df_x)
                if self.cluster_model == "dbscan":
                    # Silhouette is misleading when noise is treated as its own "cluster";
                    # restrict to non-noise rows so the score reflects actual cluster quality.
                    mask = labels != -1
                    if mask.sum() > 0 and len(np.unique(labels[mask])) >= 2:
                        silhouette = calculate_silhouette_score(x_transformed[mask], labels[mask])
                    else:
                        silhouette = float("nan")
                else:
                    silhouette = calculate_silhouette_score(x_transformed, labels)
                # np.bincount rejects negative labels; DBSCAN uses -1 for noise. np.unique
                # returns counts sorted by label value (noise first, then clusters).
                _, cluster_counts = np.unique(labels, return_counts=True)
                return param, labels, inertia_val, silhouette, cluster_counts

            # Hierarchical fits are memory-heavy even after subsampling; DBSCAN uses n_jobs=-1
            # internally. Run both serially to avoid CPU/memory contention.
            max_workers = 1 if self.cluster_model in ("hierarchical", "dbscan") else DEFAULT_MAX_JOBS
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_param = {executor.submit(fit_one_param, p): p for p in param_range}
                for future in as_completed(future_to_param):
                    param, labels, inertia_val, silhouette, cluster_counts = future.result()
                    results.append((param, labels, inertia_val, silhouette, cluster_counts))

            # Sort results by param to maintain plot order
            results.sort(key=lambda x: param_range.index(x[0]))
            for param, labels, inertia_val, silhouette, cluster_counts in results:
                cluster_indices_by_param[param] = labels
                inertia.append(inertia_val)
                silhouette_scores.append(silhouette)
                cluster_sizes.append(cluster_counts)

            cluster_indices[n] = cluster_indices_by_param
            self._plot_elbow_method(param_range, inertia, silhouette_scores, cluster_sizes, n, param_label)

        save_list(upper_bounds, str(self.output_path / "features" / "upper_bounds.json"))
        save_list(lower_bounds, str(self.output_path / "features" / "lower_bounds.json"))
        return cluster_indices, upper_bounds, lower_bounds

    def _plot_elbow_method(self, param_range, inertia, silhouette_scores, cluster_sizes, n_features, param_label="k"):
        """Plot elbow method results.

        ``param_label`` is "k" for kmeans/hierarchical (number of clusters) or "eps" for DBSCAN
        (neighborhood radius). The x-axis label and cluster-size column headers follow it.
        """

        title = f"Elbow Method for Optimal {param_label} n_features({n_features})"
        fig, ax1 = plt.subplots(figsize=(12, 9))

        x_label = "Number of Clusters (k)" if param_label == "k" else "DBSCAN eps"
        (line1,) = ax1.plot(param_range, inertia, marker="o", linestyle="-", label="Inertia")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Inertia")
        ax1.set_title(title, fontsize=16)

        ax2 = ax1.twinx()
        (line2,) = ax2.plot(
            param_range, silhouette_scores, marker="s", linestyle="-", color="red", label="Silhouette Score"
        )
        ax2.set_ylabel("Silhouette Score")

        lines = [line1, line2]
        labels = [str(line.get_label()) for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        # Add cluster sizes table
        self._add_cluster_sizes_table(plt, param_range, cluster_sizes, param_label)

        plt.subplots_adjust(left=0.1, bottom=0.3)
        plt.savefig(self.output_path / "figures" / f"{title}.png")
        plt.close()

    def _add_cluster_sizes_table(self, plt, param_range, cluster_sizes, param_label="k"):
        """Add cluster sizes table to elbow method plot.

        For DBSCAN the first row corresponds to noise (label ``-1``) since np.unique sorts
        labels ascending; label it "noise" for clarity.
        """
        max_clusters = max(len(sizes) for sizes in cluster_sizes)
        cluster_sizes_str = []

        for sizes in cluster_sizes:
            row = [f"{count} ({count/sum(sizes):.3f})".replace("(0.", "(.") for count in sizes]
            row += [""] * (max_clusters - len(row))
            cluster_sizes_str.append(row)

        cluster_sizes_table = list(map(list, zip(*cluster_sizes_str)))
        if param_label == "eps":
            row_labels = ["noise"] + [f"C{i}" for i in range(max_clusters - 1)]
        else:
            row_labels = [f"C{i + 1}" for i in range(max_clusters)]

        plt.table(
            cellText=cluster_sizes_table,
            rowLabels=row_labels,
            colLabels=[f"{param_label}={p}" for p in param_range],
            cellLoc="center",
            loc="bottom",
            bbox=[0.0, -0.5, 1, 0.3],
        )

    def cluster_analysis(
        self, data: pd.DataFrame, features_ordered_by_importance: List[str]
    ) -> Tuple[np.ndarray, Pipeline]:
        """Run K-means clustering analysis using pipeline approach."""

        feature_columns = features_ordered_by_importance[: self.n_top_features]
        # Persist the full importance-ordered feature list so a later apply_model run
        # (e.g. scoring the AI group with this Default-trained model) reuses the exact
        # same columns/order instead of re-running feature selection on different data.
        self._save_feature_order(features_ordered_by_importance)
        # Clip outliers to match elbow_method's preprocessing, so the cluster sizes
        # reported here match the elbow table for the same (n_features, k).
        clip_input = cast(pd.DataFrame, data[feature_columns])
        clipped, upper_bounds, lower_bounds = clip_outliers(
            clip_input,
            self.clip_threshold[0],
            self.clip_threshold[1],
        )
        # Persist the fitted clip bounds so apply_model scores another group with the
        # SAME bounds (a transform, not a re-fit on the new group's distribution).
        self._save_clip_bounds(clip_input, lower_bounds, upper_bounds)
        clustering_data = cast(pd.DataFrame, clipped)
        transform_columns, transform_columns_index = self.get_transform_columns(clustering_data)
        pipeline = self.create_clustering_pipeline(
            n_clusters=self.n_clusters,
            transform_columns_index=transform_columns_index,
        )
        cluster_label = pipeline.fit_predict(clustering_data)

        unique_values, counts = np.unique(cluster_label, return_counts=True)
        cluster_counts = dict(zip(unique_values, counts))
        logger.info(f"Cluster sizes: {cluster_counts}")

        # Save cluster centers
        if self.cluster_model == "kmeans":
            centroids_df = pd.DataFrame(
                pipeline.named_steps["cluster"].cluster_centers_,
                columns=pd.Index(feature_columns),
            )
            logger.info(f"Cluster centers:\n{centroids_df}")
            centroids_df.to_csv(
                self.output_path
                / "models"
                / f"cluster_centers_standardized_{self.n_top_features}_k_{self.n_clusters}.csv",
                index=False,
            )

        # Save cluster labels into the shared parquet; one column per <model>-n_features_<N>-k_<K>.
        new_labels = pd.DataFrame(data[self.merge_features].copy())
        new_labels[self.cluster_label_column] = cluster_label
        self._write_cluster_labels(new_labels)

        # Save pipeline model
        self.save_pipeline_model(pipeline)

        x_transformed = pipeline[:-1].transform(clustering_data)
        self.plot_pca(x_transformed, cluster_label)
        self.plot_radar_chart(
            pd.DataFrame(x_transformed, columns=pd.Index(feature_columns)),
            cluster_label,
        )

        scaler = self.get_scaler()
        scaled_data = scaler.fit_transform(data[features_ordered_by_importance[: self.n_top_features]])
        self.plot_pca(scaled_data, cluster_label, output_file_name="pca_cluster_raw_feature")
        self.plot_radar_chart(
            pd.DataFrame(scaled_data, columns=pd.Index(feature_columns)),
            cluster_label,
            output_file_name="radar_clusters_raw_feature",
        )

        return cluster_label, pipeline

    @property
    def feature_order_path(self):
        """JSON sidecar holding the importance-ordered feature list from training."""
        return self.output_path / "models" / "feature_order.json"

    def _save_feature_order(self, features_ordered_by_importance: List[str]) -> None:
        self.feature_order_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feature_order_path, "w") as f:
            json.dump(list(features_ordered_by_importance), f, indent=2)
        logger.info(
            f"Saved feature order ({len(features_ordered_by_importance)} features) to {self.feature_order_path}"
        )

    def _load_feature_order(self) -> List[str]:
        if not self.feature_order_path.exists():
            raise FileNotFoundError(
                f"feature order not found at {self.feature_order_path}; run cluster_analysis (training) first."
            )
        with open(self.feature_order_path) as f:
            return list(json.load(f))

    @property
    def clip_bounds_path(self):
        """JSON sidecar of the per-column clip bounds fitted during training."""
        return self.output_path / "models" / "clip_bounds.json"

    def _save_clip_bounds(self, clip_input: pd.DataFrame, lower_bounds: List[float], upper_bounds: List[float]) -> None:
        """Persist the clip bounds clip_outliers fitted on the training data.

        ``lower_bounds`` / ``upper_bounds`` are aligned to the input frame's numeric
        columns (clip_outliers' iteration order). Stored keyed by column name (robust
        to column reordering) with NaN -> None for "no clip on this tail".
        """
        numeric_cols = clip_input.select_dtypes(include="number").columns.tolist()
        bounds = {
            col: {
                "lower": None if pd.isna(lower_bounds[i]) else float(lower_bounds[i]),
                "upper": None if pd.isna(upper_bounds[i]) else float(upper_bounds[i]),
            }
            for i, col in enumerate(numeric_cols)
        }
        self.clip_bounds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.clip_bounds_path, "w") as f:
            json.dump(bounds, f, indent=2)
        logger.info(f"Saved clip bounds ({len(bounds)} columns) to {self.clip_bounds_path}")

    def _apply_clip_bounds(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip ``data`` with the training bounds, or fall back to refitting on ``data``.

        Reusing the fitted bounds keeps clipping a pure transform so a new group (AI) is
        scored in the same feature space the model was trained on. If the sidecar is
        missing (model trained before clip-bound persistence) we recompute on ``data``
        and warn, matching the legacy behaviour.
        """
        if not self.clip_bounds_path.exists():
            logger.warning(
                f"clip bounds not found at {self.clip_bounds_path}; recomputing on this group's data "
                "(re-fits clipping on the inference set — retrain to persist bounds)."
            )
            clipped, _, _ = clip_outliers(data, self.clip_threshold[0], self.clip_threshold[1])
            return cast(pd.DataFrame, clipped)

        with open(self.clip_bounds_path) as f:
            bounds = json.load(f)
        clipped = data.copy()
        for col in clipped.columns:
            col_bounds = bounds.get(col)
            if not col_bounds:
                continue
            clipped[col] = clipped[col].clip(lower=col_bounds["lower"], upper=col_bounds["upper"])
        return clipped

    def _write_cluster_labels(self, new_labels: pd.DataFrame) -> None:
        """Upsert one run's labels into the shared cluster_labels.parquet.

        One column per ``<model>-n_features_<N>-k_<K>`` run. Labels are upserted by
        ``merge_features`` key (which includes ``ai_group``): the incoming rows
        overwrite their own keys in the column, while rows for other keys are
        preserved. This matters because the train (Default) and apply (AI) runs share
        the SAME column name, so a naive drop-and-remerge would wipe the other group's
        labels out of that column — here Default and AI labels coexist side by side.
        """
        merge_cols: List[str] = (
            [self.merge_features] if isinstance(self.merge_features, str) else list(self.merge_features)
        )
        col = self.cluster_label_column
        labels_path = self.cluster_labels_file_path
        if labels_path.exists():
            existing = pd.read_parquet(labels_path)
            if col in existing.columns:
                # Outer-merge brings col (existing) and col_incoming (this run); prefer
                # incoming where present, else keep the existing value for that key.
                merged = existing.merge(
                    new_labels[merge_cols + [col]], on=merge_cols, how="outer", suffixes=("", "_incoming")
                )
                merged[col] = cast(pd.Series, merged[f"{col}_incoming"]).combine_first(cast(pd.Series, merged[col]))
                combined = merged.drop(columns=[f"{col}_incoming"])
            else:
                combined = existing.merge(new_labels, on=merge_cols, how="outer")
        else:
            combined = new_labels
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(labels_path, index=False)

    def apply_model(self, data: pd.DataFrame, model_path: Optional[str] = None) -> np.ndarray:
        """Score a new ai_group with the trained pipeline and persist its cluster labels.

        Cross-group inference stage: applies the model fit by ``cluster_analysis`` on
        the training group (e.g. Default) to another group's grouped data (e.g. AI),
        then writes the predicted labels into the shared ``cluster_labels.parquet`` so
        ``attach_cluster_label`` can join them onto that group's enriched data.

        Preprocessing mirrors training exactly so this is a pure transform, not a re-fit
        on the new group: the importance-ordered feature list is reloaded from
        ``feature_order.json`` (NOT re-selected here), sliced to ``top_features``, and
        clipped with the bounds fitted during training (``clip_bounds.json``). Requires
        that ``top_features`` / ``n_clusters`` match the trained model so the pickle path
        and column count line up.

        When ``cluster_analysis.pretrained_model_dir`` is set, the model + feature order
        + clip bounds are fetched from that prior S3 run first, so applying a previously
        trained model needs no local training run.
        """
        self.ensure_pretrained_artifacts()
        feature_columns = self._load_feature_order()[: self.n_top_features]
        clipped = self._apply_clip_bounds(cast(pd.DataFrame, data[feature_columns]))
        pipeline = self.load_trained_model(model_path)
        cluster_labels = np.asarray(pipeline.predict(cast(pd.DataFrame, clipped)))

        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        logger.info(f"apply_model cluster sizes: {dict(zip(unique_labels, counts))}")

        new_labels = pd.DataFrame(data[self.merge_features].copy())
        new_labels[self.cluster_label_column] = cluster_labels
        self._write_cluster_labels(new_labels)

        # Persist the aggregated (grouped) features alongside their predicted label, so
        # this group's per-bin stats are inspectable next to the raw per-bet output that
        # attach_cluster_label writes. Tagged + bin-size-suffixed to avoid collisions
        # across the groups scored into the same run folder.
        self._save_grouped_with_labels(data, cluster_labels)

        # Radar over the standardized features the model actually clusters on (the loaded
        # pipeline's transform == training-time power-transform + scaler), so the per-group
        # cluster profiles are comparable to the training radar.
        x_transformed = pipeline[:-1].transform(cast(pd.DataFrame, clipped))
        radar_tag = self.output_tag or "group"
        self.plot_radar_chart(
            pd.DataFrame(x_transformed, columns=pd.Index(feature_columns)),
            cluster_labels,
            output_file_name=f"radar_apply_{radar_tag}",
        )

        return cluster_labels

    def _save_grouped_with_labels(self, data: pd.DataFrame, cluster_labels: np.ndarray) -> None:
        """Write the aggregated (grouped) cluster_data with its predicted cluster label.

        Output: ``output/grouped_data_<tag>_cluster_label_binsize{N}.parquet`` (the
        merge keys + every grouped feature + the ``cluster_label_column``). This is the
        aggregated-feature counterpart to the raw per-bet files attach_cluster_label
        writes.
        """
        out = data.copy()
        out[self.cluster_label_column] = cluster_labels
        tag = f"_{self.output_tag}" if self.output_tag else ""
        file_name = f"grouped_data{tag}_cluster_label_binsize{self.bin_size}.parquet"
        out.to_parquet(self.output_path / "output" / file_name, index=False)
        logger.info(f"saved aggregated features with cluster label to {self.output_path / 'output' / file_name}")

    def ensure_pretrained_artifacts(self) -> None:
        """Fetch the pretrained model + preprocessing sidecars from S3 into this run.

        No-op when ``pretrained_model_dir`` is unset (normal two-phase flow where the
        local training run produced the artifacts). Idempotent: files already present
        are kept, so the three per-group apply runs sharing one run_id download once.

        Required artifacts (apply fails without them): the model pickle,
        ``feature_order.json``, ``clip_bounds.json``. Optional provenance copied when
        present: cluster centers and the importance-ordered feature list.
        """
        src = self.pretrained_model_dir
        if not src:
            return
        src = src.rstrip("/")
        required = {self.pickle_model_name, "feature_order.json", "clip_bounds.json"}
        artifacts = [
            ("models", self.pickle_model_name),
            ("models", "feature_order.json"),
            ("models", "clip_bounds.json"),
            ("models", f"cluster_centers_standardized_{self.n_top_features}_k_{self.n_clusters}.csv"),
            ("features", "important_features.json"),
        ]
        client = get_s3_client()
        for subdir, name in artifacts:
            dest = self.output_path / subdir / name
            if dest.exists():
                continue
            bucket, key = parse_s3_path(f"{src}/{subdir}/{name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                client.download_file(bucket, key, str(dest))
                logger.info(f"fetched pretrained artifact s3://{bucket}/{key} -> {dest}")
            except ClientError:
                if name in required:
                    raise
                logger.warning(f"optional pretrained artifact not found, skipping: s3://{bucket}/{key}")
        self.save_apply_run_info()

    def save_apply_run_info(self) -> None:
        """Write ``models/apply_run_info.json`` so the prediction is reproducible.

        Records the source model folder, the exact pickle used, the hyperparameters
        that resolve it (top_features / n_clusters / bin_size), the cluster-label column
        name, and the top-N feature order the model consumes (clip bounds + full
        importance list sit beside it as their own JSON files).
        """
        info_path = self.output_path / "models" / "apply_run_info.json"
        if info_path.exists():
            return
        try:
            feature_order = self._load_feature_order()
        except FileNotFoundError:
            feature_order = []
        info = {
            "pretrained_model_dir": self.pretrained_model_dir,
            "pickle_model_name": self.pickle_model_name,
            "cluster_model": self.cluster_model,
            "top_features": self.n_top_features,
            "n_clusters": self.n_clusters,
            "bin_size": self.bin_size,
            "cluster_label_column": self.cluster_label_column,
            "feature_order_used": feature_order[: self.n_top_features],
            "clip_bounds_file": "clip_bounds.json",
            "feature_order_file": "feature_order.json",
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"wrote apply run info to {info_path}")

    def plot_pca(
        self,
        data: pd.DataFrame,
        cluster_label: np.ndarray,
        n_components: int = 3,
        output_file_name: str = "PCA_Clusters",
    ) -> None:
        """Plot PCA visualization of clusters and pairwise plots for n_components > 3 using seaborn.pairplot."""
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(data)
        n_clusters = len(np.unique(cluster_label))
        palette = sns.color_palette("Set1", n_clusters)
        cluster_label = np.asarray(cluster_label)

        # Standard 2D PCA scatterplot for first two components.
        plt.figure(figsize=(20, 16))
        sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=cluster_label, palette=palette, alpha=0.7)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("PCA Visualization of KMeans Clusters")
        plt.legend(title="Cluster")
        plt.grid(True)
        plt.savefig(
            self.output_path
            / "figures"
            / f"{output_file_name}_k({self.n_clusters})_n_features({self.n_top_features}).png"
        )
        plt.close()

        # If n_components > 3, generate a pair plot for the PCA components
        if n_components >= 3:
            df_pca = pd.DataFrame(
                x_pca,
                columns=pd.Index([f"PC{i+1}" for i in range(n_components)]),
            )
            df_pca["cluster_label"] = cluster_label
            # Only lower triangle, diagonal = hist; hue as cluster. Disable upper triangle.
            pair_grid = sns.PairGrid(
                df_pca,
                vars=[f"PC{i+1}" for i in range(n_components)],
                hue="cluster_label",
                corner=True,
                palette=palette,
            )
            pair_grid.map_lower(sns.scatterplot, alpha=0.7, s=7)
            pair_grid.map_diag(sns.histplot, kde=False, alpha=0.6, stat="density")
            pair_grid.add_legend(title="Cluster", adjust_subtitles=True)
            plt.suptitle(f"Pair Plot of First {n_components} PCA Components by Cluster", fontsize=10, y=0.95)
            plt.tight_layout(rect=(0, 0.03, 1, 0.97))
            plt.savefig(
                self.output_path
                / "figures"
                / f"{output_file_name}_pairplot_k({self.n_clusters})_n_features({self.n_top_features})_ncomps({n_components}).png"
            )
            plt.close()

    def plot_radar_chart(
        self, data: pd.DataFrame, data_cluster: np.ndarray, output_file_name: str = "Radar_Clusters"
    ) -> None:
        """Plot radar chart of cluster feature means."""
        data = data.copy()
        data.loc[:, "_cluster"] = data_cluster
        cluster_means = data.groupby("_cluster").mean().T
        cluster_means.sort_index(ascending=True, inplace=True)
        # Persist the per-cluster feature means that the radar draws, so the plot is
        # reproducible / consumable downstream without re-deriving it from raw output.
        cluster_means.to_csv(
            self.output_path
            / "output"
            / f"{output_file_name}_cluster_means_k({self.n_clusters})_n_features({self.n_top_features}).csv"
        )
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
        ax.set_xticklabels(categories, fontsize=16)
        # Align labels by side (same approach as make_radar_plot in simulation_report):
        # left side → ha="right" so text extends left; right side → ha="left" so text extends right;
        # top and bottom → do not change.
        top_bottom_band = np.pi / 12  # no change when angle near pi/2 (top) or 3*pi/2 (bottom)
        for label, angle_rad in zip(ax.get_xticklabels(), angles[:-1]):
            in_top = np.pi / 2 - top_bottom_band <= angle_rad <= np.pi / 2 + top_bottom_band
            in_bottom = 3 * np.pi / 2 - top_bottom_band <= angle_rad <= 3 * np.pi / 2 + top_bottom_band
            if in_top or in_bottom:
                continue
            if np.pi / 2 < angle_rad < 3 * np.pi / 2:  # left side → text extends left
                label.set_horizontalalignment("right")
            else:  # right side → text extends right
                label.set_horizontalalignment("left")
            label.set_y(label.get_position()[1] + 0.05)  # slight radial nudge outward
        plt.title("Cluster Feature Means (Standardized) - Radar Chart", pad=28, fontsize=14)
        plt.legend(loc="upper right")
        plt.subplots_adjust(left=0.1, bottom=0.1, top=0.92)
        plt.savefig(
            self.output_path
            / "figures"
            / f"{output_file_name}_k({self.n_clusters})_n_features({self.n_top_features}).png"
        )
        plt.close()

    def save_cluster_data(
        self,
        data_original: pd.DataFrame,
        data_cluster: pd.DataFrame,
        merge_columns: List[str],
        cluster_column: str = "cluster_label",
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

    def load_trained_model(self, model_path: Optional[str] = None) -> KMeans:
        """Load the trained K-means model."""
        resolved_path = self.output_path / "models" / self.pickle_model_name if model_path is None else model_path

        if not os.path.exists(resolved_path):
            raise ValueError("clustering model is not trained!")
        return joblib.load(resolved_path)

    def predict_clusters(self, data: pd.DataFrame, features: List[str], model_path: Optional[str] = None) -> np.ndarray:
        """Predict clusters for new data using trained model."""
        model = self.load_trained_model(model_path)
        return np.asarray(model.predict(data[features]))

    def save_pipeline_model(self, pipeline: Pipeline) -> None:
        """Save a complete pipeline model to disk."""

        pickle_model_name = self.pickle_model_name
        joblib.dump(pipeline, self.output_path / "models" / pickle_model_name)

        logger.info(f"Clustering pipeline model saved to: {self.output_path / 'models'}")
        logger.info(f"Pickle model saved as: {pickle_model_name}")

        # Save ONNX model with version from config
        if self.cluster_model == "kmeans":
            onnx_opset_version = self.onnx_opset_version
            logger.info(f"Using ONNX opset version: {onnx_opset_version}")
            initial_type = [("float_input", FloatTensorType([None, self.n_top_features]))]
            onnx_convert_result = convert_sklearn(pipeline, initial_types=initial_type, target_opset=onnx_opset_version)
            onnx_model = onnx_convert_result[0] if isinstance(onnx_convert_result, tuple) else onnx_convert_result

            onnx_model_name = self.onnx_model_name
            with open(self.output_path / "models" / onnx_model_name, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"ONNX model saved as: {onnx_model_name}")

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
        pipeline = self.set_n_clusters(pipeline, n_clusters)
        cluster_labels = pipeline.fit_predict(data)

        logger.info(f"Fitted and predicted with n_clusters={n_clusters}")
        return np.asarray(cluster_labels)

    def attach_cluster_label(self, reload: bool = False):
        """Attach cluster labels to enriched (attach) data and save per-cluster parquet files.

        Row removal can happen in two places:
        1. load_attach_data(): keeps only (merge_features) groups with exactly bin_size rows,
           so incomplete groups are already dropped before this method.
        2. Merge with cluster labels: cluster labels come from clustering output, which used
           cluster_data after dropna(how='any'). So any attach_data key that was dropped in
           clustering (e.g. due to NaN in cluster features) has no label; with how='inner'
           those rows are dropped here.
        """
        merge_cols: List[str] = (
            [self.merge_features] if isinstance(self.merge_features, str) else list(self.merge_features)
        )
        cluster_column = self.cluster_label_column
        data_with_cluster_label = pd.read_parquet(
            self.cluster_labels_file_path,
            columns=[cluster_column] + merge_cols,
        )
        # Outer-merged shared file can leave NaN where another model clustered different rows.
        data_with_cluster_label = data_with_cluster_label.dropna(subset=[cluster_column])
        data_with_cluster_label[cluster_column] = data_with_cluster_label[cluster_column].astype(int)
        unique_clusters = data_with_cluster_label[cluster_column].unique()

        attach_data = self.load_attach_data(reload=reload)
        feature_columns = attach_data.select_dtypes(include="number").columns.to_list()

        # Log key counts to explain row removal
        attach_keys = attach_data.loc[:, merge_cols].drop_duplicates()
        label_keys = data_with_cluster_label.loc[:, merge_cols].drop_duplicates()
        keys_only_in_attach = attach_keys.merge(label_keys, on=merge_cols, how="left", indicator=True)
        keys_only_in_attach = keys_only_in_attach[keys_only_in_attach["_merge"] == "left_only"]
        n_keys_attach = len(attach_keys)
        n_keys_label = len(label_keys)
        n_keys_dropped = len(keys_only_in_attach)
        n_rows_dropped = attach_data.merge(keys_only_in_attach.loc[:, merge_cols], on=merge_cols, how="inner").shape[0]
        logger.info(
            f"attach_cluster_label: attach_data keys={n_keys_attach}, cluster_label keys={n_keys_label}, "
            f"keys in attach but not in labels={n_keys_dropped}, enriched rows dropped by merge={n_rows_dropped}"
        )
        if n_keys_dropped > 0:
            logger.warning(
                f"Dropped {n_keys_dropped} (user_id, session_group, agg_group) groups "
                f"({n_rows_dropped} rows) because they have no cluster label (e.g. were removed by dropna before clustering)."
            )

        num_samples = 0
        for cluster in unique_clusters:
            logger.info(f"save attach data for cluster: {cluster}")
            file_name = self._enriched_cluster_file_name(cluster)
            cluster_data = attach_data.merge(
                data_with_cluster_label[data_with_cluster_label[cluster_column] == cluster],
                on=merge_cols,
                how="inner",
                suffixes=("", "_y"),
            )
            cluster_data = cluster_data[list(attach_data.columns) + [cluster_column]]
            cluster_data.to_parquet(self.output_path / "output" / file_name, index=False)
            num_samples += len(cluster_data)

            if feature_columns:
                stats = cast(pd.DataFrame, cluster_data.loc[:, feature_columns]).describe().T
                logger.info(f"Cluster {cluster} feature statistics:")
                logger.info(stats[["mean", "std", "min", "25%", "50%", "75%", "max"]])

        if num_samples < len(attach_data):
            logger.warning(
                f"sum of sample size for all clusters: {num_samples} less than attach data: {len(attach_data)}!"
            )
        if num_samples > len(attach_data):
            logger.critical(
                f"sum of sample size for all clusters: {num_samples} larger than attach data: {len(attach_data)}!"
            )

    def _enriched_cluster_file_name(self, cluster: Any) -> str:
        """Per-cluster attach filename: group-tagged and bin-size suffixed.

        Empty ``output_tag`` (legacy / single-group configs) keeps the historical
        ``enriched_data_cluster_{k}`` stem; a tag (e.g. ``ai``) yields
        ``enriched_data_ai_cluster_{k}``. The ``_binsize{N}`` suffix records the bin
        size the clustering used, e.g. ``enriched_data_ai_cluster_0_binsize50.parquet``.
        """
        tag = f"_{self.output_tag}" if self.output_tag else ""
        return f"enriched_data{tag}_cluster_{cluster}_binsize{self.bin_size}.parquet"

    def get_cluster_stats(self):

        cluster_stats = {}
        for cluster_index in range(self.n_clusters):
            data = read_local_cache(
                local_cache_path=self.output_path / "output" / self._enriched_cluster_file_name(cluster_index),
                columns=self.cluster_stats_columns,
                lazy_load=False,
            )
            data = cast(pd.DataFrame, data)

            # Only set columns if the number of columns matches expected
            expected_columns = ["loginname", "billtime", "basepoint", "account", "fg_rounds"]
            if len(data.columns) == len(expected_columns):
                data.columns = expected_columns
            else:
                logger.error(
                    f"Number of columns in cluster data ({len(data.columns)}) does not match expected ({len(expected_columns)}). Skipping column rename."
                )
            cluster_stats[f"cluster_{cluster_index}"] = compute_cluster_stats(cast(pd.DataFrame, data))

        # Convert NumPy types to native Python types for JSON serialization
        cluster_stats_serializable = convert_numpy_types(cluster_stats)
        json.dump(cluster_stats_serializable, open(self.output_path / "output/cluster_stats.json", "w"), indent=4)


def compute_cluster_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """Summary stats for one cluster.

    ``data`` must already be renamed to columns ``loginname, billtime, basepoint, account, fg_rounds``;
    callers handle the rename so this helper is reusable both inside the
    :class:`ClusterAnalysisPipeline` per-cluster loop and from one-off analysis scripts that treat all
    rows as a single cluster. NumPy/Counter values are returned as-is — pass through
    :func:`bituslabs_ds.utils.convert_numpy_types` before JSON serialization.
    """
    scaler = StandardScaler()
    scaler.fit(data["basepoint"].to_frame())
    basepoint_clean = data["basepoint"].dropna().astype(float)

    stats: Dict[str, Any] = {
        "basepoint_mean": basepoint_clean.mean(),
        "basepoint_min": basepoint_clean.min(),
        "basepoint_max": basepoint_clean.max(),
        "basepoint_p5": (np.percentile(basepoint_clean, 5) if basepoint_clean.size > 0 else None),
        "basepoint_p25": (np.percentile(basepoint_clean, 25) if basepoint_clean.size > 0 else None),
        "basepoint_p75": (np.percentile(basepoint_clean, 75) if basepoint_clean.size > 0 else None),
        "basepoint_p95": (np.percentile(basepoint_clean, 95) if basepoint_clean.size > 0 else None),
        "basepoint_median": basepoint_clean.median(),
        "basepoint_skewness": skew(basepoint_clean),
        "basepoint_std": basepoint_clean.std(),
        "basepoint_count": len(data["basepoint"]),
        "basepoint_nan_count": len(data["basepoint"]) - len(basepoint_clean),
        "basepoint_nan_ratio": (len(data["basepoint"]) - len(basepoint_clean)) / len(data["basepoint"]),
        "basepoint_less_than_0_count": (basepoint_clean < 0).sum(),
        "basepoint_less_than_0_ratio": (basepoint_clean < 0).sum() / len(data["basepoint"]),
        "basepoint_scaler_mean": cast(np.ndarray, scaler.mean_)[0],
        "basepoint_scaler_std": np.sqrt(cast(np.ndarray, scaler.var_)[0]),
        "account_counter": Counter(data["account"]),
    }

    # For each loginname, select "fg_rounds" from their earliest "billtime"
    first_slottype = (
        data.sort_values(["loginname", "billtime"])
        .groupby("loginname", as_index=False)
        .first()[["loginname", "fg_rounds"]]
    )
    stats["fg_rounds_ratio"] = cast(pd.Series, first_slottype["fg_rounds"]).value_counts(normalize=False).to_dict()
    return stats


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
