"""
Shared data loading for cluster transition analysis and MCMC.

Provides: load cluster labels from S3, load features (local or S3), build merged+transitions,
and ensure_merged_parquet() which uses local cache when present and otherwise builds from S3 + features.
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from bituslabs_ds.s3_utils import read_single_file

MERGE_KEYS = ["user_id", "session_group", "agg_group"]

# Default S3 URIs for cluster label parquets (same as analysis_cluster_transition_stats)
DEFAULT_CLUSTER_LABEL_URIS = [
    "s3://bituslabs-team-ai/ss01_analysis_kmeans_2026-01-21_15-22-05/output/enriched_data_cluster_0.parquet",
    "s3://bituslabs-team-ai/ss01_analysis_kmeans_2026-01-21_15-22-05/output/enriched_data_cluster_1.parquet",
    "s3://bituslabs-team-ai/ss01_analysis_kmeans_2026-01-21_15-22-05/output/enriched_data_cluster_2.parquet",
]


def get_default_features_path() -> Path:
    """Default path to features parquet (output of etl/feature engineer)."""
    return Path(__file__).resolve().parent.parent / "output_ss01_wucaishen" / "ss01_features_grouped.parquet"


def _cluster_index_from_path(s3_uri: str) -> int:
    """Extract cluster index from path like .../enriched_data_cluster_2.parquet."""
    m = re.search(r"enriched_data_cluster_(\d+)\.parquet", s3_uri, re.I)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract cluster index from: {s3_uri}")


def load_cluster_labels(cluster_uris: list[str], merge_keys: Optional[list[str]] = None) -> pd.DataFrame:
    """Load all cluster parquet files from S3 and concat with cluster_label column."""
    merge_cols = list(merge_keys or MERGE_KEYS)
    dfs = []
    for uri in cluster_uris:
        k = _cluster_index_from_path(uri)
        df = read_single_file(uri, columns=merge_cols)
        # read_single_file returns DataFrame when lazy_load=False (default); LazyFrame otherwise
        assert isinstance(df, pd.DataFrame), "Expected DataFrame (lazy_load=False)"
        df = df.copy()
        df["cluster_label"] = k
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_features(path: str | Path) -> pd.DataFrame:
    """Load features parquet (local or S3). Uses read_single_file for Decimal->float conversion."""
    return read_single_file(str(path))


def build_merged_with_transitions(
    cluster_df: pd.DataFrame,
    features_df: pd.DataFrame,
    merge_keys: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Inner join cluster labels to features on merge_keys, then add transition labels.

    Transition = (from_cluster, to_cluster). Next cluster = following agg_group in same session.
    Drops last agg_group per session (no "to" state).
    """
    keys = list(merge_keys or MERGE_KEYS)
    merged = features_df.merge(cluster_df, on=keys, how="inner")
    merged = merged.drop_duplicates(subset=keys)
    merged = merged.sort_values(keys)
    merged["next_cluster"] = merged.groupby(["user_id", "session_group"], group_keys=False)["cluster_label"].shift(-1)
    merged = merged.dropna(subset=["next_cluster"]).copy()
    merged["next_cluster"] = merged["next_cluster"].astype(int)
    merged["transition"] = "cluster" + merged["cluster_label"].astype(str) + ":" + merged["next_cluster"].astype(str)
    return merged


def ensure_merged_parquet(
    merged_path: Path,
    cluster_uris: Optional[list[str]] = None,
    features_path: Optional[str | Path] = None,
    merge_keys: Optional[list[str]] = None,
) -> Path:
    """Ensure merged_with_transitions.parquet exists at merged_path.

    If merged_path exists, return it (use cache, skip S3). Otherwise load cluster labels
    from S3, load features, build merged DataFrame, save to merged_path, and return it.

    Call this at the start of analysis or MCMC when running locally so either script
    can run without manually running the other first.
    """
    if merged_path.exists():
        print(f"Found cached merged data at {merged_path}, skip reading S3 cluster files.")
        return merged_path
    uris = cluster_uris or DEFAULT_CLUSTER_LABEL_URIS
    feat_path = features_path or get_default_features_path()
    keys = merge_keys or MERGE_KEYS
    print("Loading cluster label files from S3...")
    cluster_df = load_cluster_labels(uris, merge_keys=keys)
    print("Loading features...")
    features_df = load_features(feat_path)
    merged = build_merged_with_transitions(cluster_df, features_df, merge_keys=keys)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_path, index=False)
    print(f"Built and saved merged data to {merged_path} (shape {merged.shape}).")
    return merged_path
