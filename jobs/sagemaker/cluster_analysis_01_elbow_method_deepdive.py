import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from cluster_analysis_01_elbow_method import (
    elbow_method,
    feature_selection_by_pca,
    feature_selection_by_variance,
    smart_feature_selection,
)
from cluster_config import CORRELATION_THRESHOLD, ELBOW_K_RANGE, OUTPUT_PATH, USE_CNY, VARIANCE_THRESHOLD, WORK_DIR
from sklearn.metrics import confusion_matrix

from bituslabs_ds.config import setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.s3_utils import list_s3_files, read_dataset, read_files
from bituslabs_ds.utils import column_iterator, df_power_transform, keep_numeric_columns, remove_outliers, save_list


def load_data(output_file: str, columns: Optional[List[str]] = None, pattern: str = ".*") -> pd.DataFrame:

    files = list_s3_files("hyber-slot", "deepdive_groupdata", pattern)
    data = read_files(
        files,
        local_cache_path=output_file,
        columns=columns,
        reload=False,
    )

    counts = data["currency_label"].value_counts()
    percentages = data["currency_label"].value_counts(normalize=True) * 100
    currency_stats = pd.DataFrame({"count": counts, "percentage": percentages.round(2)})
    print("Currency counts and percentages (%):\n", currency_stats)

    if USE_CNY:
        data = data[data["currency_label"] == 0]  # only keep CNY
        data.drop(columns="currency_label", inplace=True)

    # select data after April:
    # data = data[pd.to_datetime(data["start_time"]).dt.month >= 4]

    return data


def get_feature_names() -> Tuple[List[str], List[str], List[str]]:

    non_features = ["group_id", "loginname", "start_time"]

    base_features = [
        "bet",
        "basepoint",
        "payout",
        "profit",
        "delta_t",
        "delta_bet",
        "delta_profit",
        "streak",
        "win_streak",
        "lose_streak",
        "deposit",
        "withdrawal",
    ]
    suffixes = ["min", "max", "mean", "p25", "median", "p75"]
    features = [f"{bf}_{suf}" for bf in base_features for suf in suffixes]
    skewed_features = ["rtp_mean"] + features

    normal_features = [
        # "group_num",
        "slottype_2_count",
        "slottype_16_count",
        "slottype_17_count",
        "payout_rate",
        "profit_rate",
        "profit_stddev",
        "account_stddev",
        "morning_count",
        "afternoon_count",
        "night_count",
        "midnight_count",
        "weekend_count",
        "duration_seconds",
        "avg_time_per_bet",
        "currency_label",
    ]

    return non_features, normal_features, skewed_features


def check_cluster_index_contingency(
    cluster_indices: Dict[int, Dict[int, np.ndarray]], reference_feature: np.ndarray, output_dir: str
) -> None:
    """
    For each element in cluster_indices (which is a dict of k: cluster_labels),
    calculate and print the contingency matrix between each cluster index and the reference feature.
    """

    for idx, cluster_idx_dict in cluster_indices.items():
        print(f"\n--- Cluster Indices Set {idx+1} ---")
        for k, cluster_labels in cluster_idx_dict.items():
            print(f"\nContingency Matrix for k={k}:")
            contingency = pd.crosstab(cluster_labels, reference_feature, rownames=["Cluster"], colnames=["Reference"])
            print(contingency)

            contingency.to_csv(f"{output_dir}/cluster_index_contingency_nfeature-{idx}_k-{k}.csv", index=False)


def main():

    os.makedirs(f"{WORK_DIR}", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/output", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/features", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/figures", exist_ok=True)

    non_features, normal_features, skewed_features = get_feature_names()
    player_data = load_data(
        f"{WORK_DIR}/deepdive_grouped_stat_output_25.csv",
        columns=[*non_features, *normal_features, *skewed_features],
        pattern=r"250.?/deepdive_grouped_stat_output.*\.csv$",
    )

    DataProfiler.count_df_missing_columns(player_data)
    player_data.fillna(0, inplace=True)
    player_data = df_power_transform(player_data, skewed_features)
    save_list(normal_features + skewed_features, f"{OUTPUT_PATH}/features/log_transform_features.json")

    if USE_CNY:
        normal_features.remove("currency_label")

    viz = DataVisualizer(player_data[normal_features + skewed_features])
    viz.create_figure(fig_title="Correlation of features: deepdive", fig_size=(23, 13.5))
    viz.add_correlation_heatmap(annot=False, cmap="coolwarm")
    viz.figure.subplots_adjust(left=0.15, bottom=0.17, top=0.93, right=0.97)
    # viz.display()
    viz.save(f"{OUTPUT_PATH}/figures/deepdive_correlation.png")

    # remove highly correlated features:
    _, kept_features = smart_feature_selection(
        player_data[normal_features + skewed_features], threshold=CORRELATION_THRESHOLD
    )
    _, kept_features = feature_selection_by_variance(player_data[kept_features], threshold=VARIANCE_THRESHOLD)
    data_select = player_data[kept_features + non_features]

    important_features = feature_selection_by_pca(data_select, OUTPUT_PATH)
    cluster_indices = elbow_method(player_data, important_features, ELBOW_K_RANGE, OUTPUT_PATH)

    if not USE_CNY:
        check_cluster_index_contingency(
            cluster_indices, player_data["currency_label"].values, output_dir=f"{OUTPUT_PATH}/output"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    setup_logging(f"{OUTPUT_PATH}/log", "analysis_cluster_01_deepdive_elbow_method.log")
    main()
