import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from cluster_analysis_01_elbow_method import (
    elbow_method,
    feature_selection_by_pca,
    feature_selection_by_variance,
    smart_feature_selection,
)

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

    data = data[data["currency_label"] == 0]  # only keep CNY
    data.drop(columns="currency_label", inplace=True)
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
        # currently we combine all currencies as different currency users may have different purchase power.
        "currency_label",
    ]

    return non_features, normal_features, skewed_features


def main(output_path: str):

    os.makedirs(f"{output_path}/output", exist_ok=True)
    os.makedirs(f"{output_path}/features", exist_ok=True)
    os.makedirs(f"{output_path}/figures", exist_ok=True)

    non_features, normal_features, skewed_features = get_feature_names()
    player_data = load_data(
        f"{output_path}/output/deepdive_grouped_stat_output_25.csv",
        columns=[*non_features, *normal_features, *skewed_features],
        pattern=r"250.?/deepdive_grouped_stat_output.*\.csv$",
    )

    DataProfiler.count_df_missing_columns(player_data)
    player_data.fillna(0, inplace=True)
    player_data = df_power_transform(player_data, skewed_features)
    save_list(normal_features + skewed_features, f"{output_path}/features/log_transform_features.json")

    normal_features.remove("currency_label")
    viz = DataVisualizer(player_data[normal_features + skewed_features])
    viz.create_figure(fig_title="Correlation of features: deepdive", fig_size=(23, 14))
    viz.add_correlation_heatmap(annot=False, cmap="coolwarm")
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    viz.display()
    viz.save(f"{output_path}/figures/deepdive_correlation.png")

    # remove highly correlated features:
    _, kept_features = smart_feature_selection(player_data[normal_features + skewed_features], threshold=0.9)
    _, kept_features = feature_selection_by_variance(player_data[kept_features], threshold=0.01)
    data_select = player_data[kept_features + non_features]

    important_features = feature_selection_by_pca(data_select, output_path)
    elbow_method(player_data, important_features, [15, 20, 25, 30, 35, 40, 45, 50], output_path)


if __name__ == "__main__":

    default_output_path = Path(__file__).parent / "output_deepdive"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=False, default=default_output_path)
    args = parser.parse_args()

    setup_logging(args.output_path, "analysis_cluster_01_deepdive_elbow_method.log")

    main(args.output_path)
