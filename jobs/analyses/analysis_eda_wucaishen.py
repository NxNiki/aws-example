import glob
import os
from typing import List, Tuple, Union

import pandas as pd

from bituslabs_ds.eda import DataProfiler, plot_scatter_pairs, plot_seasonality, read_csv_cols


def load_enriched_data(files: List[str], sampling: Union[int, float], output: str, reload=False) -> pd.DataFrame:

    if os.path.exists(output) and not reload:
        return pd.read_csv(output)

    data = read_csv_cols(
        files,
        columns=[
            "group_id",
            "billtime",
            "basepoint",
            "account",
            "cus_account",
            "payout",
            "delta_t",
            "delta_bet",
            "delta_profit",
            "delta_payout",
            "streak",
            "win_streak",
            "lose_streak",
            "rtp",
        ],
        filters={"currency": "CNY"},
        sampling=sampling,
    )

    data["billtime"] = pd.to_datetime(data["billtime"])
    data["bill_day"] = data["billtime"].dt.date
    data = data.drop(columns="billtime").groupby(["group_id", "bill_day"]).mean().reset_index()
    data.drop(columns="group_id").to_csv(output, index=False)

    return data


def make_seasonality_plots(
    data: pd.DataFrame, time_column: str, value_columns: List[Tuple[List[str], List[str]]], freq="monthly"
) -> None:
    for value_column_left, value_column_right in value_columns:
        plot_seasonality(data[time_column], data[value_column_left], data[value_column_right], freq=freq)


def run_enriched_analysis():

    files = glob.glob(f"./output/wucaishen_enriched_output_24??.csv")
    sampling = 2_000_000
    data = load_enriched_data(files, sampling, f"./output/wucaishen_enriched_output_24_group_mean_{sampling}.csv")

    time_column = "bill_day"
    seasonality_data_columns = [
        (["account", "payout"], ["cus_account", "rtp"]),
        (["streak"], ["win_streak", "lose_streak"]),
        (["basepoint"], ["delta_t"]),
    ]

    logx = True
    logy = True
    pairs = [
        ("delta_t", "basepoint", logx, logy),
        ("delta_t", "account", logx, logy),
        ("delta_t", "payout", logx, logy),
        ("delta_t", "cus_account", logx, logy),
        ("delta_t", "rtp", logx, logy),
        ("streak", "basepoint", logx, logy),
        ("streak", "account", logx, logy),
        ("streak", "payout", logx, logy),
        ("streak", "cus_account", logx, logy),
        ("streak", "rtp", logx, logy),
        ("cus_account", "rtp", logx, logy),
    ]

    run_eda(data, time_column, seasonality_data_columns, pairs)


def run_grouped_analysis():
    ## grouped data (aggregate by 40 consecutive bets):
    data = pd.read_csv("./output/wucaishen_grouped_stat_output_24.csv")
    columns = [
        "start_time",
        "bet_median",
        "basepoint_median",
        "payout_median",
        "profit_median",
        "delta_t_median",
        "delta_bet_median",
        "delta_profit_median",
        "streak_median",
        "win_streak_median",
        "lose_streak_median",
        "rtp_mean",
    ]
    data = data[columns]

    time_column = "start_time"
    seasonality_data_columns = [
        (["bet_median"], ["profit_median"]),
        (["streak_median"], ["win_streak_median", "lose_streak_median"]),
        (["basepoint_median"], ["delta_t_median"]),
        (["rtp_mean"], ["payout_median"]),
    ]

    logx = True
    logy = True
    pairs = [
        ("delta_t_median", "basepoint_median", logx, logy),
        ("delta_t_median", "bet_median", logx, logy),
        ("delta_t_median", "payout_median", logx, logy),
        ("delta_t_median", "profit_median", logx, logy),
        ("delta_t_median", "rtp_mean", logx, logy),
        ("streak_median", "basepoint_median", logx, logy),
        ("streak_median", "bet_median", logx, logy),
        ("streak_median", "payout_median", logx, logy),
        ("streak_median", "profit_median", logx, logy),
        ("streak_median", "rtp_mean", logx, logy),
    ]

    run_eda(data, time_column, seasonality_data_columns, pairs)


def run_user_analysis():

    data = pd.read_csv("./output/wucaishen_grouped_stat_output_24.csv", usecols=["group_id", "start_time"])
    data["start_time"] = pd.to_datetime(data["start_time"]).dt.date

    data = data.groupby(["start_time"]).count().reset_index()
    data.rename(columns={"group_id": "num_users"}, inplace=True)
    plot_seasonality(data["start_time"], data["num_users"], freq="monthly")
    plot_seasonality(data["start_time"], data["num_users"], freq="weekly")
    plot_seasonality(data["start_time"], data["num_users"], freq="daily")


def run_eda(
    data: pd.DataFrame,
    time_column: str,
    seasonality_data_columns: List[Tuple[List[str], List[str]]],
    pairs: List[Tuple[str, str, bool, bool]],
):

    DataProfiler.plot_df_distribution(data.iloc[:, 1:], log=True)
    make_seasonality_plots(data, time_column, seasonality_data_columns, freq="monthly")
    make_seasonality_plots(data, time_column, seasonality_data_columns, freq="weekly")
    make_seasonality_plots(data, time_column, seasonality_data_columns, freq="daily")
    plot_scatter_pairs(data.iloc[:, 1:], pairs, alpha=0.2)


if __name__ == "__main__":

    # data = pd.read_csv("./output/wucaishen_grouped_stat_output_24.csv", nrows=5)
    # print(data.columns)

    run_enriched_analysis()
    run_grouped_analysis()
    run_user_analysis()
