import glob
import logging
import os
from typing import List, Tuple

import pandas as pd

from bituslabs_ds.eda import plot_df_distribution, plot_scatter_pairs, plot_seasonality, read_csv_cols


def load_enriched_data(files: List[str], sampling: int | float, output: str) -> pd.DataFrame:

    if os.path.exists(output):
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


if __name__ == "__main__":

    # files = glob.glob(f'./output/wucaishen_enriched_output_24??.csv')
    # sampling = 2_000_000
    # data = load_enriched_data(files, sampling, f'./output/wucaishen_enriched_output_24_group_mean_{sampling}.csv')
    #
    # time_column = 'bill_day'
    # seasonality_data_columns = [
    #     (['account', 'payout'], ['cus_account', 'rtp']),
    #     (['streak'], ['win_streak', 'lose_streak']),
    #     (['basepoint'], ['delta_t']),
    # ]
    #
    # pairs = [
    #     ('delta_t', 'basepoint'),
    #     ('delta_t', 'account'),
    #     ('delta_t', 'payout'),
    #     ('delta_t', 'cus_account'),
    #     ('delta_t', 'rtp'),
    #     ('streak', 'basepoint'),
    #     ('streak', 'account'),
    #     ('streak', 'payout'),
    #     ('streak', 'cus_account'),
    #     ('streak', 'rtp'),
    #     ('cus_account', 'rtp'),
    # ]

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

    pairs = [
        ("delta_t_median", "basepoint_median"),
        ("delta_t_median", "bet_median"),
        ("delta_t_median", "payout_median"),
        ("delta_t_median", "profit_median"),
        ("delta_t_median", "rtp_mean"),
        ("streak_median", "basepoint_median"),
        ("streak_median", "bet_median"),
        ("streak_median", "payout_median"),
        ("streak_median", "profit_median"),
        ("streak_median", "rtp_mean"),
    ]

    plot_df_distribution(data.iloc[:, 1:], log=True)
    make_seasonality_plots(data, time_column, seasonality_data_columns, freq="monthly")
    make_seasonality_plots(data, time_column, seasonality_data_columns, freq="weekly")
    make_seasonality_plots(data, time_column, seasonality_data_columns, freq="daily")
    # plot_scatter_pairs(data.iloc[:, 1:], pairs)
