"""
EDA on dragon-tiger game user data.
"""

import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import (
    plot_by_time,
    plot_df_distribution,
    plot_dual_axis_sorted_swarm,
    plot_heatmap,
    plot_multiple_box_swarm,
    plot_scatter_pairs,
    read_excel_sheets,
    split_column_by_multiple_separators,
    split_column_by_threshold,
)
from bituslabs_ds.s3_utils import upload_file_to_s3, write_pandas_to_s3
from bituslabs_ds.utils import add_event_group_by_gap, check_consecutive_event, group_iterator, log_transform

config = Settings()
config.plot.histogram.bins = 200

COLUMNS_TO_DROP = ["1", "遊戲類型", "日期時(文)", "日期文(2)"]

COLUMNS_TO_RENAME = {
    "帳號\nLOGINNAME": "LOGIN_NAME",
    "下注時間\nBILLTIME": "BILL_TIME",
    "局號\nGMCODE": "GM_CODE",
    "桌檯": "TABLE_ID",
    "玩法名稱": "GAME_NAME",
    "交易前餘額\nBASEPOINT": "Pre-TRANSACTION_BALANCE",
    "投注額\nACCOUNT": "BET_ACCOUNT",
    "派彩\nCUS_ACCOUNT": "CUS_ACCOUNT",
    "下注IP\nCUR_IP": "CUR_IP",
    "訂單號\nBILLNO": "BILL_NO",
    "幣種\nCURRENCY": "CURRENCY",
    "投注設備": "BET_DEVICE",
    "訂單狀態\nFLAG": "FLAG",
    "開局時間\nBEGINTIME": "BEGIN_TIME",
    "遊戲類型\nGAMETYPE": "GAME_TYPE",
    "大廳類型\nPLATFORMTYPE": "PLATFORM_TYPE",
    "投注設備\nCUTOFF": "BET_DEVICE_CUTOFF",
    "有效投注額\nVALID_ACCOUNT": "VALID_ACCOUNT",
    "結算時間\nRECKONTIME": "RECKON_TIME",
    "荷官\nDEALER": "DEALER",
    "遊戲結果\nCARDLIST": "CARD_LIST",
    "派彩間隔": "PAYOUT_INTERVAL",
    "投注額轉": "BET_TURN",
    "有效投注額轉": "VALID_BET_TURN",
    "派彩轉": "CUS_ACCOUNT_TURN",
    "最早下注": "EARLIEST_BET",
    "日期文": "DATE_TEXT",
    "時": "HOUR_OF_DAY",
    "倍投檢測": "TYPE_OF_HIGH_BET",
    "網段": "SUBNET",
    "賠率": "PAYOUT_RATIO",
    "玩法\nPLAYTYPE": "PLAY_TYPE",
    "龍牌代碼": "DRAGON_CARD",
    "虎牌代碼": "TIGER_CARD",
    "龍牌牌面數字": "DRAGON_CARD_NUMBER",
    "虎牌牌面數字": "TIGER_CARD_NUMBER",
    "牌值結果": "CARD_RESULT",
    "下注間隔": "BET_INTERVAL",
    "倒數秒數": "COUNTDOWN_SECONDS",
}

REPLACEMENTS = {
    "龍": "Dragon",
    "虎": "Tiger",
    "和": "He",
    "手機新": "cellphone_new",
    "正常": "Normal",
    "疑似重派局": "Suspicious",
    "贏後的投注": "Bet_after_win",
    "輸後重注": "High_Bet_after_lose",
    "輸後加注": "Add_bet_after_lose",
    "輸後倍投": "Multi_bet_after_lose",
    "黑桃": "Spade",
    "方塊": "Diamond",
    "紅心": "Heart",
    "梅花": "Club",
    "（": "(",
    "）": ")",
    "其它": "Others",
    "單": "Single",
    "雙": "Double",
}

COLUMNS_TO_LOG_TRANSFORM = [
    "Pre-TRANSACTION_BALANCE",
    "BET_ACCOUNT",
    "CUS_ACCOUNT",
    "VALID_ACCOUNT",
    "BET_TURN",
    "VALID_BET_TURN",
    "CUS_ACCOUNT_TURN",
    "BET_INTERVAL",
]


def time_to_seconds(t):
    if pd.isna(t):
        return None
    return t.hour * 3600 + t.minute * 60 + t.second


def read_and_preprocess(file_name: str) -> pd.DataFrame:

    df = read_excel_sheets(file_name, sheet_name_col="year_month")
    print(df.columns)

    df.drop(columns=COLUMNS_TO_DROP, inplace=True)
    df.rename(
        columns=COLUMNS_TO_RENAME,
        inplace=True,
    )

    df["BET_INTERVAL"] = df["BET_INTERVAL"].apply(time_to_seconds)
    df["COUNTDOWN_SECONDS"] = df["COUNTDOWN_SECONDS"].apply(time_to_seconds)

    for k, v in REPLACEMENTS.items():
        df = df.map(lambda x: x.replace(k, v) if isinstance(x, str) else x)

    df["TIGER_CARD_NUMBER"] = df["TIGER_CARD_NUMBER"].astype(str)
    df["DRAGON_CARD_NUMBER"] = df["DRAGON_CARD_NUMBER"].astype(str)

    return df


def create_profile_report(df: pd.DataFrame, out_file: str, s3_key: str = "") -> None:

    file_path = os.path.dirname(out_file)
    file_name = os.path.basename(out_file)
    os.makedirs(file_path, exist_ok=True)
    profile = ProfileReport(df, title=file_name, config=config)
    profile.to_file(out_file)

    if s3_key:
        upload_file_to_s3(out_file, S3_BUCKET, f"{s3_key}/{file_name}")


def create_heatmap_by_group(data: pd.DataFrame, group_by_cols: List[str], var_columns: List[str]) -> None:

    grouped_data = data.groupby(group_by_cols)[var_columns[0]].count()
    pivot_table = grouped_data.unstack(level=1)
    plot_heatmap(pivot_table, x_label="Dragon Card", y_label="Tiger Card", title="Bet count")

    for col in var_columns:
        grouped_data = data.groupby(group_by_cols)[col].mean()
        pivot_table = grouped_data.unstack(level=1)
        plot_heatmap(pivot_table, x_label="Dragon Card", y_label="Tiger Card", title=f"Avg {col}")


def create_swam_plot_by_card(data: pd.DataFrame, card_columns: List[str], card_col_name: str) -> None:

    plot_data = data.loc[
        data["LOGIN_NAME"] == "EW3u96150agent_136361155", [*card_columns, "CUS_ACCOUNT", "card_change"]
    ]

    plot_data = plot_data.melt(
        id_vars=["CUS_ACCOUNT", "card_change"],
        value_vars=card_columns,
        value_name=card_col_name,
    )

    plot_dual_axis_sorted_swarm(plot_data, y_col_left=card_col_name, hue_col="card_change", value_cols=["CUS_ACCOUNT"])


def plot_bet_time(df: pd.DataFrame, time_col: str = "timestamp", threshold_hours: float = 12):

    df = df.copy()
    df = check_consecutive_event(df, time_col=time_col, threshold=40, col_name="is_consecutive")
    df = check_consecutive_event(
        df, time_col=time_col, threshold=threshold_hours * 3600, col_name="is_gap", check_gap=True
    )
    df["seconds"] = df[time_col].dt.time.map(time_to_seconds)
    df["date_str"] = df[time_col].dt.strftime("%Y-%m-%d")
    unique_dates = df["date_str"].unique()
    date_to_y = {date: i for i, date in enumerate(unique_dates)}
    df["y"] = df["date_str"].map(date_to_y)

    fig, ax = plt.subplots(figsize=(14, 8))

    bet_counter: dict = defaultdict(int)

    # Plot short vertical lines per timestamp
    for i in range(len(df)):
        x = df.loc[i, "seconds"]
        y = df.loc[i, "y"]

        if df.loc[i, "CUS_ACCOUNT"] > 0:
            y_offset = random.randint(0, 3)
            colors = ["green", "blue"]
            counter_suffix = "win"
        else:
            y_offset = random.randint(4, 7)
            colors = ["gray", "gray"]
            counter_suffix = "lose"

        if df.loc[i, "is_gap"] != 0:
            color = "black"
            y_range = [y - 0.5, y + 0.5]
            ax.plot([x, x], y_range, color=color, alpha=0.3, linestyle="-")
            if len(bet_counter) > 0:
                con_win = bet_counter["consecutive_win"]
                con_lose = bet_counter["consecutive_lose"]
                text_x_offset = 500
                if con_win + con_lose > 10:
                    ax.text(
                        x + text_x_offset,
                        y + 0.05,
                        f"{con_win} ({con_win/(con_win + con_lose):.3f})",
                        fontsize=9,
                        ha="left",
                        va="bottom",
                    )
                    ax.text(
                        x + text_x_offset,
                        y + 0.2,
                        f"{con_lose} ({con_lose/(con_win + con_lose):.3f})",
                        fontsize=9,
                        ha="left",
                        va="bottom",
                    )
                non_con_win = bet_counter["non_consecutive_win"]
                non_con_lose = bet_counter["non_consecutive_lose"]
                if non_con_win + non_con_lose > 10:
                    ax.text(
                        x + text_x_offset,
                        y - 0.05,
                        f"{non_con_win} ({non_con_win/(non_con_win + non_con_lose):.3f})",
                        fontsize=9,
                        ha="left",
                        va="top",
                    )
                    ax.text(
                        x + text_x_offset,
                        y - 0.2,
                        f"{non_con_lose} ({non_con_lose/(non_con_win + non_con_lose):.3f})",
                        fontsize=9,
                        ha="left",
                        va="top",
                    )
                bet_counter = defaultdict(int)

        if df.loc[i, "is_consecutive"] != 0:
            color = colors[0]
            y = y + y_offset * 0.05
            y_range = [y + 0.01, y + 0.04]
            bet_counter[f"consecutive_{counter_suffix}"] += 1
        else:
            color = colors[1]
            y = y - y_offset * 0.05
            y_range = [y - 0.04, y - 0.01]
            bet_counter[f"non_consecutive_{counter_suffix}"] += 1

        ax.plot([x, x], y_range, color=color, alpha=0.7)

    # Format axes
    ax.set_xticks(range(0, 86401, 3600 * 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.set_xlabel("Time of Day")

    ax.set_yticks(range(len(unique_dates)))
    ax.set_yticklabels(unique_dates, fontsize=8)
    ax.set_ylabel("Date")

    ax.set_title(
        f"Bet Time by Day (Vertical Line: Gap > {threshold_hours} hours, Green and above: Gaps <= 40s, Blue and below: Gaps > 40s, Gray: lose bets)"
    )
    ax.grid(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    setup_logging("./eda_output/", "dragon_tiger.log")
    data = read_and_preprocess("/Users/niuxin/Downloads/N020_BetOrders.xlsx")

    # s3_path = "dragon_tiger/processed_data"
    # write_pandas_to_s3(data, S3_BUCKET, f"{s3_path}/N020_BetOrders.csv")

    gap_thresh = 6 * 3600  # seconds

    data.drop(columns=["BILL_NO", "BET_DEVICE", "GAME_TYPE", "BET_DEVICE_CUTOFF", "GM_CODE"], inplace=True)
    data.loc[data["BET_INTERVAL"] > gap_thresh, "BET_INTERVAL"] = pd.NA

    ## plot cus_account (bet return) for each block to bets, separately for consecutive and non-consecutive bets:
    data_bet = add_event_group_by_gap(
        data.loc[data["LOGIN_NAME"] == "EW3u96150agent_136361155", ["BILL_TIME", "CUS_ACCOUNT"]],
        time_col="BILL_TIME",
        threshold=gap_thresh,
        col_name="gap_group",
    )

    data_bet = check_consecutive_event(data_bet, time_col="BILL_TIME", threshold=40, col_name="is_consecutive")
    data_bet["is_consecutive"] = data_bet["is_consecutive"].map({True: "consecutive", False: "non-consecutive"})

    for data_group, group, _ in group_iterator(data_bet, "gap_group", count_thresh=10):
        plot_by_time(
            data_group,
            time_col="BILL_TIME",
            var_columns=["CUS_ACCOUNT"],
            group_col="is_consecutive",
            time_window=timedelta(seconds=10),
            title=f"cus account, group: {group}",
        )

    # s3_path = "dragon_tiger/eda_output"
    # create_profile_report(data, "./eda_output/eda_dragon_tiger.html", s3_path)

    # data = log_transform(data, col_names=COLUMNS_TO_LOG_TRANSFORM)
    # create_profile_report(data, "./eda_output/eda_dragon_tiger_log.html", s3_path)

    # plot_df_distribution(data)
    # plot_scatter_pairs(data)

    print(data.groupby(["LOGIN_NAME", "year_month"])["BILL_TIME"].count())

    # data = split_column_by_threshold(data, columns=["CUS_ACCOUNT", "CUS_ACCOUNT_TURN"])

    ## plot the sliding window average for selected columns:
    # plot_by_time(
    #     data[data["LOGIN_NAME"] == "EW3u96150agent_136361155"],
    #     time_col="BILL_TIME",
    #     var_columns=["Pre-TRANSACTION_BALANCE", "VALID_ACCOUNT", "BET_INTERVAL"],
    #     time_window=timedelta(hours=1),
    #     title="user: EW3u96150agent_136361155",
    #     vline_time=datetime(year=2025, month=6, day=6, hour=21, minute=6),
    #     vline_label="change card",
    #     vars_in_logscale={"BET_INTERVAL"},
    # )

    ## plot win rate for each card number:
    # data["card_change"] = data["BILL_TIME"] > datetime(year=2025, month=6, day=6, hour=21, minute=6)
    # data = split_column_by_multiple_separators(data, column="CARD_RESULT")
    # create_swam_plot_by_card(data, card_columns=["CARD_RESULT_1", "CARD_RESULT_2"], card_col_name="CARD_RESULT")
    # create_swam_plot_by_card(data, card_columns=["DRAGON_CARD_NUMBER", "TIGER_CARD_NUMBER"], card_col_name="CARD_NUMBER")
    # plot_dual_axis_sorted_swarm(
    #     data,
    #     y_col_right="DRAGON_CARD_NUMBER",
    #     y_col_left="TIGER_CARD_NUMBER",
    #     hue_col="card_change",
    #     value_cols=["CUS_ACCOUNT"],
    # )

    # create_heatmap_by_group(
    #     data[data["LOGIN_NAME"] == "EW3u96150agent_136361155"],
    #     group_by_cols=["CARD_RESULT_1", "CARD_RESULT_2"],
    #     var_columns=["CUS_ACCOUNT", "PAYOUT_RATIO"]
    # )
    #
    # create_heatmap_by_group(
    #     data[data["LOGIN_NAME"] != "EW3u96150agent_136361155"],
    #     group_by_cols=["CARD_RESULT_1", "CARD_RESULT_2"],
    #     var_columns=["CUS_ACCOUNT", "PAYOUT_RATIO"]
    # )

    # for x_col in ["TABLE_ID", "GAME_NAME", "PLATFORM_TYPE"]:
    #     plot_multiple_box_swarm(
    #         data[data["LOGIN_NAME"] == "EW3u96150agent_136361155"],
    #         x_cols=[x_col],
    #         y_cols=COLUMNS_TO_LOG_TRANSFORM,
    #         group_col="year_month",
    #         fig_title=x_col,
    #     )

    # for login_name, df in data.groupby("LOGIN_NAME"):
    #     plot_multiple_box_swarm(
    #         df,
    #         x_cols=["TABLE_ID", "GAME_NAME", "PLATFORM_TYPE"],
    #         y_cols=COLUMNS_TO_LOG_TRANSFORM,
    #         group_col="year_month",
    #         fig_title=f"LIGIN_NAME: {str(login_name)}",
    #     )

    ## plot distribution of bet intervals:
    # plot_df_distribution(data.loc[data["LOGIN_NAME"] == "EW3u96150agent_136361155", "BET_INTERVAL"], log=True)
    # print(data.loc[data["LOGIN_NAME"] == "EW3u96150agent_136361155", "BET_INTERVAL"].describe())

    # data_log = log_transform(data, col_names="BET_INTERVAL")
    # plot_df_distribution(data_log.loc[data_log["LOGIN_NAME"] == "EW3u96150agent_136361155", "BET_INTERVAL"], log=True)

    ## plot the start and end time for each day:
    # plot_bet_time(
    #     data.loc[data["LOGIN_NAME"] == "EW3u96150agent_136361155", ["BILL_TIME", "CUS_ACCOUNT"]],
    #     time_col="BILL_TIME",
    #     threshold_hours=6,
    # )
