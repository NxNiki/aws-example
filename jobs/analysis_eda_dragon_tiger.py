"""
EDA on dragon-tiger game user data.
"""

import os
from typing import List

import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.eda import plot_df_distribution, read_excel_sheets
from bituslabs_ds.s3_utils import write_pandas_to_s3
from bituslabs_ds.utils import log_transform

config = Settings()
config.plot.histogram.bins = 200

COLUMNS_TO_DROP = ["1", "遊戲類型", "下注間隔", "倒數秒數", "日期時(文)", "日期文(2)"]

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
    "PAYOUT_RATIO",
    "BET_INTERVAL",
]


def time_to_seconds(t):
    if pd.isna(t):
        return None
    return t.hour * 3600 + t.minute * 60 + t.second


def read_and_preprocess(file_name: str) -> pd.DataFrame:

    df = read_excel_sheets(file_name, sheet_name_col="year_month")
    print(df.columns)

    df["BET_INTERVAL"] = df["下注間隔"].apply(time_to_seconds)
    df["COUNTDOWN_SECONDS"] = df["倒數秒數"].apply(time_to_seconds)
    df.drop(columns=COLUMNS_TO_DROP, inplace=True)
    df.rename(
        columns=COLUMNS_TO_RENAME,
        inplace=True,
    )

    for k, v in REPLACEMENTS.items():
        df = df.map(lambda x: x.replace(k, v) if isinstance(x, str) else x)

    return df


def create_profile_report(df: pd.DataFrame, out_file: str) -> None:

    file_path = os.path.dirname(out_file)
    file_name = os.path.basename(out_file)
    os.makedirs(file_path, exist_ok=True)
    profile = ProfileReport(df, title=file_name, config=config)
    profile.to_file(out_file)


if __name__ == "__main__":

    setup_logging("./eda_output/", "dragon_tiger.log")
    data = read_and_preprocess("/Users/niuxin/Downloads/N020_BetOrders.xlsx")
    write_pandas_to_s3(data, S3_BUCKET, "dragon_tiger/processed_data/N020_BetOrders.csv")

    data.drop(
        columns=["year_month", "BILL_NO", "BET_DEVICE", "GAME_TYPE", "BET_DEVICE_CUTOFF", "GM_CODE"], inplace=True
    )
    create_profile_report(data, "./eda_output/eda_dragon_tiger.html")

    data = log_transform(data[COLUMNS_TO_LOG_TRANSFORM])
    create_profile_report(data, "./eda_output/eda_dragon_tiger_log.html")
