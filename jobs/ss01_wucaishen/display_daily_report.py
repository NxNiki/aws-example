from textwrap import dedent
from typing import List

import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT


def generate_daily_report(df: pd.DataFrame, df_pa: pd.DataFrame) -> None:
    # Convert activity_date columns to datetime if they aren't already
    df["activity_date"] = pd.to_datetime(df["activity_date"])
    df_pa["activity_date"] = pd.to_datetime(df_pa["activity_date"])

    def get_value(df, col, group=None):
        if group:
            vals = df.loc[df["ai_group"] == group, col]
        else:
            vals = df[col]
        return vals.values[0] if not vals.empty else ""

    metrics = [
        ("玩家数量（投注次数 >= 40）", "num_active_users"),
        ("玩家数量（All）", "day0_num_users"),
        ("前一日留存玩家数量 (Day -1 Retention)", "day1_num_users"),
        ("前三日留存玩家数量 (Day -3 Retention)", "day3_num_users"),
        ("玩家平均投注次数 (Avg. Bets/User)", "num_bets_per_user"),
        ("玩家平均投注总额度 (Avg. Total Bet/User)", "total_bet_per_user"),
        ("RTP (Return to Player)", "rtp"),
    ]

    dates = sorted(df["activity_date"].unique(), reverse=True)

    for date in dates[1:4]:
        df_date = df[df["activity_date"] == date]
        # Subtract one year from the target date for df_pa selection
        pa_date = date - pd.DateOffset(years=1)
        df_pa_date = df_pa[df_pa["activity_date"] == pa_date]

        rows = []
        for metric_cn, col in metrics:
            default_val = get_value(df_date, col, "Default")
            ai_val = get_value(df_date, col, "AI")
            pa_val = get_value(df_pa_date, col)
            rows.append(f"| {metric_cn} | {default_val} | {ai_val} | {pa_val:.4f} |")

        # Remove extra leading/trailing spaces in header lines and ensure no extra leading spaces in the first n rows
        lines = [
            f"SS01 AI调控上线后数据跟进 [{date}]",
            "统计时间：（北京时间）",
            "",
            f"| \U0001F4CA 统计指标 (Metric) | \U0001F9EA 对照组 (Control) | \U0001F916 AI组 (AI) | PA ({pa_date}) |",
            "| :--- | :--- | :--- | :--- |",
        ] + rows

        report = "\n" + "\n".join([line.strip() for line in lines]) + "\n"

        print(report)


if __name__ == "__main__":
    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/stats_by_date.parquet"
    df_hg = pd.read_parquet(file_path)

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/stats_by_day_pa.parquet"
    df_pa = pd.read_parquet(file_path)

    generate_daily_report(df_hg, df_pa)
