from textwrap import dedent
from typing import List

import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT


def generate_daily_report(df: pd.DataFrame, dates: List[str]) -> None:

    def get_group_value(df, group, col):
        vals = df.loc[df["ai_group"] == group, col]
        return vals.values[0] if not vals.empty else ""

    metrics = [
        ("玩家数量（投注次数 >= 40）", "num_active_users"),
        ("玩家数量（All）", "day0_num_users"),
        ("前一日留存玩家数量 (Day 1 Retention)", "day1_num_users"),
        ("前三日留存玩家数量 (Day 3 Retention)", "day3_num_users"),
        ("玩家平均投注次数 (Avg. Bets/User)", "num_bets_per_user"),
        ("玩家平均投注总额度 (Avg. Total Bet/User)", "total_bet_per_user"),
        ("RTP (Return to Player)", "rtp"),
    ]

    for date in dates:
        df_date = df[df["activity_date"].astype(str) == date]

        rows = []
        for metric_cn, col in metrics:
            default_val = get_group_value(df_date, "Default", col)
            ai_val = get_group_value(df_date, "AI", col)
            rows.append(f"| {metric_cn} | {default_val} | {ai_val} |")

        # Remove extra leading/trailing spaces in header lines and ensure no extra leading spaces in the first n rows
        lines = [
            f"SS01 AI调控上线后数据跟进 [{date}]",
            "统计时间：（北京时间）",
            "",
            "| \U0001F4CA 统计指标 (Metric) | \U0001F9EA 对照组 (Control) | \U0001F916 AI组 (AI) | PA () |",
            "| :--- | :--- | :--- | :--- |",
        ] + rows

        report = "\n".join([line.strip() for line in lines])

        print(report)


if __name__ == "__main__":
    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/stats_by_date.parquet"
    df_hg = pd.read_parquet(file_path)

    generate_daily_report(df_hg, dates=["2025-12-09"])
