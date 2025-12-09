from textwrap import dedent

import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT


def generate_daily_report(df: pd.DataFrame, row=1):

    report = dedent(
        f"""
        SS01 AI调控上线后数据跟进 [{df['activity_date'][row]}]
        统计时间：（北京时间）

        | \U0001F4CA 统计指标 (Metric) | \U0001F9EA 对照组 (Control) | \U0001F916 AI组 (AI) | PA ()
        | :--- | :--- | :--- | :--- |
        | 玩家数量（投注次数 >= 40）| {df['total_daily_users'][row]} | {df['ai_group_users'][row]} |
        | 玩家数量（All）| {df['day0_num_users'][row]} | {df['ai_day0_num_users'][row]} |
        | 前一日留存玩家数量 (Day 1 Retention) | {df['day1_num_users'][row+1]} | {df['ai_day1_num_users'][row+1]} |
        | 前三日留存玩家数量 (Day 3 Retention) | {df['day1_num_users'][row+3]} | {df['ai_day1_num_users'][row+3]} | 
        | 玩家平均投注次数 (Avg. Bets/User) | {df['default_num_bets_per_user'][row]} | {df['ai_num_bets_per_user'][row]} |
        | 玩家平均投注总额度 (Avg. Total Bet/User) | {df['default_total_bet_per_user'][row]} | {df['ai_total_bet_per_user'][row]} |
        | RTP (Return to Player) | {df['default_rtp'][row]} | {df['ai_rtp'][row]} |
        """
    )

    print(report)


if __name__ == "__main__":
    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/stats_by_date.parquet"
    df_hg = pd.read_parquet(file_path)

    generate_daily_report(df_hg, row=1)
    generate_daily_report(df_hg, row=2)
    generate_daily_report(df_hg, row=3)
