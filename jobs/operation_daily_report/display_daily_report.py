"""
Display SS01 daily report (Control vs AI vs PA).
"""

import argparse
from pathlib import Path

import pandas as pd

from bituslabs_ds.config import DATE_START_HOUR


def generate_daily_report(df: pd.DataFrame, df_pa: pd.DataFrame, lookback_days: int = 7) -> str:
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
        ("活跃玩家平均投注次数 (Avg. Bets/User)", "num_bets_per_user"),
        ("活跃玩家平均投注总额度 (Avg. Total Bet/User)", "total_bet_per_user"),
        ("RTP (Return to Player)", "rtp"),
    ]

    dates = sorted(df["activity_date"].unique(), reverse=True)
    # dates[0] is most recent; we skip it (dates[1] is yesterday). Show last lookback_days.
    chunks = []
    for date in dates[1 : lookback_days + 1]:
        df_date = df[df["activity_date"] == date]
        pa_date = date - pd.DateOffset(years=1)
        df_pa_date = df_pa[df_pa["activity_date"] == pa_date]

        rows = []
        for i, (metric_cn, col) in enumerate(metrics):
            default_val = get_value(df_date, col, "Default")
            ai_val = get_value(df_date, col, "AI")
            pa_val = get_value(df_pa_date, col)
            if i > 4:
                try:
                    pa_fmt = f"{float(pa_val):.4f}" if pa_val != "" else pa_val
                except (TypeError, ValueError):
                    pa_fmt = str(pa_val)
                rows.append(f"| {metric_cn} | {default_val} | {ai_val} | {pa_fmt} |")
            else:
                rows.append(f"| {metric_cn} | {default_val} | {ai_val} | {pa_val} |")

        date = date + pd.Timedelta(hours=DATE_START_HOUR)
        pa_date = pa_date + pd.Timedelta(hours=DATE_START_HOUR)
        lines = [
            f"SS01 AI调控上线后数据跟进 [{date}]",
            "统计时间：（北京时间）",
            "",
            f"| \U0001F4CA 统计指标 (Metric) | \U0001F9EA 对照组 (Control) | \U0001F916 AI组 (AI) | PA ({pa_date}) |",
            "| :--- | :--- | :--- | :--- |",
        ] + rows

        report = "\n" + "\n".join([line.strip() for line in lines]) + "\n"
        chunks.append(report)
    return "\n".join(chunks)


def _load_data(output_dir: Path):
    """Load HG and PA data from output_dir."""
    hg_path = output_dir / "stats_by_date.parquet"
    pa_path = output_dir / "stats_by_day_pa.parquet"
    df_hg = pd.read_parquet(hg_path)
    df_pa = pd.read_parquet(pa_path)
    return df_hg, df_pa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=7, help="Number of past days to report (default: 7)")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "output"
    df_hg, df_pa = _load_data(output_dir)
    report = generate_daily_report(df_hg, df_pa, lookback_days=args.lookback_days)
    print(report)
