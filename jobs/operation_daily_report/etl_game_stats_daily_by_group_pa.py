"""
ETL game stats daily by group PA (Athena).
"""

from pathlib import Path
from textwrap import dedent

from bituslabs_ds.config import (
    ETL_CURRENCY_CODES,
    LOCAL_ROOT,
    S3_BUCKET,
    TIMEZONE_SHANGHAI,
    setup_logging,
)
from bituslabs_ds.etl import AthenaBackend, DataLoader

stats_agg_col = "activity_date"
output_file = "stats_by_day_pa"

query = dedent(
    f"""
    WITH user_bets AS (
    SELECT
        t.loginname AS user_id,
        t.account AS bet_amount,
        t.account + t.cus_account AS payout,
        t.slottype AS bet_type,
        t.cus_account AS profit,
        date_trunc('day', from_unixtime(t.billtime / 1.0E9) 
            AT TIME ZONE 'UTC' 
            AT TIME ZONE '{TIMEZONE_SHANGHAI}' - INTERVAL '6' HOUR) AS activity_date,
        date_trunc('month', from_unixtime(t.billtime / 1.0E9) 
            AT TIME ZONE 'UTC' 
            AT TIME ZONE '{TIMEZONE_SHANGHAI}' - INTERVAL '6' HOUR) AS activity_month,
        t.billtime / 1.0E9 - LAG(t.billtime / 1.0E9) OVER (PARTITION BY t.loginname ORDER BY t.billtime) AS delta_t,
        COUNT(t.loginname) OVER (PARTITION BY t.loginname) AS user_bet_count
    FROM
        ag_share_data.slotorders AS t
    WHERE
        t.currency IN {ETL_CURRENCY_CODES}
        AND gametype = 'SB28' 
        AND flag != -8.0
    ),

    daily_login AS (
        SELECT DISTINCT
            t.activity_date,
            t.activity_month,
            t.user_id
        FROM
            user_bets AS t
    ),

    user_retention AS (
        SELECT
            t1.{stats_agg_col},
            COUNT(DISTINCT t1.user_id) AS day0_num_users,
            COUNT(DISTINCT t2.user_id) AS day1_num_users,
            COUNT(DISTINCT t3.user_id) AS day3_num_users
        FROM
            daily_login AS t1
        LEFT JOIN daily_login AS t2
            ON t2.activity_date = DATE_ADD('day', -1, t1.activity_date) AND t1.user_id = t2.user_id
        LEFT JOIN daily_login AS t3
            ON t3.activity_date = DATE_ADD('day', -3, t1.activity_date) AND t1.user_id = t3.user_id
        GROUP BY t1.{stats_agg_col}
    ),

    daily_stats AS (
        SELECT
            t.{stats_agg_col},
            COUNT(DISTINCT t.user_id) AS total_daily_users,
            COUNT(t.user_id) AS num_bets,
            COUNT(CASE WHEN t.bet_type = 1 THEN t.user_id END) AS num_bets_bg,
            COUNT(CASE WHEN t.bet_type = 2 THEN t.user_id END) AS num_bets_fg,
            SUM(t.bet_amount) AS total_bet,
            SUM(CASE WHEN t.bet_type = 1 THEN t.bet_amount END) AS total_bet_bg,
            SUM(t.payout) AS total_payout,
            SUM(CASE WHEN t.bet_type = 1 THEN t.payout END) AS total_payout_bg,
            SUM(CASE WHEN t.bet_type = 2 THEN t.payout END) AS total_payout_fg,
            AVG(CASE WHEN t.delta_t BETWEEN 0 AND 86400 THEN t.delta_t END) AS avg_delta_t_seconds
        FROM user_bets AS t
        WHERE t.user_bet_count >= 40
        GROUP BY t.{stats_agg_col}
    )

    SELECT
        CAST(ds.{stats_agg_col} AS DATE) AS {stats_agg_col},
        'PA' AS ai_group,
        ds.total_daily_users AS num_active_users,
        ds.total_bet,
        ds.total_bet_bg,
        ds.total_payout,
        ds.total_payout_bg,
        ds.total_payout_fg,
        ds.avg_delta_t_seconds,
        ur.day0_num_users,
        ur.day1_num_users,
        ur.day3_num_users,
        ds.num_bets_fg * 1.0 / NULLIF(ds.num_bets, 0) AS fg_ratio,
        ds.num_bets / NULLIF(ds.total_daily_users, 0) AS num_bets_per_user,
        ds.total_bet / NULLIF(ds.total_daily_users, 0) AS total_bet_per_user,
        (ds.total_payout - ds.total_bet) AS total_profit,
        (ds.total_payout - ds.total_bet) / NULLIF(ds.total_daily_users, 0) AS profit_per_user,
        ds.total_payout / NULLIF(ds.total_bet, 0) AS rtp,
        ds.total_payout_bg / NULLIF(ds.total_bet, 0) AS rtp_bg
    FROM daily_stats AS ds
    INNER JOIN user_retention AS ur
        ON ds.{stats_agg_col} = ur.{stats_agg_col}
    ORDER BY ds.{stats_agg_col} DESC;
    """
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true", help="Force re-run query (ignore local cache)")
    args = parser.parse_args()

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=f"ss01_etl_{output_file}.log")

    data_loader = DataLoader(
        backend=AthenaBackend(
            database="agfish",
            output_location=f"s3://{S3_BUCKET}/ds-data-ss01/{output_file}",
        )
    )

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{output_file}.parquet"
    df_rs = data_loader.query_to_df(query=query, local_cache=str(file_path), reload=args.reload)
    print(df_rs)
    data_loader.close()
