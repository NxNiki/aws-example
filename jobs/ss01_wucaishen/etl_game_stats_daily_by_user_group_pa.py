import argparse
import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, TIMEZONE_SHANGHAI, setup_logging
from bituslabs_ds.etl import AthenaBackend, DataLoader, ETLScheduler


def generate_query(stats_agg_col: str, start_date: str) -> str:
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
            date_trunc('week', from_unixtime(t.billtime / 1.0E9)
                AT TIME ZONE 'UTC' 
                AT TIME ZONE '{TIMEZONE_SHANGHAI}' - INTERVAL '6' HOUR) AS activity_week,
            date_trunc('month', from_unixtime(t.billtime / 1.0E9)
                AT TIME ZONE 'UTC' 
                AT TIME ZONE '{TIMEZONE_SHANGHAI}' - INTERVAL '6' HOUR) AS activity_month,
            t.billtime / 1.0E9 - LAG(t.billtime / 1.0E9) OVER (PARTITION BY t.loginname ORDER BY t.billtime) AS delta_t,
            COUNT(t.loginname) OVER (PARTITION BY t.loginname) AS user_bet_count
        FROM
            ag_share_data.slotorders AS t
        WHERE
            t.currency = 'CNY'
            AND gametype = 'SB28' 
            AND flag != -8.0
            AND date_trunc('day', from_unixtime(t.billtime / 1.0E9)
                AT TIME ZONE 'UTC' 
                AT TIME ZONE '{TIMEZONE_SHANGHAI}' - INTERVAL '6' HOUR) > CAST('{start_date}' AS timestamp)
        ),

        daily_login AS (
            SELECT DISTINCT
                t.activity_date,
                t.activity_week,
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
                ON t2.activity_date = DATE_ADD('day', 1, t1.activity_date) AND t1.user_id = t2.user_id
            LEFT JOIN daily_login AS t3
                ON t3.activity_date = DATE_ADD('day', 3, t1.activity_date) AND t1.user_id = t3.user_id
            GROUP BY t1.{stats_agg_col}
        ),


        user_stats AS (
            SELECT
                t.{stats_agg_col},
                t.user_id,

                -- number of users:
                COUNT(DISTINCT t.user_id) AS total_daily_users,

                -- total number of bets:
                COUNT(t.user_id) AS user_num_bets,
                COUNT(CASE WHEN t.bet_type = 1 THEN t.user_id END) AS user_num_bets_bg,
                COUNT(CASE WHEN t.bet_type = 2 THEN t.user_id END) AS user_num_bets_fg,

                -- total bet amount:
                SUM(t.bet_amount) AS user_total_bet,
                SUM(CASE WHEN t.bet_type = 1 THEN t.bet_amount END) AS user_total_bet_bg,

                -- total payout amount:
                SUM(t.payout) AS user_total_payout,
                SUM(CASE WHEN t.bet_type = 1 THEN t.payout END) AS user_total_payout_bg,
                SUM(CASE WHEN t.bet_type = 2 THEN t.payout END) AS user_total_payout_fg,
                
                AVG(CASE WHEN t.delta_t BETWEEN 0 AND 86400 THEN t.delta_t END)
                    AS user_avg_delta_t_seconds
                
            FROM user_bets AS t
            WHERE t.user_bet_count >= 40
            GROUP BY t.{stats_agg_col}, t.user_id
        ),

        group_stats AS (
            SELECT
                t.{stats_agg_col},
                COUNT(DISTINCT t.user_id) AS num_active_users,
                COUNT(t.user_id) AS total_num_bets,
                COUNT(CASE WHEN t.bet_type = 1 THEN t.user_id END) AS total_num_bets_bg,
                COUNT(CASE WHEN t.bet_type = 2 THEN t.user_id END) AS total_num_bets_fg,

                -- total bet amount:
                SUM(t.bet_amount) AS total_bet,
                SUM(CASE WHEN t.bet_type = 1 THEN t.bet_amount END) AS total_bet_bg,

                SUM(t.payout) AS total_payout,
                SUM(CASE WHEN t.bet_type = 1 THEN t.payout END) AS total_payout_bg,
                SUM(CASE WHEN t.bet_type = 2 THEN t.payout END) AS total_payout_fg
            FROM user_bets AS t
            WHERE t.user_bet_count >= 40
            GROUP BY t.{stats_agg_col}
        )
        ,

        group_user_rtp_median AS (
            SELECT
                t.{stats_agg_col},
                PERCENTILE_CONT(0.50) WITHIN GROUP (
                    ORDER BY t.user_total_payout * 1.0 / NULLIF(t.user_total_bet, 0)
                ) AS user_rtp_median
            FROM user_stats AS t
            GROUP BY t.{stats_agg_col}
        )

        SELECT
            CAST(us.{stats_agg_col} AS DATE) AS activity_date,
            'PA' AS ai_group,
            us.user_id,
            0 AS user_mathtable_change,

            gs.num_active_users,

            gs.total_num_bets,
            gs.total_num_bets_bg,
            gs.total_num_bets_fg,
            gs.total_num_bets * 1.0 / NULLIF(gs.num_active_users, 0) AS total_num_bets_per_user,

            gs.total_bet,
            gs.total_bet_bg,
            gs.total_bet * 1.0 / NULLIF(gs.num_active_users, 0) AS total_bet_per_user,

            gs.total_payout,
            gs.total_payout_bg,
            gs.total_payout_fg,

            us.user_avg_delta_t_seconds,

            -- Retention:
            ur.day0_num_users,
            ur.day1_num_users,
            ur.day3_num_users,
            ur.day1_num_users * 1.0 / NULLIF(ur.day0_num_users, 0) AS retention_rate_day1,
            ur.day3_num_users * 1.0 / NULLIF(ur.day0_num_users, 0) AS retention_rate_day3,

            -- Number of bets and total bet per user:
            us.user_num_bets,
            us.user_num_bets_bg,
            us.user_num_bets_fg,

            us.user_total_bet,

            -- free game ratio
            us.user_num_bets_fg * 1.0 / NULLIF(us.user_num_bets, 0) AS user_fg_ratio,
            gs.total_num_bets_fg * 1.0 / NULLIF(gs.total_num_bets, 0) AS fg_ratio,

            -- Profit Calculations
            (gs.total_payout - gs.total_bet) AS total_profit,
            (gs.total_payout - gs.total_bet) * 1.0 / gs.num_active_users AS total_profit_per_user,
            (us.user_total_payout - us.user_total_bet) AS user_total_profit,

            -- RTP (Return to Player) Calculations (Fix: NULLIF for bet amounts AND RTP numerator error)
            gs.total_payout / NULLIF(gs.total_bet, 0) AS rtp,
            gm.user_rtp_median,
            gm.user_rtp_median / NULLIF(gs.total_payout / NULLIF(gs.total_bet, 0), 0) AS user_rtp_ultilization_ratio,
            -- total bet for base game is same to total bet and free game has 0 bet amount:
            gs.total_payout_bg / NULLIF(gs.total_bet, 0) AS rtp_bg,
            us.user_total_payout / NULLIF(us.user_total_bet, 0) AS user_rtp,
            us.user_total_payout_bg / NULLIF(us.user_total_bet, 0) AS user_rtp_bg

        FROM user_stats AS us
        INNER JOIN group_stats AS gs
            ON us.{stats_agg_col} = gs.{stats_agg_col} 
        LEFT JOIN group_user_rtp_median AS gm
            ON us.{stats_agg_col} = gm.{stats_agg_col}
        INNER JOIN user_retention AS ur
            ON us.{stats_agg_col} = ur.{stats_agg_col} 
        ORDER BY us.{stats_agg_col} DESC, us.user_id DESC;
        """
    )

    return query


def execute_query(output_file, stats_agg_col):

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/{output_file}.parquet"
    query = generate_query(stats_agg_col)
    df_rs = data_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Game Stats Daily by User Group (Athena)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing S3/local output (full reload from default start date)",
    )
    args = parser.parse_args()

    data_loader = DataLoader(
        backend=AthenaBackend(
            database="agfish",
            output_location=f"s3://{S3_BUCKET}/ds-data-pa_wucaishen/",
        )
    )

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    # Initialize Scheduler
    scheduler = ETLScheduler(
        data_loader=data_loader,
        storage_root=f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen",
        lookback_days=3,
        overwrite=args.overwrite,
    )

    scheduler.run_incremental_job(
        job_name="daily_stats_pa",
        query_func=lambda start_date: generate_query("activity_date", start_date),
        key_cols=["activity_date", "user_id", "ai_group"],
        date_col="activity_date",
        partition_level="none",
    )

    # Overrides to 7 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="weekly_stats_pa",
        query_func=lambda start_date: generate_query("activity_week", start_date),
        key_cols=["activity_date", "user_id", "ai_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=7,
    )

    # Overrides to 31 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="monthly_stats_pa",
        query_func=lambda start_date: generate_query("activity_month", start_date),
        key_cols=["activity_date", "user_id", "ai_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=31,
    )

    data_loader.close()
