import argparse
import os
from textwrap import dedent

import pandas as pd

from bituslabs_ds.config import (
    DATE_START_HOUR,
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    ETL_CURRENCY_CODES,
    ETL_EXCLUDED_OP_CODES,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    TIMEZONE_SHANGHAI,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import AggCol, DataLoader, ETLScheduler, RedshiftBackend, effective_start_date

GAME_ID = "SS02"
AI_GROUP_ID = "jojpin-9mokha-rexQug"

# Shared column list for user_bets_group UNION (reused across group variants)
_USER_BETS_GROUP_COLS = """
                t.created_at,
                t.activity_date,
                t.activity_week,
                t.activity_month,
                t.user_id,
                t.bet_amount,
                t.payout,
                t.bet_type,
                t.profit,
                t.delta_t,
                t.user_bet_count,
                t.mathtable_change,
                t.prev_bet_type,
                t.prev_bet_amount,
                CASE WHEN t.bet_type = 'FREE' AND t.prev_bet_type = 'BASE' THEN t.prev_bet_amount END AS fg_session_trigger_bet,
                """

# TODO:
# add max/min user daily profit


def generate_query(stats_agg_col: AggCol, start_date: str):
    effective_start = effective_start_date(stats_agg_col, start_date)
    query = dedent(
        f"""
        WITH bet_events AS (
            SELECT
                t.user_id,
                t.created_at,
                CASE
                    WHEN t.math_table_id = 'FourScatter'
                    THEN LEAD(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.created_at)
                    ELSE t.math_table_id
                END AS math_table_id,
                CASE WHEN t.math_table_id = 'FourScatter' THEN 1 ELSE 0 END AS buy_free_game,
                t.bet_amount,
                t.actual_payout,
                t.bet_type,
                t.partition_ab
            FROM
                public.fct_bet_orders AS t
            WHERE
                t.game_id = '{GAME_ID}'
                AND CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at) >= '{effective_start}'
                AND t.currency_type IN {ETL_CURRENCY_CODES}
                AND t.status = 'COMPLETED'
                AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
        ),

        user_bets AS (
        SELECT
            t.user_id,
            t.created_at,
            t.math_table_id AS mathtable,
            t.buy_free_game,
            t.bet_amount,
            t.actual_payout AS payout,
            t.bet_type,
            t.actual_payout - t.bet_amount AS profit,
            CAST(DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_date,
            CAST(DATE_TRUNC('week', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_week,
            CAST(DATE_TRUNC('month', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_month,
            t.partition_ab[0] AS ab_group_id,
            t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS delta_t,
            LAG(t.bet_type) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS prev_bet_type,
            LAG(t.bet_amount) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS prev_bet_amount,
            COUNT(t.user_id) OVER (PARTITION BY t.user_id) AS user_bet_count,
            CASE
                WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.created_at) IS NULL THEN 0
                WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.created_at) <> t.math_table_id THEN 1
                ELSE 0
            END AS mathtable_change
        FROM
            bet_events AS t
        ),

        user_bets_group AS (
            SELECT
                {_USER_BETS_GROUP_COLS}
                CASE
                    WHEN t.ab_group_id = '{AI_GROUP_ID}' THEN 'AI'
                    ELSE 'Default'
                END AS ai_group
            FROM
                user_bets AS t

            UNION ALL

            SELECT
                {_USER_BETS_GROUP_COLS}
                CASE
                    WHEN t.ab_group_id != '{AI_GROUP_ID}' THEN CONCAT('Default_', t.mathtable)
                END AS ai_group
            FROM
                user_bets AS t
                WHERE t.ab_group_id != '{AI_GROUP_ID}'

            UNION ALL

            SELECT
                {_USER_BETS_GROUP_COLS}
                t.mathtable AS ai_group
            FROM
                user_bets AS t
            WHERE
                t.ab_group_id = '{AI_GROUP_ID}'
        ),

        user_stats AS (
            SELECT
                t.{stats_agg_col},
                t.ai_group,
                t.user_id,

                -- total number of bets:
                COUNT(t.user_id) AS user_num_bets,
                COUNT(CASE WHEN t.bet_type = 'BASE' THEN t.user_id END) AS user_num_bets_bg,
                COUNT(CASE WHEN t.bet_type = 'FREE' THEN t.user_id END) AS user_num_bets_fg,

                -- total bet amount:
                SUM(t.bet_amount) AS user_total_bet,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS user_total_bet_bg,

                -- total payout amount:
                SUM(t.payout) AS user_total_payout,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS user_total_payout_bg,
                SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS user_total_payout_fg,

                SUM(t.fg_session_trigger_bet) AS user_total_bet_fg,

                COUNT(CASE WHEN t.payout > 0 THEN 1 END) AS user_num_bets_with_payout,
                COUNT(CASE WHEN t.bet_type = 'BASE' AND t.payout > 0 THEN 1 END) AS user_num_bets_bg_with_payout,
                COUNT(CASE WHEN t.bet_type = 'FREE' AND t.payout > 0 THEN 1 END) AS user_num_bets_fg_with_payout,

                AVG(CASE WHEN EXTRACT(EPOCH FROM t.delta_t) BETWEEN 0 AND 86400 THEN EXTRACT(EPOCH FROM t.delta_t) END)
                    AS user_avg_delta_t_seconds,

                SUM(t.mathtable_change) AS user_mathtable_change

            FROM user_bets_group AS t
            GROUP BY t.{stats_agg_col}, t.ai_group, t.user_id
        ),

        user_first_bet AS (
            SELECT
                user_id,
                MIN(CAST(DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', created_at))) AS DATE)) AS first_bet_date
            FROM public.fct_bet_orders
            WHERE game_id = '{GAME_ID}'
              AND currency_type IN {ETL_CURRENCY_CODES}
              AND status = 'COMPLETED'
              AND op_code NOT IN {ETL_EXCLUDED_OP_CODES}
            GROUP BY user_id
        )

        SELECT
            us.{stats_agg_col} AS activity_date,
            us.ai_group,
            us.user_id,
            us.user_mathtable_change,
            CASE
                WHEN DATEDIFF('day', fb.first_bet_date, us.{stats_agg_col}) <= 3 THEN 'new'
                WHEN DATEDIFF('day', fb.first_bet_date, us.{stats_agg_col}) <= 7 THEN 'beginner'
                ELSE 'old'
            END AS user_group,

            CASE
                WHEN DATEDIFF('day', fb.first_bet_date, us.{stats_agg_col}) < 1 THEN 'day0_user'
                WHEN DATEDIFF('day', fb.first_bet_date, us.{stats_agg_col}) < 7 THEN 'day1-6_user'
                ELSE 'day7+_user'
            END AS user_group2,

            -- DataMetrics input columns (user-level raw stats):
            us.user_num_bets,
            us.user_num_bets_bg,
            us.user_num_bets_fg,
            us.user_total_bet,
            us.user_total_bet_bg,
            us.user_total_payout,
            us.user_total_payout_bg,
            us.user_total_payout_fg,
            us.user_total_bet_fg,
            us.user_num_bets_with_payout,
            us.user_num_bets_bg_with_payout,
            us.user_num_bets_fg_with_payout,
            us.user_total_payout * 1.0 / NULLIF(us.user_total_bet, 0) AS user_rtp,

            -- User-level derived (not computed by DataMetrics):
            us.user_avg_delta_t_seconds,
            us.user_num_bets_fg * 1.0 / NULLIF(us.user_num_bets, 0) AS user_fg_ratio,
            (us.user_total_payout - us.user_total_bet) AS user_total_profit,
            us.user_total_payout_bg * 1.0 / NULLIF(us.user_total_bet, 0) AS user_rtp_bg,
            us.user_total_payout_fg * 1.0 / NULLIF(us.user_total_bet_fg, 0) AS user_rtp_fg,
            us.user_num_bets_with_payout * 1.0 / NULLIF(us.user_num_bets, 0) AS user_hit_rate,
            us.user_num_bets_bg_with_payout * 1.0 / NULLIF(us.user_num_bets_bg, 0) AS user_hit_rate_bg,
            us.user_num_bets_fg_with_payout * 1.0 / NULLIF(us.user_num_bets_fg, 0) AS user_hit_rate_fg

        FROM user_stats AS us
        LEFT JOIN user_first_bet AS fb ON us.user_id = fb.user_id
        WHERE us.{stats_agg_col} >= '{effective_start}'
        ORDER BY us.{stats_agg_col} DESC, us.ai_group DESC, us.user_id DESC;

        """
    )

    return query


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(description="ETL Game Stats Daily by User Group")
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP address for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing S3/local output (full reload from default start date)",
    )
    args = parser.parse_args()

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="slot-machine",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=args.bastion_ip,
        )
    )

    # Initialize Scheduler with a default 3-day lookback
    scheduler = ETLScheduler(
        redshift_loader,
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_ss02_deepdive",
        lookback_days=3,
        overwrite=args.overwrite,
    )

    scheduler.run_incremental_job(
        job_name="daily_stats",
        query_func=lambda start_date: generate_query("activity_date", start_date),
        key_cols=["activity_date", "user_id", "ai_group"],
        date_col="activity_date",
        partition_level="none",
    )

    # Overrides to 7 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="weekly_stats",
        query_func=lambda start_date: generate_query("activity_week", start_date),
        key_cols=["activity_date", "user_id", "ai_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=7,
    )

    # Overrides to 31 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="monthly_stats",
        query_func=lambda start_date: generate_query("activity_month", start_date),
        key_cols=["activity_date", "user_id", "ai_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=31,
    )

    redshift_loader.close()
