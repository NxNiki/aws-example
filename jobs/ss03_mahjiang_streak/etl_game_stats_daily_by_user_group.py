import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd

# Allow "jobs" package to be found when script is run directly (e.g. python jobs/ss03_mahjiang_streak/...)
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, ETLScheduler, RedshiftBackend
from jobs.etl_utils import AggCol, effective_start_date

# Day boundary: 6 AM Shanghai time (same as fish_hunter)
DATE_START_HOUR = 6

GAME_ID = "SS03"
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
        WITH user_bets AS (
        SELECT
            t.user_id,
            t.created_at,
            t.script_id AS mathtable,
            t.bet_amount,
            t.actual_payout AS payout,
            t.bet_type,
            t.actual_payout - t.bet_amount AS profit,
            CAST(DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_date,
            CAST(DATE_TRUNC('week', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_week,
            CAST(DATE_TRUNC('month', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_month,
            t.partition_ab[0] AS ab_group_id,
            t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS delta_t,
            LAG(t.bet_type) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS prev_bet_type,
            LAG(t.bet_amount) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS prev_bet_amount,
            COUNT(t.user_id) OVER (PARTITION BY t.user_id) AS user_bet_count,
            CASE
                WHEN LAG(t.script_id) OVER (PARTITION BY t.user_id ORDER BY t.created_at) IS NULL THEN 0
                WHEN LAG(t.script_id) OVER (PARTITION BY t.user_id ORDER BY t.created_at) <> t.script_id THEN 1
                ELSE 0
            END AS mathtable_change
        FROM
            public.fct_bet_orders AS t
        WHERE
            t.game_id = '{GAME_ID}'
            AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) >= '{effective_start}'
            AND t.currency_type = 'CNY'
            AND t.status = 'COMPLETED'
            AND t.op_code not in ('B26','TST','TSB','TSO') 
        ),

        user_bets_group AS (
            SELECT
                {_USER_BETS_GROUP_COLS}
                CASE
                    WHEN t.ab_group_id != '{AI_GROUP_ID}' THEN 'Default'
                    ELSE 'AI'
                END AS ai_group
            FROM
                user_bets AS t
        ),

        daily_login AS (
            SELECT 
                t.user_id,
                MIN(t.created_at) as first_bet_time,
                MAX(t.created_at) as last_bet_time,
                t.activity_date,
                t.activity_week,
                t.activity_month,
                t.ai_group
            FROM
                user_bets_group AS t
            GROUP BY t.user_id, t.activity_date, t.activity_week, t.activity_month, t.ai_group
        ),

        user_retention AS (
            SELECT
                t1.{stats_agg_col},
                t1.ai_group,
                COUNT(DISTINCT t1.user_id) AS day0_num_users,
                COUNT(DISTINCT t2.user_id) AS day1_num_users,
                COUNT(DISTINCT t3.user_id) AS day3_num_users
            FROM
                daily_login AS t1
            LEFT JOIN daily_login AS t2
                ON t1.user_id = t2.user_id AND t2.activity_date = DATEADD(day, 1, t1.activity_date)
            LEFT JOIN daily_login AS t3
                ON t1.user_id = t3.user_id AND t3.activity_date = DATEADD(day, 3, t1.activity_date)
            GROUP BY t1.{stats_agg_col}, t1.ai_group
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

                SUM(t.fg_session_trigger_bet) AS user_total_bet_fg_for_rtp,

                COUNT(CASE WHEN t.payout > 0 THEN 1 END) AS user_num_bets_with_payout,
                COUNT(CASE WHEN t.bet_type = 'BASE' AND t.payout > 0 THEN 1 END) AS user_num_bets_bg_with_payout,
                COUNT(CASE WHEN t.bet_type = 'FREE' AND t.payout > 0 THEN 1 END) AS user_num_bets_fg_with_payout,

                AVG(CASE WHEN EXTRACT(EPOCH FROM t.delta_t) BETWEEN 0 AND 86400 THEN EXTRACT(EPOCH FROM t.delta_t) END)
                    AS user_avg_delta_t_seconds,

                SUM(t.mathtable_change) AS user_mathtable_change

            FROM user_bets_group AS t
            WHERE t.user_bet_count >= 40
            GROUP BY t.{stats_agg_col}, t.ai_group, t.user_id
        ),

        group_stats AS (
            SELECT
                t.{stats_agg_col},
                t.ai_group,

                COUNT(DISTINCT t.user_id) AS num_active_users,
                COUNT(t.user_id) AS total_num_bets,
                COUNT(CASE WHEN t.bet_type = 'BASE' THEN t.user_id END) AS total_num_bets_bg,
                COUNT(CASE WHEN t.bet_type = 'FREE' THEN t.user_id END) AS total_num_bets_fg,

                -- total bet amount:
                SUM(t.bet_amount) AS total_bet,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS total_bet_bg,

                SUM(t.payout) AS total_payout,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS total_payout_bg,
                SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS total_payout_fg,

                SUM(t.fg_session_trigger_bet) AS total_bet_fg_for_rtp,

                COUNT(CASE WHEN t.payout > 0 THEN 1 END) AS total_num_bets_with_payout,
                COUNT(CASE WHEN t.bet_type = 'BASE' AND t.payout > 0 THEN 1 END) AS total_num_bets_bg_with_payout,
                COUNT(CASE WHEN t.bet_type = 'FREE' AND t.payout > 0 THEN 1 END) AS total_num_bets_fg_with_payout
            FROM user_bets_group AS t
            WHERE t.user_bet_count >= 40
            GROUP BY t.{stats_agg_col}, t.ai_group
        ),

        group_user_rtp_median AS (
            SELECT
                t.{stats_agg_col},
                t.ai_group,
                PERCENTILE_CONT(0.50) WITHIN GROUP (
                    ORDER BY t.user_total_payout * 1.0 / NULLIF(t.user_total_bet, 0)
                ) AS user_rtp_median
            FROM user_stats AS t
            GROUP BY t.{stats_agg_col}, t.ai_group
        ),

        group_no_fg AS (
            SELECT
                t.{stats_agg_col},
                t.ai_group,
                COUNT(*) AS num_active_users_no_fg
            FROM user_stats AS t
            WHERE t.user_num_bets_fg = 0
            GROUP BY t.{stats_agg_col}, t.ai_group
        ),

        group_0_rtp AS (
            SELECT
                t.{stats_agg_col},
                t.ai_group,
                COUNT(*) AS num_active_users_0_rtp
            FROM user_stats AS t
            WHERE t.user_total_payout = 0
            GROUP BY t.{stats_agg_col}, t.ai_group
        ),

        group_rtp_thresholds AS (
            SELECT
                t.{stats_agg_col},
                t.ai_group,
                COUNT(CASE WHEN t.user_total_payout * 1.0 / NULLIF(t.user_total_bet, 0) < 0.1 THEN 1 END) AS num_rtp_lt_01,
                COUNT(CASE WHEN t.user_total_payout * 1.0 / NULLIF(t.user_total_bet, 0) < 0.2 THEN 1 END) AS num_rtp_lt_02,
                COUNT(CASE WHEN t.user_total_payout * 1.0 / NULLIF(t.user_total_bet, 0) < 0.3 THEN 1 END) AS num_rtp_lt_03,
                COUNT(CASE WHEN t.user_total_payout * 1.0 / NULLIF(t.user_total_bet, 0) < 0.4 THEN 1 END) AS num_rtp_lt_04,
                COUNT(CASE WHEN t.user_total_payout * 1.0 / NULLIF(t.user_total_bet, 0) < 0.5 THEN 1 END) AS num_rtp_lt_05
            FROM user_stats t
            GROUP BY t.{stats_agg_col}, t.ai_group
        )

        SELECT
            us.{stats_agg_col} AS activity_date,
            us.ai_group,
            us.user_id,
            us.user_mathtable_change,

            gs.num_active_users,
            CASE WHEN us.user_num_bets_fg = 0 THEN 1 ELSE 0 END AS active_user_no_fg,
            COALESCE(gn.num_active_users_no_fg, 0) * 1.0 / NULLIF(gs.num_active_users, 0) AS active_user_no_fg_ratio,
            COALESCE(gn0.num_active_users_0_rtp, 0) AS num_active_user_0_rtp,
            COALESCE(gn0.num_active_users_0_rtp, 0) * 1.0 / NULLIF(gs.num_active_users, 0) AS active_user_0_rtp_ratio,
            COALESCE(gr.num_rtp_lt_01, 0) * 1.0 / NULLIF(gs.num_active_users, 0) AS active_user_rtp_less_0_1_ratio,
            COALESCE(gr.num_rtp_lt_02, 0) * 1.0 / NULLIF(gs.num_active_users, 0) AS active_user_rtp_less_0_2_ratio,
            COALESCE(gr.num_rtp_lt_03, 0) * 1.0 / NULLIF(gs.num_active_users, 0) AS active_user_rtp_less_0_3_ratio,
            COALESCE(gr.num_rtp_lt_04, 0) * 1.0 / NULLIF(gs.num_active_users, 0) AS active_user_rtp_less_0_4_ratio,
            COALESCE(gr.num_rtp_lt_05, 0) * 1.0 / NULLIF(gs.num_active_users, 0) AS active_user_rtp_less_0_5_ratio,

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
            us.user_total_payout_bg / NULLIF(us.user_total_bet, 0) AS user_rtp_bg,

            -- Free game RTP: payout of free games / trigger bet (bet before free game)
            us.user_total_payout_fg / NULLIF(us.user_total_bet_fg_for_rtp, 0) AS user_rtp_fg,
            gs.total_payout_fg / NULLIF(gs.total_bet_fg_for_rtp, 0) AS rtp_fg,

            -- Hit rate: proportion of bets with payout > 0
            us.user_num_bets_with_payout * 1.0 / NULLIF(us.user_num_bets, 0) AS user_hit_rate,
            gs.total_num_bets_with_payout * 1.0 / NULLIF(gs.total_num_bets, 0) AS hit_rate,
            us.user_num_bets_bg_with_payout * 1.0 / NULLIF(us.user_num_bets_bg, 0) AS user_hit_rate_bg,
            us.user_num_bets_fg_with_payout * 1.0 / NULLIF(us.user_num_bets_fg, 0) AS user_hit_rate_fg,
            gs.total_num_bets_bg_with_payout * 1.0 / NULLIF(gs.total_num_bets_bg, 0) AS hit_rate_bg,
            gs.total_num_bets_fg_with_payout * 1.0 / NULLIF(gs.total_num_bets_fg, 0) AS hit_rate_fg

        FROM user_stats AS us
        INNER JOIN group_stats AS gs
            ON us.{stats_agg_col} = gs.{stats_agg_col} AND us.ai_group = gs.ai_group
        LEFT JOIN group_no_fg AS gn
            ON us.{stats_agg_col} = gn.{stats_agg_col} AND us.ai_group = gn.ai_group
        LEFT JOIN group_0_rtp AS gn0
            ON us.{stats_agg_col} = gn0.{stats_agg_col} AND us.ai_group = gn0.ai_group
        LEFT JOIN group_rtp_thresholds AS gr
            ON us.{stats_agg_col} = gr.{stats_agg_col} AND us.ai_group = gr.ai_group
        LEFT JOIN group_user_rtp_median AS gm
            ON us.{stats_agg_col} = gm.{stats_agg_col} AND us.ai_group = gm.ai_group
        INNER JOIN user_retention AS ur
            ON us.{stats_agg_col} = ur.{stats_agg_col} AND us.ai_group = ur.ai_group
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
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_ss03_mahjiang_streak",
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
