"""Per-user game stats for SS06 (pocket soccer), by period, ab_group and mathtable.

ETL job: writes output_ss06_pocket_soccer_v2/{daily,weekly,monthly}_stats
(the ``user_*`` metrics consumed by the SS06 dashboard / DataMetrics).
``stats_agg_col`` selects the period grain: activity_date / _week / _month.

Rows are one per (period, user, ab_group, mathtable, bet_level) — every bet
lives in exactly one row, so sums are correct under ANY selection and no label ever
re-partitions the same bets. The dashboard exposes ab_group and mathtable as
two separate cohort dimensions; when a dimension is unselected, the API
collapses the grain back to one row per user (``user_row_grain`` in the
config + ``collapse_user_rows``), summing the additive components and
recomputing ratios. ``user_num_delta_bet`` / ``user_num_delta_t_bg`` are the
exact denominators that make the AVG-type columns recombinable.
"""

import argparse
import os
from textwrap import dedent

from bituslabs_ds.config import (
    AB_TEST_GROUP_A,
    AB_TEST_GROUP_B,
    AI_GROUP_ID,
    DATE_START_HOUR,
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    ETL_CURRENCY_CODES,
    ETL_DELTA_T_MAX_SECONDS,
    ETL_DELTA_T_MIN_SECONDS,
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

GAME_ID = "SS06"


def generate_query(stats_agg_col: AggCol, start_date: str):
    effective_start = effective_start_date(stats_agg_col, start_date)
    query = dedent(
        f"""
        WITH bets AS (
        SELECT
            t.user_id,
            t.spin_id,
            t.created_at,
            COALESCE(NULLIF(t.math_table_id, ''), '(none)') AS mathtable,
            t.bet_amount,
            t.actual_payout AS payout,
            t.bet_type,
            t.actual_payout - t.bet_amount AS profit,
            CAST(DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_date,
            CAST(DATE_TRUNC('week', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_week,
            CAST(DATE_TRUNC('month', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_month,
            CASE
                WHEN t.partition_ab[0] = '{AI_GROUP_ID}' THEN 'AI'
                WHEN t.partition_ab[0] = '{AB_TEST_GROUP_A}' THEN 'AB_TEST_A'
                WHEN t.partition_ab[0] = '{AB_TEST_GROUP_B}' THEN 'AB_TEST_B'
                ELSE 'Default'
            END AS ab_group
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
            -- Sequence metrics (delta_t / delta_bet / mathtable_change / FG
            -- trigger) are DAY-partitioned: each activity_date is
            -- self-contained, so incremental pulls, full reloads, and ad-hoc
            -- SQL over any date range agree exactly, and weekly/monthly
            -- sequence metrics equal the sum of their days. The first bet of a
            -- day has no predecessor by definition (a session crossing the day
            -- boundary contributes no delta to the new day).
            SELECT
                t.*,
                EXTRACT(EPOCH FROM (t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at))) AS delta_t_seconds,
                LAG(t.bet_type) OVER (PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at) AS prev_bet_type,
                LAG(t.bet_amount) OVER (PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at) AS prev_bet_amount,
                -- Bet-level grain: BASE spins use their own bet_amount; FREE
                -- spins inherit the day's prevailing BASE bet (their trigger),
                -- so FG metrics stay meaningful inside a bet-level range. A
                -- FREE spin with no prior BASE that day keeps its own amount.
                COALESCE(
                    LAST_VALUE(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END IGNORE NULLS) OVER (
                        PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ),
                    t.bet_amount
                ) AS bet_level,
                CASE
                    WHEN LAG(t.mathtable) OVER (PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at) IS NULL THEN 0
                    WHEN LAG(t.mathtable) OVER (PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at) <> t.mathtable THEN 1
                    ELSE 0
                END AS mathtable_change
            FROM bets AS t
        ),

        user_stats AS (
            SELECT
                t.{stats_agg_col},
                t.ab_group,
                t.mathtable,
                t.bet_level,
                t.user_id,

                -- total number of bets:
                COUNT(t.user_id) AS user_num_bets,
                COUNT(CASE WHEN t.bet_type = 'BASE' THEN t.user_id END) AS user_num_bets_bg,
                COUNT(CASE WHEN t.bet_type = 'FREE' THEN t.user_id END) AS user_num_bets_fg,

                -- total bet amount:
                SUM(t.bet_amount) AS user_total_bet,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS user_total_bet_bg,
                AVG(t.bet_amount) AS user_avg_bet_amount,

                -- total payout amount:
                SUM(t.payout) AS user_total_payout,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS user_total_payout_bg,
                SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS user_total_payout_fg,

                SUM(CASE WHEN t.bet_type = 'FREE' AND t.prev_bet_type = 'BASE' THEN t.prev_bet_amount END) AS user_total_bet_fg,

                COUNT(CASE WHEN t.payout > 0 THEN 1 END) AS user_num_bets_with_payout,
                COUNT(CASE WHEN t.bet_type = 'BASE' AND t.payout > 0 THEN 1 END) AS user_num_bets_bg_with_payout,
                COUNT(CASE WHEN t.bet_type = 'FREE' AND t.payout > 0 THEN 1 END) AS user_num_bets_fg_with_payout,

                -- delta-t between consecutive BASE bets: AVG plus its exact
                -- denominator so the average recombines across grain rows.
                AVG(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= {ETL_DELTA_T_MAX_SECONDS} THEN GREATEST(t.delta_t_seconds, {ETL_DELTA_T_MIN_SECONDS}) END)
                    AS user_avg_delta_t_seconds_bg,
                COUNT(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= {ETL_DELTA_T_MAX_SECONDS} THEN 1 END)
                    AS user_num_delta_t_bg,

                SUM(t.mathtable_change) AS user_mathtable_change,

                -- delta bet amount metrics:
                SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet,
                SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet,
                AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet_avg,
                AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet_avg,
                SUM(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet,
                AVG(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet_avg,
                COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN 1 END) AS user_pos_delta_bet_num,
                COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN 1 END) AS user_neg_delta_bet_num,
                COUNT(t.prev_bet_amount) AS user_num_delta_bet

            FROM user_bets AS t
            GROUP BY t.{stats_agg_col}, t.ab_group, t.mathtable, t.bet_level, t.user_id
        )

        SELECT
            us.{stats_agg_col} AS activity_date,
            us.ab_group,
            us.mathtable,
            us.bet_level,
            us.user_id,
            us.user_mathtable_change,
            -- DataMetrics input columns (user-level raw stats):
            us.user_num_bets,
            us.user_num_bets_bg,
            us.user_num_bets_fg,
            us.user_total_bet,
            us.user_total_bet_bg,
            us.user_avg_bet_amount,
            us.user_total_payout,
            us.user_total_payout_bg,
            us.user_total_payout_fg,
            us.user_total_bet_fg,
            us.user_num_bets_with_payout,
            us.user_num_bets_bg_with_payout,
            us.user_num_bets_fg_with_payout,
            us.user_total_payout * 1.0 / NULLIF(us.user_total_bet, 0) AS user_rtp,

            -- User-level derived (not computed by DataMetrics):
            us.user_avg_delta_t_seconds_bg,
            us.user_num_delta_t_bg,
            us.user_num_bets_fg * 1.0 / NULLIF(us.user_num_bets, 0) AS user_fg_ratio,
            (us.user_total_payout - us.user_total_bet) AS user_total_profit,
            us.user_total_payout_bg * 1.0 / NULLIF(us.user_total_bet, 0) AS user_rtp_bg,
            us.user_total_payout_fg * 1.0 / NULLIF(us.user_total_bet_fg, 0) AS user_rtp_fg,
            us.user_num_bets_with_payout * 1.0 / NULLIF(us.user_num_bets, 0) AS user_hit_rate,
            us.user_num_bets_bg_with_payout * 1.0 / NULLIF(us.user_num_bets_bg, 0) AS user_hit_rate_bg,
            us.user_num_bets_fg_with_payout * 1.0 / NULLIF(us.user_num_bets_fg, 0) AS user_hit_rate_fg,

            -- delta bet amount metrics:
            us.user_accu_pos_delta_bet,
            us.user_accu_neg_delta_bet,
            us.user_accu_pos_delta_bet_avg,
            us.user_accu_neg_delta_bet_avg,
            us.user_accu_delta_bet,
            us.user_accu_delta_bet_avg,
            us.user_pos_delta_bet_num,
            us.user_neg_delta_bet_num,
            us.user_num_delta_bet

        FROM user_stats AS us
        WHERE us.{stats_agg_col} >= '{effective_start}'
        ORDER BY us.{stats_agg_col} DESC, us.ab_group DESC, us.mathtable DESC, us.user_id DESC;

        """
    )

    return query


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(description="ETL SS06 Game Stats by Period, AB Group and Mathtable")
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
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_ss06_pocket_soccer_v2",
        lookback_days=3,
        overwrite=args.overwrite,
    )

    scheduler.run_incremental_job(
        job_name="daily_stats",
        query_func=lambda start_date: generate_query("activity_date", start_date),
        key_cols=["activity_date", "user_id", "ab_group", "mathtable", "bet_level"],
        date_col="activity_date",
        partition_level="none",
    )

    # Overrides to 7 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="weekly_stats",
        query_func=lambda start_date: generate_query("activity_week", start_date),
        key_cols=["activity_date", "user_id", "ab_group", "mathtable", "bet_level"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=7,
    )

    # Overrides to 31 days because monthly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="monthly_stats",
        query_func=lambda start_date: generate_query("activity_month", start_date),
        key_cols=["activity_date", "user_id", "ab_group", "mathtable", "bet_level"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=31,
    )

    redshift_loader.close()
