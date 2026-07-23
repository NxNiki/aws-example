"""Per-user game stats for SS01A (Golden Goal), by period and ai_group.

ETL job: writes output_ss01a_golden_goal/{daily,weekly,monthly}_stats
(the ``user_*`` metrics consumed by the SS01A dashboard / DataMetrics).
``stats_agg_col`` selects the period grain: activity_date / _week / _month.

IMPORTANT — rows are intentionally duplicated, so NEVER aggregate across
all ``ai_group`` values. ``user_bets_group`` UNION ALLs each bet into:
  * one COMBINED group: AI / Default, and
  * one PER-MATHTABLE group: ``<mathtable>`` for AI users,
    ``Default_<mathtable>`` for the rest.
The per-mathtable rows are just a re-partition of the same bets, so every
bet is counted twice. The two COMBINED groups partition each bet exactly
once and have one row per user per period; the dashboard's "all" cohort
aggregates only them (``group_col_partition`` in the config).
"""

import argparse
import os
from textwrap import dedent

from bituslabs_ds.config import (
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

GAME_ID = "SS01A"

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
                t.delta_t_seconds,
                t.user_bet_count,
                t.mathtable_change,
                t.prev_bet_type,
                t.prev_bet_amount,
                CASE WHEN t.bet_type = 'FREE' AND t.prev_bet_type = 'BASE' THEN t.prev_bet_amount END AS fg_session_trigger_bet,
                """


def generate_query(stats_agg_col: AggCol, start_date: str):
    effective_start = effective_start_date(stats_agg_col, start_date)
    query = dedent(
        f"""
        WITH user_bets AS (
        SELECT
            t.user_id,
            t.spin_id,
            t.created_at,
            t.math_table_id AS mathtable,
            t.bet_amount,
            t.actual_payout AS payout,
            t.bet_type,
            t.actual_payout - t.bet_amount AS profit,
            CAST(DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_date,
            CAST(DATE_TRUNC('week', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_week,
            CAST(DATE_TRUNC('month', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at))) AS DATE) AS activity_month,
            t.partition_ab[0] AS ab_group_id,
            LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_created_at,
            EXTRACT(EPOCH FROM (t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at))) AS delta_t_seconds,
            LAG(t.bet_type) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_type,
            LAG(t.bet_amount) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_amount,
            COUNT(t.user_id) OVER (PARTITION BY t.user_id) AS user_bet_count,
            CASE
                WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) IS NULL THEN 0
                WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) <> t.math_table_id THEN 1
                ELSE 0
            END AS mathtable_change
        FROM
            public.fct_bet_orders AS t
        WHERE
            t.game_id = '{GAME_ID}'
            AND CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.created_at) >= '{effective_start}'
            AND t.currency_type IN {ETL_CURRENCY_CODES}
            AND t.status = 'COMPLETED'
            AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
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
                    WHEN t.ab_group_id = '{AI_GROUP_ID}' THEN t.mathtable
                    ELSE CONCAT('Default_', t.mathtable)
                END AS ai_group
            FROM
                user_bets AS t
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
                AVG(t.bet_amount) AS user_avg_bet_amount,

                -- total payout amount:
                SUM(t.payout) AS user_total_payout,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS user_total_payout_bg,
                SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS user_total_payout_fg,

                SUM(t.fg_session_trigger_bet) AS user_total_bet_fg,

                COUNT(CASE WHEN t.payout > 0 THEN 1 END) AS user_num_bets_with_payout,
                COUNT(CASE WHEN t.bet_type = 'BASE' AND t.payout > 0 THEN 1 END) AS user_num_bets_bg_with_payout,
                COUNT(CASE WHEN t.bet_type = 'FREE' AND t.payout > 0 THEN 1 END) AS user_num_bets_fg_with_payout,

                AVG(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= {ETL_DELTA_T_MAX_SECONDS} THEN GREATEST(t.delta_t_seconds, {ETL_DELTA_T_MIN_SECONDS}) END)
                    AS user_avg_delta_t_seconds_bg,

                SUM(t.mathtable_change) AS user_mathtable_change,

                -- delta bet amount metrics:
                SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet,
                SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet,
                AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet_avg,
                AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet_avg,
                SUM(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet,
                AVG(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet_avg,
                COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN 1 END) AS user_pos_delta_bet_num,
                COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN 1 END) AS user_neg_delta_bet_num

            FROM user_bets_group AS t
            GROUP BY t.{stats_agg_col}, t.ai_group, t.user_id
        ),

        user_mathtable_last_spin_ai AS (
            -- Last spin per (period, AI user, mathtable). Restricted to AI bets so the
            -- downstream metric only applies to AI per-mathtable ai_groups.
            SELECT
                t.{stats_agg_col},
                t.user_id,
                t.mathtable,
                MAX(t.spin_id) AS mathtable_last_spin_id
            FROM user_bets AS t
            WHERE t.ab_group_id = '{AI_GROUP_ID}'
            GROUP BY t.{stats_agg_col}, t.user_id, t.mathtable
        ),

        user_remaining_bet AS (
            -- Avg bet on bets the AI user made AFTER their last bet on the math table,
            -- keyed by ai_group (= mathtable for the AI per-mathtable variant). All
            -- other ai_groups are absent here, so the LEFT JOIN below yields NULL.
            SELECT
                m.{stats_agg_col},
                m.user_id,
                m.mathtable AS ai_group,
                AVG(b.bet_amount) AS user_avg_remaining_bet_amount
            FROM user_mathtable_last_spin_ai AS m
            LEFT JOIN user_bets AS b
                ON b.user_id = m.user_id
                AND b.{stats_agg_col} = m.{stats_agg_col}
                AND b.spin_id > m.mathtable_last_spin_id
                AND b.ab_group_id = '{AI_GROUP_ID}'
            GROUP BY m.{stats_agg_col}, m.user_id, m.mathtable
        )

        SELECT
            us.{stats_agg_col} AS activity_date,
            us.ai_group,
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

            -- average bet amount on bets after the user's last bet on the ai_group's math table
            -- (only populated for per-mathtable ai_groups; NULL for combined groups)
            rb.user_avg_remaining_bet_amount

        FROM user_stats AS us
        LEFT JOIN user_remaining_bet AS rb
            ON rb.user_id = us.user_id
            AND rb.{stats_agg_col} = us.{stats_agg_col}
            AND rb.ai_group = us.ai_group
        WHERE us.{stats_agg_col} >= '{effective_start}'
        ORDER BY us.{stats_agg_col} DESC, us.ai_group DESC, us.user_id DESC;

        """
    )

    return query


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(description="ETL SS01A Game Stats by Period and User Group")
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
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_ss01a_golden_goal",
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

    # Overrides to 31 days because monthly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="monthly_stats",
        query_func=lambda start_date: generate_query("activity_month", start_date),
        key_cols=["activity_date", "user_id", "ai_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=31,
    )

    redshift_loader.close()
