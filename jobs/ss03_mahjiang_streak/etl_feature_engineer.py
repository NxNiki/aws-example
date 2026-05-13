"""
-- sql script to get features for bet segmentation

-- key steps:

-- to be consistent with GAIL model, we merge free game to the previous base game.
-- aggregate stats in each single run of free game (usually 10 rounds) and the previous base game. The result stats is
-- treated as a single bet.
-- use the timestamp (created_at) of the base game before the free game session in the aggregated free game, the
-- delta_t_seconds for the aggregated bet is that of the base game (before free game)
-- delta_t_seconds for the base game after a free game is the interval between last round of free game and the timestamp
-- of the current bet.

-- split segments (session_group) of bets if interval is larger 7 days (same as previous pyspark job). We propose to
-- reduce this to 1 hour in the future. Need to sync with MLE team. Currently, we walk around by ignoring delta t larger
-- than 1 hour in aggregated stats related to delta t.

-- only consider bet_amount in base game to calculate related aggregated stats. (setting bet_amount to null for free
-- game)

-- define streak of bet based on a threshold of 200 seconds in delta_t. starting from 1.
-- define win streak (200 seconds threshold) starting from 1. lose bet will be NULL
-- define lose streak (200 seconds threshold) starting from 1. win bet will be NULL

-- derive metrics:
-- delta_t_seconds: the difference of timestamp between current bet and previous bet
-- delta_bet_amount
-- delta_payout
-- deposit: the deposit into balance before current bet. (current balance before bet - previous balance after payout)
-- withdraw: -deposit if it is negative.

-- aggregate 100 consecutive raw bet rounds in the same session_group (free game aggregated into the previous base game
-- so won't be counted). agg_group = ROUND((row_number - 1) / 100), so the last group in a session may have < 100 rows.
-- and get statistics such as mean, median, std, and:
-- accumulative positive delta bet amount
-- accumulative negative delta bet amount
-- accumulative positive profit (payout - bet_amount)
-- accumulative negative profit (payout - bet_amount)

-- add coefficient of variation, spike ratio and drawdown ratio for the following metrics:
-- bet_amount
-- balance_after_bet

-- normalized by avg(bet_amount) for the following metrics:
-- payout
-- profit
-- accum_pos_profit
-- accum_deposit
-- accum_withdraw
-- delta_bet_amount
-- delta_payout

-- group selection:
-- SELECTED_GROUPS controls which ab_partition buckets are included. Each row gets an ai_group label
-- ('AI', 'AB_TEST_A', 'AB_TEST_B', 'Default'); set SELECTED_GROUPS to AI_GROUPS to keep all.

-- incomplete tail groups:
-- When DROP_INCOMPLETE_TAIL_GROUPS is True, the grouped output keeps only agg_groups with exactly
-- SESSION_LENGTH rows -- the final (incomplete) group in each session is dropped, so unique
-- (user_id, ai_group, math_table_id, session_group, agg_group) in ss03_features_grouped <= ss03_features_enriched.
-- When False, the tail group is kept even if it has fewer rows.
"""

import argparse
import os
from datetime import datetime
from textwrap import dedent

from bituslabs_ds.config import (
    AB_TEST_GROUP_A,
    AB_TEST_GROUP_B,
    AI_GROUP_ID,
    DEFAULT_BASTION_IP,
    ETL_CURRENCY_CODES,
    ETL_EXCLUDED_OP_CODES,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, RedshiftBackend

# select a short period as ss03 has large number of users, and calculating percentiles is time-consuming.
DATE_START = "2026-01-01 00:00:00"
DATE_END = "2026-05-01 00:00:00"

MAX_SESSION_INTERVAL = 60 * 60 * 24 * 7
STREAK_THRESHOLD = 200
MAX_SESSION_GAP = 60 * 60  # 1 hour
SESSION_LENGTH = 100  # number of consecutive bets in the same session

AI_GROUPS = ("AI", "AB_TEST_A", "AB_TEST_B", "Default")
# Restrict the query to these ai_group labels. Set to AI_GROUPS to keep all.
SELECTED_GROUPS = "Default"

# Drop the final agg_group in each session if it has fewer than SESSION_LENGTH rows.
DROP_INCOMPLETE_TAIL_GROUPS = False


def _normalize_groups(groups):
    return (groups,) if isinstance(groups, str) else tuple(groups)


def _build_output_suffix(selected_groups):
    normalized = _normalize_groups(selected_groups)
    if set(normalized) >= set(AI_GROUPS):
        groups_tag = "all"
    else:
        groups_tag = "_".join(sorted(normalized))
    return f"{groups_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _build_ai_group_filter(selected_groups):
    selected_groups = _normalize_groups(selected_groups)
    if set(selected_groups) >= set(AI_GROUPS):
        return ""
    conditions = []
    if "AI" in selected_groups:
        conditions.append(f"t.partition_ab[0] = '{AI_GROUP_ID}'")
    if "AB_TEST_A" in selected_groups:
        conditions.append(f"t.partition_ab[0] = '{AB_TEST_GROUP_A}'")
    if "AB_TEST_B" in selected_groups:
        conditions.append(f"t.partition_ab[0] = '{AB_TEST_GROUP_B}'")
    if "Default" in selected_groups:
        conditions.append(
            f"(t.partition_ab[0] IS NULL OR t.partition_ab[0] NOT IN "
            f"('{AI_GROUP_ID}', '{AB_TEST_GROUP_A}', '{AB_TEST_GROUP_B}'))"
        )
    return "AND (" + " OR ".join(conditions) + ")"


def generate_query():

    ai_group_filter = _build_ai_group_filter(SELECTED_GROUPS)
    having_clause = f"HAVING COUNT(t.user_id) = {SESSION_LENGTH}" if DROP_INCOMPLETE_TAIL_GROUPS else ""

    query_ctes = dedent(
        f"""
        WITH user_bets AS (
            SELECT
                t.spin_id,
                t.user_id,
                t.math_table_id,
                t.created_at,
                t.bet_type,
                -- ignore bet amount in free game (0). This will influence avg and std stats
                CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END AS bet_amount,
                t.actual_payout AS payout,
                t.balance_after_bet,
                t.balance_after_payout,
                LAG(t.created_at, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_time,
                CASE
                    WHEN t.bet_type = 'BASE' THEN 1
                    ELSE 0
                END AS is_new_game_group,
                CASE
                    WHEN t.partition_ab[0] = '{AI_GROUP_ID}' THEN 'AI'
                    WHEN t.partition_ab[0] = '{AB_TEST_GROUP_A}' THEN 'AB_TEST_A'
                    WHEN t.partition_ab[0] = '{AB_TEST_GROUP_B}' THEN 'AB_TEST_B'
                    ELSE 'Default'
                END AS ai_group
            FROM public.fct_bet_orders AS t
            WHERE
                t.created_at >= '{DATE_START}'
                AND t.created_at < '{DATE_END}'
                AND t.currency_type IN {ETL_CURRENCY_CODES}
                AND t.status = 'COMPLETED'
                AND t.game_id = 'SS03'
                AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
                {ai_group_filter}
        ),

        free_game_group AS (
            SELECT
                t.spin_id,
                t.user_id,
                t.ai_group,
                t.math_table_id,
                t.created_at,
                t.bet_type,
                t.bet_amount,
                t.payout,
                t.balance_after_bet,
                t.balance_after_payout,
                t.prev_bet_time,
                SUM(t.is_new_game_group) OVER (
                    PARTITION BY t.user_id
                    ORDER BY t.spin_id, t.created_at
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS fg_group
            FROM user_bets AS t
        ),

        agg_free_game AS (
            SELECT
                t.user_id,
                t.ai_group,
                t.math_table_id,
                MIN(t.spin_id) AS spin_id,
                MIN(t.created_at) AS min_created_at,
                MAX(t.created_at) AS max_created_at,
                SUM(CASE WHEN t.bet_type = 'FREE' THEN 1 ELSE 0 END) AS fg_rounds,
                MIN(t.prev_bet_time) AS prev_bet_time,
                SUM(t.bet_amount) AS bet_amount,
                SUM(t.payout) AS payout,
                MIN(t.balance_after_bet) AS balance_after_bet,
                MAX(t.balance_after_payout) AS balance_after_payout
            FROM free_game_group AS t
            GROUP BY t.user_id, t.ai_group, t.math_table_id, t.fg_group
        ),

        delta_stats AS (
            SELECT
                t.user_id,
                t.ai_group,
                t.math_table_id,
                t.spin_id,
                t.min_created_at,
                t.max_created_at,
                t.fg_rounds,
                t.bet_amount,
                t.payout,
                t.balance_after_bet,
                DATEDIFF(SECONDS, t.prev_bet_time, t.min_created_at) AS delta_t_seconds,
                t.bet_amount - LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_bet_amount,
                t.payout - LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_payout,
                t.balance_after_bet
                + t.bet_amount
                - LAG(t.balance_after_payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS balance_transaction,
                CASE WHEN t.payout > t.bet_amount THEN 1 ELSE 0 END AS is_win,
                CASE WHEN t.payout < t.bet_amount THEN 1 ELSE 0 END AS is_lose,
                CASE
                    WHEN
                        LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                        > LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                        THEN 1
                    ELSE 0
                END AS prev_win,
                CASE
                    WHEN
                        LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                        < LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                        THEN 1
                    ELSE 0
                END AS prev_lose
            FROM agg_free_game AS t
        ),

        user_group AS (
            SELECT
                t.user_id,
                t.ai_group,
                t.math_table_id,
                t.spin_id,
                t.min_created_at,
                t.max_created_at,
                t.fg_rounds,
                t.delta_t_seconds,
                t.bet_amount,
                t.delta_bet_amount,
                t.payout,
                t.delta_payout,
                t.balance_after_bet,
                t.balance_transaction,
                t.is_win,
                t.is_lose,
                t.prev_win,
                t.prev_lose,
                SUM(CASE WHEN t.delta_t_seconds <= {MAX_SESSION_INTERVAL} THEN 0 ELSE 1 END)
                    OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                    AS session_group,
                SUM(CASE WHEN t.delta_t_seconds <= {STREAK_THRESHOLD} THEN 0 ELSE 1 END)
                    OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                    AS streak_group,
                SUM(
                    CASE
                        WHEN t.delta_t_seconds <= {STREAK_THRESHOLD} AND t.prev_win = 1 AND t.is_win = 1 THEN 0 ELSE 1
                    END
                )
                    OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                    AS win_streak_group,
                SUM(
                    CASE
                        WHEN
                            t.delta_t_seconds <= {STREAK_THRESHOLD} AND t.prev_lose = 1 AND t.is_lose = 1
                            THEN 0
                        ELSE 1
                    END
                )
                    OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                    AS lose_streak_group
            FROM delta_stats AS t
        ),

        raw_stats AS (
            SELECT
                t.user_id,
                t.ai_group,
                t.math_table_id,
                t.session_group,
                t.min_created_at,
                t.max_created_at,
                t.spin_id,
                t.fg_rounds,
                t.bet_amount,
                t.delta_bet_amount,
                t.payout,
                t.delta_payout,
                t.balance_after_bet,
                t.delta_t_seconds,
                -- ignore delta t larger than 1 hour.
                CASE WHEN t.delta_t_seconds <= {MAX_SESSION_GAP} THEN t.delta_t_seconds END AS delta_t_seconds_nogap,
                t.payout - t.bet_amount AS profit,
                CASE WHEN t.balance_transaction > .1 THEN t.balance_transaction END AS deposit,
                CASE WHEN t.balance_transaction < -.1 THEN -t.balance_transaction END AS withdraw,
                ROW_NUMBER() OVER (PARTITION BY t.user_id, t.streak_group ORDER BY t.spin_id, t.min_created_at) AS streak,
                CASE
                    WHEN
                        t.payout > t.bet_amount
                        THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.win_streak_group ORDER BY t.spin_id, t.min_created_at)
                END AS win_streak,
                CASE
                    WHEN
                        t.payout < t.bet_amount
                        THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.lose_streak_group ORDER BY t.spin_id, t.min_created_at)
                END AS lose_streak,
                ROUND((ROW_NUMBER() OVER (PARTITION BY t.user_id, t.session_group ORDER BY t.spin_id, t.min_created_at) - 1) / {SESSION_LENGTH})
                    AS agg_group
            FROM user_group AS t
        )
        """
    )

    query_raw_stats = dedent(
        f"""

        {query_ctes}
        SELECT * FROM raw_stats;
        """
    )

    query_grouped_stats = dedent(
        f"""

        {query_ctes},

        -- CTEs to separate percentile calculations by sort order
        perc_delta_t AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_t_seconds) AS delta_t_seconds_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_t_seconds) AS delta_t_seconds_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_t_seconds) AS delta_t_seconds_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_delta_t_ng AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_t_seconds_nogap) AS delta_t_seconds_nogap_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_t_seconds_nogap) AS delta_t_seconds_nogap_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_t_seconds_nogap) AS delta_t_seconds_nogap_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_bet_amount AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bet_amount) AS bet_amount_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY bet_amount) AS bet_amount_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bet_amount) AS bet_amount_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_delta_bet AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_bet_amount) AS delta_bet_amount_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_bet_amount) AS delta_bet_amount_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_bet_amount) AS delta_bet_amount_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_payout AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY payout) AS payout_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY payout) AS payout_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY payout) AS payout_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_rtp AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_profit AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY profit) AS profit_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY profit) AS profit_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY profit) AS profit_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_delta_payout AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_payout) AS delta_payout_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_payout) AS delta_payout_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_payout) AS delta_payout_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_balance AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY balance_after_bet) AS balance_after_bet_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY balance_after_bet) AS balance_after_bet_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY balance_after_bet) AS balance_after_bet_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_streak AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY streak) AS streak_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY streak) AS streak_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY streak) AS streak_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_win_streak AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY win_streak) AS win_streak_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY win_streak) AS win_streak_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY win_streak) AS win_streak_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),
        perc_lose_streak AS (
            SELECT
                user_id, ai_group, math_table_id, session_group, agg_group,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY lose_streak) AS lose_streak_p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY lose_streak) AS lose_streak_median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY lose_streak) AS lose_streak_p75
            FROM raw_stats GROUP BY user_id, ai_group, math_table_id, session_group, agg_group
        ),

    -- Assemble standard metrics and join with percentiles
        stats_base AS (
            SELECT
                t.user_id,
                t.ai_group,
                t.math_table_id,
                t.session_group,
                t.agg_group,
                MIN(t.min_created_at) AS min_created_at,
                MAX(t.max_created_at) AS max_created_at,
                COUNT(t.user_id) AS bet_rounds,
                SUM(t.fg_rounds) AS fg_rounds,
                AVG(t.delta_t_seconds) AS delta_t_seconds_avg,
                STDDEV(t.delta_t_seconds) AS delta_t_seconds_std,
                MIN(t.delta_t_seconds) AS delta_t_seconds_min,
                MAX(t.delta_t_seconds) AS delta_t_seconds_max,
                AVG(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_avg,
                STDDEV(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_std,
                MIN(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_min,
                MAX(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_max,
                AVG(t.bet_amount) AS bet_amount_avg,
                STDDEV(t.bet_amount) AS bet_amount_std,
                MIN(t.bet_amount) AS bet_amount_min,
                MAX(t.bet_amount) AS bet_amount_max,
                AVG(t.delta_bet_amount) AS delta_bet_amount_avg,
                STDDEV(t.delta_bet_amount) AS delta_bet_amount_std,
                MIN(t.delta_bet_amount) AS delta_bet_amount_min,
                MAX(t.delta_bet_amount) AS delta_bet_amount_max,
                SUM(CASE WHEN t.delta_bet_amount > 0 THEN t.delta_bet_amount ELSE 0 END) AS accum_pos_delta_bet_amount,
                SUM(CASE WHEN t.delta_bet_amount < 0 THEN t.delta_bet_amount ELSE 0 END) AS accum_neg_delta_bet_amount,
                AVG(t.payout) AS payout_avg,
                STDDEV(t.payout) AS payout_std,
                MIN(t.payout) AS payout_min,
                MAX(t.payout) AS payout_max,
                SUM(CASE WHEN t.payout > 0 THEN 1 END) * 1.0 / NULLIF(COUNT(t.user_id), 0) AS payout_rate,
                AVG(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_mean,
                MAX(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_max,
                MIN(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_min,
                AVG(t.profit) AS profit_avg,
                STDDEV(t.profit) AS profit_std,
                MIN(t.profit) AS profit_min,
                MAX(t.profit) AS profit_max,
                SUM(CASE WHEN t.profit > 0 THEN t.profit ELSE 0 END) AS accum_pos_profit,
                SUM(CASE WHEN t.profit < 0 THEN t.profit ELSE 0 END) AS accum_neg_profit,
                SUM(CASE WHEN t.profit > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(t.user_id), 0) AS profit_rate,
                AVG(t.delta_payout) AS delta_payout_avg,
                STDDEV(t.delta_payout) AS delta_payout_std,
                MIN(t.delta_payout) AS delta_payout_min,
                MAX(t.delta_payout) AS delta_payout_max,
                AVG(t.balance_after_bet) AS balance_after_bet_avg,
                STDDEV(t.balance_after_bet) AS balance_after_bet_std,
                MIN(t.balance_after_bet) AS balance_after_bet_min,
                MAX(t.balance_after_bet) AS balance_after_bet_max,
                SUM(CASE WHEN t.deposit > 0 THEN 1 ELSE 0 END) AS num_deposit,
                SUM(CASE WHEN t.deposit > 0 THEN t.deposit ELSE 0 END) AS accum_deposit,
                SUM(CASE WHEN t.withdraw > 0 THEN 1 ELSE 0 END) AS num_withdraw,
                SUM(CASE WHEN t.withdraw > 0 THEN t.withdraw ELSE 0 END) AS accum_withdraw,
                AVG(t.streak) AS streak_avg,
                STDDEV(t.streak) AS streak_std,
                MIN(t.streak) AS streak_min,
                MAX(t.streak) AS streak_max,
                AVG(t.win_streak) AS win_streak_avg,
                STDDEV(t.win_streak) AS win_streak_std,
                MIN(t.win_streak) AS win_streak_min,
                MAX(t.win_streak) AS win_streak_max,
                AVG(t.lose_streak) AS lose_streak_avg,
                STDDEV(t.lose_streak) AS lose_streak_std,
                MIN(t.lose_streak) AS lose_streak_min,
                MAX(t.lose_streak) AS lose_streak_max
            FROM raw_stats AS t
            GROUP BY
                t.user_id,
                t.ai_group,
                t.math_table_id,
                t.session_group,
                t.agg_group
            {having_clause}
        )

    -- Final assembly join
    SELECT
        b.user_id,
        b.ai_group,
        b.math_table_id,
        b.session_group,
        b.agg_group,
        b.min_created_at,
        b.max_created_at,
        b.bet_rounds,
        b.fg_rounds,
        -- delta bet time
        b.delta_t_seconds_avg,
        p1.delta_t_seconds_p25,
        p1.delta_t_seconds_median,
        p1.delta_t_seconds_p75,
        b.delta_t_seconds_std,
        b.delta_t_seconds_min,
        b.delta_t_seconds_max,
        -- delta bet time nogap
        b.delta_t_seconds_nogap_avg,
        p2.delta_t_seconds_nogap_p25,
        p2.delta_t_seconds_nogap_median,
        p2.delta_t_seconds_nogap_p75,
        b.delta_t_seconds_nogap_std,
        b.delta_t_seconds_nogap_min,
        b.delta_t_seconds_nogap_max,
        -- bet amount
        b.bet_amount_avg,
        p3.bet_amount_p25,
        p3.bet_amount_median,
        p3.bet_amount_p75,
        b.bet_amount_std,
        b.bet_amount_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_cv,
        b.bet_amount_min,
        b.bet_amount_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_drawdown_ratio,
        b.bet_amount_max,
        b.bet_amount_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_spike_ratio,
        -- delta bet amount
        b.delta_bet_amount_avg,
        p4.delta_bet_amount_p25,
        p4.delta_bet_amount_median,
        p4.delta_bet_amount_p75,
        b.delta_bet_amount_std,
        COALESCE(b.delta_bet_amount_std * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_cv,
        b.delta_bet_amount_min,
        COALESCE(b.delta_bet_amount_min * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_drawdown_ratio,
        b.delta_bet_amount_max,
        COALESCE(b.delta_bet_amount_max * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_spike_ratio,
        -- accumulative delta bet
        b.accum_pos_delta_bet_amount,
        b.accum_neg_delta_bet_amount,
        b.accum_pos_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_pos_delta_bet_amount_ratio,
        b.accum_neg_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_neg_delta_bet_amount_ratio,
        -- payout
        b.payout_avg,
        p5.payout_p25,
        p5.payout_median,
        p5.payout_p75,
        b.payout_std,
        b.payout_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_cv,
        b.payout_min,
        b.payout_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_drawdown_ratio,
        b.payout_max,
        b.payout_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_spike_ratio,
        b.payout_rate,
        -- rtp
        b.rtp_mean,
        b.rtp_max,
        b.rtp_min,
        p6.rtp_p25,
        p6.rtp_median,
        p6.rtp_p75,
        -- profit
        b.profit_avg,
        p7.profit_p25,
        p7.profit_median,
        p7.profit_p75,
        b.profit_std,
        b.profit_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS profit_cv,
        b.profit_min,
        b.profit_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS min_profit_ratio,
        b.profit_max,
        b.profit_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS max_profit_ratio,
        b.accum_pos_profit,
        b.accum_pos_profit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_pos_profit_ratio,
        b.accum_neg_profit,
        b.accum_neg_profit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_neg_profit_ratio,
        b.profit_rate,
        -- delta payout
        b.delta_payout_avg,
        p8.delta_payout_p25,
        p8.delta_payout_median,
        p8.delta_payout_p75,
        b.delta_payout_std,
        COALESCE(b.delta_payout_std * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_cv,
        b.delta_payout_min,
        COALESCE(b.delta_payout_min * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_drawdown_ratio,
        b.delta_payout_max,
        COALESCE(b.delta_payout_max * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_spike_ratio,
        -- balance
        b.balance_after_bet_avg,
        p9.balance_after_bet_p25,
        p9.balance_after_bet_median,
        p9.balance_after_bet_p75,
        b.balance_after_bet_std,
        b.balance_after_bet_std * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_cv,
        b.balance_after_bet_min,
        b.balance_after_bet_min * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_drawdown_ratio,
        b.balance_after_bet_max,
        b.balance_after_bet_max * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_spike_ratio,
        -- deposit/withdraw
        b.num_deposit,
        b.accum_deposit,
        b.accum_deposit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_deposit_ratio,
        b.num_withdraw,
        b.accum_withdraw,
        b.accum_withdraw * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_withdraw_ratio,
        -- streaks
        COALESCE(b.streak_avg, 0) AS streak_avg,
        COALESCE(p10.streak_p25, 0) AS streak_p25,
        COALESCE(p10.streak_median, 0) AS streak_median,
        COALESCE(p10.streak_p75, 0) AS streak_p75,
        COALESCE(b.streak_std, 0) AS streak_std,
        COALESCE(b.streak_min, 0) AS streak_min,
        COALESCE(b.streak_max, 0) AS streak_max,
        COALESCE(b.win_streak_avg, 0) AS win_streak_avg,
        COALESCE(p11.win_streak_p25, 0) AS win_streak_p25,
        COALESCE(p11.win_streak_median, 0) AS win_streak_median,
        COALESCE(p11.win_streak_p75, 0) AS win_streak_p75,
        COALESCE(b.win_streak_std, 0) AS win_streak_std,
        COALESCE(b.win_streak_min, 0) AS win_streak_min,
        COALESCE(b.win_streak_max, 0) AS win_streak_max,
        COALESCE(b.lose_streak_avg, 0) AS lose_streak_avg,
        COALESCE(p12.lose_streak_p25, 0) AS lose_streak_p25,
        COALESCE(p12.lose_streak_median, 0) AS lose_streak_median,
        COALESCE(p12.lose_streak_p75, 0) AS lose_streak_p75,
        COALESCE(b.lose_streak_std, 0) AS lose_streak_std,
        COALESCE(b.lose_streak_min, 0) AS lose_streak_min,
        COALESCE(b.lose_streak_max, 0) AS lose_streak_max
    FROM stats_base AS b
            JOIN perc_delta_t AS p1
                ON b.user_id = p1.user_id
                    AND b.ai_group = p1.ai_group
                    AND b.math_table_id = p1.math_table_id
                    AND b.session_group = p1.session_group
                    AND b.agg_group = p1.agg_group
            JOIN perc_delta_t_ng AS p2
                ON b.user_id = p2.user_id
                    AND b.ai_group = p2.ai_group
                    AND b.math_table_id = p2.math_table_id
                    AND b.session_group = p2.session_group
                    AND b.agg_group = p2.agg_group
            JOIN perc_bet_amount AS p3
                ON b.user_id = p3.user_id
                    AND b.ai_group = p3.ai_group
                    AND b.math_table_id = p3.math_table_id
                    AND b.session_group = p3.session_group
                    AND b.agg_group = p3.agg_group
            JOIN perc_delta_bet AS p4
                ON b.user_id = p4.user_id
                    AND b.ai_group = p4.ai_group
                    AND b.math_table_id = p4.math_table_id
                    AND b.session_group = p4.session_group
                    AND b.agg_group = p4.agg_group
            JOIN perc_payout AS p5
                ON b.user_id = p5.user_id
                    AND b.ai_group = p5.ai_group
                    AND b.math_table_id = p5.math_table_id
                    AND b.session_group = p5.session_group
                    AND b.agg_group = p5.agg_group
            JOIN perc_rtp AS p6
                ON b.user_id = p6.user_id
                    AND b.ai_group = p6.ai_group
                    AND b.math_table_id = p6.math_table_id
                    AND b.session_group = p6.session_group
                    AND b.agg_group = p6.agg_group
            JOIN perc_profit AS p7
                ON b.user_id = p7.user_id
                    AND b.ai_group = p7.ai_group
                    AND b.math_table_id = p7.math_table_id
                    AND b.session_group = p7.session_group
                    AND b.agg_group = p7.agg_group
            JOIN perc_delta_payout AS p8
                ON b.user_id = p8.user_id
                    AND b.ai_group = p8.ai_group
                    AND b.math_table_id = p8.math_table_id
                    AND b.session_group = p8.session_group
                    AND b.agg_group = p8.agg_group
            JOIN perc_balance AS p9
                ON b.user_id = p9.user_id
                    AND b.ai_group = p9.ai_group
                    AND b.math_table_id = p9.math_table_id
                    AND b.session_group = p9.session_group
                    AND b.agg_group = p9.agg_group
            JOIN perc_streak AS p10
                ON b.user_id = p10.user_id
                    AND b.ai_group = p10.ai_group
                    AND b.math_table_id = p10.math_table_id
                    AND b.session_group = p10.session_group
                    AND b.agg_group = p10.agg_group
            JOIN perc_win_streak AS p11
                ON b.user_id = p11.user_id
                    AND b.ai_group = p11.ai_group
                    AND b.math_table_id = p11.math_table_id
                    AND b.session_group = p11.session_group
                    AND b.agg_group = p11.agg_group
            JOIN perc_lose_streak AS p12
                ON b.user_id = p12.user_id
                    AND b.ai_group = p12.ai_group
                    AND b.math_table_id = p12.math_table_id
                    AND b.session_group = p12.session_group
                    AND b.agg_group = p12.agg_group
        ORDER BY
            b.user_id,
            b.ai_group,
            b.math_table_id,
            b.session_group,
            b.agg_group
        ;

        """
    )

    return query_raw_stats, query_grouped_stats


def execute_query(redshift_loader, output_file, query) -> None:

    file_path = f"{LOCAL_ROOT}/jobs/output_ss03_mahjiang_streak/{output_file}.parquet"
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)

    print(df_rs.head(5))


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(description="ETL Feature Engineer for SS03 Mahjiang Streak")
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP address for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
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

    query_raw_stats, query_grouped_stats = generate_query()
    output_suffix = _build_output_suffix(SELECTED_GROUPS)
    execute_query(redshift_loader, f"ss03_features_enriched_{output_suffix}", query_raw_stats)
    # execute_query(redshift_loader, f"ss03_features_grouped_{output_suffix}", query_grouped_stats)

    redshift_loader.close()
