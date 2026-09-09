"""SS03 per-bet feature engineering from the slot_orders_ab_group dataset (PySpark).

ETL job: SageMaker port of ``jobs/etl/redshift/ss03_mahjiang_streak/
etl_feature_engineer.py`` — same two outputs per ai_group slice
(``features_enriched`` per aggregated bet, ``features_grouped_binsize_{N}``
per session bin), same output roots, ``year=/month=`` layout, column order
and parquet dtypes (see OUTPUT dtype maps), reading the
``output_slot_orders_ab_group/orders`` dataset (raw bets + the
policy-correct ``ab_group`` — see etl_slot_orders_ab_group.py) instead of
Redshift through the bastion, so group membership is derived exactly once
upstream.

Semantics ported 1:1 from the Redshift SQL with these deliberate translations:
``DATEDIFF(SECONDS, ...)`` becomes a truncating
``unix_timestamp`` difference; ``LAST_VALUE(... IGNORE NULLS)`` becomes
``LAST(expr, TRUE)``; ``ROUND((idx-1)/N)`` — integer division on Redshift —
becomes explicit ``FLOOR``; the 12 single-percentile CTEs collapse into one
``percentile()`` aggregation; Redshift's integer-AVG truncation is reproduced
with BIGINT casts; grouped rows with NULL math_table_id are excluded (the
Redshift inner joins dropped them). Source decimals stay DECIMAL through the
whole computation (exact win/lose ties and the ±0.1 deposit dead zone) and
are cast to DOUBLE only at write time, like the Redshift → pandas path.
Unlike the Redshift config there is no hard ``date_end`` cutoff — the whole
point of the monthly schedule is to keep extending the datasets; the sidecar's
``date_end`` records the latest run's horizon (it is not a semantic field).

Incremental model: month-aligned recompute with dynamic partition overwrite,
replacing the Redshift ETLScheduler's key-dedup compaction (and structurally
fixing its mutable-key leak — an old-label row cannot survive a whole-month
rewrite). Each run recomputes whole months in [output-start, output-end);
the default window self-heals: it starts at the earlier of the previous
calendar month and the month containing the newest row already on S3, so a
missed monthly firing widens the next run instead of leaving a silent gap.
The scan begins LOOKBACK_DAYS before the window so sessions straddling the
window start keep their true ``session_start_ts`` (stable keys, no
re-keying/double counting for any session shorter than the lookback).

Accepted edge (grouped outputs only): a session bin whose FIRST bet falls in
the month before the window keeps the aggregates from that month's final
recompute — bets landing after that moment are in the enriched rows (written
to their own month with stable session keys) but not in that bin's stats.
Months before COLD_DATA_FLOOR stay owned by the Redshift-era data: cold data
starts 2026-02-10, so recomputing 2026-02 would erase Feb 1-9.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone

from pyspark.sql import functions as F

# Ships via submit_py_files on SageMaker; for local runs/tests put
# jobs/etl/sagemaker on the path first (the snapshot tests already do).
from spark_etl_common import (
    BJ_UTC_OFFSET_HOURS,
    build_spark_session,
    check_schema,
    month_start,
    prev_month_start,
    prune_period_days,
    utc_today,
)

# Sessions straddling the window start keep stable keys as long as they are
# shorter than this (a session only breaks after a 12h gap, so multi-day
# sessions exist; 14 days of continuous <12h-gap betting is negligible).
LOOKBACK_DAYS = 14

# First month fully covered by cold data (bet_order starts 2026-02-10;
# 2026-02 on S3 contains Redshift-era Feb 1-9 rows a rewrite would erase).
COLD_DATA_FLOOR = date(2026, 3, 1)

SESSION_BREAK_SECONDS = 43200
STREAK_THRESHOLD_SECONDS = 200
MAX_DELTA_T_GAP_SECONDS = 3600

# The AB grouping policy flipped from partition_ab ids to user-id last digits
# across these Beijing days (announced 08-03, effective 08-04), so group
# membership DURING them is ambiguous — sessions STARTING on them are dropped
# from every output. Keying on the session start keeps cross-midnight
# sessions whole: a session opened 08-04 23:50 BJ is excluded entirely, one
# opened 08-05 00:10 is kept entirely.
EXCLUDED_SESSION_START_DATES_BJ = ("2026-08-03", "2026-08-04")

REQUIRED_COLUMNS = [
    "spin_id",
    "user_id",
    "math_table_id",
    "created_at",
    "bet_type",
    "bet_amount",
    "actual_payout",
    "balance_after_bet",
    "balance_after_payout",
    "ab_group",
    "currency_type",
    "status",
    "op_code",
]

# ai_group slice -> (output_prefix, ai_group filter SQL, bin sizes, selected label).
# Mirrors jobs/etl/redshift/ss03_mahjiang_streak/etl_feature_engineer.py GROUPS;
# the ai_group label itself is the date-gated policy (group_policy.py).
GROUPS = {
    "default": ("output_ss03_feature_engineer", "ai_group = 'Default'", [50, 70], "Default"),
    "ai": ("output_ss03_feature_engineer_ai", "ai_group = 'AI'", [30, 50, 70, 100], "AI"),
    "ab_test_a": ("output_ss03_feature_engineer_ab_test_a", "ai_group = 'AB_TEST_A'", [50, 70], "AB_TEST_A"),
    "ab_test_b": ("output_ss03_feature_engineer_ab_test_b", "ai_group = 'AB_TEST_B'", [50, 70], "AB_TEST_B"),
}

SIDECAR_SEMANTIC_FIELDS = [
    "game_id",
    "ai_groups",
    "selected_groups",
    "partition_cols",
    "bin_size",
    "session_break_threshold_seconds",
    "streak_threshold_seconds",
    "max_delta_t_gap_seconds",
    "drop_incomplete_tail_groups",
    "extra_where_clauses",
]

# Written column order + parquet dtype per dataset. MUST stay byte-compatible
# with the Redshift/awswrangler-era files that share these dataset roots:
# mixed dtypes across files break every downstream dataset read. Redshift
# AVG(int) truncates to BIGINT — hence the long avg columns.
ENRICHED_DTYPES = {
    "user_id": "bigint",
    "ai_group": "string",
    "math_table_id": "string",
    "session_start_date": "date",
    "session_group": "bigint",
    "session_start_ts": "timestamp",
    "min_created_at": "timestamp",
    "max_created_at": "timestamp",
    "activity_date": "timestamp",
    "spin_id": "bigint",
    "fg_rounds": "bigint",
    "bet_amount": "double",
    "delta_bet_amount": "double",
    "payout": "double",
    "delta_payout": "double",
    "balance_after_bet": "double",
    "delta_t_seconds": "bigint",
    "delta_t_seconds_nogap": "bigint",
    "profit": "double",
    "deposit": "double",
    "withdraw": "double",
    "streak": "bigint",
    "win_streak": "bigint",
    "lose_streak": "bigint",
    "session_bet_index": "bigint",
}

_GROUPED_KEYS = {
    "user_id": "bigint",
    "ai_group": "string",
    "math_table_id": "string",
    "session_start_date": "date",
    "session_group": "bigint",
    "agg_group": "double",
    "activity_date": "timestamp",
    "min_created_at": "timestamp",
    "max_created_at": "timestamp",
    "bet_rounds": "bigint",
    "fg_rounds": "bigint",
}


def _stat_family(base: str, int_valued: bool, ratios_of: str | None = None, coalesce: bool = False) -> dict:
    long_or_double = "bigint" if int_valued else "double"
    cols = {
        f"{base}_avg": long_or_double,
        f"{base}_p25": "double",
        f"{base}_median": "double",
        f"{base}_p75": "double",
        f"{base}_std": "double",
    }
    if ratios_of:
        cols[f"{base}_cv"] = "double"
    cols[f"{base}_min"] = long_or_double
    if ratios_of:
        cols[f"{base}_drawdown_ratio"] = "double"
    cols[f"{base}_max"] = long_or_double
    if ratios_of:
        cols[f"{base}_spike_ratio"] = "double"
    return cols


def grouped_dtypes() -> dict:
    cols = dict(_GROUPED_KEYS)
    cols.update(_stat_family("delta_t_seconds", int_valued=True))
    cols.update(_stat_family("delta_t_seconds_nogap", int_valued=True))
    cols.update(_stat_family("bet_amount", int_valued=False, ratios_of="bet_amount_avg"))
    cols.update(_stat_family("delta_bet_amount", int_valued=False, ratios_of="bet_amount_avg"))
    cols.update(
        {
            "accum_pos_delta_bet_amount": "double",
            "accum_neg_delta_bet_amount": "double",
            "accum_pos_delta_bet_amount_ratio": "double",
            "accum_neg_delta_bet_amount_ratio": "double",
        }
    )
    cols.update(_stat_family("payout", int_valued=False, ratios_of="bet_amount_avg"))
    cols["payout_rate"] = "double"
    cols.update({"rtp_mean": "double", "rtp_max": "double", "rtp_min": "double"})
    cols.update({"rtp_p25": "double", "rtp_median": "double", "rtp_p75": "double"})
    cols.update(
        {
            "profit_avg": "double",
            "profit_p25": "double",
            "profit_median": "double",
            "profit_p75": "double",
            "profit_std": "double",
            "profit_cv": "double",
            "profit_min": "double",
            "min_profit_ratio": "double",
            "profit_max": "double",
            "max_profit_ratio": "double",
            "accum_pos_profit": "double",
            "accum_pos_profit_ratio": "double",
            "accum_neg_profit": "double",
            "accum_neg_profit_ratio": "double",
            "profit_rate": "double",
        }
    )
    cols.update(_stat_family("delta_payout", int_valued=False, ratios_of="bet_amount_avg"))
    cols.update(_stat_family("balance_after_bet", int_valued=False, ratios_of="balance_after_bet_avg"))
    cols.update(
        {
            "num_deposit": "bigint",
            "accum_deposit": "double",
            "accum_deposit_ratio": "double",
            "num_withdraw": "bigint",
            "accum_withdraw": "double",
            "accum_withdraw_ratio": "double",
        }
    )
    for streak in ("streak", "win_streak", "lose_streak"):
        cols.update(_stat_family(streak, int_valued=True))
    return cols


def enriched_sql(group_filter: str, scan_start: date, scan_end: date) -> str:
    source_cols = ",\n            ".join(f"t.{c}" for c in REQUIRED_COLUMNS if c != "ab_group")
    excluded_session_days = ", ".join(f"DATE '{d}'" for d in EXCLUDED_SESSION_START_DATES_BJ)
    return f"""
WITH user_bets AS (
    SELECT
        spin_id,
        user_id,
        math_table_id,
        created_at,
        bet_type,
        CASE WHEN bet_type = 'BASE' THEN bet_amount END AS bet_amount,
        actual_payout AS payout,
        balance_after_bet,
        balance_after_payout,
        LAG(created_at, 1) OVER (PARTITION BY user_id ORDER BY spin_id, created_at) AS prev_bet_time,
        CASE WHEN bet_type = 'BASE' THEN 1 ELSE 0 END AS is_new_game_group,
        ai_group
    FROM (
        SELECT
            {source_cols},
            t.ab_group AS ai_group
        FROM bet_order_raw AS t
        WHERE
            t.created_at >= TIMESTAMP '{scan_start} 00:00:00'
            AND t.created_at < TIMESTAMP '{scan_end} 00:00:00'
            AND t.currency_type IN ('CNY')
            AND t.status = 'COMPLETED'
            AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
    )
    WHERE {group_filter}
),

free_game_group AS (
    SELECT
        t.spin_id,
        t.user_id,
        t.math_table_id,
        t.created_at,
        t.bet_type,
        t.bet_amount,
        t.payout,
        t.balance_after_bet,
        t.balance_after_payout,
        t.prev_bet_time,
        t.is_new_game_group,
        t.ai_group,
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
        unix_timestamp(t.min_created_at) - unix_timestamp(t.prev_bet_time) AS delta_t_seconds,
        t.bet_amount - LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_bet_amount,
        t.payout - LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_payout,
        t.balance_after_bet
            + t.bet_amount
            - LAG(t.balance_after_payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS balance_transaction,
        CASE WHEN t.payout > t.bet_amount THEN 1 ELSE 0 END AS is_win,
        CASE WHEN t.payout < t.bet_amount THEN 1 ELSE 0 END AS is_lose,
        CASE
            WHEN LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                 > LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                THEN 1
            ELSE 0
        END AS prev_win,
        CASE
            WHEN LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
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
        t.bet_amount,
        t.payout,
        t.balance_after_bet,
        t.delta_t_seconds,
        t.delta_bet_amount,
        t.delta_payout,
        t.balance_transaction,
        t.is_win,
        t.is_lose,
        t.prev_win,
        t.prev_lose,
        LAST(
            CASE
                WHEN t.delta_t_seconds > {SESSION_BREAK_SECONDS} OR t.delta_t_seconds IS NULL
                    THEN t.min_created_at
            END,
            TRUE
        )
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS session_start_ts,
        SUM(CASE WHEN t.delta_t_seconds <= {STREAK_THRESHOLD_SECONDS} THEN 0 ELSE 1 END)
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS streak_group,
        SUM(
            CASE
                WHEN t.delta_t_seconds <= {STREAK_THRESHOLD_SECONDS} AND t.prev_win = 1 AND t.is_win = 1 THEN 0
                ELSE 1
            END
        )
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS win_streak_group,
        SUM(
            CASE
                WHEN t.delta_t_seconds <= {STREAK_THRESHOLD_SECONDS} AND t.prev_lose = 1 AND t.is_lose = 1 THEN 0
                ELSE 1
            END
        )
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS lose_streak_group
    FROM delta_stats AS t
)

SELECT
    t.user_id,
    t.ai_group,
    t.math_table_id,
    CAST(t.session_start_ts AS DATE) AS session_start_date,
    DENSE_RANK() OVER (
        PARTITION BY t.user_id, CAST(t.session_start_ts AS DATE)
        ORDER BY t.session_start_ts
    ) - 1 AS session_group,
    t.session_start_ts,
    t.min_created_at,
    t.max_created_at,
    CAST(t.min_created_at AS DATE) AS activity_date,
    t.spin_id,
    t.fg_rounds,
    t.bet_amount,
    t.delta_bet_amount,
    t.payout,
    t.delta_payout,
    t.balance_after_bet,
    t.delta_t_seconds,
    CASE WHEN t.delta_t_seconds <= {MAX_DELTA_T_GAP_SECONDS} THEN t.delta_t_seconds END AS delta_t_seconds_nogap,
    t.payout - t.bet_amount AS profit,
    CASE WHEN t.balance_transaction > .1 THEN t.balance_transaction END AS deposit,
    CASE WHEN t.balance_transaction < -.1 THEN -t.balance_transaction END AS withdraw,
    ROW_NUMBER() OVER (PARTITION BY t.user_id, t.streak_group ORDER BY t.spin_id, t.min_created_at) AS streak,
    CASE
        WHEN t.payout > t.bet_amount
            THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.win_streak_group ORDER BY t.spin_id, t.min_created_at)
    END AS win_streak,
    CASE
        WHEN t.payout < t.bet_amount
            THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.lose_streak_group ORDER BY t.spin_id, t.min_created_at)
    END AS lose_streak,
    ROW_NUMBER() OVER (PARTITION BY t.user_id, t.session_start_ts ORDER BY t.spin_id, t.min_created_at)
        AS session_bet_index
FROM user_group AS t
WHERE CAST(t.session_start_ts + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR AS DATE)
    NOT IN ({excluded_session_days})
"""


# Percentiles are exact interpolated (Spark percentile == Redshift
# PERCENTILE_CONT, NULLs ignored); a single GROUP BY replaces Redshift's 12
# one-percentile CTE joins. CAST(AVG(...) AS BIGINT) reproduces Redshift's
# integer-AVG truncation on the five integer-valued families. The
# math_table_id IS NOT NULL filter reproduces the Redshift inner joins
# silently dropping NULL-key groups from the grouped output.
def grouped_sql(enriched_view: str, bin_size: int) -> str:
    enriched_cols = ",\n        ".join(f"t.{c}" for c in ENRICHED_DTYPES)
    return f"""
WITH binned AS (
    SELECT
        {enriched_cols},
        FLOOR((t.session_bet_index - 1) / {bin_size}) AS agg_group
    FROM {enriched_view} AS t
),

stats_base AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        MIN(activity_date) AS activity_date,
        MIN(min_created_at) AS min_created_at,
        MAX(max_created_at) AS max_created_at,
        COUNT(user_id) AS bet_rounds,
        SUM(fg_rounds) AS fg_rounds,

        CAST(AVG(delta_t_seconds) AS BIGINT) AS delta_t_seconds_avg,
        percentile(delta_t_seconds, 0.25) AS delta_t_seconds_p25,
        percentile(delta_t_seconds, 0.50) AS delta_t_seconds_median,
        percentile(delta_t_seconds, 0.75) AS delta_t_seconds_p75,
        STDDEV(delta_t_seconds) AS delta_t_seconds_std,
        MIN(delta_t_seconds) AS delta_t_seconds_min,
        MAX(delta_t_seconds) AS delta_t_seconds_max,

        CAST(AVG(delta_t_seconds_nogap) AS BIGINT) AS delta_t_seconds_nogap_avg,
        percentile(delta_t_seconds_nogap, 0.25) AS delta_t_seconds_nogap_p25,
        percentile(delta_t_seconds_nogap, 0.50) AS delta_t_seconds_nogap_median,
        percentile(delta_t_seconds_nogap, 0.75) AS delta_t_seconds_nogap_p75,
        STDDEV(delta_t_seconds_nogap) AS delta_t_seconds_nogap_std,
        MIN(delta_t_seconds_nogap) AS delta_t_seconds_nogap_min,
        MAX(delta_t_seconds_nogap) AS delta_t_seconds_nogap_max,

        AVG(bet_amount) AS bet_amount_avg,
        percentile(bet_amount, 0.25) AS bet_amount_p25,
        percentile(bet_amount, 0.50) AS bet_amount_median,
        percentile(bet_amount, 0.75) AS bet_amount_p75,
        STDDEV(bet_amount) AS bet_amount_std,
        MIN(bet_amount) AS bet_amount_min,
        MAX(bet_amount) AS bet_amount_max,

        AVG(delta_bet_amount) AS delta_bet_amount_avg,
        percentile(delta_bet_amount, 0.25) AS delta_bet_amount_p25,
        percentile(delta_bet_amount, 0.50) AS delta_bet_amount_median,
        percentile(delta_bet_amount, 0.75) AS delta_bet_amount_p75,
        STDDEV(delta_bet_amount) AS delta_bet_amount_std,
        MIN(delta_bet_amount) AS delta_bet_amount_min,
        MAX(delta_bet_amount) AS delta_bet_amount_max,
        SUM(CASE WHEN delta_bet_amount > 0 THEN delta_bet_amount ELSE 0 END) AS accum_pos_delta_bet_amount,
        SUM(CASE WHEN delta_bet_amount < 0 THEN delta_bet_amount ELSE 0 END) AS accum_neg_delta_bet_amount,

        AVG(payout) AS payout_avg,
        percentile(payout, 0.25) AS payout_p25,
        percentile(payout, 0.50) AS payout_median,
        percentile(payout, 0.75) AS payout_p75,
        STDDEV(payout) AS payout_std,
        MIN(payout) AS payout_min,
        MAX(payout) AS payout_max,
        SUM(CASE WHEN payout > 0 THEN 1 END) * 1.0 / NULLIF(COUNT(user_id), 0) AS payout_rate,

        AVG(payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_mean,
        MAX(payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_max,
        MIN(payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_min,
        percentile(payout * 1.0 / NULLIF(bet_amount, 0), 0.25) AS rtp_p25,
        percentile(payout * 1.0 / NULLIF(bet_amount, 0), 0.50) AS rtp_median,
        percentile(payout * 1.0 / NULLIF(bet_amount, 0), 0.75) AS rtp_p75,

        AVG(profit) AS profit_avg,
        percentile(profit, 0.25) AS profit_p25,
        percentile(profit, 0.50) AS profit_median,
        percentile(profit, 0.75) AS profit_p75,
        STDDEV(profit) AS profit_std,
        MIN(profit) AS profit_min,
        MAX(profit) AS profit_max,
        SUM(CASE WHEN profit > 0 THEN profit ELSE 0 END) AS accum_pos_profit,
        SUM(CASE WHEN profit < 0 THEN profit ELSE 0 END) AS accum_neg_profit,
        SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(user_id), 0) AS profit_rate,

        AVG(delta_payout) AS delta_payout_avg,
        percentile(delta_payout, 0.25) AS delta_payout_p25,
        percentile(delta_payout, 0.50) AS delta_payout_median,
        percentile(delta_payout, 0.75) AS delta_payout_p75,
        STDDEV(delta_payout) AS delta_payout_std,
        MIN(delta_payout) AS delta_payout_min,
        MAX(delta_payout) AS delta_payout_max,

        AVG(balance_after_bet) AS balance_after_bet_avg,
        percentile(balance_after_bet, 0.25) AS balance_after_bet_p25,
        percentile(balance_after_bet, 0.50) AS balance_after_bet_median,
        percentile(balance_after_bet, 0.75) AS balance_after_bet_p75,
        STDDEV(balance_after_bet) AS balance_after_bet_std,
        MIN(balance_after_bet) AS balance_after_bet_min,
        MAX(balance_after_bet) AS balance_after_bet_max,

        SUM(CASE WHEN deposit > 0 THEN 1 ELSE 0 END) AS num_deposit,
        SUM(CASE WHEN deposit > 0 THEN deposit ELSE 0 END) AS accum_deposit,
        SUM(CASE WHEN withdraw > 0 THEN 1 ELSE 0 END) AS num_withdraw,
        SUM(CASE WHEN withdraw > 0 THEN withdraw ELSE 0 END) AS accum_withdraw,

        CAST(AVG(streak) AS BIGINT) AS streak_avg,
        percentile(streak, 0.25) AS streak_p25,
        percentile(streak, 0.50) AS streak_median,
        percentile(streak, 0.75) AS streak_p75,
        STDDEV(streak) AS streak_std,
        MIN(streak) AS streak_min,
        MAX(streak) AS streak_max,

        CAST(AVG(win_streak) AS BIGINT) AS win_streak_avg,
        percentile(win_streak, 0.25) AS win_streak_p25,
        percentile(win_streak, 0.50) AS win_streak_median,
        percentile(win_streak, 0.75) AS win_streak_p75,
        STDDEV(win_streak) AS win_streak_std,
        MIN(win_streak) AS win_streak_min,
        MAX(win_streak) AS win_streak_max,

        CAST(AVG(lose_streak) AS BIGINT) AS lose_streak_avg,
        percentile(lose_streak, 0.25) AS lose_streak_p25,
        percentile(lose_streak, 0.50) AS lose_streak_median,
        percentile(lose_streak, 0.75) AS lose_streak_p75,
        STDDEV(lose_streak) AS lose_streak_std,
        MIN(lose_streak) AS lose_streak_min,
        MAX(lose_streak) AS lose_streak_max
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
)

SELECT
    b.user_id, b.ai_group, b.math_table_id, b.session_start_date, b.session_group, b.agg_group,
    b.activity_date, b.min_created_at, b.max_created_at, b.bet_rounds, b.fg_rounds,
    b.delta_t_seconds_avg, b.delta_t_seconds_p25, b.delta_t_seconds_median, b.delta_t_seconds_p75,
    b.delta_t_seconds_std, b.delta_t_seconds_min, b.delta_t_seconds_max,
    b.delta_t_seconds_nogap_avg, b.delta_t_seconds_nogap_p25, b.delta_t_seconds_nogap_median,
    b.delta_t_seconds_nogap_p75, b.delta_t_seconds_nogap_std, b.delta_t_seconds_nogap_min, b.delta_t_seconds_nogap_max,
    b.bet_amount_avg, b.bet_amount_p25, b.bet_amount_median, b.bet_amount_p75, b.bet_amount_std,
    b.bet_amount_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_cv,
    b.bet_amount_min,
    b.bet_amount_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_drawdown_ratio,
    b.bet_amount_max,
    b.bet_amount_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_spike_ratio,
    b.delta_bet_amount_avg, b.delta_bet_amount_p25, b.delta_bet_amount_median, b.delta_bet_amount_p75,
    b.delta_bet_amount_std,
    COALESCE(b.delta_bet_amount_std * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_cv,
    b.delta_bet_amount_min,
    COALESCE(b.delta_bet_amount_min * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_drawdown_ratio,
    b.delta_bet_amount_max,
    COALESCE(b.delta_bet_amount_max * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_spike_ratio,
    b.accum_pos_delta_bet_amount,
    b.accum_neg_delta_bet_amount,
    b.accum_pos_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_pos_delta_bet_amount_ratio,
    b.accum_neg_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_neg_delta_bet_amount_ratio,
    b.payout_avg, b.payout_p25, b.payout_median, b.payout_p75, b.payout_std,
    b.payout_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_cv,
    b.payout_min,
    b.payout_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_drawdown_ratio,
    b.payout_max,
    b.payout_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_spike_ratio,
    b.payout_rate,
    b.rtp_mean, b.rtp_max, b.rtp_min, b.rtp_p25, b.rtp_median, b.rtp_p75,
    b.profit_avg, b.profit_p25, b.profit_median, b.profit_p75, b.profit_std,
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
    b.delta_payout_avg, b.delta_payout_p25, b.delta_payout_median, b.delta_payout_p75, b.delta_payout_std,
    COALESCE(b.delta_payout_std * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_cv,
    b.delta_payout_min,
    COALESCE(b.delta_payout_min * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_drawdown_ratio,
    b.delta_payout_max,
    COALESCE(b.delta_payout_max * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_spike_ratio,
    b.balance_after_bet_avg, b.balance_after_bet_p25, b.balance_after_bet_median, b.balance_after_bet_p75,
    b.balance_after_bet_std,
    b.balance_after_bet_std * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_cv,
    b.balance_after_bet_min,
    b.balance_after_bet_min * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_drawdown_ratio,
    b.balance_after_bet_max,
    b.balance_after_bet_max * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_spike_ratio,
    b.num_deposit, b.accum_deposit,
    b.accum_deposit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_deposit_ratio,
    b.num_withdraw, b.accum_withdraw,
    b.accum_withdraw * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_withdraw_ratio,
    COALESCE(b.streak_avg, 0) AS streak_avg,
    COALESCE(b.streak_p25, 0) AS streak_p25,
    COALESCE(b.streak_median, 0) AS streak_median,
    COALESCE(b.streak_p75, 0) AS streak_p75,
    COALESCE(b.streak_std, 0) AS streak_std,
    COALESCE(b.streak_min, 0) AS streak_min,
    COALESCE(b.streak_max, 0) AS streak_max,
    COALESCE(b.win_streak_avg, 0) AS win_streak_avg,
    COALESCE(b.win_streak_p25, 0) AS win_streak_p25,
    COALESCE(b.win_streak_median, 0) AS win_streak_median,
    COALESCE(b.win_streak_p75, 0) AS win_streak_p75,
    COALESCE(b.win_streak_std, 0) AS win_streak_std,
    COALESCE(b.win_streak_min, 0) AS win_streak_min,
    COALESCE(b.win_streak_max, 0) AS win_streak_max,
    COALESCE(b.lose_streak_avg, 0) AS lose_streak_avg,
    COALESCE(b.lose_streak_p25, 0) AS lose_streak_p25,
    COALESCE(b.lose_streak_median, 0) AS lose_streak_median,
    COALESCE(b.lose_streak_p75, 0) AS lose_streak_p75,
    COALESCE(b.lose_streak_std, 0) AS lose_streak_std,
    COALESCE(b.lose_streak_min, 0) AS lose_streak_min,
    COALESCE(b.lose_streak_max, 0) AS lose_streak_max
FROM stats_base AS b
WHERE b.math_table_id IS NOT NULL
"""


def existing_max_activity_month(spark, root: str) -> date | None:
    """Newest month already materialized under features_enriched, for the
    self-healing default window; None when the dataset is missing/unreadable."""
    try:
        df = spark.read.parquet(f"{root}/features_enriched")
        row = df.select(F.max("activity_date").alias("mx")).collect()[0]
        if row["mx"] is None:
            return None
        mx = row["mx"]
        return month_start(mx.date() if isinstance(mx, datetime) else mx)
    except Exception as e:
        print(f"watermark read failed for {root} ({e}); falling back to previous-month window")
        return None


def read_sidecar(spark, root: str) -> dict | None:
    sc = spark.sparkContext
    hadoop_path = sc._jvm.org.apache.hadoop.fs.Path(f"{root}/_feature_config.json")
    fs = hadoop_path.getFileSystem(sc._jsc.hadoopConfiguration())
    if not fs.exists(hadoop_path):
        return None
    # Copy the stream JVM-side: FSDataInputStream.read(buf) can't work from
    # py4j (bytearray args pass BY VALUE, the Python buffer stays empty) and
    # spark.read.text skips underscore-prefixed files as hidden.
    stream = fs.open(hadoop_path)
    try:
        out = sc._jvm.java.io.ByteArrayOutputStream()
        sc._jvm.org.apache.hadoop.io.IOUtils.copyBytes(stream, out, 65536, False)
        return json.loads(out.toString("UTF-8"))
    finally:
        stream.close()


def sidecar_payload(prefix: str, group: str, bins: list, output_end: date) -> dict:
    return {
        "config": {
            "game_id": "SS03",
            "output_prefix": prefix,
            "date_start": "2026-01-01" if group in ("default", "ai") else "2026-06-10",
            "date_end": str(output_end),
            "ai_groups": ["AI", "AB_TEST_A", "AB_TEST_B", "Default"],
            "selected_groups": [GROUPS[group][3]],
            "partition_cols": ["math_table_id"],
            "bin_size": bins,
            "session_break_threshold_seconds": SESSION_BREAK_SECONDS,
            "streak_threshold_seconds": STREAK_THRESHOLD_SECONDS,
            "max_delta_t_gap_seconds": MAX_DELTA_T_GAP_SECONDS,
            "drop_incomplete_tail_groups": False,
            # Recorded as a semantic field: the grouping-policy transition
            # days are excluded by BJ session start (see the constant).
            "extra_where_clauses": [f"session_start_date_bj NOT IN {EXCLUDED_SESSION_START_DATES_BJ}"],
            "requires_full_history": False,
            "lookback_days": None,
        },
        "semantic_fields": SIDECAR_SEMANTIC_FIELDS,
        "engine": "sagemaker-cold-data",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def check_semantic_drift(spark, root: str, payload: dict, allow_drift: bool) -> None:
    """Port of the Redshift runner's sidecar guard: appending months computed
    under changed semantics (e.g. a different bin_size set) next to old-
    semantics history corrupts the dataset."""
    existing = read_sidecar(spark, root)
    if existing is None:
        return
    old_cfg, new_cfg = existing.get("config", {}), payload["config"]
    drift = {f: (old_cfg.get(f), new_cfg[f]) for f in SIDECAR_SEMANTIC_FIELDS if old_cfg.get(f) != new_cfg[f]}
    if drift and not allow_drift:
        raise SystemExit(
            f"semantic drift vs {root}/_feature_config.json: {drift}. "
            "Recompute the full dataset under the new semantics (or pass --allow-semantic-drift "
            "if the sidecar itself is wrong)."
        )
    if drift:
        print(f"WARNING: proceeding despite semantic drift: {drift}")


def write_sidecar(spark, root: str, payload: dict) -> None:
    sc = spark.sparkContext
    hadoop_path = sc._jvm.org.apache.hadoop.fs.Path(f"{root}/_feature_config.json")
    fs = hadoop_path.getFileSystem(sc._jsc.hadoopConfiguration())
    stream = fs.create(hadoop_path, True)
    try:
        stream.write(json.dumps(payload, indent=2).encode("utf-8"))
    finally:
        stream.close()


def write_dataset(df, dtypes: dict, root: str, dataset: str) -> None:
    out = df.select([F.col(c).cast(t).alias(c) for c, t in dtypes.items()])
    out = (
        out.withColumn("year", F.year("activity_date"))
        .withColumn("month", F.month("activity_date"))
        .withColumn("_processed_at", F.current_timestamp())
    )
    # One shuffle partition per month: matches the existing one-file-per-month
    # layout (a month of per-bet SS03 rows is a few hundred MB at most).
    (out.repartition("year", "month").write.partitionBy("year", "month").mode("overwrite").parquet(f"{root}/{dataset}"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="s3://... partition_cold_data/bet_order root")
    parser.add_argument("--output-root-base", required=True, help="s3://... prefix that holds the four output roots")
    parser.add_argument("--groups", default="default,ai,ab_test_a,ab_test_b", help="comma-separated subset of groups")
    parser.add_argument(
        "--output-start",
        default=None,
        help="recompute whole months from this date's month, YYYY-MM-DD"
        " (default: self-healing — the earlier of the previous month and the month after the newest data on S3)",
    )
    parser.add_argument(
        "--output-end",
        default=None,
        help="exclusive end date, YYYY-MM-DD; must be month-aligned or in the future"
        " (a mid-month historical end would truncate that month's partition). Default: tomorrow, UTC",
    )
    parser.add_argument(
        "--input-region",
        default="us-west-2",
        help="region of the input bucket (the slot_orders_ab_group dataset lives in our us-west-2 bucket)",
    )
    parser.add_argument(
        "--allow-semantic-drift",
        action="store_true",
        help="proceed when the sidecar's semantic fields differ (only when the sidecar itself is wrong)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    today = utc_today()
    output_end = date.fromisoformat(args.output_end) if args.output_end else today + timedelta(days=1)
    if args.output_end and output_end < today and output_end != month_start(output_end):
        raise SystemExit(
            f"--output-end {output_end} is historical and not month-aligned: overwriting its month with a"
            " partial recompute would truncate the partition. Use the 1st of the following month."
        )
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise SystemExit(f"unknown groups: {unknown}; valid: {sorted(GROUPS)}")

    spark = build_spark_session("SS03_feature_engineer_cold_data", args.input_root, args.input_region)

    if args.output_start:
        output_start = month_start(date.fromisoformat(args.output_start))
    else:
        output_start = prev_month_start(today)
        # Self-heal a missed firing: never leave a gap between the newest
        # month on S3 and the window (watermark from the 'default' group).
        watermark = existing_max_activity_month(spark, f"{args.output_root_base}/{GROUPS['default'][0]}")
        if watermark is not None and watermark < output_start:
            print(f"self-heal: newest data month {watermark} predates default window; widening")
            output_start = watermark
    if output_start < COLD_DATA_FLOOR:
        raise SystemExit(
            f"refusing to recompute months before {COLD_DATA_FLOOR}: cold data starts 2026-02-10, and"
            " overwriting 2026-02 (or earlier) would erase Redshift-era rows the warehouse cannot reproduce."
        )
    scan_start = output_start - timedelta(days=LOOKBACK_DAYS)
    print(f"output window: [{output_start}, {output_end}) | scan from {scan_start} | groups: {groups}")

    # Reading inside game_id=SS03/ pins the game by path (no game_id column
    # survives — it is the consumed partition level), and prunes other games.
    bet_order = spark.read.parquet(f"{args.input_root}/game_id=SS03")
    check_schema(bet_order, REQUIRED_COLUMNS, table_name="slot_orders_ab_group orders")
    raw = prune_period_days(bet_order, scan_start, output_end, margin_days=1)
    if raw.limit(1).count() == 0:
        raise SystemExit(f"no bet_order rows for SS03 in [{scan_start}, {output_end}); refusing to overwrite")
    raw.createOrReplaceTempView("bet_order_raw")

    for group in groups:
        prefix, group_filter, bins, _ = GROUPS[group]
        root = f"{args.output_root_base}/{prefix}"
        payload = sidecar_payload(prefix, group, bins, output_end)
        check_semantic_drift(spark, root, payload, args.allow_semantic_drift)
        print(f"===== {group} -> {root} (bins {bins}, window [{output_start}, {output_end})) =====")

        enriched = spark.sql(enriched_sql(group_filter, scan_start, output_end)).persist()
        enriched.createOrReplaceTempView(f"enriched_{group}")
        # Lookback rows participate in windows/bins above but are not written:
        # their months are not being overwritten.
        out_enriched = enriched.filter(F.col("activity_date") >= F.lit(output_start))
        write_dataset(out_enriched, ENRICHED_DTYPES, root, "features_enriched")
        print(f"[{group}] features_enriched written")

        for n in bins:
            grouped = spark.sql(grouped_sql(f"enriched_{group}", n))
            out_grouped = grouped.filter(F.col("activity_date") >= F.lit(output_start))
            write_dataset(out_grouped, grouped_dtypes(), root, f"features_grouped_binsize_{n}")
            print(f"[{group}] features_grouped_binsize_{n} written")

        write_sidecar(spark, root, payload)
        enriched.unpersist()

    spark.stop()
    print("done")


if __name__ == "__main__":
    main()
