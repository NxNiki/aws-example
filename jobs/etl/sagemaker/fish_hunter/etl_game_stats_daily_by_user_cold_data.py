"""FM01 daily/weekly/monthly per-user game stats from S3 cold data — PySpark job.

ETL job: cold-data source for the fish_hunter dashboard tab metrics
(``ab_test_group`` dimension plus the ``user_*`` metrics). Same query logic as
etl_game_stats_daily_by_user.py, but reads bullet parquet directly from
``s3://oceanhunter-production-data-warehouse/transformed_data/cold_data/bullet``
(partition-pruned by year=/month=/day=) instead of Redshift ``public.bullet``.
Output datasets: ``<output-root>/{daily,weekly,monthly}_stats/period=YYYY-MM-DD/``
where ``period`` is the day / week start / month start of ``activity_date``.

Behavior: one row per (user, period, ab_test_group, fish_value) for EVERY betting
user -- not only fish-killers (kill-specific metrics are NULL/0 for non-killers;
the ``user_killed_fish`` flag segments killers). ``ab_test_group`` collapses each
(user, day) to exactly ONE strategy group, priority RC_*/CR_* strategies
(literal strategy_name kept, MIN tie-break) > RISK_CONTROLLED > BOOST_POOL >
DYNAMIC_RTP family (MIN tie-break) > DEFAULT_FALLBACK. The companion
``ab_test_group_combined`` column carries the same assignment with each family
collapsed to its rollup label (RC_* -> ``RC_ALL``, CR_* -> ``CR_ALL``, all
other labels pass through), giving a per-family dimension without changing
the row grain. Each run
recomputes whole periods in [output-start, output-end) and dynamic-partition-
overwrites exactly the ``period=`` directories it produced, so windowed re-runs
are idempotent; a partial trailing period (output-end not on a period boundary)
self-heals when a later run covers the full period. The scan reaches
SCAN_LOOKBACK_DAYS before the (period-aligned) start, matching the Redshift
job's raw window, so kill streaks touching the start boundary are complete.

Runs standalone on the SageMaker Spark container -- no bituslabs_ds imports
(ETL constants from bituslabs_ds.config are inlined below). Submit with
etl_game_stats_daily_by_user_cold_data_submit.py.
"""

import argparse
from datetime import date, datetime, time, timedelta
from textwrap import dedent

from pyspark.sql import SparkSession, functions as F

# Inlined from bituslabs_ds.config (this job runs without the package).
EXCLUDED_OP_CODES = "('B26', 'TST', 'TSB', 'TSO')"
DELTA_T_MAX_SECONDS = 1800
DELTA_T_MIN_SECONDS = 0.25
# Asia/Shanghai has no DST, so a fixed offset equals CONVERT_TIMEZONE.
BJ_UTC_OFFSET_HOURS = 8

# A strategy_name starting with one of these prefixes claims the whole
# user-day (top priority); ab_test_group_combined collapses each family into its
# rollup label while ab_test_group keeps the literal strategy_name.
RC_PREFIX = "RC_"
CR_PREFIX = "CR_"
RC_ROLLUP_GROUP = "RC_ALL"
CR_ROLLUP_GROUP = "CR_ALL"

# Same knobs as the Redshift job.
SCAN_LOOKBACK_DAYS = 30
STREAK_SESSION_THRESH = 600
STREAK_KILL_THRESH = 3  # nearly 10% of all killing intervals.

AGG_LEVELS = {
    "daily": ("activity_date", "daily_stats"),
    "weekly": ("activity_week", "weekly_stats"),
    "monthly": ("activity_month", "monthly_stats"),
}

REQUIRED_COLUMNS = [
    "user_id",
    "room_id",
    "bullet_id",
    "strategy_name",
    "event_timestamp",
    "created_at",
    "bet",
    "payout",
    "profit",
    "fish_value",
    "killed",
    "op_code",
    "currency_type",
    "game_id",
    "year",
    "month",
    "day",
]


def generate_query(
    stats_agg_col: str,
    effective_start: date,
    output_end: date,
    scan_start_utc: datetime,
    scan_end_utc: datetime,
    game_id: str,
    currency: str,
) -> str:
    """Spark-SQL version of the etl_game_stats_daily_by_user.py query over ``bullet_raw``.

    Translation notes versus the Redshift dialect: BJ time is
    ``created_at + 8h`` (session timezone is UTC); ``DATEDIFF(SECOND, ...)``
    becomes a ``unix_timestamp`` difference (both floor to whole seconds);
    ``EXTRACT(EPOCH FROM interval)`` becomes a ``CAST(ts AS DOUBLE)``
    difference to keep the sub-second deltas that DELTA_T_MIN_SECONDS guards.
    """
    is_rc_cr = f"substr(strategy_name, 1, 3) IN ('{RC_PREFIX}', '{CR_PREFIX}')"
    return dedent(
        f"""
        -- 1. FETCH RAW DATA (Keep strictly RAW columns to enable partition pruning)
        WITH base_data AS (
            SELECT
                b.user_id,
                b.room_id,
                b.bullet_id,
                b.strategy_name, -- Keep Raw
                CAST(b.event_timestamp AS TIMESTAMP) AS bet_time,
                CAST(b.payout AS DOUBLE) AS payout,
                CAST(b.bet AS DOUBLE) AS bet,
                CAST(b.fish_value AS DOUBLE) AS fish_value,
                CAST(b.killed AS INT) AS killed,
                COALESCE(CAST(b.profit AS DOUBLE), CAST(b.payout AS DOUBLE) - CAST(b.bet AS DOUBLE)) AS profit,
                -- Sequence metrics are DAY-partitioned (each activity_date is
                -- self-contained, so incremental pulls and full reloads agree);
                -- bj_date_last_bet below stays cross-day on purpose.
                LAG(CAST(b.event_timestamp AS TIMESTAMP)) OVER (
                    PARTITION BY b.user_id, CAST(CAST(b.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR AS DATE)
                    ORDER BY b.bullet_id, CAST(b.event_timestamp AS TIMESTAMP)
                ) AS prev_bet_time,
                LAG(CAST(b.bet AS DOUBLE)) OVER (
                    PARTITION BY b.user_id, CAST(CAST(b.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR AS DATE)
                    ORDER BY b.bullet_id, CAST(b.event_timestamp AS TIMESTAMP)
                ) AS prev_bet_amount,
                CAST(CAST(b.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR AS DATE) AS activity_date,
                CAST(DATE_TRUNC('week', CAST(b.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR) AS DATE) AS activity_week,
                CAST(DATE_TRUNC('month', CAST(b.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR) AS DATE) AS activity_month
            FROM bullet_raw b
            WHERE
                b.currency_type = '{currency}'
                AND b.game_id = '{game_id}'
                AND b.op_code NOT IN {EXCLUDED_OP_CODES}
                AND CAST(b.created_at AS TIMESTAMP) >= TIMESTAMP '{scan_start_utc:%Y-%m-%d %H:%M:%S}'
                AND CAST(b.created_at AS TIMESTAMP) < TIMESTAMP '{scan_end_utc:%Y-%m-%d %H:%M:%S}'
        ),

        -- 2. DETERMINE USER DAILY GROUP (Logic applied inside SUM)
        user_ab_test_group AS (
            SELECT
                user_id,
                activity_date,
                activity_week,
                activity_month,
                -- Assign the whole user-day to its highest-priority strategy_name.
                -- A single bet in a higher tier claims the day. RC_*/CR_* keep
                -- their literal label; MIN is the deterministic tie-break when
                -- several coexist in a day (spanning both families, so a day
                -- with both lands on the lexicographically smallest label).
                CASE
                    WHEN MAX(CASE WHEN {is_rc_cr} THEN 1 ELSE 0 END) > 0
                        THEN MIN(CASE WHEN {is_rc_cr} THEN strategy_name END)
                    WHEN MAX(CASE WHEN strategy_name = 'RISK_CONTROLLED' THEN 1 ELSE 0 END) > 0 THEN 'RISK_CONTROLLED'
                    WHEN MAX(CASE WHEN strategy_name = 'BOOST_POOL' THEN 1 ELSE 0 END) > 0 THEN 'BOOST_POOL'
                    WHEN MAX(CASE WHEN strategy_name IN ('DYNAMIC_RTP', 'DYNAMIC_RTP_V2', 'DYNAMIC_RTP_V3') THEN 1 ELSE 0 END) > 0
                        THEN MIN(CASE WHEN strategy_name IN ('DYNAMIC_RTP', 'DYNAMIC_RTP_V2', 'DYNAMIC_RTP_V3') THEN strategy_name END)
                    ELSE 'DEFAULT_FALLBACK'
                END AS ab_test_group,
                LAG(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date) AS bj_date_last_bet
            FROM base_data
            GROUP BY user_id, activity_date, activity_week, activity_month
        ),

        -- GET KILL STREAK LENGTH:
        base_kills AS (
            SELECT
                user_id,
                activity_date,
                bullet_id,
                bet_time
            FROM base_data
            WHERE killed = 1
        ),

        calculate_islands AS (
            SELECT
                user_id,
                activity_date,
                bullet_id,
                bet_time,
                -- Checks if ANY fish was killed recently.
                CASE
                    WHEN unix_timestamp(bet_time) - unix_timestamp(LAG(bet_time) OVER (PARTITION BY user_id ORDER BY bullet_id, bet_time)) > {STREAK_KILL_THRESH}
                        OR LAG(bet_time) OVER (PARTITION BY user_id ORDER BY bullet_id, bet_time) IS NULL
                    THEN 1 ELSE 0
                END AS is_new_global_streak
            FROM base_kills
        ),

        streak_ids AS (
            SELECT
                user_id,
                activity_date,
                bet_time,
                SUM(is_new_global_streak) OVER (
                    PARTITION BY user_id ORDER BY bullet_id, bet_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS global_streak_id
            FROM calculate_islands
        ),

        streak_lengths AS (
            SELECT
                user_id,
                activity_date,
                -- Calculate the length of the specific streak instance this row belongs to
                COUNT(*) OVER (PARTITION BY user_id, global_streak_id) AS global_streak_len
            FROM streak_ids
        ),

        max_kill_streak_length AS (
            SELECT
                user_id,
                activity_date,
                MAX(global_streak_len) AS max_kill_streak,
                AVG(global_streak_len) AS avg_kill_streak
            FROM streak_lengths
            GROUP BY user_id, activity_date
        ),

        -- GET BET SESSION STATS:
        user_session_id AS (
            SELECT
                t.activity_date,
                t.user_id,
                t.bet_time,
                t.killed,
                SUM(CASE
                        WHEN unix_timestamp(t.bet_time) - unix_timestamp(t.prev_bet_time) < {STREAK_SESSION_THRESH} THEN 0
                        ELSE 1
                    END) OVER (
                        PARTITION BY t.activity_date, t.user_id
                        ORDER BY t.bullet_id, t.bet_time
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS session_id,
                ROW_NUMBER() OVER (
                        PARTITION BY t.activity_date, t.user_id
                        ORDER BY t.bullet_id, t.bet_time
                    ) AS bet_index

            FROM base_data t
        ),

        user_session_length AS (
            SELECT
                t.activity_date,
                t.user_id,
                t.session_id,

                unix_timestamp(MIN(CASE WHEN t.killed = 1 THEN t.bet_time END)) - unix_timestamp(MIN(t.bet_time)) AS seconds_to_kill_fish,

                MIN(CASE WHEN t.killed = 1 THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish,

                COUNT(t.user_id) AS session_length
            FROM user_session_id t
            GROUP BY t.user_id, t.activity_date, t.session_id
        ),

        user_session_stats AS (
            SELECT
                t.activity_date,
                t.user_id,
                COUNT(DISTINCT t.session_id) AS num_streak_sessions,
                AVG(CAST(t.session_length AS DOUBLE)) AS avg_streak_length,
                MAX(t.session_length) AS max_streak_length,
                MIN(t.session_length) AS min_streak_length,

                AVG(seconds_to_kill_fish) AS seconds_to_kill_fish,

                AVG(bets_to_kill_fish) AS bets_to_kill_fish

            FROM user_session_length t
            GROUP BY t.user_id, t.activity_date
        ),

        user_session_stats_agg AS (
            SELECT
                u.ab_test_group,
                t.user_id,
                u.{stats_agg_col},
                SUM(t.num_streak_sessions)      AS user_num_streak_sessions,
                AVG(t.avg_streak_length)        AS user_avg_streak_length,
                MAX(t.max_streak_length)        AS user_max_streak_length,
                MIN(t.min_streak_length)        AS user_min_streak_length,

                AVG(t.seconds_to_kill_fish)         AS user_seconds_to_kill_fish,

                AVG(t.bets_to_kill_fish)        AS user_bets_to_kill_fish
            FROM user_session_stats t
            JOIN user_ab_test_group u ON t.user_id = u.user_id AND t.activity_date = u.activity_date
            GROUP BY u.ab_test_group, t.user_id, u.{stats_agg_col}
        ),

        max_kill_streak_length_agg AS (
            SELECT
                u.ab_test_group,
                t.user_id,
                u.{stats_agg_col},
                MAX(t.max_kill_streak)          AS user_max_kill_streak,

                AVG(t.avg_kill_streak)          AS user_avg_kill_streak
            FROM max_kill_streak_length t
            JOIN user_ab_test_group u ON t.user_id = u.user_id AND t.activity_date = u.activity_date
            GROUP BY u.ab_test_group, t.user_id, u.{stats_agg_col}
        ),

        -- 4a. USER-DAY LEVEL COLUMNS that cannot be sliced by fish_value
        -- (distinct rooms overlap across slices; the CV needs the full sample).
        user_day_stats AS (
            SELECT
                b.user_id,
                u.ab_test_group,
                b.{stats_agg_col},
                COUNT(DISTINCT CONCAT(CAST(b.user_id AS STRING), '-', CAST(b.room_id AS STRING))) AS user_num_rooms,
                STDDEV(b.profit) / NULLIF(ABS(AVG(b.profit)), 0)  AS user_profit_coef_var
            FROM base_data b
            JOIN user_ab_test_group u ON b.user_id = u.user_id AND b.activity_date = u.activity_date
            GROUP BY b.user_id, u.ab_test_group, b.{stats_agg_col}
        ),

        -- 4. USER-LEVEL STATS BY DAILY GROUP AND FISH VALUE. One row per
        -- (period, user, ab_test_group, fish_value): each bullet in exactly one
        -- row, so the dashboard's custom fish-level ranges ([min, max]
        -- inclusive over fish_value) recombine every sliced metric exactly.
        stats_by_user_date AS (
            SELECT
                b.user_id,
                u.ab_test_group,
                b.fish_value,
                b.{stats_agg_col},
                COUNT(b.user_id)                              AS user_num_bets,

                SUM(b.killed)                                 AS user_num_killed_bullets,

                SUM(b.bet)                                                             AS user_total_bet,
                AVG(b.bet)                                                             AS user_avg_bet_amount,
                AVG(CASE WHEN (CAST(b.bet_time AS DOUBLE) - CAST(b.prev_bet_time AS DOUBLE)) <= {DELTA_T_MAX_SECONDS} THEN GREATEST(CAST(b.bet_time AS DOUBLE) - CAST(b.prev_bet_time AS DOUBLE), {DELTA_T_MIN_SECONDS}) END)
                                                                                       AS user_avg_delta_t_seconds,
                COUNT(CASE WHEN (CAST(b.bet_time AS DOUBLE) - CAST(b.prev_bet_time AS DOUBLE)) <= {DELTA_T_MAX_SECONDS} THEN 1 END)
                                                                                       AS user_num_delta_t,
                SUM(b.payout)                                                          AS user_total_payout,
                SUM(b.profit)                                                          AS user_total_profit,
                MAX(b.profit)                                                          AS user_max_profit,
                ROUND(SUM(b.payout) / NULLIF(SUM(b.bet), 0), 3)                       AS user_rtp,
                MAX(CASE WHEN b.killed >= 1 THEN 1 ELSE 0 END)                        AS user_killed_fish,

                AVG(b.fish_value)                                                      AS user_avg_fish_value,
                AVG(CASE WHEN b.killed = 1 THEN b.fish_value END)                     AS user_avg_killed_fish_value,
                AVG(b.profit)                                                          AS user_bullet_avg_profit,
                AVG(CASE WHEN b.killed = 1 THEN b.profit END)                         AS user_bullet_kill_avg_profit,

                -- delta bet amount metrics:
                SUM(CASE WHEN (b.bet - b.prev_bet_amount) > 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_pos_delta_bet,
                SUM(CASE WHEN (b.bet - b.prev_bet_amount) < 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_neg_delta_bet,
                AVG(CASE WHEN (b.bet - b.prev_bet_amount) > 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_pos_delta_bet_avg,
                AVG(CASE WHEN (b.bet - b.prev_bet_amount) < 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_neg_delta_bet_avg,
                SUM(b.bet - b.prev_bet_amount) AS user_accu_delta_bet,
                AVG(b.bet - b.prev_bet_amount) AS user_accu_delta_bet_avg,
                COUNT(CASE WHEN (b.bet - b.prev_bet_amount) > 0 THEN 1 END) AS user_pos_delta_bet_num,
                COUNT(CASE WHEN (b.bet - b.prev_bet_amount) < 0 THEN 1 END) AS user_neg_delta_bet_num,
                COUNT(b.prev_bet_amount) AS user_num_delta_bet
            FROM base_data b
            JOIN user_ab_test_group u ON b.user_id = u.user_id AND b.activity_date = u.activity_date
            GROUP BY b.user_id, u.ab_test_group, b.fish_value, b.{stats_agg_col}
            -- Include ALL betting users, not only fish-killers. Kill-specific metrics
            -- already degrade to NULL/0 for non-killers (CASE WHEN killed / NULLIF), while
            -- downstream user counts & retention (day0_num_users, num_active_users, ...) need
            -- the full active-user set; a `HAVING MAX(b.killed) > 0` here undercounted them.
            -- Segment to killers downstream via the user_killed_fish flag when needed.
        )

        -- 5. FINAL JOIN & FORMATTING. Row grain: (period, user, ab_test_group,
        -- fish_value). The session/streak columns (t4/t5) and the user-day
        -- columns (t6) are computed per user-day and repeat identically on
        -- each of the user's fish_value rows; the dashboard keeps their first
        -- value when collapsing.
        SELECT
            t1.user_id,
            t1.ab_test_group,
            -- Family rollup dimension: RC_*/CR_* variants collapse to RC_ALL /
            -- CR_ALL, every other label passes through unchanged. Functionally
            -- dependent on ab_test_group, so the row grain does not change.
            CASE
                WHEN substr(t1.ab_test_group, 1, 3) = '{RC_PREFIX}' THEN '{RC_ROLLUP_GROUP}'
                WHEN substr(t1.ab_test_group, 1, 3) = '{CR_PREFIX}' THEN '{CR_ROLLUP_GROUP}'
                ELSE t1.ab_test_group
            END AS ab_test_group_combined,
            t1.fish_value,
            t1.{stats_agg_col} AS activity_date,
            t6.user_num_rooms,
            t1.user_num_bets,
            t1.user_num_killed_bullets,
            t1.user_killed_fish,

            -- Bet / payout / profit:
            t1.user_total_bet,
            t1.user_avg_bet_amount,
            t1.user_avg_delta_t_seconds,
            t1.user_num_delta_t,
            t1.user_total_payout,
            t1.user_total_profit,
            t1.user_max_profit,
            t1.user_rtp,
            t6.user_profit_coef_var,
            ROUND(CAST(t1.user_num_killed_bullets AS DOUBLE) / NULLIF(t1.user_num_bets, 0), 3) AS user_bullet_kill_ratio,

            -- Fish value:
            t1.user_avg_fish_value,
            t1.user_avg_killed_fish_value,
            t1.user_bullet_avg_profit,
            t1.user_bullet_kill_avg_profit,

            -- delta bet amount metrics:
            t1.user_accu_pos_delta_bet,
            t1.user_accu_neg_delta_bet,
            t1.user_accu_pos_delta_bet_avg,
            t1.user_accu_neg_delta_bet_avg,
            t1.user_accu_delta_bet,
            t1.user_accu_delta_bet_avg,
            t1.user_pos_delta_bet_num,
            t1.user_neg_delta_bet_num,
            t1.user_num_delta_bet,

            -- Session stats:
            t4.user_num_streak_sessions,
            t4.user_avg_streak_length,
            t4.user_max_streak_length,
            t4.user_min_streak_length,
            t4.user_seconds_to_kill_fish,
            t4.user_bets_to_kill_fish,

            -- Kill streak stats:
            t5.user_max_kill_streak,
            t5.user_avg_kill_streak
        FROM stats_by_user_date t1
        LEFT JOIN user_session_stats_agg t4
            ON t1.user_id = t4.user_id
            AND t1.{stats_agg_col} = t4.{stats_agg_col}
            AND t1.ab_test_group = t4.ab_test_group
        LEFT JOIN max_kill_streak_length_agg t5
            ON t1.user_id = t5.user_id
            AND t1.{stats_agg_col} = t5.{stats_agg_col}
            AND t1.ab_test_group = t5.ab_test_group
        LEFT JOIN user_day_stats t6
            ON t1.user_id = t6.user_id
            AND t1.{stats_agg_col} = t6.{stats_agg_col}
            AND t1.ab_test_group = t6.ab_test_group
        WHERE t1.{stats_agg_col} >= DATE '{effective_start}'
          AND t1.{stats_agg_col} < DATE '{output_end}'
        """
    )


def period_start(stats_agg_col: str, start: date) -> date:
    """Truncate to the period start so every output period is complete (mirrors
    bituslabs_ds.etl.effective_start_date)."""
    if stats_agg_col == "activity_month":
        return start.replace(day=1)
    if stats_agg_col == "activity_week":
        return start - timedelta(days=start.weekday())
    return start


# Columns coerced to the exact Arrow types of the Redshift-written datasets in
# output_fish_hunter_v2, so files from both writers concat cleanly in the
# dashboard loader and in ETLScheduler compaction. The BIGINT casts also
# reproduce Redshift's integer-AVG truncation for the avg-of-int metrics.
OUTPUT_BIGINT_COLUMNS = [
    "user_id",
    "fish_value",
    "user_num_rooms",
    "user_num_bets",
    "user_num_killed_bullets",
    "user_killed_fish",
    "user_num_delta_t",
    "user_avg_fish_value",
    "user_avg_killed_fish_value",
    "user_pos_delta_bet_num",
    "user_neg_delta_bet_num",
    "user_num_delta_bet",
    "user_num_streak_sessions",
    "user_max_streak_length",
    "user_min_streak_length",
    "user_seconds_to_kill_fish",
    "user_bets_to_kill_fish",
    "user_max_kill_streak",
    "user_avg_kill_streak",
]


def align_output_schema(df):
    for col in OUTPUT_BIGINT_COLUMNS:
        df = df.withColumn(col, F.col(col).cast("bigint"))
    df = df.withColumn("activity_date", F.col("activity_date").cast("timestamp"))
    # Lets ETLScheduler's key-dedup keep the freshest row when the nightly
    # Redshift lookback overlaps dates this job produced.
    return df.withColumn("_processed_at", F.current_timestamp())


def check_schema(bullet) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in bullet.columns]
    if missing:
        raise SystemExit(
            f"cold data bullet table is missing required columns: {missing}; "
            f"available columns: {sorted(bullet.columns)}"
        )
    print("input schema (required columns):")
    for name, dtype in bullet.select(*REQUIRED_COLUMNS).dtypes:
        print(f"  {name}: {dtype}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="s3://... root of bullet parquet (year=/month=/day=)")
    parser.add_argument("--output-root", required=True, help="s3://... root for the stats datasets")
    parser.add_argument("--output-start", required=True, help="keep rows with activity date >= this, YYYY-MM-DD")
    parser.add_argument(
        "--output-end",
        required=True,
        help="keep rows with activity date < this, YYYY-MM-DD; align to a period"
        " start (Monday / 1st) so weekly/monthly rows are complete",
    )
    parser.add_argument("--agg", choices=["all", *AGG_LEVELS], default="all", help="which aggregation levels to run")
    parser.add_argument("--game-id", default="FM01")
    parser.add_argument("--currency", default="CNY")
    return parser.parse_args()


def main():
    args = parse_args()
    output_start = date.fromisoformat(args.output_start)
    output_end = date.fromisoformat(args.output_end)
    levels = list(AGG_LEVELS) if args.agg == "all" else [args.agg]

    spark = (
        SparkSession.builder.appName("FM01_game_stats_cold_data")  # type: ignore[attr-defined]
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    bullet = spark.read.parquet(args.input_root)
    check_schema(bullet)
    partition_date = F.make_date("year", "month", "day")

    for level in levels:
        stats_agg_col, job_name = AGG_LEVELS[level]
        effective_start = period_start(stats_agg_col, output_start)
        # Raw scan window in UTC: BJ midnight minus the fixed offset. The
        # lookback matches the Redshift job so start-boundary kill streaks
        # (the only cross-day sequence metric) are complete.
        scan_start_utc = datetime.combine(effective_start - timedelta(days=SCAN_LOOKBACK_DAYS), time()) - timedelta(
            hours=BJ_UTC_OFFSET_HOURS
        )
        scan_end_utc = datetime.combine(output_end, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)

        raw = bullet.filter(partition_date.between(F.lit(scan_start_utc.date()), F.lit(scan_end_utc.date())))
        if raw.limit(1).count() == 0:
            print("no data, skip:", job_name)
            continue
        raw.createOrReplaceTempView("bullet_raw")

        df = spark.sql(
            generate_query(
                stats_agg_col=stats_agg_col,
                effective_start=effective_start,
                output_end=output_end,
                scan_start_utc=scan_start_utc,
                scan_end_utc=scan_end_utc,
                game_id=args.game_id,
                currency=args.currency,
            )
        )
        df = align_output_schema(df)
        df = df.withColumn("period", F.date_format("activity_date", "yyyy-MM-dd"))

        out = f"{args.output_root}/{job_name}"
        df.repartition("period").write.partitionBy("period").mode("overwrite").parquet(out)
        print("saved:", out)
        summary = (
            spark.read.parquet(out)
            .where((F.col("activity_date") >= F.lit(effective_start)) & (F.col("activity_date") < F.lit(output_end)))
            .selectExpr(
                "count(*) AS rows",
                "count(distinct user_id) AS users",
                "sum(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END) AS null_user_ids",
                "min(activity_date) AS min_date",
                "max(activity_date) AS max_date",
            )
            .collect()[0]
        )
        print(summary)
        if summary["null_user_ids"]:
            raise SystemExit(
                f"{job_name}: {summary['null_user_ids']} rows have NULL user_id after the BIGINT cast -- "
                "the cold data user_id column is probably not numeric; output is unusable."
            )

    spark.stop()


if __name__ == "__main__":
    main()
