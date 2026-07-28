"""Slot-machine per-user game stats from S3 cold data — PySpark job, all games.

ETL job: cold-data source for the per-game dashboards' ``user_*`` metrics
(grain: one row per period, user, ab_group, mathtable, bet_level). Same query logic as
the per-game ``jobs/ss*/etl_game_stats_daily_by_user_group.py`` Redshift jobs,
parameterized by ``--game-id`` (SS01, SS01A, SS02, SS03, SS06), reading the
``cold_data_bet_order`` parquet from the slotmachine production warehouse
instead of Redshift ``public.fct_bet_orders``. Output datasets:
``<output-root>/{daily,weekly,monthly}_stats/period=YYYY-MM-DD/``.

Behavior: every bet lives in exactly one (period, user, ab_group, mathtable,
bet_level) row, so sums recombine correctly under any dashboard cohort
selection. ``bet_level`` backs the dashboards' bet-level range picker
(bet_low/medium/high/ultra ranges over existing bet amounts, like the
fish_hunter fish-level groups): BASE spins carry their own bet_amount and
FREE spins inherit the day's prevailing BASE bet.
Per-game variations (from the Redshift originals): SS02 attributes FourScatter
free-game buy-ins to the mathtable of the FOLLOWING spin (full-stream LEAD, so
its scan window extends one day past output-end); SS03/SS06 map the AB-test
partition ids to AB_TEST_A/B groups on top of the AI/Default mapping. Each run
recomputes whole periods in [output-start, output-end) and dynamic-partition-
overwrites exactly the ``period=`` directories it produced. Sequence metrics
(delta-t / delta-bet / mathtable_change / FG trigger) are DAY-partitioned, so
windowed runs compose exactly.

Runs standalone on the SageMaker Spark container -- no bituslabs_ds imports
(ETL constants from bituslabs_ds.config are inlined below). Submit with
etl_game_stats_daily_by_user_group_cold_data_submit.py.
"""

import argparse
from datetime import date, datetime, time, timedelta
from textwrap import dedent

from pyspark.sql import SparkSession, functions as F

# Inlined from bituslabs_ds.config (this job runs without the package).
EXCLUDED_OP_CODES = "('B26', 'TST', 'TSB', 'TSO')"
DELTA_T_MAX_SECONDS = 1800
DELTA_T_MIN_SECONDS = 1
AI_GROUP_ID = "jojpin-9mokha-rexQug"
AB_TEST_GROUP_A = "4f1a46ca-7baa-4452-9a40-ef21d9b33b57"
AB_TEST_GROUP_B = "4a04df21-c749-4808-8e55-3a0b74c084d2"
# Asia/Shanghai has no DST, so a fixed offset equals CONVERT_TIMEZONE.
BJ_UTC_OFFSET_HOURS = 8
# Rolling window for scheduled no-args runs; matches the old ETLScheduler lookback.
INCREMENTAL_LOOKBACK_DAYS = 3

# The cold data stores partition_ab as binary JSON (b'["<group-id>"]'), not a
# parquet list, so the first element is extracted via get_json_object.
PARTITION_AB_FIRST = "get_json_object(CAST(t.partition_ab AS STRING), '$[0]')"

GAME_CONFIG = {
    "SS01": {},
    "SS01A": {},
    "SS02": {"fourscatter_lead": True},
    "SS03": {"ab_test_groups": True},
    "SS06": {"ab_test_groups": True},
}

AGG_LEVELS = {
    "daily": ("activity_date", "daily_stats"),
    "weekly": ("activity_week", "weekly_stats"),
    "monthly": ("activity_month", "monthly_stats"),
}

REQUIRED_COLUMNS = [
    "user_id",
    "spin_id",
    "created_at",
    "math_table_id",
    "bet_amount",
    "actual_payout",
    "bet_type",
    "partition_ab",
    "game_id",
    "currency_type",
    "status",
    "op_code",
]

# Coerced to the exact Arrow types of the Redshift-written output_ss*_v2
# datasets so files from both writers concat cleanly downstream.
OUTPUT_BIGINT_COLUMNS = [
    "user_id",
    "user_mathtable_change",
    "user_num_bets",
    "user_num_bets_bg",
    "user_num_bets_fg",
    "user_num_bets_with_payout",
    "user_num_bets_bg_with_payout",
    "user_num_bets_fg_with_payout",
    "user_num_delta_t_bg",
    "user_pos_delta_bet_num",
    "user_neg_delta_bet_num",
    "user_num_delta_bet",
]


def _ab_group_case(game_id: str) -> str:
    ab_test_lines = (
        f"""
                WHEN {PARTITION_AB_FIRST} = '{AB_TEST_GROUP_A}' THEN 'AB_TEST_A'
                WHEN {PARTITION_AB_FIRST} = '{AB_TEST_GROUP_B}' THEN 'AB_TEST_B'"""
        if GAME_CONFIG[game_id].get("ab_test_groups")
        else ""
    )
    return f"""CASE
                WHEN {PARTITION_AB_FIRST} = '{AI_GROUP_ID}' THEN 'AI'{ab_test_lines}
                ELSE 'Default'
            END AS ab_group"""


def _math_table_expr(game_id: str) -> str:
    if GAME_CONFIG[game_id].get("fourscatter_lead"):
        # FourScatter rows are free-game buy-ins: attribute them to the
        # mathtable of the FOLLOWING spin (full-stream LEAD on purpose --
        # this is mathtable attribution, not a sequence metric).
        return """CASE
                    WHEN t.math_table_id = 'FourScatter'
                    THEN LEAD(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, CAST(t.created_at AS TIMESTAMP))
                    ELSE t.math_table_id
                END"""
    return "t.math_table_id"


def generate_query(
    stats_agg_col: str,
    game_id: str,
    effective_start: date,
    output_end: date,
    scan_start_utc: datetime,
    scan_end_utc: datetime,
    currency: str,
) -> str:
    """Spark-SQL version of the per-game generate_query over ``bet_order_raw``.

    Translation notes versus the Redshift dialect: BJ time is
    ``created_at + 8h`` (session timezone is UTC); ``EXTRACT(EPOCH FROM
    interval)`` becomes a ``CAST(ts AS DOUBLE)`` difference to keep
    sub-second deltas.
    """
    bj_ts = f"CAST(t.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR"
    day_window = "PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at"
    return dedent(
        f"""
        WITH bet_events AS (
            SELECT
                t.user_id,
                t.spin_id,
                CAST(t.created_at AS TIMESTAMP) AS created_at,
                {_math_table_expr(game_id)} AS math_table_id,
                CAST(t.bet_amount AS DOUBLE) AS bet_amount,
                CAST(t.actual_payout AS DOUBLE) AS actual_payout,
                t.bet_type,
                t.partition_ab
            FROM
                bet_order_raw AS t
            WHERE
                t.game_id = '{game_id}'
                AND t.currency_type = '{currency}'
                AND t.status = 'COMPLETED'
                AND t.op_code NOT IN {EXCLUDED_OP_CODES}
                AND CAST(t.created_at AS TIMESTAMP) >= TIMESTAMP '{scan_start_utc:%Y-%m-%d %H:%M:%S}'
                AND CAST(t.created_at AS TIMESTAMP) < TIMESTAMP '{scan_end_utc:%Y-%m-%d %H:%M:%S}'
        ),

        bets AS (
            SELECT
                t.user_id,
                t.spin_id,
                t.created_at,
                COALESCE(NULLIF(t.math_table_id, ''), '(none)') AS mathtable,
                t.bet_amount,
                t.actual_payout AS payout,
                t.bet_type,
                t.actual_payout - t.bet_amount AS profit,
                CAST({bj_ts} AS DATE) AS activity_date,
                CAST(DATE_TRUNC('week', {bj_ts}) AS DATE) AS activity_week,
                CAST(DATE_TRUNC('month', {bj_ts}) AS DATE) AS activity_month,
                {_ab_group_case(game_id)}
            FROM bet_events AS t
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
                -- Bet-level grain: BASE spins use their own bet_amount; FREE
                -- spins inherit the day's prevailing BASE bet (their trigger),
                -- so FG metrics stay meaningful inside a bet-level range. A
                -- FREE spin with no prior BASE that day keeps its own amount.
                COALESCE(
                    LAST(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END, TRUE) OVER (
                        {day_window}
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ),
                    t.bet_amount
                ) AS bet_level,
                CAST(t.created_at AS DOUBLE) - CAST(LAG(t.created_at) OVER ({day_window}) AS DOUBLE) AS delta_t_seconds,
                LAG(t.bet_type) OVER ({day_window}) AS prev_bet_type,
                LAG(t.bet_amount) OVER ({day_window}) AS prev_bet_amount,
                CASE
                    WHEN LAG(t.mathtable) OVER ({day_window}) IS NULL THEN 0
                    WHEN LAG(t.mathtable) OVER ({day_window}) <> t.mathtable THEN 1
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
                AVG(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= {DELTA_T_MAX_SECONDS} THEN GREATEST(t.delta_t_seconds, {DELTA_T_MIN_SECONDS}) END)
                    AS user_avg_delta_t_seconds_bg,
                COUNT(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= {DELTA_T_MAX_SECONDS} THEN 1 END)
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
        WHERE us.{stats_agg_col} >= DATE '{effective_start}'
          AND us.{stats_agg_col} < DATE '{output_end}'
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


def align_output_schema(df):
    for col in OUTPUT_BIGINT_COLUMNS:
        df = df.withColumn(col, F.col(col).cast("bigint"))
    df = df.withColumn("activity_date", F.col("activity_date").cast("timestamp"))
    # Lets ETLScheduler's key-dedup keep the freshest row when a Redshift
    # lookback run overlaps dates this job produced.
    return df.withColumn("_processed_at", F.current_timestamp())


def check_schema(bet_order) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in bet_order.columns]
    if missing:
        raise SystemExit(
            f"cold data bet_order table is missing required columns: {missing}; "
            f"available columns: {sorted(bet_order.columns)}"
        )
    print("input schema (required columns):")
    for name, dtype in bet_order.select(*REQUIRED_COLUMNS).dtypes:
        print(f"  {name}: {dtype}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True, choices=sorted(GAME_CONFIG))
    parser.add_argument("--input-root", required=True, help="s3://... root of bet_order parquet")
    parser.add_argument("--output-root", required=True, help="s3://... root for the stats datasets")
    parser.add_argument(
        "--output-start",
        default=None,
        help="keep rows with activity date >= this, YYYY-MM-DD"
        " (default: rolling daily-incremental window, last INCREMENTAL_LOOKBACK_DAYS Beijing days)",
    )
    parser.add_argument(
        "--output-end",
        default=None,
        help="keep rows with activity date < this, YYYY-MM-DD; align to a period"
        " start (Monday / 1st) so weekly/monthly rows are complete"
        " (default: tomorrow in Beijing time)",
    )
    parser.add_argument("--agg", choices=["all", *AGG_LEVELS], default="all", help="which aggregation levels to run")
    parser.add_argument("--currency", default="CNY")
    parser.add_argument(
        "--input-region",
        default="ap-southeast-1",
        help="region of the input bucket (s3a needs it spelled out for cross-region reads)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Rolling daily-incremental defaults: the job recomputes only the periods
    # in the window (dynamic period= overwrite), so the scheduled no-args run
    # refreshes recent days without touching history.
    bj_today = (datetime.utcnow() + timedelta(hours=BJ_UTC_OFFSET_HOURS)).date()
    output_start = (
        date.fromisoformat(args.output_start)
        if args.output_start
        else bj_today - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
    )
    output_end = date.fromisoformat(args.output_end) if args.output_end else bj_today + timedelta(days=1)
    print(f"output window: [{output_start}, {output_end})")
    levels = list(AGG_LEVELS) if args.agg == "all" else [args.agg]
    # The SS02 FourScatter LEAD looks at the next spin, so scan one day past
    # the output window to attribute buy-ins on the last output day.
    scan_end_margin = timedelta(days=1 if GAME_CONFIG[args.game_id].get("fourscatter_lead") else 0)

    builder = (
        SparkSession.builder.appName(f"{args.game_id}_game_stats_cold_data")  # type: ignore[attr-defined]
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    )
    if args.input_root.startswith("s3"):
        input_bucket = args.input_root.split("/")[2]
        builder = builder.config(f"spark.hadoop.fs.s3a.bucket.{input_bucket}.endpoint.region", args.input_region)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    bet_order = spark.read.parquet(args.input_root)
    check_schema(bet_order)
    prunable = {"year", "month", "day"} <= set(bet_order.columns)
    if not prunable:
        print("no year/month/day partition columns; relying on created_at filter only")

    for level in levels:
        stats_agg_col, job_name = AGG_LEVELS[level]
        effective_start = period_start(stats_agg_col, output_start)
        scan_start_utc = datetime.combine(effective_start, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
        scan_end_utc = datetime.combine(output_end + scan_end_margin, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)

        raw = bet_order
        if prunable:
            partition_date = F.make_date("year", "month", "day")
            raw = raw.filter(partition_date.between(F.lit(scan_start_utc.date()), F.lit(scan_end_utc.date())))
        if raw.limit(1).count() == 0:
            print("no data, skip:", job_name)
            continue
        raw.createOrReplaceTempView("bet_order_raw")

        df = spark.sql(
            generate_query(
                stats_agg_col=stats_agg_col,
                game_id=args.game_id,
                effective_start=effective_start,
                output_end=output_end,
                scan_start_utc=scan_start_utc,
                scan_end_utc=scan_end_utc,
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
