"""Slot-machine per-user game stats from the slot_orders_ab_group dataset.

ETL job: cold-data source for the per-game dashboards' ``user_*`` metrics
(grain: one row per period, user, ab_group, mathtable, bet_level). Same query logic as
the per-game ``jobs/ss*/etl_game_stats_daily_by_user_group.py`` Redshift jobs,
parameterized by ``--game-id`` (SS01, SS01A, SS02, SS03, SS06), reading the
``output_slot_orders_ab_group/orders`` dataset (the data behind the Athena
table ``bituslabs_ds.slot_orders_ab_group`` — raw bet orders plus the
policy-correct ``ab_group``, see etl_slot_orders_ab_group.py), so group
membership is derived exactly once upstream. Output datasets:
``<output-root>/{daily,weekly,monthly}_stats/period=YYYY-MM-DD/``.

Behavior: ``activity_date`` is the SESSION-START Beijing date (bet_session =
30 min without a bet) for every game — the daily-stats midnight fix, so
cross-midnight play stays on the day it started. Every bet lives in exactly
one (period, user, ab_group, mathtable,
bet_level) row, so sums recombine correctly under any dashboard cohort
selection. ``bet_level`` backs the dashboards' bet-level range picker
(bet_low/medium/high/ultra ranges over existing bet amounts, like the
fish_hunter fish-level groups): BASE spins carry their own bet_amount and
FREE spins inherit the day's prevailing BASE bet.
Per-game variations (from the Redshift originals): SS02 attributes FourScatter
free-game buy-ins to the mathtable of the FOLLOWING spin (full-stream LEAD, so
its scan window extends one day past output-end); SS03/SS06 carry the AB_TEST_A/B
groups; games without AB tests collapse those labels into Default. Each run
recomputes whole periods in [output-start, output-end) and dynamic-partition-
overwrites exactly the ``period=`` directories it produced. Sequence metrics
(delta-t / delta-bet / mathtable_change / FG trigger) are DAY-partitioned, so
windowed runs compose exactly.

Group policy comes from ``group_policy.py`` (every game's rules in one
config module): SS03's dashboard uses user-day groups under the announced
cutover, every other game the stored row-level ``ab_group``; the run/cohort
variants always keep the stored label. Details: docs/ab_group_policy.md.

``--min-mathtable-run N`` keeps only bets inside a stretch of >= N consecutive
same-mathtable bets (per user per Beijing day) — the ss03 AI mathtable-combo
dashboard reads a run-filtered output so brief mathtable dabbles don't count
toward a user's daily combination. ``--ab-group`` restricts the scan to one
AB group label (e.g. AI) so a cohort-specific dataset stays small. Write
filtered outputs to their own suffixed root (``..._ai_run30``) so the
unfiltered dashboards stay untouched.

Runs standalone on the SageMaker Spark container -- no bituslabs_ds imports
(ETL constants from bituslabs_ds.config are inlined below). Submit with
etl_game_stats_daily_by_user_group_cold_data_submit.py.
"""

import argparse
from datetime import date, datetime, time, timedelta
from textwrap import dedent

# Ships via submit_py_files on SageMaker; for local runs/tests put
# jobs/etl/sagemaker on the path first (the snapshot tests already do).
from group_policy_sql import slot_grouping, slot_stored_label_case
from pyspark.sql import functions as F
from spark_etl_common import (
    BJ_UTC_OFFSET_HOURS,
    EXCLUDED_OP_CODES,
    beijing_today,
    build_spark_session,
    check_schema,
    prune_period_days,
    session_day_ctes,
    warn_on_schema_drift,
)

DELTA_T_MAX_SECONDS = 1800
DELTA_T_MIN_SECONDS = 1
# Rolling window for scheduled no-args runs; matches the old ETLScheduler lookback.
INCREMENTAL_LOOKBACK_DAYS = 3

GAME_CONFIG = {
    "SS01": {},
    "SS01A": {},
    "SS02": {"fourscatter_lead": True},
    "SS03": {},
    "SS06": {},
}

AB_GROUP_LABELS = ["AI", "AB_TEST_A", "AB_TEST_B", "Default"]

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
    "ab_group",
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


def _mathtable_run_ctes(day_window: str, min_run: int) -> str:
    """CTE chain keeping only bets inside a run of >= min_run consecutive
    same-mathtable bets. Runs are per user per activity_date like every other
    sequence computation here, so windowed ETL runs compose exactly; a stretch
    crossing Beijing midnight counts as two separate runs."""
    return f"""mathtable_run_flags AS (
            SELECT
                t.*,
                CASE
                    WHEN LAG(t.mathtable) OVER ({day_window}) IS NULL THEN 1
                    WHEN LAG(t.mathtable) OVER ({day_window}) <> t.mathtable THEN 1
                    ELSE 0
                END AS mathtable_run_start
            FROM bets AS t
        ),

        mathtable_runs AS (
            SELECT
                t.*,
                SUM(t.mathtable_run_start) OVER (
                    {day_window}
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS mathtable_run_id
            FROM mathtable_run_flags AS t
        ),

        long_run_bets AS (
            SELECT * FROM (
                SELECT
                    t.*,
                    COUNT(*) OVER (
                        PARTITION BY t.user_id, t.activity_date, t.mathtable_run_id
                    ) AS mathtable_run_len
                FROM mathtable_runs AS t
            ) AS runs
            WHERE runs.mathtable_run_len >= {min_run}
        )"""


def generate_query(
    stats_agg_col: str,
    game_id: str,
    effective_start: date,
    output_end: date,
    scan_start_utc: datetime,
    scan_end_utc: datetime,
    currency: str,
    min_mathtable_run: int = 0,
    ab_group: str = "",
) -> str:
    """Spark-SQL version of the per-game generate_query over ``bet_order_raw``.

    ``activity_date`` is always the SESSION-START Beijing date (bet_session,
    docs/ab_group_policy.md "Day basis"). The game's group policy comes from
    ``group_policy.GROUP_POLICY``; the run/cohort variants keep the
    stored row-level label (``group_policy_sql.slot_grouping``).

    Translation notes versus the Redshift dialect: BJ time is
    ``created_at + 8h`` (session timezone is UTC); ``EXTRACT(EPOCH FROM
    interval)`` becomes a ``CAST(ts AS DOUBLE)`` difference to keep
    sub-second deltas.
    """
    sess_bj = f"t.session_start_ts + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR"
    day_window = "PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at"
    # With the run filter on, sequence metrics are computed on the RETAINED
    # stream: surviving bets on either side of a dropped short run count as
    # adjacent (their delta-t spans the dropped run's wall time).
    run_ctes = f"\n        {_mathtable_run_ctes(day_window, min_mathtable_run)},\n" if min_mathtable_run > 0 else ""
    bets_source = "long_run_bets" if min_mathtable_run > 0 else "bets"
    # Filtering in bet_events keeps the scan cheap and, with the run filter
    # on, computes runs over the cohort's own bet stream only (the stored
    # slot_orders_ab_group label is already policy-correct).
    ab_where = f"\n                AND t.ab_group = '{ab_group}'" if ab_group else ""
    group_source_col, bet_label_case, day_collapse_case = slot_grouping(
        game_id, row_level=bool(min_mathtable_run or ab_group)
    )
    day_cols = f"""CAST({sess_bj} AS DATE) AS activity_date,
                CAST(DATE_TRUNC('week', {sess_bj}) AS DATE) AS activity_week,
                CAST(DATE_TRUNC('month', {sess_bj}) AS DATE) AS activity_month"""
    if day_collapse_case:
        bets_chain = f"""bets_labeled AS (
            SELECT
                t.user_id,
                t.spin_id,
                t.created_at,
                COALESCE(NULLIF(t.math_table_id, ''), '(none)') AS mathtable,
                t.bet_amount,
                t.actual_payout AS payout,
                t.bet_type,
                t.actual_payout - t.bet_amount AS profit,
                {day_cols},
                {bet_label_case} AS bet_ab_group
            FROM session_days AS t
        ),

        bets AS (
            SELECT
                t.user_id,
                t.spin_id,
                t.created_at,
                t.mathtable,
                t.bet_amount,
                t.payout,
                t.bet_type,
                t.profit,
                t.activity_date,
                t.activity_week,
                t.activity_month,
                {day_collapse_case} AS ab_group
            FROM bets_labeled AS t
        )"""
    else:
        bets_chain = f"""bets AS (
            SELECT
                t.user_id,
                t.spin_id,
                t.created_at,
                COALESCE(NULLIF(t.math_table_id, ''), '(none)') AS mathtable,
                t.bet_amount,
                t.actual_payout AS payout,
                t.bet_type,
                t.actual_payout - t.bet_amount AS profit,
                {day_cols},
                {slot_stored_label_case(game_id)}
            FROM session_days AS t
        )"""
    bets_chain = f"""{session_day_ctes("bet_events", "spin_id")},

        {bets_chain}"""
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
                {group_source_col}
            FROM
                bet_order_raw AS t
            WHERE
                t.currency_type = '{currency}'
                AND t.status = 'COMPLETED'
                AND t.op_code NOT IN {EXCLUDED_OP_CODES}
                AND CAST(t.created_at AS TIMESTAMP) >= TIMESTAMP '{scan_start_utc:%Y-%m-%d %H:%M:%S}'
                AND CAST(t.created_at AS TIMESTAMP) < TIMESTAMP '{scan_end_utc:%Y-%m-%d %H:%M:%S}'{ab_where}
        ),

        {bets_chain},
{run_ctes}
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
            FROM {bets_source} AS t
        ),

        user_stats AS (
            SELECT
                t.{stats_agg_col},
                t.ab_group,
                t.mathtable,
                t.bet_level,
                t.user_id,

                -- first-appearance order of this grain row within the user's
                -- period (dashboards break combination-label ties with it):
                MIN(t.spin_id) AS user_first_spin_id,

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
            us.user_first_spin_id,
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True, choices=sorted(GAME_CONFIG))
    parser.add_argument(
        "--input-root", required=True, help="s3://... root of the slot_orders_ab_group 'orders' dataset"
    )
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
    parser.add_argument(
        "--min-mathtable-run",
        type=int,
        default=0,
        help="keep only bets inside a run of >= N consecutive same-mathtable bets"
        " (per user per Beijing day); 0 disables the filter. Point --output-root"
        " at a dedicated _runN root so the unfiltered datasets stay untouched",
    )
    parser.add_argument(
        "--ab-group",
        choices=AB_GROUP_LABELS,
        default="",
        help="only keep bets whose date-gated AB group label equals this; default: all groups",
    )
    parser.add_argument("--currency", default="CNY")
    parser.add_argument(
        "--input-region",
        default="us-west-2",
        help="region of the input bucket (the slot_orders_ab_group dataset lives in our us-west-2 bucket)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Rolling daily-incremental defaults: the job recomputes only the periods
    # in the window (dynamic period= overwrite), so the scheduled no-args run
    # refreshes recent days without touching history.
    bj_today = beijing_today()
    output_start = (
        date.fromisoformat(args.output_start)
        if args.output_start
        else bj_today - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
    )
    output_end = date.fromisoformat(args.output_end) if args.output_end else bj_today + timedelta(days=1)
    print(
        f"output window: [{output_start}, {output_end})"
        f" min_mathtable_run={args.min_mathtable_run} ab_group={args.ab_group or 'all'}"
    )
    levels = list(AGG_LEVELS) if args.agg == "all" else [args.agg]
    # Session-day attribution scans one day on BOTH sides so boundary
    # sessions are whole (a 30min-gap bet_session spans hours at most; the
    # output filter on the session-start date keeps windowed runs composing
    # exactly). The SS02 FourScatter LEAD needs the same one-day end margin.
    session_margin = timedelta(days=1)
    scan_end_margin = session_margin

    spark = build_spark_session(f"{args.game_id}_game_stats_cold_data", args.input_root, args.input_region)

    # Reading inside game_id=<G>/ pins the game by path (no game_id column
    # inside, matching REQUIRED_COLUMNS).
    bet_order = spark.read.parquet(f"{args.input_root}/game_id={args.game_id}")
    group_source_col = slot_grouping(args.game_id, row_level=bool(args.min_mathtable_run or args.ab_group))[0]
    required = REQUIRED_COLUMNS + (["partition_ab_label"] if group_source_col == "t.partition_ab_label" else [])
    check_schema(bet_order, required, table_name="slot_orders_ab_group orders")

    for level in levels:
        stats_agg_col, job_name = AGG_LEVELS[level]
        effective_start = period_start(stats_agg_col, output_start)
        scan_start_utc = (
            datetime.combine(effective_start, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS) - session_margin
        )
        scan_end_utc = datetime.combine(output_end + scan_end_margin, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)

        raw = prune_period_days(bet_order, scan_start_utc.date(), scan_end_utc.date(), margin_days=1)
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
                min_mathtable_run=args.min_mathtable_run,
                ab_group=args.ab_group,
            )
        )
        df = align_output_schema(df)
        df = df.withColumn("period", F.date_format("activity_date", "yyyy-MM-dd"))

        out = f"{args.output_root}/{job_name}"
        warn_on_schema_drift(spark, out, df)
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
