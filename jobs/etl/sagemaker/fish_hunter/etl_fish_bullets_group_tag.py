"""FM01 bullets with the policy-correct ``group_tag`` label (PySpark).

ETL job: bullet-level copy of the oceanhunter warehouse's ``cold_data/bullet``
rows plus the ``group_tag`` column the raw data lacks — the fish grouping
policy (``spark_etl_common.fish_group_tag_sql``: BOOST_POOL ->
``boost_pool``, DYNAMIC_RTP family -> ``dynamic_rtp``,
RC_FISHING_*/'%RISK_CONTROL%' -> ``risk_control``,
CR_FISHING_* or user-id last digit 0/1 from the 2026-07-31 00:00 UTC
retention launch -> ``retention``, else ``default``; see
docs/fish_group_tag_policy.md). Backs the Athena table
``bituslabs_ds.fish_bullets_group_tag`` (register with
``infra/etl/register_fish_bullets_catalog.py``), so ad-hoc queries get
correct group membership without re-deriving the policy.

Behavior: test bets (op_code B26/TST/TSB/TSO) are dropped at the source so
downstream consumers and ad-hoc Athena queries need no op-code filter;
currency/game filtering stays downstream (those are selections, not junk).
Keeps only the columns the daily-stats and feature-engineering ETLs consume
plus ``strategy_name`` for auditing the tag. ``activity_date`` is the bullet's
Beijing calendar date (from ``created_at``) and drives the ``period``
partition. Output layout: ``<output-root>/bullets/period=YYYY-MM-DD/``.
Each run recomputes the window's periods with dynamic partition overwrite,
so the scheduled no-args run (rolling last 3 Beijing days) composes with
backfills exactly like the game-stats job.

Runs standalone on the SageMaker Spark container. Submit with
etl_fish_bullets_group_tag_submit.py.
"""

import argparse
from datetime import date, datetime, time, timedelta
from textwrap import dedent

from pyspark.sql import functions as F
from spark_etl_common import (
    BJ_UTC_OFFSET_HOURS,
    EXCLUDED_OP_CODES,
    beijing_today,
    build_spark_session,
    check_schema,
    fish_group_tag_sql,
    prune_partition_days,
    warn_on_schema_drift,
)

INCREMENTAL_LOOKBACK_DAYS = 3

REQUIRED_COLUMNS = [
    "user_id",
    "bullet_id",
    "event_id",
    "room_id",
    "strategy_name",
    "event_timestamp",
    "created_at",
    "bet",
    "payout",
    "profit",
    "prev_balance",
    "curr_balance",
    "fish_value",
    "killed",
    "bullet_level",
    "multiplier",
    "op_code",
    "currency_type",
    "game_id",
]


def generate_query(effective_start: date, output_end: date) -> str:
    bj_ts = f"CAST(t.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR"
    bj_date = f"CAST({bj_ts} AS DATE)"
    group_tag = fish_group_tag_sql("t.strategy_name", "t.user_id", "CAST(t.event_timestamp AS TIMESTAMP)")
    scan_start_utc = datetime.combine(effective_start, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    scan_end_utc = datetime.combine(output_end, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    return dedent(
        f"""
        SELECT
            CAST(t.user_id AS BIGINT) AS user_id,
            CAST(t.bullet_id AS BIGINT) AS bullet_id,
            t.event_id,
            t.room_id,
            t.strategy_name,
            CAST(t.event_timestamp AS TIMESTAMP) AS event_timestamp,
            CAST(t.created_at AS TIMESTAMP) AS created_at,
            CAST(t.bet AS DOUBLE) AS bet,
            CAST(t.payout AS DOUBLE) AS payout,
            CAST(t.profit AS DOUBLE) AS profit,
            CAST(t.prev_balance AS DOUBLE) AS prev_balance,
            CAST(t.curr_balance AS DOUBLE) AS curr_balance,
            CAST(t.fish_value AS DOUBLE) AS fish_value,
            CAST(t.killed AS INT) AS killed,
            t.bullet_level,
            t.multiplier,
            t.op_code,
            t.currency_type,
            t.game_id,
            {group_tag} AS group_tag,
            {bj_date} AS activity_date
        FROM bullet_raw AS t
        WHERE
            t.op_code NOT IN {EXCLUDED_OP_CODES}
            AND CAST(t.created_at AS TIMESTAMP) >= TIMESTAMP '{scan_start_utc:%Y-%m-%d %H:%M:%S}'
            AND CAST(t.created_at AS TIMESTAMP) < TIMESTAMP '{scan_end_utc:%Y-%m-%d %H:%M:%S}'
        """
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="s3://... root of bullet parquet (year=/month=/day=)")
    parser.add_argument("--output-root", required=True, help="s3://... root for the bullets dataset")
    parser.add_argument(
        "--output-start",
        default=None,
        help="keep rows with Beijing activity date >= this, YYYY-MM-DD"
        " (default: rolling window, last INCREMENTAL_LOOKBACK_DAYS Beijing days)",
    )
    parser.add_argument("--output-end", default=None, help="exclusive end, YYYY-MM-DD (default: tomorrow Beijing)")
    parser.add_argument(
        "--input-region",
        default="ap-southeast-1",
        help="region of the input bucket (s3a needs it spelled out for cross-region reads)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bj_today = beijing_today()
    output_start = (
        date.fromisoformat(args.output_start)
        if args.output_start
        else bj_today - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
    )
    output_end = date.fromisoformat(args.output_end) if args.output_end else bj_today + timedelta(days=1)
    print(f"output window: [{output_start}, {output_end})")

    spark = build_spark_session("fish_bullets_group_tag", args.input_root, args.input_region)
    bullet = spark.read.parquet(args.input_root)
    check_schema(bullet, REQUIRED_COLUMNS, table_name="cold data bullet")

    scan_start_utc = datetime.combine(output_start, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    scan_end_utc = datetime.combine(output_end, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    raw = prune_partition_days(bullet, scan_start_utc.date(), scan_end_utc.date())
    if raw.limit(1).count() == 0:
        print("no data in window, nothing to do")
        spark.stop()
        return
    raw.createOrReplaceTempView("bullet_raw")

    df = spark.sql(generate_query(output_start, output_end))
    df = df.withColumn("period", F.date_format("activity_date", "yyyy-MM-dd"))
    out = f"{args.output_root}/bullets"
    warn_on_schema_drift(spark, out, df)
    df.repartition("period").write.partitionBy("period").mode("overwrite").parquet(out)
    print("saved:", out)
    # Per-tag first/last event timestamps double as the policy audit: whether
    # risk_control strategies predate the 2026-07-31 retention launch is read
    # straight off this summary.
    summary = (
        spark.read.parquet(out)
        .where((F.col("activity_date") >= F.lit(output_start)) & (F.col("activity_date") < F.lit(output_end)))
        .groupBy("group_tag")
        .agg(
            F.count("*").alias("rows"),
            F.min("event_timestamp").alias("first_event"),
            F.max("event_timestamp").alias("last_event"),
        )
        .orderBy("group_tag")
    )
    summary.show(truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
