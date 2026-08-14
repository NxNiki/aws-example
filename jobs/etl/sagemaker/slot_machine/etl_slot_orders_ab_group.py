"""Slot-machine bet orders with the policy-correct ``ab_group`` label (PySpark).

ETL job: order-level copy of the slot warehouse's ``bet_order`` rows plus the
``ab_group`` column the raw data lacks — the timestamp-gated grouping policy
(``spark_etl_common.ab_group_sql``: partition_ab ids before the empirically
located cutover, 2026-08-04 05:30 Beijing; user-id last digit from that
moment on). Backs the Athena table
``bituslabs_ds.slot_orders_ab_group`` (register with
``infra/etl/register_slot_orders_catalog.py``), so ad-hoc queries get correct
group membership without re-deriving the policy.

Behavior: incomplete bets (status != COMPLETED) and test bets (op_code
B26/TST/TSB/TSO) are dropped at the source so downstream consumers and
ad-hoc Athena queries need no status/op-code filter; currency filtering
stays downstream (a selection, not junk). ``activity_date``
is the bet's Beijing calendar date and drives the ``period`` partition.
Output layout: ``<output-root>/orders/game_id=<GAME>/period=YYYY-MM-DD/``.
Each run recomputes the window's periods with dynamic partition overwrite,
so the scheduled no-args run (rolling last 3 Beijing days) composes with
backfills exactly like the game-stats job.

Runs standalone on the SageMaker Spark container. Submit with
etl_slot_orders_ab_group_submit.py.
"""

import argparse
from datetime import date, datetime, time, timedelta
from textwrap import dedent

from pyspark.sql import functions as F
from spark_etl_common import (
    BJ_UTC_OFFSET_HOURS,
    EXCLUDED_OP_CODES,
    PARTITION_AB_FIRST,
    ab_group_sql,
    beijing_today,
    build_spark_session,
    check_schema,
    prune_partition_days,
    warn_on_schema_drift,
)

INCREMENTAL_LOOKBACK_DAYS = 3

SLOT_GAMES = ["SS01", "SS01A", "SS02", "SS03", "SS06"]

REQUIRED_COLUMNS = [
    "game_id",
    "spin_id",
    "user_id",
    "created_at",
    "math_table_id",
    "bet_type",
    "bet_amount",
    "actual_payout",
    "balance_after_bet",
    "balance_after_payout",
    "partition_ab",
    "currency_type",
    "status",
    "op_code",
]


def generate_query(games: list[str], effective_start: date, output_end: date) -> str:
    bj_ts = f"CAST(t.created_at AS TIMESTAMP) + INTERVAL '{BJ_UTC_OFFSET_HOURS}' HOUR"
    bj_date = f"CAST({bj_ts} AS DATE)"
    ab_group = ab_group_sql(PARTITION_AB_FIRST, "t.user_id", bj_ts)
    games_in = ", ".join(f"'{g}'" for g in games)
    scan_start_utc = datetime.combine(effective_start, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    scan_end_utc = datetime.combine(output_end, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    return dedent(
        f"""
        SELECT
            t.game_id,
            t.spin_id,
            CAST(t.user_id AS BIGINT) AS user_id,
            CAST(t.created_at AS TIMESTAMP) AS created_at,
            t.math_table_id,
            t.bet_type,
            CAST(t.bet_amount AS DOUBLE) AS bet_amount,
            CAST(t.actual_payout AS DOUBLE) AS actual_payout,
            CAST(t.balance_after_bet AS DOUBLE) AS balance_after_bet,
            CAST(t.balance_after_payout AS DOUBLE) AS balance_after_payout,
            t.currency_type,
            t.status,
            t.op_code,
            {PARTITION_AB_FIRST} AS partition_ab_label,
            {ab_group} AS ab_group,
            {bj_date} AS activity_date
        FROM bet_order_raw AS t
        WHERE
            t.game_id IN ({games_in})
            AND t.status = 'COMPLETED'
            AND t.op_code NOT IN {EXCLUDED_OP_CODES}
            AND CAST(t.created_at AS TIMESTAMP) >= TIMESTAMP '{scan_start_utc:%Y-%m-%d %H:%M:%S}'
            AND CAST(t.created_at AS TIMESTAMP) < TIMESTAMP '{scan_end_utc:%Y-%m-%d %H:%M:%S}'
        """
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="s3://... root of bet_order parquet")
    parser.add_argument("--output-root", required=True, help="s3://... root for the orders dataset")
    parser.add_argument("--games", default=",".join(SLOT_GAMES), help="comma-separated game ids")
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
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    print(f"output window: [{output_start}, {output_end}) games={games}")

    spark = build_spark_session("slot_orders_ab_group", args.input_root, args.input_region)
    bet_order = spark.read.parquet(args.input_root)
    check_schema(bet_order, REQUIRED_COLUMNS, table_name="cold data bet_order")

    scan_start_utc = datetime.combine(output_start, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    scan_end_utc = datetime.combine(output_end, time()) - timedelta(hours=BJ_UTC_OFFSET_HOURS)
    raw = prune_partition_days(bet_order, scan_start_utc.date(), scan_end_utc.date())
    if raw.limit(1).count() == 0:
        print("no data in window, nothing to do")
        spark.stop()
        return
    raw.createOrReplaceTempView("bet_order_raw")

    df = spark.sql(generate_query(games, output_start, output_end))
    df = df.withColumn("period", F.date_format("activity_date", "yyyy-MM-dd"))
    out = f"{args.output_root}/orders"
    warn_on_schema_drift(spark, out, df)
    (df.repartition("game_id", "period").write.partitionBy("game_id", "period").mode("overwrite").parquet(out))
    print("saved:", out)
    summary = (
        spark.read.parquet(out)
        .where((F.col("activity_date") >= F.lit(output_start)) & (F.col("activity_date") < F.lit(output_end)))
        .groupBy("ab_group")
        .count()
        .collect()
    )
    print({r["ab_group"]: r["count"] for r in summary})
    spark.stop()


if __name__ == "__main__":
    main()
