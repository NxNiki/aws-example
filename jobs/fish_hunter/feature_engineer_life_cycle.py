"""FM01 lifecycle (HMM) feature engineering — PySpark job, runs on the cluster.

ETL job: builds the per-user-per-day lifecycle feature table used for
HMM user-lifecycle modeling of fish_hunter (FM01). Output datasets under
``--output-root``: ``user_day_base_fm01_cny/month=*`` (monthly base
aggregates), ``user_day_base_fm01_cny_dedup`` (deduped base table),
``selected_hmm_features_fm01_cny`` (final features), plus single-file
CSV exports under ``csv/``.

Behavior: reads bullet order parquet directly from S3 (partition-pruned
by year=/month=/day=; no local download), sessionizes bets (180s gap),
aggregates to user-day month by month, dedups the month-boundary scan
overlap, then derives history/trend features with per-user windows.
Runs standalone on the SageMaker Spark container — no bituslabs_ds
imports. Submit with feature_engineer_life_cycle_submit.py.
"""

import argparse
from datetime import date

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

USER_DAY_BASE_SQL = """
WITH order_base AS (
    SELECT
        bullet_id,
        event_id,
        user_id,
        CAST(CAST(created_at AS TIMESTAMP) + INTERVAL 8 HOURS AS DATE) AS natural_bet_date,
        CAST(created_at AS TIMESTAMP) + INTERVAL 8 HOURS AS ts_bj,
        bullet_level,
        multiplier,
        fish_value,
        killed,
        CAST(bet AS DOUBLE) AS bet,
        CAST(payout AS DOUBLE) AS payout,
        COALESCE(CAST(profit AS DOUBLE), CAST(payout AS DOUBLE) - CAST(bet AS DOUBLE)) AS profit,
        CAST(prev_balance AS DOUBLE) AS prev_balance,
        CAST(curr_balance AS DOUBLE) AS curr_balance,
        LAG(CAST(created_at AS TIMESTAMP) + INTERVAL 8 HOURS) OVER (
            PARTITION BY CAST(user_id AS STRING)
            ORDER BY
                CAST(created_at AS TIMESTAMP) + INTERVAL 8 HOURS,
                COALESCE(CAST(event_id AS STRING), CAST(bullet_id AS STRING), '')
        ) AS prev_ts_bj
    FROM bullet_raw
    WHERE op_code NOT IN ('B26','TST','TSB','TSO')
      AND currency_type = '{currency}'
      AND game_id = '{game_id}'
),
order_enriched AS (
    SELECT
        *,
        CASE
            WHEN prev_ts_bj IS NULL
              OR unix_timestamp(ts_bj) - unix_timestamp(prev_ts_bj) > 180
            THEN 1
            ELSE 0
        END AS new_session_flag
    FROM order_base
),
sessionized AS (
    SELECT
        *,
        SUM(new_session_flag) OVER (
            PARTITION BY user_id
            ORDER BY ts_bj, COALESCE(CAST(event_id AS STRING), CAST(bullet_id AS STRING), '')
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_seq
    FROM order_enriched
),
session_day_assigned AS (
    SELECT
        *,
        MIN(natural_bet_date) OVER (
            PARTITION BY user_id, session_seq
        ) AS bet_date
    FROM sessionized
),
day_order_enriched AS (
    SELECT
        *,
        LAG(bullet_level) OVER (
            PARTITION BY user_id, bet_date
            ORDER BY ts_bj, COALESCE(CAST(event_id AS STRING), CAST(bullet_id AS STRING), '')
        ) AS prev_bullet_level_day,
        LAG(multiplier) OVER (
            PARTITION BY user_id, bet_date
            ORDER BY ts_bj, COALESCE(CAST(event_id AS STRING), CAST(bullet_id AS STRING), '')
        ) AS prev_multiplier_day
    FROM session_day_assigned
),
loss_runs AS (
    SELECT
        *,
        SUM(CASE WHEN profit < 0 THEN 0 ELSE 1 END) OVER (
            PARTITION BY user_id, bet_date
            ORDER BY ts_bj, COALESCE(CAST(event_id AS STRING), CAST(bullet_id AS STRING), '')
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS non_loss_run_group
    FROM day_order_enriched
),
loss_run_lengths AS (
    SELECT
        user_id,
        bet_date,
        non_loss_run_group,
        COUNT(*) AS loss_run_length
    FROM loss_runs
    WHERE profit < 0
    GROUP BY user_id, bet_date, non_loss_run_group
),
loss_day AS (
    SELECT
        user_id,
        bet_date,
        MAX(loss_run_length) AS max_consecutive_loss_count_today
    FROM loss_run_lengths
    GROUP BY user_id, bet_date
),
target_day AS (
    SELECT
        user_id,
        bet_date,
        CAST(SUM(CASE WHEN fish_value BETWEEN 2 AND 10 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS low_ratio,
        CAST(SUM(CASE WHEN fish_value BETWEEN 15 AND 130 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS medium_ratio,
        CAST(SUM(CASE WHEN fish_value BETWEEN 150 AND 200 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS high_ratio,
        CAST(SUM(CASE WHEN fish_value BETWEEN 500 AND 1000 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS ultra_ratio
    FROM day_order_enriched
    GROUP BY user_id, bet_date
),
day_base AS (
    SELECT
        user_id,
        bet_date,
        COUNT(*) AS bet_count_today,
        SUM(bet) AS bet_amount_today,
        AVG(bet) AS avg_bet_one_time_today,
        SUM(payout) AS payout_today,
        SUM(profit) AS profit_today,
        CASE WHEN SUM(bet) > 0 THEN SUM(payout) / SUM(bet) END AS rtp_day,
        MAX(curr_balance) AS current_balance_day_max,
        SUM(CASE
            WHEN prev_bullet_level_day IS NOT NULL AND bullet_level <> prev_bullet_level_day THEN 1
            ELSE 0
        END) AS bullet_level_change_count_day,
        SUM(CASE
            WHEN prev_multiplier_day IS NOT NULL AND multiplier <> prev_multiplier_day THEN 1
            ELSE 0
        END) AS multiplier_change_count_day
    FROM day_order_enriched
    GROUP BY user_id, bet_date
)
SELECT
    d.user_id,
    d.bet_date,
    d.bet_count_today,
    d.bet_amount_today,
    d.avg_bet_one_time_today,
    d.payout_today,
    d.profit_today,
    d.rtp_day,
    d.current_balance_day_max,
    COALESCE(l.max_consecutive_loss_count_today, 0) AS max_consecutive_loss_count_today,
    d.bullet_level_change_count_day,
    d.multiplier_change_count_day,
    t.low_ratio,
    t.medium_ratio,
    t.high_ratio,
    t.ultra_ratio
FROM day_base d
LEFT JOIN loss_day l
    ON d.user_id = l.user_id
   AND d.bet_date = l.bet_date
LEFT JOIN target_day t
    ON d.user_id = t.user_id
   AND d.bet_date = t.bet_date
WHERE d.bet_date >= CAST('{output_start}' AS DATE)
  AND d.bet_date < CAST('{output_end}' AS DATE)
ORDER BY d.user_id, d.bet_date
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="s3://... root of bullet parquet (year=/month=/day=)")
    parser.add_argument("--output-root", required=True, help="s3://... root for all outputs")
    parser.add_argument("--scan-start", required=True, help="first day of raw data to scan, YYYY-MM-DD")
    parser.add_argument("--scan-end", required=True, help="last day of raw data to scan (inclusive), YYYY-MM-DD")
    parser.add_argument("--output-start", required=True, help="keep rows with bet_date >= this, YYYY-MM-DD")
    parser.add_argument("--output-end", required=True, help="keep rows with bet_date < this, YYYY-MM-DD")
    parser.add_argument("--game-id", default="FM01")
    parser.add_argument("--currency", default="CNY")
    return parser.parse_args()


def month_jobs(scan_start, scan_end):
    """Split [scan_start, scan_end] into per-month scan windows.

    Each window deliberately extends one day into the next month
    (inclusive end = next month's 1st) so sessions running past the last
    midnight are complete; the duplicated boundary user-days this creates
    are removed later by the native-month dedup.
    """
    jobs = []
    cur = date(scan_start.year, scan_start.month, 1)
    while cur <= scan_end:
        nxt = date(cur.year + cur.month // 12, cur.month % 12 + 1, 1)
        jobs.append((cur.strftime("%Y-%m"), max(cur, scan_start), min(nxt, scan_end)))
        cur = nxt
    return jobs


def show_summary(df, label):
    print(f"--- {label} ---")
    df.selectExpr(
        "count(*) as rows",
        "count(distinct user_id) as users",
        "min(bet_date) as min_date",
        "max(bet_date) as max_date",
    ).show(truncate=False)


def main():
    args = parse_args()
    scan_start = date.fromisoformat(args.scan_start)
    scan_end = date.fromisoformat(args.scan_end)

    spark = (
        SparkSession.builder.appName("FM01_lifecycle_feature_engineering")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    base_root = f"{args.output_root}/user_day_base_fm01_cny"
    dedup_root = f"{args.output_root}/user_day_base_fm01_cny_dedup"
    feature_root = f"{args.output_root}/selected_hmm_features_fm01_cny"

    # ---- stage 1: monthly user-day base aggregates -------------------------
    bullet = spark.read.parquet(args.input_root)
    partition_date = F.make_date("year", "month", "day")

    for label, month_start, month_end in month_jobs(scan_start, scan_end):
        raw = bullet.filter(
            partition_date.between(F.to_date(F.lit(str(month_start))), F.to_date(F.lit(str(month_end))))
        )
        if raw.limit(1).count() == 0:
            print("no data, skip:", label)
            continue
        raw.createOrReplaceTempView("bullet_raw")
        df = spark.sql(
            USER_DAY_BASE_SQL.format(
                output_start=args.output_start,
                output_end=args.output_end,
                game_id=args.game_id,
                currency=args.currency,
            )
        )
        out = f"{base_root}/month={label}"
        df.repartition(64).write.mode("overwrite").parquet(out)
        show_summary(spark.read.parquet(out), f"saved {out}")

    # ---- stage 2: cross-month dedup ----------------------------------------
    # A user-day scanned in two adjacent month windows keeps the row from its
    # own month's window (the complete one); ties broken by activity volume.
    base_all = spark.read.parquet(base_root)
    w = Window.partitionBy("user_id", "bet_date").orderBy(
        (F.col("month") == F.date_format("bet_date", "yyyy-MM")).cast("int").desc(),
        F.col("bet_count_today").desc(),
        F.col("bet_amount_today").desc(),
    )
    base_dedup = base_all.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn")
    base_dedup.repartition(128).write.mode("overwrite").parquet(dedup_root)

    base = spark.read.parquet(dedup_root)
    show_summary(base, f"saved {dedup_root}")
    remaining_dups = base.groupBy("user_id", "bet_date").count().filter("count > 1").count()
    print("duplicate user-day groups after dedup:", remaining_dups)

    # ---- stage 3: selected HMM lifecycle features ---------------------------
    w_user_day = Window.partitionBy("user_id").orderBy("bet_date")
    w_hist = w_user_day.rowsBetween(Window.unboundedPreceding, -1)
    w_7 = w_user_day.rowsBetween(-6, 0)

    def entropy_term(col_name):
        c = F.coalesce(F.col(col_name), F.lit(0.0))
        return F.when(c > 0, c * F.log(c)).otherwise(F.lit(0.0))

    selected_features = (
        base.withColumn("prev_bet_date", F.lag("bet_date").over(w_user_day))
        .withColumn("no_bet_streak_days", F.datediff("bet_date", "prev_bet_date") - F.lit(1))
        .withColumn("hist_avg_bet_amount_per_bet_day", F.avg("bet_amount_today").over(w_hist))
        .withColumn(
            "bet_amount_ratio_today_vs_history",
            F.when(
                F.col("hist_avg_bet_amount_per_bet_day") > 0,
                F.col("bet_amount_today") / F.col("hist_avg_bet_amount_per_bet_day"),
            ),
        )
        .withColumn("hist_avg_bet_count_per_bet_day", F.avg("bet_count_today").over(w_hist))
        .withColumn(
            "bet_count_ratio_today_vs_history",
            F.when(
                F.col("hist_avg_bet_count_per_bet_day") > 0,
                F.col("bet_count_today") / F.col("hist_avg_bet_count_per_bet_day"),
            ),
        )
        .withColumn(
            "avg_bet_one_time_today_log",
            F.log1p(F.greatest(F.col("avg_bet_one_time_today"), F.lit(0.0))),
        )
        .withColumn("payout_7_bet_days", F.sum("payout_today").over(w_7))
        .withColumn("bet_amount_7_bet_days", F.sum("bet_amount_today").over(w_7))
        .withColumn(
            "rtp_7_bet_days",
            F.when(
                F.col("bet_amount_7_bet_days") > 0,
                F.col("payout_7_bet_days") / F.col("bet_amount_7_bet_days"),
            ),
        )
        .withColumn(
            "loss_streak_ratio_today",
            F.col("max_consecutive_loss_count_today") / F.col("bet_count_today"),
        )
        .withColumn("hist_avg_bet_one_time", F.avg("avg_bet_one_time_today").over(w_hist))
        .withColumn(
            "current_balance_max_to_avg_bet_ratio",
            F.when(
                F.col("hist_avg_bet_one_time") > 0,
                F.col("current_balance_day_max") / F.col("hist_avg_bet_one_time"),
            ),
        )
        .withColumn(
            "target_selection_entropy",
            -(
                entropy_term("low_ratio")
                + entropy_term("medium_ratio")
                + entropy_term("high_ratio")
                + entropy_term("ultra_ratio")
            )
            / F.log(F.lit(4.0)),
        )
        .withColumn(
            "multiplier_change_count_ratio",
            F.col("multiplier_change_count_day") / F.col("bet_count_today"),
        )
        .withColumn(
            "bullet_level_change_count_ratio",
            F.col("bullet_level_change_count_day") / F.col("bet_count_today"),
        )
        .select(
            "user_id",
            "bet_date",
            "no_bet_streak_days",
            "bet_amount_ratio_today_vs_history",
            "bet_count_ratio_today_vs_history",
            "avg_bet_one_time_today_log",
            "rtp_7_bet_days",
            "loss_streak_ratio_today",
            "current_balance_max_to_avg_bet_ratio",
            "target_selection_entropy",
            "multiplier_change_count_ratio",
            "bullet_level_change_count_ratio",
            # debug/raw
            "bet_count_today",
            "bet_amount_today",
            "avg_bet_one_time_today",
            "payout_today",
            "profit_today",
            "rtp_day",
            "current_balance_day_max",
            "max_consecutive_loss_count_today",
            "bullet_level_change_count_day",
            "multiplier_change_count_day",
            "low_ratio",
            "medium_ratio",
            "high_ratio",
            "ultra_ratio",
        )
    )

    selected_features.repartition(128).write.mode("overwrite").parquet(feature_root)
    show_summary(spark.read.parquet(feature_root), f"saved {feature_root}")

    # ---- stage 4: single-file CSV exports -----------------------------------
    for name, df in [
        ("selected_hmm_features_fm01_cny", selected_features),
        ("user_day_base_fm01_cny", base),
    ]:
        csv_out = f"{args.output_root}/csv/{name}"
        (df.orderBy("user_id", "bet_date").coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_out))
        print("saved csv:", csv_out)

    spark.stop()


if __name__ == "__main__":
    main()
