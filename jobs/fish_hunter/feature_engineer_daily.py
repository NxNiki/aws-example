"""FM01 per-user-day feature engineering — PySpark job, runs on the cluster.

ETL job: builds the fish_hunter analogue of the slot-machine grouped feature
pipeline (``bituslabs_ds.features`` / ss03 ``features_grouped``), aggregated on
the same daily grain as the lifecycle HMM features: one row of distributional
statistics per ``(user_id, bet_date)`` over ALL of that user-day's bullets.
Output dataset: ``<output-root>/features_user_daily/month=YYYY-MM/``.

Behavior: reads bullet order parquet directly from S3 (partition-pruned by
year=/month=/day=). Sessionization matches feature_engineer_life_cycle.py:
sessions break on gaps > 180s, and ``bet_date`` is the session's START date,
so play crossing midnight stays on the day it started. Per-bullet metrics
mirror the slot CTE chain -- bet-to-bet deltas, deposit/withdraw detection
from balance jumps, win/lose streaks (streak gap = the 180s session break, so
streaks never span sessions) -- rolled up with the same column names as the
ss03 grouped features (avg/p25/median/p75/std/cv, drawdown/spike ratios,
accumulators, rtp, payout/profit rates; percentiles are percentile_approx).
Fish-specific additions per fish_hunter_daily.sql: ``kill_rate`` and
fish-type share columns (``low/medium/high/ultra_ratio``; fish_value
<=10/<=130/<=200/else). Rows are attributed to the month of ``bet_date``;
each month scans one margin day on both sides and outputs only its own
bet_dates (exactly-once tiling). ``bet_rounds`` carries the user-day's bullet
count. A final full-window pass adds ``day_index`` (the user's 1-based
bet-day rank, the HMM's tenure ``k``) and ``is_first_bet_day`` so consumers
can match the HMM's k>=2 training population by filtering
``is_first_bet_day == 0``.

Submit with feature_engineer_daily_submit.py.
"""

import argparse
from datetime import date, timedelta

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window as W

# Ships via submit_py_files (see feature_engineer_daily_submit.py).
from spark_etl_common import prune_partition_days, prune_period_days

# Same session definition as the lifecycle HMM feature job.
SESSION_BREAK_SECONDS = 180
# Streaks reset with the session (ss03 uses 200s; here the session break is
# shorter than that, so the session boundary is the binding threshold).
STREAK_SECONDS = SESSION_BREAK_SECONDS
NOGAP_SECONDS = 60 * 60
# Sessions are 180s-gap bounded, so they cross at most a midnight or two;
# one margin day on each side of the month keeps boundary sessions whole.
SCAN_MARGIN_DAYS = 1

PRE_AGG_SQL = """
WITH user_bets AS (
    SELECT
        CAST(bullet_id AS STRING) AS bullet_id,
        user_id,
        CAST(created_at AS TIMESTAMP) + INTERVAL 8 HOURS AS ts_bj,
        CAST(bet AS DOUBLE) AS bet_amount,
        CAST(payout AS DOUBLE) AS payout,
        CAST(prev_balance AS DOUBLE) AS prev_balance,
        CAST(curr_balance AS DOUBLE) AS balance_after_bet,
        COALESCE(CAST(profit AS DOUBLE), CAST(payout AS DOUBLE) - CAST(bet AS DOUBLE)) AS profit,
        CAST(killed AS INT) AS killed,
        CASE
            WHEN fish_value <= 10 THEN 'low'
            WHEN fish_value <= 130 THEN 'medium'
            WHEN fish_value <= 200 THEN 'high'
            ELSE 'ultra'
        END AS fish_type
    FROM bullet_raw
    WHERE op_code NOT IN ('B26','TST','TSB','TSO')
      AND currency_type = '{currency}'
      AND game_id = '{game_id}'
),

delta_stats AS (
    SELECT
        *,
        unix_timestamp(ts_bj)
            - unix_timestamp(LAG(ts_bj) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id))
            AS delta_t_seconds,
        bet_amount - LAG(bet_amount) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id)
            AS delta_bet_amount,
        payout - LAG(payout) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id) AS delta_payout,
        prev_balance - LAG(balance_after_bet) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id)
            AS balance_transaction,
        CASE WHEN payout > bet_amount THEN 1 ELSE 0 END AS is_win,
        CASE WHEN payout < bet_amount THEN 1 ELSE 0 END AS is_lose,
        CASE
            WHEN LAG(payout) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id)
                 > LAG(bet_amount) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id)
            THEN 1 ELSE 0
        END AS prev_win,
        CASE
            WHEN LAG(payout) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id)
                 < LAG(bet_amount) OVER (PARTITION BY user_id ORDER BY ts_bj, bullet_id)
            THEN 1 ELSE 0
        END AS prev_lose
    FROM user_bets
),

user_group AS (
    SELECT
        *,
        LAST_VALUE(
            CASE
                WHEN delta_t_seconds > {session_break} OR delta_t_seconds IS NULL THEN ts_bj
            END,
            TRUE
        ) OVER (
            PARTITION BY user_id ORDER BY ts_bj, bullet_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_start_ts,
        SUM(CASE WHEN delta_t_seconds <= {streak} THEN 0 ELSE 1 END) OVER (
            PARTITION BY user_id ORDER BY ts_bj, bullet_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS streak_group,
        SUM(
            CASE WHEN delta_t_seconds <= {streak} AND prev_win = 1 AND is_win = 1 THEN 0 ELSE 1 END
        ) OVER (
            PARTITION BY user_id ORDER BY ts_bj, bullet_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS win_streak_group,
        SUM(
            CASE WHEN delta_t_seconds <= {streak} AND prev_lose = 1 AND is_lose = 1 THEN 0 ELSE 1 END
        ) OVER (
            PARTITION BY user_id ORDER BY ts_bj, bullet_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS lose_streak_group
    FROM delta_stats
),

raw_stats AS (
    SELECT
        user_id,
        CAST(session_start_ts AS DATE) AS bet_date,
        ts_bj,
        CAST(ts_bj AS DATE) AS activity_date,
        bet_amount,
        delta_bet_amount,
        payout,
        delta_payout,
        balance_after_bet,
        delta_t_seconds,
        CASE WHEN delta_t_seconds <= {nogap} THEN delta_t_seconds END AS delta_t_seconds_nogap,
        profit,
        killed,
        fish_type,
        CASE WHEN balance_transaction > .1 THEN balance_transaction END AS deposit,
        CASE WHEN balance_transaction < -.1 THEN -balance_transaction END AS withdraw,
        ROW_NUMBER() OVER (PARTITION BY user_id, streak_group ORDER BY ts_bj, bullet_id) AS streak,
        CASE
            WHEN payout > bet_amount
                THEN ROW_NUMBER() OVER (PARTITION BY user_id, win_streak_group ORDER BY ts_bj, bullet_id)
        END AS win_streak,
        CASE
            WHEN payout < bet_amount
                THEN ROW_NUMBER() OVER (PARTITION BY user_id, lose_streak_group ORDER BY ts_bj, bullet_id)
        END AS lose_streak
    FROM user_group
)
"""


def _percentile_lines(metric):
    return [
        f"percentile_approx({metric}, 0.25) AS {metric}_p25",
        f"percentile_approx({metric}, 0.5) AS {metric}_median",
        f"percentile_approx({metric}, 0.75) AS {metric}_p75",
    ]


def _basic_lines(metric):
    return [
        f"AVG({metric}) AS {metric}_avg",
        *_percentile_lines(metric),
        f"STDDEV({metric}) AS {metric}_std",
        f"MIN({metric}) AS {metric}_min",
        f"MAX({metric}) AS {metric}_max",
    ]


def _ratio_lines(metric, norm_by="bet_amount_avg", coalesce=False):
    """cv / drawdown / spike ratios computed from the same SELECT's aggregates."""

    def wrap(expr):
        return f"COALESCE({expr}, 0)" if coalesce else expr

    norm = {"bet_amount_avg": "AVG(bet_amount)", "balance_after_bet_avg": "AVG(balance_after_bet)"}[norm_by]
    return [
        f"{wrap(f'STDDEV({metric}) * 1.0 / NULLIF({norm}, 0)')} AS {metric}_cv",
        f"{wrap(f'MIN({metric}) * 1.0 / NULLIF({norm}, 0)')} AS {metric}_drawdown_ratio",
        f"{wrap(f'MAX({metric}) * 1.0 / NULLIF({norm}, 0)')} AS {metric}_spike_ratio",
    ]


def _streak_lines(metric):
    return [
        f"COALESCE(AVG({metric}), 0) AS {metric}_avg",
        f"COALESCE(percentile_approx({metric}, 0.25), 0) AS {metric}_p25",
        f"COALESCE(percentile_approx({metric}, 0.5), 0) AS {metric}_median",
        f"COALESCE(percentile_approx({metric}, 0.75), 0) AS {metric}_p75",
        f"COALESCE(STDDEV({metric}), 0) AS {metric}_std",
        f"COALESCE(MIN({metric}), 0) AS {metric}_min",
        f"COALESCE(MAX({metric}), 0) AS {metric}_max",
    ]


def build_grouped_select(output_start, output_end):
    """Per-user-day rollup with the same column names as the ss03 grouped features."""
    agg = [
        "MIN(activity_date) AS activity_date",
        "MIN(ts_bj) AS min_created_at",
        "MAX(ts_bj) AS max_created_at",
        "COUNT(*) AS bet_rounds",
        # delta bet time
        *_basic_lines("delta_t_seconds"),
        *_basic_lines("delta_t_seconds_nogap"),
        # bet amount
        *_basic_lines("bet_amount"),
        *_ratio_lines("bet_amount"),
        # delta bet amount
        *_basic_lines("delta_bet_amount"),
        *_ratio_lines("delta_bet_amount", coalesce=True),
        "SUM(CASE WHEN delta_bet_amount > 0 THEN delta_bet_amount ELSE 0 END) AS accum_pos_delta_bet_amount",
        "SUM(CASE WHEN delta_bet_amount < 0 THEN delta_bet_amount ELSE 0 END) AS accum_neg_delta_bet_amount",
        "SUM(CASE WHEN delta_bet_amount > 0 THEN delta_bet_amount ELSE 0 END) * 1.0"
        " / NULLIF(AVG(bet_amount), 0) AS accum_pos_delta_bet_amount_ratio",
        "SUM(CASE WHEN delta_bet_amount < 0 THEN delta_bet_amount ELSE 0 END) * 1.0"
        " / NULLIF(AVG(bet_amount), 0) AS accum_neg_delta_bet_amount_ratio",
        # payout
        *_basic_lines("payout"),
        *_ratio_lines("payout"),
        "SUM(CASE WHEN payout > 0 THEN 1 END) * 1.0 / NULLIF(COUNT(*), 0) AS payout_rate",
        # rtp
        "AVG(payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_mean",
        "MAX(payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_max",
        "MIN(payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_min",
        "percentile_approx(payout * 1.0 / NULLIF(bet_amount, 0), 0.25) AS rtp_p25",
        "percentile_approx(payout * 1.0 / NULLIF(bet_amount, 0), 0.5) AS rtp_median",
        "percentile_approx(payout * 1.0 / NULLIF(bet_amount, 0), 0.75) AS rtp_p75",
        # profit (min/max ratios named per the slot pipeline)
        *_basic_lines("profit"),
        "STDDEV(profit) * 1.0 / NULLIF(AVG(bet_amount), 0) AS profit_cv",
        "MIN(profit) * 1.0 / NULLIF(AVG(bet_amount), 0) AS min_profit_ratio",
        "MAX(profit) * 1.0 / NULLIF(AVG(bet_amount), 0) AS max_profit_ratio",
        "SUM(CASE WHEN profit > 0 THEN profit ELSE 0 END) AS accum_pos_profit",
        "SUM(CASE WHEN profit < 0 THEN profit ELSE 0 END) AS accum_neg_profit",
        "SUM(CASE WHEN profit > 0 THEN profit ELSE 0 END) * 1.0"
        " / NULLIF(AVG(bet_amount), 0) AS accum_pos_profit_ratio",
        "SUM(CASE WHEN profit < 0 THEN profit ELSE 0 END) * 1.0"
        " / NULLIF(AVG(bet_amount), 0) AS accum_neg_profit_ratio",
        "SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS profit_rate",
        # delta payout
        *_basic_lines("delta_payout"),
        *_ratio_lines("delta_payout", coalesce=True),
        # balance after bet (normalized by its own avg)
        *_basic_lines("balance_after_bet"),
        *_ratio_lines("balance_after_bet", norm_by="balance_after_bet_avg"),
        # deposit / withdraw
        "SUM(CASE WHEN deposit > 0 THEN 1 ELSE 0 END) AS num_deposit",
        "SUM(CASE WHEN deposit > 0 THEN deposit ELSE 0 END) AS accum_deposit",
        "SUM(CASE WHEN deposit > 0 THEN deposit ELSE 0 END) * 1.0"
        " / NULLIF(AVG(bet_amount), 0) AS accum_deposit_ratio",
        "SUM(CASE WHEN withdraw > 0 THEN 1 ELSE 0 END) AS num_withdraw",
        "SUM(CASE WHEN withdraw > 0 THEN withdraw ELSE 0 END) AS accum_withdraw",
        "SUM(CASE WHEN withdraw > 0 THEN withdraw ELSE 0 END) * 1.0"
        " / NULLIF(AVG(bet_amount), 0) AS accum_withdraw_ratio",
        # streaks
        *_streak_lines("streak"),
        *_streak_lines("win_streak"),
        *_streak_lines("lose_streak"),
        # fish_hunter-specific (fish_value buckets per fish_hunter_daily.sql)
        "AVG(killed * 1.0) AS kill_rate",
        "AVG(CASE WHEN fish_type = 'low' THEN 1.0 ELSE 0.0 END) AS low_ratio",
        "AVG(CASE WHEN fish_type = 'medium' THEN 1.0 ELSE 0.0 END) AS medium_ratio",
        "AVG(CASE WHEN fish_type = 'high' THEN 1.0 ELSE 0.0 END) AS high_ratio",
        "AVG(CASE WHEN fish_type = 'ultra' THEN 1.0 ELSE 0.0 END) AS ultra_ratio",
    ]
    cols = ",\n    ".join(agg)
    return f"""
SELECT
    user_id,
    bet_date,
    {cols}
FROM raw_stats
WHERE bet_date >= CAST('{output_start}' AS DATE)
  AND bet_date < CAST('{output_end}' AS DATE)
GROUP BY user_id, bet_date
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="s3://... root of bullet parquet (year=/month=/day=)")
    parser.add_argument("--output-root", required=True, help="s3://... root for the daily grouped features")
    parser.add_argument("--output-start", required=True, help="keep rows with bet_date >= this")
    parser.add_argument("--output-end", required=True, help="keep rows with bet_date < this")
    parser.add_argument("--game-id", default="FM01")
    parser.add_argument("--currency", default="CNY")
    return parser.parse_args()


def month_jobs(output_start, output_end):
    """Split [output_start, output_end) into per-month (label, scan window, output window).

    Rows belong to the month of their session-start bet_date. Each month scans
    SCAN_MARGIN_DAYS beyond both edges so sessions crossing the boundary are
    complete, and outputs only its own bet_dates, so every user-day is emitted
    by exactly one month job.
    """
    jobs = []
    cur = date(output_start.year, output_start.month, 1)
    while cur < output_end:
        nxt = date(cur.year + cur.month // 12, cur.month % 12 + 1, 1)
        out_lo = max(cur, output_start)
        out_hi = min(nxt, output_end)
        margin = timedelta(days=SCAN_MARGIN_DAYS)
        jobs.append((cur.strftime("%Y-%m"), out_lo - margin, out_hi + margin, out_lo, out_hi))
        cur = nxt
    return jobs


def main():
    args = parse_args()
    output_start = date.fromisoformat(args.output_start)
    output_end = date.fromisoformat(args.output_end)

    spark = (
        SparkSession.builder.appName("FM01_feature_engineer_daily")  # type: ignore[attr-defined]
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    daily_root = f"{args.output_root}/features_user_daily"
    bullet = spark.read.parquet(args.input_root)

    pre_agg = PRE_AGG_SQL.format(
        currency=args.currency,
        game_id=args.game_id,
        session_break=SESSION_BREAK_SECONDS,
        streak=STREAK_SECONDS,
        nogap=NOGAP_SECONDS,
    )

    staging_root = f"{daily_root}_staging"
    for label, scan_lo, scan_hi, out_lo, out_hi in month_jobs(output_start, output_end):
        # Each pruner no-ops without its partition columns, so the job reads
        # the fish_bullets_group_tag dataset (period=) or the raw warehouse
        # (year=/month=/day=) alike.
        raw = prune_period_days(prune_partition_days(bullet, scan_lo, scan_hi), scan_lo, scan_hi)
        if raw.limit(1).count() == 0:
            print("no data, skip:", label)
            continue
        raw.createOrReplaceTempView("bullet_raw")
        grouped = spark.sql(pre_agg + build_grouped_select(out_lo, out_hi))
        out = f"{staging_root}/month={label}"
        grouped.repartition(1).write.mode("overwrite").parquet(out)
        check = spark.read.parquet(out)
        check.selectExpr(
            "count(*) as user_days",
            "count(distinct user_id) as users",
            "min(bet_date) as min_date",
            "max(bet_date) as max_date",
        ).show(truncate=False)
        print("staged:", out)

    # ---- global per-user day_index (the HMM's tenure k) needs all months ----
    # is_first_bet_day == 1 marks each user's first bet-day within the window;
    # the cluster config filters it out to match the HMM's k>=2 population.
    full = spark.read.parquet(staging_root)
    w_user = W.partitionBy("user_id").orderBy("bet_date")
    full = full.withColumn("day_index", F.row_number().over(w_user).cast("int"))
    full = full.withColumn("is_first_bet_day", (F.col("day_index") == 1).cast("int"))
    full.repartition("month").write.partitionBy("month").mode("overwrite").parquet(daily_root)
    print("saved:", daily_root)
    spark.read.parquet(daily_root).selectExpr(
        "count(*) as user_days",
        "count(distinct user_id) as users",
        "sum(is_first_bet_day) as first_bet_days",
    ).show(truncate=False)

    hadoop_path = spark._jvm.org.apache.hadoop.fs.Path(staging_root)  # type: ignore[attr-defined]
    fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())  # type: ignore[attr-defined]
    fs.delete(hadoop_path, True)
    print("removed staging:", staging_root)

    spark.stop()


if __name__ == "__main__":
    main()
