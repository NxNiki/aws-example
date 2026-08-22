"""SS03 AB-test metrics by math table and daily-bet bucket, with D1/D3 retention.

One-off analysis (not wired into ``run_scheduled_etl_jobs.py``): for the
window 2026-07-29..2026-08-17 (Beijing bet dates), reports per
(bet_date, ab_group, mathtable, daily total-bet bucket):
day-1 / day-3 retention, total_spin (BASE bets) and total bet (CNY).
Goal: identify which math table suits users of each daily-spend tier,
using retention as the indicator.

Definitions:
- **Session**: 30-min bet gap per user (repo ``bet_session`` convention).
  A session is labeled with the math table of its FIRST bet; within-session
  math-table changes are ignored.
- **bet_date**: Beijing calendar date of the session's first bet, so a
  session crossing midnight belongs wholly to the day it started.
- **User-day mathtable**: winner-take-all across the day's sessions — the
  session math table with the most BASE bets that day (ties: more total
  rows, then earliest session).
- **User-day ab_group**: branch-order collapse AI > AB_TEST_A > AB_TEST_B >
  Default (constant within a day except at the policy cutover).
- **Currency**: all currencies kept; bet amounts converted to CNY at the
  rates in ``FX_TO_CNY`` (exchangerate-api.com, as of ``FX_AS_OF``).
- **Retention**: cohort = users with >= 1 bet on day D in a cell; retained
  if the user has ANY SS03 bet on D+1 / D+3 (any group/table/currency).
  D+N beyond ``PRESENCE_END`` (last complete Beijing day) reports null.
- The AB policy cutover days 2026-08-03/04 (Beijing) are excluded from
  metric rows and cohorts (group membership ambiguous — same as the
  feature ETL), but still count as presence for the retained-on-D+N check.

Source: Athena ``bituslabs_ds.slot_orders_ab_group`` (cold data + the
policy-correct ``ab_group``; status/op-code junk pre-filtered upstream).

Usage:
    poetry run python jobs/ss03_mahjiang_streak/analysis_ab_mathtable_retention.py --extract
    poetry run python jobs/ss03_mahjiang_streak/analysis_ab_mathtable_retention.py --analyze
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import awswrangler as wr
import boto3
import polars as pl

from bituslabs_ds.config import DEFAULT_ATHENA_OUTPUT, REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WINDOW_START = date(2026, 7, 29)
WINDOW_END = date(2026, 8, 17)  # inclusive
# Beijing dates droppped from metrics/cohorts: AB grouping policy flipped
# partition_ab -> user-id digits across these days, membership is ambiguous.
EXCLUDED_COHORT_DATES = (date(2026, 8, 3), date(2026, 8, 4))
# Last COMPLETE Beijing day on S3 (the daily 09:30 LA refresh = 00:30 BJ
# next day); presence beyond it is partial, so D+N past it reports null.
PRESENCE_END = date(2026, 8, 19)

# Scan margin: 2 days before the window so sessions straddling the window
# start keep their true first-bet date; through 08-21 for retention presence.
SCAN_PERIOD_START = "2026-07-27"
SCAN_PERIOD_END = "2026-08-21"

SESSION_GAP_SECONDS = 1800

FX_AS_OF = "2026-08-20"
FX_TO_CNY = {
    "CNY": 1.0,
    "USD": 6.7460,
    "INR": 0.070504,
    "IDR": 0.000378,
    "KRW": 0.0047890,
    "MYR": 1.6690,
    "PHP": 0.108900,
    "THB": 0.204844,
    "VND": 0.000258,
}

AB_GROUPS = ("AB_TEST_A", "AB_TEST_B", "Default", "AI")

# AI per-mathtable analysis: a user-day counts FULLY toward every math table
# the user had a session on that day (multi-attribution, NOT winner-take-all),
# and user-days with fewer than this many BASE bets are dropped — the AI group
# rotates tables within a day, so low-volume days say little about any single
# table and add noise.
AI_MIN_DAILY_BETS = 50

# Daily total-bet (CNY) buckets (see --distribution: median ~16, p75 ~80,
# p90 ~360; these edges split user-days ~42/36/18/5%).
BET_BUCKET_EDGES = (10, 100, 1000)

# First date where A/B are fully on their new math tables (switched 07-30,
# a mixed day); the "which table for which tier" comparison uses this range
# so each group maps 1:1 to one table.
POST_SWITCH_START = date(2026, 7, 31)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output_ss03_mahjiang_streak" / "ab_mathtable_retention"
USER_DAY_DIR = OUTPUT_DIR / "user_day"
USER_DAY_MT_DIR = OUTPUT_DIR / "user_day_mathtable"


def _rate_case() -> str:
    whens = "\n            ".join(f"WHEN '{c}' THEN {r!r}" for c, r in FX_TO_CNY.items())
    return f"CASE currency_type\n            {whens}\n        END"


def _base_ctes() -> str:
    """Shared CTEs: sessionized bets and the per-user-day aggregate."""
    return f"""
WITH bets AS (
    SELECT
        user_id,
        spin_id,
        created_at,
        COALESCE(NULLIF(math_table_id, ''), '(none)') AS mathtable,
        bet_type,
        bet_amount,
        currency_type,
        ab_group
    FROM slot_orders_ab_group
    WHERE game_id = 'SS03'
      AND period BETWEEN '{SCAN_PERIOD_START}' AND '{SCAN_PERIOD_END}'
),

seq AS (
    SELECT
        t.*,
        LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at, spin_id) AS prev_ts
    FROM bets AS t
),

sessions AS (
    SELECT
        *,
        SUM(
            CASE
                WHEN prev_ts IS NULL OR date_diff('second', prev_ts, created_at) > {SESSION_GAP_SECONDS} THEN 1
                ELSE 0
            END
        ) OVER (PARTITION BY user_id ORDER BY created_at, spin_id ROWS UNBOUNDED PRECEDING) AS session_seq
    FROM seq
),

sess AS (
    SELECT
        *,
        MIN(created_at) OVER (PARTITION BY user_id, session_seq) AS session_start_ts,
        FIRST_VALUE(mathtable) OVER (
            PARTITION BY user_id, session_seq
            ORDER BY created_at, spin_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS session_mathtable
    FROM sessions
),

labeled AS (
    SELECT *, CAST(session_start_ts + INTERVAL '8' HOUR AS DATE) AS bet_date
    FROM sess
),

user_day AS (
    SELECT
        bet_date,
        user_id,
        CASE MIN(CASE ab_group WHEN 'AI' THEN 0 WHEN 'AB_TEST_A' THEN 1 WHEN 'AB_TEST_B' THEN 2 ELSE 3 END)
            WHEN 0 THEN 'AI' WHEN 1 THEN 'AB_TEST_A' WHEN 2 THEN 'AB_TEST_B' ELSE 'Default'
        END AS ab_group,
        COUNT(CASE WHEN bet_type = 'BASE' THEN 1 END) AS total_spin,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN bet_type = 'BASE' THEN bet_amount * {_rate_case()} END) AS total_bet_cny,
        COUNT(DISTINCT session_seq) AS n_sessions,
        COUNT(DISTINCT session_mathtable) AS n_mathtables,
        COUNT(CASE WHEN currency_type NOT IN ({", ".join(f"'{c}'" for c in FX_TO_CNY)}) THEN 1 END)
            AS n_unknown_currency
    FROM labeled
    GROUP BY bet_date, user_id
)
"""


def extraction_sql() -> str:
    return f"""{_base_ctes()},

mt_counts AS (
    SELECT
        bet_date,
        user_id,
        session_mathtable AS mathtable,
        COUNT(CASE WHEN bet_type = 'BASE' THEN 1 END) AS n_base,
        COUNT(*) AS n_rows,
        MIN(session_start_ts) AS first_ts
    FROM labeled
    GROUP BY bet_date, user_id, session_mathtable
),

mt_winner AS (
    SELECT bet_date, user_id, mathtable
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY bet_date, user_id
                ORDER BY n_base DESC, n_rows DESC, first_ts ASC
            ) AS rn
        FROM mt_counts
    )
    WHERE rn = 1
)

SELECT
    u.bet_date,
    u.user_id,
    u.ab_group,
    w.mathtable,
    u.total_spin,
    u.total_rows,
    u.total_bet_cny,
    u.n_sessions,
    u.n_mathtables,
    u.n_unknown_currency
FROM user_day AS u
JOIN mt_winner AS w ON u.bet_date = w.bet_date AND u.user_id = w.user_id
WHERE u.bet_date BETWEEN DATE '{WINDOW_START}' AND DATE '{SCAN_PERIOD_END}'
"""


def mathtable_sql() -> str:
    """One row per (user-day, BET-LEVEL mathtable), carrying the WHOLE day's totals.

    Multi-attribution source for the AI per-mathtable analysis: a user whose
    bets touched tables m1 and m2 the same day yields two rows, each with the
    day's full total_spin / total_bet_cny. Uses the per-bet math_table_id
    (not the session label) because the AI group rotates tables WITHIN a
    session (avg 2.1 bet-level tables per AI user-day vs 1.1 session-labeled);
    the day itself is still the session-start Beijing date.
    """
    return f"""{_base_ctes()},

mt_sets AS (
    SELECT bet_date, user_id, mathtable
    FROM labeled
    GROUP BY bet_date, user_id, mathtable
)

SELECT
    m.bet_date,
    m.user_id,
    u.ab_group,
    m.mathtable,
    u.total_spin,
    u.total_rows,
    u.total_bet_cny,
    u.n_sessions
FROM mt_sets AS m
JOIN user_day AS u ON m.bet_date = u.bet_date AND m.user_id = u.user_id
WHERE m.bet_date BETWEEN DATE '{WINDOW_START}' AND DATE '{SCAN_PERIOD_END}'
"""


def _run_and_save(sql: str, out_dir: Path, label: str) -> None:
    session = boto3.Session(region_name=REGION)
    logger.info("running Athena extraction for %s (scans ~13M SS03 bet rows)", label)
    df = wr.athena.read_sql_query(
        sql,
        database="bituslabs_ds",
        s3_output=DEFAULT_ATHENA_OUTPUT,
        ctas_approach=False,
        boto3_session=session,
    )
    logger.info("fetched %d %s rows", len(df), label)
    pdf = pl.from_pandas(df).with_columns(pl.col("bet_date").cast(pl.Date))
    out_dir.mkdir(parents=True, exist_ok=True)
    for day, part in pdf.partition_by("bet_date", as_dict=True).items():
        part.write_parquet(out_dir / f"bet_date={day[0]}.parquet")
    logger.info("saved %d daily parquet files under %s", pdf["bet_date"].n_unique(), out_dir)


def extract() -> None:
    """Run the Athena extractions and save locally, one parquet per Beijing bet_date."""
    _run_and_save(extraction_sql(), USER_DAY_DIR, "user-day")
    _run_and_save(mathtable_sql(), USER_DAY_MT_DIR, "user-day-mathtable")


def bootstrap_mean_ci(
    df: pl.DataFrame,
    keys: list[str],
    col: str = "total_bet_cny",
    n_boot: int = 500,
    prefix: str = "avg_bet_cny",
) -> pl.DataFrame:
    """Percentile-bootstrap 95% CI of the per-user-day mean of ``col`` per key.

    The per-user-day distributions inside a cell are heavily right-skewed, so
    a normal approximation would misstate the interval. Fixed seed for
    reproducibility. Output columns: ``{prefix}_ci_lo`` / ``{prefix}_ci_hi``.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for key, grp in df.partition_by(keys, as_dict=True).items():
        vals = grp[col].to_numpy()
        means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
        rows.append(
            dict(zip(keys, key))
            | {
                f"{prefix}_ci_lo": float(np.percentile(means, 2.5)),
                f"{prefix}_ci_hi": float(np.percentile(means, 97.5)),
            }
        )
    return pl.DataFrame(rows)


def load_user_day() -> pl.DataFrame:
    files = sorted(USER_DAY_DIR.glob("bet_date=*.parquet"))
    if not files:
        raise SystemExit(f"no extracted data under {USER_DAY_DIR}; run with --extract first")
    return pl.concat([pl.read_parquet(f) for f in files])


def bucket_labels() -> list[str]:
    edges = BET_BUCKET_EDGES
    labels = [f"<{edges[0]}"]
    labels += [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
    return labels + [f"{edges[-1]}+"]


def bucket_expr() -> pl.Expr:
    labels = bucket_labels()
    expr: pl.Expr = pl.lit(labels[-1])
    for edge, label in reversed(list(zip(BET_BUCKET_EDGES, labels))):
        expr = pl.when(pl.col("total_bet_cny") < edge).then(pl.lit(label)).otherwise(expr)
    return expr.alias("bet_bucket")


def analyze() -> dict[str, pl.DataFrame]:
    """Compute the metric tables; returns them keyed by name and writes CSVs."""
    ud = load_user_day()
    if ud["n_unknown_currency"].sum() > 0:
        raise SystemExit("rows with un-mapped currency present — extend FX_TO_CNY and re-extract")

    presence = ud.select("user_id", "bet_date").unique()

    cells = ud.filter(
        pl.col("bet_date").is_between(WINDOW_START, WINDOW_END)
        & ~pl.col("bet_date").is_in(list(EXCLUDED_COHORT_DATES))
        & pl.col("ab_group").is_in(list(AB_GROUPS))
        & (pl.col("total_spin") > 0)
    ).with_columns(bucket_expr())

    for n in (1, 3):
        target = cells.select(
            "user_id",
            (pl.col("bet_date") + timedelta(days=n)).alias("target_date"),
        )
        hit = target.join(
            presence.rename({"bet_date": "target_date"}),
            on=["user_id", "target_date"],
            how="left",
            coalesce=False,
        ).select(pl.col("user_id_right").is_not_null().alias(f"retained_d{n}"))
        cells = cells.with_columns(hit[f"retained_d{n}"])
        cells = cells.with_columns(
            pl.when(pl.col("bet_date") + timedelta(days=n) > PRESENCE_END)
            .then(None)
            .otherwise(pl.col(f"retained_d{n}"))
            .alias(f"retained_d{n}")
        )

    group_cols = ["ab_group", "mathtable", "bet_bucket"]
    aggs = [
        pl.len().alias("user_days"),
        pl.col("user_id").n_unique().alias("n_users"),
        pl.col("total_spin").sum().alias("total_spin"),
        pl.col("total_bet_cny").sum().alias("total_bet_cny"),
        (pl.col("total_bet_cny").sum() / pl.col("total_spin").sum()).alias("avg_bet_cny"),
        pl.col("total_bet_cny").mean().alias("avg_bet_cny_per_user_day"),
        pl.col("retained_d1").mean().alias("retention_d1"),
        pl.col("retained_d1").count().alias("d1_cohort"),
        pl.col("retained_d3").mean().alias("retention_d3"),
        pl.col("retained_d3").count().alias("d3_cohort"),
    ]

    tables = {
        "summary_group_mathtable_bucket": cells.group_by(group_cols).agg(aggs).sort(group_cols),
        "summary_group_mathtable": cells.group_by(["ab_group", "mathtable"]).agg(aggs).sort(["ab_group", "mathtable"]),
        "summary_group_bucket": cells.group_by(["ab_group", "bet_bucket"]).agg(aggs).sort(["ab_group", "bet_bucket"]),
        "daily_group": cells.group_by(["bet_date", "ab_group"]).agg(aggs).sort(["bet_date", "ab_group"]),
        "daily_group_mathtable": cells.group_by(["bet_date", "ab_group", "mathtable"])
        .agg(aggs)
        .sort(["bet_date", "ab_group", "mathtable"]),
        "summary_group_bucket_postswitch": cells.filter(pl.col("bet_date") >= POST_SWITCH_START)
        .group_by(["ab_group", "bet_bucket"])
        .agg(aggs)
        .join(
            bootstrap_mean_ci(cells.filter(pl.col("bet_date") >= POST_SWITCH_START), ["ab_group", "bet_bucket"]),
            on=["ab_group", "bet_bucket"],
            how="left",
        )
        .join(
            bootstrap_mean_ci(
                cells.filter(pl.col("bet_date") >= POST_SWITCH_START),
                ["ab_group", "bet_bucket"],
                col="total_spin",
                prefix="avg_spin",
            ),
            on=["ab_group", "bet_bucket"],
            how="left",
        )
        .sort(["ab_group", "bet_bucket"]),
        "summary_group_mathtable_bucket_postswitch": cells.filter(pl.col("bet_date") >= POST_SWITCH_START)
        .group_by(group_cols)
        .agg(aggs)
        .sort(group_cols),
        "cells_user_day": cells,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name == "cells_user_day":
            continue
        table.write_csv(OUTPUT_DIR / f"{name}.csv")
        logger.info("wrote %s (%d rows)", OUTPUT_DIR / f"{name}.csv", len(table))
    return tables


def analyze_ai_mathtable() -> pl.DataFrame:
    """Per-mathtable stats inside the AI group, full-day multi-attribution.

    Each user-day with >= AI_MIN_DAILY_BETS BASE bets counts fully toward
    EVERY math table the user had a session on that day (tables sum to more
    than the group total by design). Retention is per (mathtable, day) cohort
    with the same any-SS03-bet-on-D+N definition as the main analysis.
    """
    files = sorted(USER_DAY_MT_DIR.glob("bet_date=*.parquet"))
    if not files:
        raise SystemExit(f"no extracted data under {USER_DAY_MT_DIR}; run with --extract first")
    mt = pl.concat([pl.read_parquet(f) for f in files])
    presence = load_user_day().select("user_id", "bet_date").unique()

    cells = mt.filter(
        (pl.col("ab_group") == "AI")
        & pl.col("bet_date").is_between(WINDOW_START, WINDOW_END)
        & ~pl.col("bet_date").is_in(list(EXCLUDED_COHORT_DATES))
        & (pl.col("total_spin") >= AI_MIN_DAILY_BETS)
    )
    for n in (1, 3):
        target = cells.select("user_id", (pl.col("bet_date") + timedelta(days=n)).alias("target_date"))
        hit = target.join(
            presence.rename({"bet_date": "target_date"}),
            on=["user_id", "target_date"],
            how="left",
            coalesce=False,
        ).select(pl.col("user_id_right").is_not_null().alias(f"retained_d{n}"))
        cells = cells.with_columns(hit[f"retained_d{n}"]).with_columns(
            pl.when(pl.col("bet_date") + timedelta(days=n) > PRESENCE_END)
            .then(None)
            .otherwise(pl.col(f"retained_d{n}"))
            .alias(f"retained_d{n}")
        )

    aggs = [
        pl.len().alias("user_days"),
        pl.col("user_id").n_unique().alias("n_users"),
        pl.col("total_spin").sum().alias("total_spin"),
        pl.col("total_spin").mean().alias("avg_spin_per_user_day"),
        pl.col("total_bet_cny").sum().alias("total_bet_cny"),
        pl.col("total_bet_cny").mean().alias("avg_bet_cny_per_user_day"),
        pl.col("retained_d1").mean().alias("retention_d1"),
        pl.col("retained_d1").count().alias("d1_cohort"),
        pl.col("retained_d3").mean().alias("retention_d3"),
        pl.col("retained_d3").count().alias("d3_cohort"),
    ]
    summary = cells.group_by("mathtable").agg(aggs).sort("user_days", descending=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.write_csv(OUTPUT_DIR / "summary_ai_mathtable.csv")
    logger.info("wrote %s (%d rows)", OUTPUT_DIR / "summary_ai_mathtable.csv", len(summary))

    # Same stats per (mathtable, daily total-bet bucket) — the bucket is the
    # DAY's total spend, so a multi-attributed user-day lands in the same
    # bucket for every table it touched.
    bucketed = cells.with_columns(bucket_expr())
    by_bucket = (
        bucketed.group_by(["mathtable", "bet_bucket"])
        .agg(aggs)
        .join(bootstrap_mean_ci(bucketed, ["mathtable", "bet_bucket"]), on=["mathtable", "bet_bucket"], how="left")
        .sort(["mathtable", "bet_bucket"])
    )
    by_bucket.write_csv(OUTPUT_DIR / "summary_ai_mathtable_bucket.csv")
    logger.info("wrote %s (%d rows)", OUTPUT_DIR / "summary_ai_mathtable_bucket.csv", len(by_bucket))
    dropped = mt.filter(
        (pl.col("ab_group") == "AI")
        & pl.col("bet_date").is_between(WINDOW_START, WINDOW_END)
        & ~pl.col("bet_date").is_in(list(EXCLUDED_COHORT_DATES))
        & (pl.col("total_spin") < AI_MIN_DAILY_BETS)
    )
    logger.info(
        "AI multi-attribution rows kept: %d (dropped %d below %d daily BASE bets)",
        len(cells),
        len(dropped),
        AI_MIN_DAILY_BETS,
    )
    return summary


def print_distribution() -> None:
    """Print the daily total-bet distribution used to choose BET_BUCKET_EDGES."""
    ud = load_user_day().filter(
        pl.col("bet_date").is_between(WINDOW_START, WINDOW_END)
        & pl.col("ab_group").is_in(list(AB_GROUPS))
        & (pl.col("total_spin") > 0)
    )
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print("user-day total_bet_cny quantiles:")
    for q in qs:
        print(f"  p{int(q * 100):02d}: {ud['total_bet_cny'].quantile(q):.2f}")
    mean = cast(float, ud["total_bet_cny"].mean())
    mx = cast(float, ud["total_bet_cny"].max())
    print(f"  mean: {mean:.1f}  max: {mx:.0f}  user-days: {len(ud)}")
    print(f"  zero-BASE user-days dropped: {len(load_user_day().filter(pl.col('total_spin') == 0))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="run the Athena extraction and cache locally")
    parser.add_argument("--distribution", action="store_true", help="print the daily-bet distribution")
    parser.add_argument("--analyze", action="store_true", help="compute metric tables and write CSVs")
    args = parser.parse_args()
    if args.extract:
        extract()
    if args.distribution:
        print_distribution()
    if args.analyze:
        analyze()
        analyze_ai_mathtable()
    if not (args.extract or args.distribution or args.analyze):
        parser.print_help()


if __name__ == "__main__":
    main()
