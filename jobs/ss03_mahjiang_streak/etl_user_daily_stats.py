"""ETL: SS03 per-user daily betting stats from the enriched per-bet data.

ETL job: "SS03 user daily stats" — per (user_id, activity_date, ai_group):
    total_bet     = sum(bet_amount)
    total_spin    = number of bets (one enriched row == one spin)
    total_payout  = sum(payout)
    user_rtp      = total_payout / total_bet   (null when total_bet == 0)

Reads the per-bet enriched datasets produced by
``jobs/ss03_mahjiang_streak/etl_feature_engineer.py`` for the Default / AB_TEST_A /
AB_TEST_B cohorts (S3 parquet, no Redshift) and aggregates to user-day grain across
all math tables within each cohort. Output is written for the dashboard to:

    s3://bituslabs-team-ai/ds-data-ss03_majiang_streak/user_daily_stats/user_daily_stats.parquet

Run:
    poetry run python jobs/ss03_mahjiang_streak/etl_user_daily_stats.py
"""

import argparse
import logging
import os

import awswrangler as wr
import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging

logger = logging.getLogger(__name__)

# Cohort -> enriched dataset prefix. Each prefix already contains only its own ai_group
# (the feature ETL writes one cohort per prefix), so reading per-prefix keeps the cohorts
# cleanly separated; we still group by ai_group so the output carries the label.
ENRICHED_SOURCES = {
    "Default": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_feature_engineer/features_enriched/",
    "AB_TEST_A": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_feature_engineer_ab_test_a/features_enriched/",
    "AB_TEST_B": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_feature_engineer_ab_test_b/features_enriched/",
}

OUTPUT_PATH = f"s3://{S3_BUCKET}/ds-data-ss03_majiang_streak/user_daily_stats/user_daily_stats.parquet"

# Only the columns the aggregation needs; projecting keeps the (large) enriched read cheap.
READ_COLS = ["user_id", "ai_group", "activity_date", "bet_amount", "payout", "spin_id"]

GROUP_KEYS = ["user_id", "ai_group", "activity_date"]

# Enriched datasets are tens of millions of rows; read in chunks and aggregate each chunk
# so peak memory stays bounded. total_bet/total_payout/total_spin are all additive, so
# per-chunk partials re-combine exactly with a second sum (see _combine).
_CHUNK_ROWS = 2_000_000


def _aggregate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(GROUP_KEYS, observed=True)
        .agg(total_bet=("bet_amount", "sum"), total_payout=("payout", "sum"), total_spin=("spin_id", "size"))
        .reset_index()
    )


def _combine(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Re-aggregate per-chunk partials into one row per group (sum of sums / counts)."""
    combined = pd.concat(partials, ignore_index=True)
    return (
        combined.groupby(GROUP_KEYS, observed=True)
        .agg(total_bet=("total_bet", "sum"), total_payout=("total_payout", "sum"), total_spin=("total_spin", "sum"))
        .reset_index()
    )


def compute_user_daily_stats() -> pd.DataFrame:
    """Aggregate the enriched per-bet rows to per (user, day, cohort) betting stats."""
    partials: list[pd.DataFrame] = []
    for cohort, prefix in ENRICHED_SOURCES.items():
        logger.info("reading enriched data for cohort=%s from %s", cohort, prefix)
        n_rows = 0
        for chunk in wr.s3.read_parquet(prefix, dataset=True, columns=READ_COLS, chunked=_CHUNK_ROWS):
            n_rows += len(chunk)
            partials.append(_aggregate_chunk(chunk))
        logger.info("cohort=%s: read %d enriched rows", cohort, n_rows)

    stats = _combine(partials)
    # RTP per user-day; guard divide-by-zero (a day with only 0-amount FREE bets) -> null.
    stats["user_rtp"] = stats["total_payout"] / stats["total_bet"].where(stats["total_bet"] > 0)
    stats = stats.sort_values(["activity_date", "ai_group", "user_id"]).reset_index(drop=True)
    logger.info(
        "computed user daily stats: %d rows; cohorts=%s; date range %s..%s",
        len(stats),
        sorted(stats["ai_group"].unique().tolist()),
        stats["activity_date"].min(),
        stats["activity_date"].max(),
    )
    return stats


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )
    parser = argparse.ArgumentParser(description="SS03 per-user daily betting stats from enriched data.")
    parser.add_argument("--output-path", default=OUTPUT_PATH, help=f"S3 output parquet path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    stats = compute_user_daily_stats()
    wr.s3.to_parquet(df=stats, path=args.output_path, index=False)
    logger.info("wrote %d rows to %s", len(stats), args.output_path)
    print(f"User daily stats: {args.output_path} ({len(stats)} rows)")


if __name__ == "__main__":
    main()
