"""ETL: SS03 per-user daily betting stats, tagged with cluster + AB-test group.

ETL job: "SS03 user daily stats" — per (user_id, session_start_date, cluster):
    user_num_bets       = number of bets (one enriched row == one spin)
    user_total_bet      = sum(bet_amount)
    user_total_payout   = sum(payout)
    user_avg_bet_amount = user_total_bet / user_num_bets
    user_rtp            = user_total_payout / user_total_bet   (null when bet == 0)

Plus two user-group dimensions for the dashboard (config user_group_cols):
    cluster  -- the kmeans segment of the bet's 50-bet bin (0/1/2), from the apply
                run's cluster_labels; bets not in a labelled complete bin -> "NA".
    ab_group -- the user's dominant AB-test cohort by bet count (Default /
                AB_TEST_A / AB_TEST_B), constant across that user's rows.

Rows are kept at per-user grain (NOT pre-aggregated to date x cluster) so the
dashboard's user-row pipeline can both aggregate to date x {cluster|ab_group}
(avg bet amount, rtp, totals) AND compute retention from the user_id sets.

Reads the Default / AB_TEST_A / AB_TEST_B enriched datasets produced by
etl_feature_engineer.py (S3 parquet, no Redshift) and the cluster_labels from the
apply run. Output (one parquet, dashboard "day" source):

    s3://bituslabs-team-ai/ds-data-ss03_majiang_streak/user_cluster_daily_stats/

Run:
    poetry run python jobs/ss03_mahjiang_streak/etl_user_daily_stats.py
"""

import argparse
import logging
import os

import awswrangler as wr
import numpy as np
import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging

logger = logging.getLogger(__name__)

# Cohort -> enriched dataset prefix (one ai_group per prefix).
ENRICHED_SOURCES = {
    "Default": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_feature_engineer/features_enriched/",
    "AB_TEST_A": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_feature_engineer_ab_test_a/features_enriched/",
    "AB_TEST_B": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_feature_engineer_ab_test_b/features_enriched/",
}

# Per-bin cluster labels from the apply run (uploaded next to the run folder).
CLUSTER_LABELS_PATH = (
    f"s3://{S3_BUCKET}/ss03_cluster_analysis/apply_pretrained_2026-06-29_binsize50/cluster_labels.parquet"
)
CLUSTER_LABEL_COLUMN = "kmeans-n_features_40-k_3"
NA_CLUSTER = "NA"

OUTPUT_PATH = f"s3://{S3_BUCKET}/ds-data-ss03_majiang_streak/user_cluster_daily_stats/"

BIN_SIZE = 50  # agg_group = (session_bet_index - 1) // BIN_SIZE, matching the grouped feature SQL.

# Keys joining a bet to its bin's cluster label (= cluster_labels merge_on).
JOIN_KEYS = ["user_id", "ai_group", "math_table_id", "session_start_date", "session_group", "agg_group"]
READ_COLS = [
    "user_id",
    "ai_group",
    "math_table_id",
    "session_start_date",
    "session_group",
    "session_bet_index",
    "bet_amount",
    "payout",
]
STAT_KEYS = ["user_id", "session_start_date", "cluster"]

_CHUNK_ROWS = 2_000_000


def _cast_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise join-key dtypes so the bet<->label merge lines up across sources."""
    for col in ("user_id", "session_group", "agg_group"):
        if col in df.columns:
            df[col] = df[col].astype("int64")
    for col in ("ai_group", "math_table_id", "session_start_date"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _load_cluster_labels() -> pd.DataFrame:
    """Per-bin (merge_on) -> integer cluster index from the apply run."""
    labels = wr.s3.read_parquet(CLUSTER_LABELS_PATH, columns=JOIN_KEYS + [CLUSTER_LABEL_COLUMN])
    labels = labels.dropna(subset=[CLUSTER_LABEL_COLUMN])
    labels["cluster_idx"] = labels[CLUSTER_LABEL_COLUMN].astype("int64")
    labels = _cast_keys(labels).drop(columns=[CLUSTER_LABEL_COLUMN])
    logger.info("loaded %d labelled bins from %s", len(labels), CLUSTER_LABELS_PATH)
    return labels


def _aggregate_chunk(chunk: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Tag each bet with its bin's cluster (NA if unlabelled), aggregate to (user, day, cluster)."""
    chunk = _cast_keys(chunk.copy())
    chunk["agg_group"] = (chunk["session_bet_index"].astype("int64") - 1) // BIN_SIZE
    merged = chunk.merge(labels, on=JOIN_KEYS, how="left")
    merged["cluster"] = np.where(
        merged["cluster_idx"].isna(), NA_CLUSTER, merged["cluster_idx"].astype("Int64").astype(str)
    )
    return (
        merged.groupby(STAT_KEYS, observed=True)
        .agg(
            user_total_bet=("bet_amount", "sum"), user_total_payout=("payout", "sum"), user_num_bets=("user_id", "size")
        )
        .reset_index()
    )


def compute_user_cluster_daily_stats() -> pd.DataFrame:
    labels = _load_cluster_labels()
    stat_partials: list[pd.DataFrame] = []
    # Per-(user, cohort) bet counts -> each user's dominant ab_group (most bets).
    cohort_partials: list[pd.DataFrame] = []

    for cohort, prefix in ENRICHED_SOURCES.items():
        logger.info("reading enriched data for cohort=%s from %s", cohort, prefix)
        n_rows = 0
        for chunk in wr.s3.read_parquet(prefix, dataset=True, columns=READ_COLS, chunked=_CHUNK_ROWS):
            n_rows += len(chunk)
            stat_partials.append(_aggregate_chunk(chunk, labels))
            cohort_partials.append(
                chunk.assign(ai_group=cohort)
                .groupby(["user_id", "ai_group"], observed=True)
                .size()
                .reset_index(name="n")
            )
        logger.info("cohort=%s: read %d enriched rows", cohort, n_rows)

    # Re-aggregate per (user, day, cluster); counts/sums are additive across chunks.
    stats = (
        pd.concat(stat_partials, ignore_index=True)
        .groupby(STAT_KEYS, observed=True)
        .agg(
            user_total_bet=("user_total_bet", "sum"),
            user_total_payout=("user_total_payout", "sum"),
            user_num_bets=("user_num_bets", "sum"),
        )
        .reset_index()
    )

    # Dominant ab_group per user = cohort with the most bets (ties -> first by cohort order).
    cohort_counts = (
        pd.concat(cohort_partials, ignore_index=True)
        .groupby(["user_id", "ai_group"], observed=True)["n"]
        .sum()
        .reset_index()
    )
    cohort_counts = cohort_counts.sort_values(["user_id", "n"], ascending=[True, False])
    ab_group = cohort_counts.drop_duplicates("user_id", keep="first")[["user_id", "ai_group"]].rename(
        columns={"ai_group": "ab_group"}
    )
    stats = stats.merge(ab_group, on="user_id", how="left")

    stats["user_avg_bet_amount"] = stats["user_total_bet"] / stats["user_num_bets"].where(stats["user_num_bets"] > 0)
    stats["user_rtp"] = stats["user_total_payout"] / stats["user_total_bet"].where(stats["user_total_bet"] > 0)
    # A few enriched bets have no session_start_date (no session assigned); they can't be a
    # daily stat, so coerce-and-drop rather than poison the date column.
    stats["session_start_date"] = pd.to_datetime(stats["session_start_date"], errors="coerce")
    before = len(stats)
    stats = stats.dropna(subset=["session_start_date"])
    if before != len(stats):
        logger.warning("dropped %d rows with no session_start_date", before - len(stats))
    stats = stats.sort_values(["session_start_date", "ab_group", "cluster", "user_id"]).reset_index(drop=True)

    logger.info(
        "user cluster daily stats: %d rows; clusters=%s; ab_groups=%s; dates %s..%s",
        len(stats),
        sorted(stats["cluster"].unique().tolist()),
        sorted(stats["ab_group"].dropna().unique().tolist()),
        stats["session_start_date"].min(),
        stats["session_start_date"].max(),
    )
    return stats


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )
    parser = argparse.ArgumentParser(description="SS03 per-user daily stats tagged with cluster + AB group.")
    parser.add_argument("--output-path", default=OUTPUT_PATH, help=f"S3 output dataset dir (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    stats = compute_user_cluster_daily_stats()
    wr.s3.to_parquet(df=stats, path=args.output_path, dataset=True, mode="overwrite", index=False)
    logger.info("wrote %d rows to %s", len(stats), args.output_path)
    print(f"User cluster daily stats: {args.output_path} ({len(stats)} rows)")


if __name__ == "__main__":
    main()
