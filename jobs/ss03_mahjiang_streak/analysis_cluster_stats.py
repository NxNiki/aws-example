"""SS03 mahjong streak: single-cluster ``cluster_stats`` over base-game bets.

Pipeline
--------
1. **ETL** (run once; cached on S3): query Redshift for SS03 per-bet rows. The query produces a
   ``BASE``-only result with one extra column, ``fg_rounds`` — for each ``BASE`` bet, the count of
   immediately following ``FREE`` bets per user (until the next ``BASE`` bet). Output is written to
   ``s3://<S3_BUCKET>/ds-data-ss03_majiang_streak/cluster_stats/raw_features_base_game.parquet``.
2. **Stats**: read the parquet with the 5 columns ``[user_id, created_at, balance_after_bet,
   bet_amount, fg_rounds]`` in that order so :func:`bituslabs_ds.ml.ClusterAnalysisPipeline.get_cluster_stats`
   's positional rename to ``[loginname, billtime, basepoint, account, fg_rounds]`` lines up.
   All rows are treated as a single cluster (``cluster_0``); stats JSON is written next to the
   parquet at ``cluster_stats.json``.
"""

from __future__ import annotations

import argparse
import logging
import os
from textwrap import dedent
from typing import cast

import awswrangler as wr
import pandas as pd

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    ETL_CURRENCY_CODES,
    ETL_EXCLUDED_OP_CODES,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    S3_BUCKET,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, RedshiftBackend
from bituslabs_ds.ml import compute_cluster_stats
from bituslabs_ds.s3_utils import read_single_file, write_json_to_s3
from bituslabs_ds.utils import convert_numpy_types

logger = logging.getLogger(__name__)

GAME_ID = "SS03"

S3_OUTPUT_PREFIX = f"s3://{S3_BUCKET}/ds-data-ss03_majiang_streak/cluster_stats"
S3_RAW_FEATURES_PATH = f"{S3_OUTPUT_PREFIX}/raw_features_base_game.parquet"
S3_CLUSTER_STATS_PATH = f"{S3_OUTPUT_PREFIX}/cluster_stats.json"

# Order matters: positional rename in get_cluster_stats() maps these to
# [loginname, billtime, basepoint, account, fg_rounds].
CLUSTER_STATS_COLUMNS = ["user_id", "created_at", "balance_after_bet", "bet_amount", "fg_rounds"]
EXPECTED_COLUMNS = ["loginname", "billtime", "basepoint", "account", "fg_rounds"]


def _sql_raw_base_with_fg_rounds() -> str:
    """SS03 per-bet rows for ``BASE`` only with ``fg_rounds`` (FREE bets after each BASE before next BASE).

    Window logic: ``base_group`` is a running count of ``BASE`` rows per user (ordered by
    ``created_at, spin_id``). All ``FREE`` rows that fall between two consecutive ``BASE`` rows
    share the preceding BASE's ``base_group`` value, so summing ``bet_type='FREE'`` over
    ``PARTITION BY user_id, base_group`` gives the FREE-bet count attributable to that BASE.
    """
    return dedent(
        f"""
        WITH bets AS (
            SELECT
                t.user_id,
                t.created_at,
                t.spin_id,
                t.bet_amount,
                t.balance_after_bet,
                t.bet_type,
                SUM(CASE WHEN t.bet_type = 'BASE' THEN 1 ELSE 0 END) OVER (
                    PARTITION BY t.user_id
                    ORDER BY t.created_at, t.spin_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS base_group
            FROM public.fct_bet_orders t
            WHERE t.game_id = '{GAME_ID}'
                AND t.currency_type IN {ETL_CURRENCY_CODES}
                AND t.status = 'COMPLETED'
                AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
        )
        SELECT
            user_id,
            created_at,
            spin_id,
            bet_amount,
            balance_after_bet,
            bet_type,
            SUM(CASE WHEN bet_type = 'FREE' THEN 1 ELSE 0 END) OVER (
                PARTITION BY user_id, base_group
            ) AS fg_rounds
        FROM bets
        WHERE bet_type = 'BASE'
        """
    ).strip()


def run_etl() -> pd.DataFrame:
    """Run the Redshift query via SSH bastion and return the resulting DataFrame."""
    loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="slot-machine",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=DEFAULT_BASTION_IP,
        )
    )
    try:
        df = cast(pd.DataFrame, loader.query_to_df(_sql_raw_base_with_fg_rounds()))
    finally:
        loader.close()

    for col in ("bet_amount", "balance_after_bet", "fg_rounds"):
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_or_run_etl(reload: bool) -> pd.DataFrame:
    """Read cached parquet from S3, or run the ETL and persist it when missing/``reload``."""
    if not reload and wr.s3.does_object_exist(S3_RAW_FEATURES_PATH):
        logger.info("Reading existing raw features from %s", S3_RAW_FEATURES_PATH)
        return cast(
            pd.DataFrame,
            read_single_file(S3_RAW_FEATURES_PATH, columns=CLUSTER_STATS_COLUMNS),
        )

    logger.info("Running Redshift ETL for SS03 BASE-game raw features...")
    df = run_etl()
    if df is None or df.empty:
        raise RuntimeError("Redshift query returned no rows; cannot compute cluster stats.")
    wr.s3.to_parquet(df=df, path=S3_RAW_FEATURES_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(df), S3_RAW_FEATURES_PATH)
    return cast(pd.DataFrame, df[CLUSTER_STATS_COLUMNS].copy())


def compute_single_cluster_stats(df: pd.DataFrame) -> dict:
    """All rows as one cluster: rename to the schema expected by :func:`compute_cluster_stats` and delegate."""
    data = df[CLUSTER_STATS_COLUMNS].copy()
    data.columns = EXPECTED_COLUMNS
    return {"cluster_0": compute_cluster_stats(cast(pd.DataFrame, data))}


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    parser = argparse.ArgumentParser(description="SS03 single-cluster cluster_stats from BASE-game bets.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Re-run the Redshift ETL even if the S3 raw features parquet already exists.",
    )
    args = parser.parse_args()

    df = load_or_run_etl(reload=args.reload)

    stats = compute_single_cluster_stats(df)
    write_json_to_s3(convert_numpy_types(stats), S3_CLUSTER_STATS_PATH)

    print(f"Raw features: {S3_RAW_FEATURES_PATH}")
    print(f"Cluster stats: {S3_CLUSTER_STATS_PATH}")


if __name__ == "__main__":
    main()
