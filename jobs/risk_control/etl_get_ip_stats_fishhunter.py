"""
One-shot ETL: aggregate ``public.bullet`` (Fish Hunter / transform-agfish-game) by
``(ip, currency_type)`` and write Parquet to S3.

**Final dataset columns**

Columns produced by the SQL ``SELECT`` (in order), then one column added in Python
before upload:

``ip``
    Client IP (non-null, non-blank rows only).

``currency_type``
    Wallet / table currency for the bullet rows aggregated in this row.

``num_user_id``
    Count of distinct ``user_id`` with activity on this IP and currency.

``max_user_concurrency``
    Over 10-second buckets of each user's first-seen time on this IP+currency
    (``round(EXTRACT(EPOCH FROM MIN(created_at)) / 10) * 10`` per user), the maximum
    number of distinct users sharing the same bucket—i.e. peak concurrent user starts
    in that coarse time grid.

``rtp``
    ``SUM(payout) / SUM(bet)`` for all qualifying bullets on this IP+currency.

``total_profit``
    ``SUM(profit)`` for those bullets.

``min_bet`` / ``max_bet``
    Minimum and maximum single ``bet`` value in the aggregate.

``avg_user_id_duration_hours``
    Mean, across users on this IP+currency, of each user's session span in hours:
    ``(max(created_at) - min(created_at))`` per ``(ip, user_id, currency_type)``.

``std_user_id_duration_hours``
    Population-style spread (``STDDEV``) of those per-user spans.

``num_bets``
    Row count of bullets (bets) in the aggregate.

``num_unique_bets``
    Count of distinct ``bet`` values.

``num_unique_fish_value``
    Count of distinct ``fish_value`` values.

``mean_of_user_mean_bet_interval_sec``
    For each user, mean gap in seconds between consecutive bets (same IP+currency),
    using only gaps with ``0 < gap < MAX_BET_INTERVAL_SEC``; then the mean of those
    per-user means across users on this IP+currency.

``std_of_user_mean_bet_interval_sec``
    Sample standard deviation of the per-user mean gaps across users; ``0`` if only
    one user contributed interval stats.

``mean_of_user_std_bet_interval_sec``
    Mean of each user's sample std of qualifying gaps (per-user std is ``0`` when
    that user has only one qualifying gap).

``std_of_user_std_bet_interval_sec``
    Sample standard deviation of those per-user stds; ``0`` if only one user
    contributed interval stats.

``mean_of_user_mean_bet_interval_sec`` … ``std_of_user_std_bet_interval_sec`` are
NULL for an IP+currency when no user has at least two bets with a qualifying gap.

``_processed_at`` *(Python only)*
    Timestamp when the dataframe was finalized before ``to_parquet``.
"""

import argparse
import logging
import os
from datetime import datetime
from textwrap import dedent

import awswrangler as wr
import pandas as pd

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, RedshiftBackend

logger = logging.getLogger(__name__)

S3_OUTPUT_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/ip_stats"


MAX_BET_INTERVAL_SEC = 1800


def build_query() -> str:
    """Return the Redshift SQL for the snapshot query; output columns are documented in the module docstring."""
    max_gap = MAX_BET_INTERVAL_SEC
    return dedent(
        f"""
        WITH user_lifespans AS (
            SELECT
                ip,
                user_id,
                currency_type,
                round(EXTRACT(EPOCH FROM MIN(created_at)) / 10) * 10 AS start_time_window,
                EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 3600.0
                    AS individual_duration_hours
            FROM public.bullet
            WHERE op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
              AND ip IS NOT NULL
              AND TRIM(ip) <> ''
            GROUP BY
                ip,
                user_id,
                currency_type
        ),
        ip_concurrency AS (
            SELECT
                ip,
                currency_type,
                start_time_window,
                COUNT(DISTINCT user_id) AS num_concurrent_user_id
            FROM user_lifespans
            GROUP BY
                ip,
                currency_type,
                start_time_window
        ),
        ip_max_concurrency AS (
            SELECT
                ip,
                currency_type,
                MAX(num_concurrent_user_id) AS max_user_concurrency
            FROM ip_concurrency
            GROUP BY
                ip,
                currency_type
        ),
        ip_user_stats AS (
            SELECT
                ip,
                currency_type,
                AVG(individual_duration_hours) AS avg_user_id_duration_hours,
                STDDEV(individual_duration_hours) AS std_user_id_duration_hours
            FROM user_lifespans
            GROUP BY
                ip,
                currency_type
        ),
        bet_ordered AS (
            SELECT
                b.ip,
                b.user_id,
                b.currency_type,
                b.created_at,
                LAG(b.created_at) OVER (
                    PARTITION BY b.ip, b.user_id, b.currency_type
                    ORDER BY b.created_at
                ) AS prev_created_at
            FROM public.bullet AS b
            WHERE b.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
              AND b.ip IS NOT NULL
              AND TRIM(b.ip) <> ''
        ),
        bet_intervals AS (
            SELECT
                o.ip,
                o.user_id,
                o.currency_type,
                EXTRACT(EPOCH FROM (o.created_at - o.prev_created_at)) AS gap_sec
            FROM bet_ordered AS o
            WHERE o.prev_created_at IS NOT NULL
              AND EXTRACT(EPOCH FROM (o.created_at - o.prev_created_at)) > 0
              AND EXTRACT(EPOCH FROM (o.created_at - o.prev_created_at)) < {max_gap}
        ),
        user_bet_interval_stats AS (
            SELECT
                i.ip,
                i.user_id,
                i.currency_type,
                AVG(i.gap_sec) AS user_mean_bet_interval_sec,
                COALESCE(STDDEV_SAMP(i.gap_sec), 0) AS user_std_bet_interval_sec
            FROM bet_intervals AS i
            GROUP BY
                i.ip,
                i.user_id,
                i.currency_type
        ),
        ip_bet_interval_stats AS (
            SELECT
                u.ip,
                u.currency_type,
                AVG(u.user_mean_bet_interval_sec) AS mean_of_user_mean_bet_interval_sec,
                CASE
                    WHEN COUNT(*) <= 1 THEN 0
                    ELSE STDDEV_SAMP(u.user_mean_bet_interval_sec)
                END AS std_of_user_mean_bet_interval_sec,
                AVG(u.user_std_bet_interval_sec) AS mean_of_user_std_bet_interval_sec,
                CASE
                    WHEN COUNT(*) <= 1 THEN 0
                    ELSE STDDEV_SAMP(u.user_std_bet_interval_sec)
                END AS std_of_user_std_bet_interval_sec
            FROM user_bet_interval_stats AS u
            GROUP BY
                u.ip,
                u.currency_type
        )
        SELECT
            t.ip,
            t.currency_type,
            COUNT(DISTINCT t.user_id) AS num_user_id,
            t2.max_user_concurrency,
            SUM(t.payout) / NULLIF(SUM(t.bet), 0) AS rtp,
            SUM(t.profit) AS total_profit,
            MIN(t.bet) AS min_bet,
            MAX(t.bet) AS max_bet,
            stats.avg_user_id_duration_hours,
            stats.std_user_id_duration_hours,
            COUNT(t.bet) AS num_bets,
            COUNT(DISTINCT t.bet) AS num_unique_bets,
            COUNT(DISTINCT t.fish_value) AS num_unique_fish_value,
            MAX(intervals.mean_of_user_mean_bet_interval_sec) AS mean_of_user_mean_bet_interval_sec,
            MAX(intervals.std_of_user_mean_bet_interval_sec) AS std_of_user_mean_bet_interval_sec,
            MAX(intervals.mean_of_user_std_bet_interval_sec) AS mean_of_user_std_bet_interval_sec,
            MAX(intervals.std_of_user_std_bet_interval_sec) AS std_of_user_std_bet_interval_sec
        FROM public.bullet AS t
        INNER JOIN ip_user_stats AS stats
            ON t.ip = stats.ip
            AND t.currency_type = stats.currency_type
        LEFT JOIN ip_max_concurrency AS t2
            ON t.ip = t2.ip
            AND t.currency_type = t2.currency_type
        LEFT JOIN ip_bet_interval_stats AS intervals
            ON t.ip = intervals.ip
            AND t.currency_type = intervals.currency_type
        WHERE t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
          AND t.ip IS NOT NULL
          AND TRIM(t.ip) <> ''
        GROUP BY
            t.ip,
            t.currency_type,
            t2.max_user_concurrency,
            stats.avg_user_id_duration_hours,
            stats.std_user_id_duration_hours
        ORDER BY
            num_user_id DESC;
        """
    ).strip()


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    parser = argparse.ArgumentParser(
        description="One-shot IP stats from Redshift bullet → Parquet on S3 (by currency_type)."
    )
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
    )
    parser.add_argument(
        "--s3-path",
        type=str,
        default=None,
        help=f"Override full S3 URI for the Parquet file (default: {S3_OUTPUT_PREFIX}/ip_stats_fishhunter.parquet)",
    )
    args = parser.parse_args()

    sql = build_query()

    loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="transform-agfish-game",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=args.bastion_ip,
        )
    )
    try:
        df = loader.query_to_df(query=sql)
    finally:
        loader.close()

    if df is None or df.empty:
        logger.warning("Query returned no rows; skipping S3 write.")
        return

    df["_processed_at"] = datetime.now()

    skip_numeric = {"ip", "currency_type", "_processed_at"}
    for col in df.columns:
        if col in skip_numeric or df[col].dtype != "object":
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception:
            pass

    out_uri = args.s3_path or f"{S3_OUTPUT_PREFIX}/ip_stats_fishhunter.parquet"
    wr.s3.to_parquet(df=df, path=out_uri, index=False)
    logger.info("Wrote %s rows to %s", len(df), out_uri)


if __name__ == "__main__":
    main()
