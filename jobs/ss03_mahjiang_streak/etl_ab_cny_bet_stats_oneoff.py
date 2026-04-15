"""
One-off Redshift export: SS03 CNY completed bets for two AB groups, aggregated by
activity day / AB / user / mathtable / bet_type / incentivized (spin_count, RTP, avg bet).

Not wired into ``run_scheduled_etl_jobs.py``. Run manually when needed:

    poetry run python jobs/ss03_mahjiang_streak/etl_ab_cny_bet_stats_oneoff.py
    poetry run python jobs/ss03_mahjiang_streak/etl_ab_cny_bet_stats_oneoff.py --use-cache
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import dedent

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, RedshiftBackend

AB_GROUP_A = "4a04df21-c749-4808-8e55-3a0b74c084d2"
AB_GROUP_B = "4f1a46ca-7baa-4452-9a40-ef21d9b33b57"

OUTPUT_DIR = Path(LOCAL_ROOT).parent / "data_ss03"
DEFAULT_OUTPUT_PARQUET = OUTPUT_DIR / "ab_cny_bet_rtp_by_user_mathtable.parquet"


def query_sql() -> str:
    # WHERE uses t.partition_ab[0] (not the ab_group_id alias — invalid in the same SELECT level).
    return dedent(
        f"""
        WITH user_bets AS (
            SELECT
                t.user_id,
                t.math_table_id AS mathtable,
                t.bet_amount,
                t.actual_payout AS payout,
                t.bet_type,
                t.currency_type,
                CAST(
                    DATE_TRUNC('day', DATEADD(HOUR, 0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at)))
                    AS DATE
                ) AS activity_date,
                t.partition_ab[0] AS ab_group_id,
                t.incentivized
            FROM
                public.fct_bet_orders AS t
            WHERE
                t.game_id = 'SS03'
                AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) >= '2026-03-20'
                AND t.currency_type = 'CNY'
                AND t.status = 'COMPLETED'
                AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
                AND t.partition_ab[0] IN ('{AB_GROUP_A}', '{AB_GROUP_B}')
        )
        SELECT
            activity_date,
            ab_group_id,
            user_id,
            mathtable,
            bet_type,
            COUNT(*) AS spin_count,
            SUM(bet_amount) AS total_bet_amount,
            SUM(payout) AS total_payout,
            SUM(payout) / NULLIF(SUM(bet_amount), 0) AS rtp,
            SUM(bet_amount) / COUNT(DISTINCT user_id) AS avg_bet,
            incentivized
        FROM user_bets
        GROUP BY activity_date, ab_group_id, mathtable, bet_type, user_id, incentivized
        ORDER BY activity_date, ab_group_id, user_id, mathtable, bet_type, incentivized
        """
    )


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    parser = argparse.ArgumentParser(
        description="One-off SS03 AB CNY bet stats export (not scheduled).",
    )
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP for Redshift SSH tunnel (default: {DEFAULT_BASTION_IP})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help=f"Local parquet path (default: {DEFAULT_OUTPUT_PARQUET})",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="If the output parquet exists, load it instead of re-querying Redshift.",
    )
    args = parser.parse_args()

    loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="slot-machine",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=args.bastion_ip,
        )
    )
    try:
        df = loader.query_to_df(
            query=query_sql(),
            local_cache=str(args.output),
            reload=not args.use_cache,
        )
        print(f"Rows: {len(df)}, columns: {list(df.columns)}")
        print(df.head())
    finally:
        loader.close()


if __name__ == "__main__":
    main()
