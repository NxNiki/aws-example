"""
One-off Redshift export: SS03 script_id distribution across all math tables.

Counts bets/users per (math_table_id, bet_type, snapshot.script_id)
(uuid-format ids only) for completed SS03 bets between 2026-06-10 and
2026-07-28 (Asia/Shanghai activity dates), with first/last seen timestamps
per script. bet_type ('BASE'/'FREE') separates free-game spins from base
spins.

Not wired into ``run_scheduled_etl_jobs.py``. Run manually when needed:

    poetry run python jobs/etl/redshift/ss03_mahjiang_streak/etl_script_id_distribution_oneoff.py
    poetry run python jobs/etl/redshift/ss03_mahjiang_streak/etl_script_id_distribution_oneoff.py --use-cache
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

OUTPUT_DIR = Path(LOCAL_ROOT).parent / "data_ss03"
DEFAULT_OUTPUT_PARQUET = OUTPUT_DIR / "script_id_distribution_all_math_tables.parquet"


def query_sql() -> str:
    return dedent(
        """
        WITH user_bets AS (
            SELECT
                t.user_id,
                t.created_at,
                t.math_table_id,
                t.bet_type,
                t."snapshot".script_id::varchar AS snapshot_script_id,
                (DATE_TRUNC('day', CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at)))::date AS activity_date
            FROM public.fct_bet_orders AS t
            WHERE
                t.game_id = 'SS03'
                AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) >= '2026-06-10'
                AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) < '2026-07-29'
                AND t.status = 'COMPLETED'
                AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
        )

        SELECT
            t.math_table_id,
            t.bet_type,
            t.snapshot_script_id,
            COUNT(*) AS num_bets,
            COUNT(DISTINCT t.user_id) AS num_users,
            MIN(t.created_at) AS first_seen_at,
            MAX(t.created_at) AS last_seen_at
        FROM user_bets AS t
        WHERE
            t.snapshot_script_id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        GROUP BY t.math_table_id, t.bet_type, t.snapshot_script_id
        ORDER BY t.math_table_id, t.bet_type, t.snapshot_script_id
        """
    )


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    parser = argparse.ArgumentParser(
        description="One-off SS03 script_id distribution export (not scheduled).",
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
        print(df.head(20))
    finally:
        loader.close()


if __name__ == "__main__":
    main()
