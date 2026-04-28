"""
Incremental ETL: platform ops daily report for FM01 (fish hunter), SS01, SS03.
This script is used to generate the weekly report for all games, adpated from stehpanie's code.
Incorporated into the dashboard by Xin.

Reads from Redshift `platform.public.fct_platform_ops_daily_report`, writes Parquet to S3
under ``{DEFAULT_ETL_OUTPUT}/jobs/output_weekly_report_all_games/ops_daily_report``.

The game stats dashboard loads this dataset per ``weekly_report`` section in dashboard_config-*.yaml.

Usage:
  poetry run python jobs/operation_daily_report/etl_weekly_report_all_games.py [--bastion-ip IP] [--overwrite]
"""

from __future__ import annotations

import argparse
import os
from textwrap import dedent

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    ETL_CURRENCY_CODES,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, ETLScheduler, RedshiftBackend

GAME_IDS = ("FM01", "SS01", "SS03")


def generate_query(start_date: str) -> str:
    games_sql = ", ".join(f"'{g}'" for g in GAME_IDS)
    return dedent(
        f"""
        SELECT *
        FROM public.fct_platform_ops_daily_report
        WHERE bj_date_key >= DATE '{start_date}'
          AND currency_type IN {ETL_CURRENCY_CODES}
          AND game_id IN ({games_sql})
        ORDER BY bj_date_key, game_id
        """
    )


if __name__ == "__main__":
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    parser = argparse.ArgumentParser(description="ETL weekly ops report (all games → S3)")
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing S3/local output (full reload from default start date)",
    )
    args = parser.parse_args()

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="platform",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=args.bastion_ip,
        )
    )

    scheduler = ETLScheduler(
        redshift_loader,
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_weekly_report_all_games",
        lookback_days=14,
        overwrite=args.overwrite,
    )

    scheduler.run_incremental_job(
        job_name="ops_daily_report",
        query_func=generate_query,
        key_cols=["bj_date_key", "game_id"],
        date_col="bj_date_key",
        partition_level="none",
        lookback=14,
    )

    redshift_loader.close()
