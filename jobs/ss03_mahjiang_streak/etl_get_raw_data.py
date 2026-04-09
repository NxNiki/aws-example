"""
This script is used to get the raw data from the Redshift database. WIP
"""

import argparse
import os
from textwrap import dedent

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
from bituslabs_ds.etl import DataLoader, ETLScheduler, RedshiftBackend

# Day boundary: 6 AM Shanghai time (same as fish_hunter)
DATE_START_HOUR = 6
GAME_ID = "SS03"


def generate_query(start_date: str):
    query = dedent(
        f"""
        SELECT
            cast(DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_date,
            cast(DATE_TRUNC('week', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_week,
            cast(DATE_TRUNC('month', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_month,
            t.user_id,
            t.bet_type,
            -- t."snapshot",
            t.created_at,
            t.trigger_type,
            t.retrigger,
            t.bet_amount,
            t.actual_payout,
            t.balance_after_bet,
            t.balance_after_payout,
            t.base_bet,
            t.factor,
            t.multiplier,
            t.credit 
        FROM public.fct_bet_orders t
        WHERE t.game_id = '{GAME_ID}'
            AND t.currency_type = 'CNY'
            AND t.status = 'COMPLETED'
            AND t.op_code not in ('B26','TST','TSB','TSO')
            AND t.created_at >= '{start_date}'
        """
    )

    return query


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(description="ETL Game Stats Daily by User Group")
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP address for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
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
            database="slot-machine",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=args.bastion_ip,
        )
    )

    # Initialize Scheduler with a default 3-day lookback
    scheduler = ETLScheduler(
        redshift_loader,
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_ss03_mahjiang_raw_data",
        lookback_days=3,
        overwrite=args.overwrite,
    )

    scheduler.run_incremental_job(
        job_name="raw_data",
        query_func=lambda start_date: generate_query(start_date),
        key_cols=["activity_date", "user_id"],
        date_col="activity_date",
        partition_level="month",
    )

    redshift_loader.close()
