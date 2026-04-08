"""
One-shot ETL: ``public.fct_user_bet_aggregate_daily`` (Fish Hunter / transform-agfish-game)
→ single CSV on S3 under ``{DEFAULT_ETL_OUTPUT}/jobs/fishunter/risk_control/``.

Output object name matches the table: ``fct_user_bet_aggregate_daily.csv``.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from textwrap import dedent

import awswrangler as wr

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

TABLE_NAME = "fct_user_bet_aggregate_daily"
S3_OUTPUT_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_fish_hunter/risk_control"
DEFAULT_CSV_NAME = f"{TABLE_NAME}.csv"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_query(min_date: str) -> str:
    if not _DATE_RE.match(min_date):
        raise ValueError(f"min_date must be YYYY-MM-DD, got {min_date!r}")
    return dedent(
        f"""
        SELECT *
        FROM public.{TABLE_NAME}
        WHERE date >= '{min_date}'
        ORDER BY date DESC
        """
    ).strip()


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    parser = argparse.ArgumentParser(description=f"Export {TABLE_NAME} from Redshift to one CSV on S3.")
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default="2026-01-01",
        help="Lower bound for date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--s3-path",
        type=str,
        default=None,
        help=(f"Full S3 URI for the CSV object " f"(default: {S3_OUTPUT_PREFIX}/{DEFAULT_CSV_NAME})"),
    )
    args = parser.parse_args()

    sql = build_query(args.min_date)

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

    out_uri = args.s3_path or f"{S3_OUTPUT_PREFIX}/{DEFAULT_CSV_NAME}"
    wr.s3.to_csv(df=df, path=out_uri, index=False, dataset=False)
    logger.info("Wrote %s rows to %s", len(df), out_uri)


if __name__ == "__main__":
    main()
