"""
Check PID difference between Redshift and Athena.
Outputs: operation_daily_report/output/ with prefixes (fishhunter_, ss01_, ss03_).
SS01 and SS03 share the same Athena query (ag_share_data.slotorders SB28).
"""

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

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
from bituslabs_ds.etl import AthenaBackend, DataLoader, RedshiftBackend

logger = logging.getLogger(__name__)

# Stored in the results dict (in place of a PID list) when a single game's check
# raises. Per-game isolation: one game failing -- e.g. corrupt Athena data -- must
# not abort the others, and the failure has to stay visible in the final report.
# run_daily_report captures stdout while calling main(), so a printed traceback
# would be swallowed; a marker carried in the returned report reaches Slack.
_CHECK_FAILED_PREFIX = "CHECK FAILED:"


def _failed_marker(exc: Exception) -> list[str]:
    return [f"{_CHECK_FAILED_PREFIX} {exc!r}"]


pids_to_ignore = {
    "B26",
    "LV1",
    "E52",
    "ER5",
    "LQ6",
    "LF1",
    "DS2",
    "MD6",
    "JL1",
    "ER2",
    "F19",
    "KD9",
    "LL0",
    "LQ5",
    "LJ8",
    "KY5",
    "JL5",
    "E41",
    "FS8",
    "LT4",
    "LO1",
    "J31",
    "JW3",
    "D70",
    "MJ4",
    "E31",
    "ED9",
    "KJ1",
    "G63",
    "H48",
    "KI6",
    "N27",
    "GN7",
    "GP7",
    "FE0",
    "KG1",
    "LO7",
    "FR7",
    "N89",
    "KE6",
    "LK2",
    "FN8",
    "MJ0",
    "LP2",
    "LX9",
    "EH9",
    "KU5",
    "A05",
    "LB9",
    "HC3",
    "LN1",
    "KW9",
    "HX1",
    "GN9",
    "LO5",
    "LJ3",
    "D13",
    "LO3",
    "A51",
    "DV5",
    "MI1",
    "LK1",
    "JW4",
    "HS9",
    "LR0",
    "HC2",
    "GG2",
    "E59",
    "HS7",
    "LO0",
    "R09",
    "CJ9",
    "DW1",
    "J75",
    "JQ9",
    "JS6",
    "MU8",
    "E04",
    "JJ3",
    "EE3",
    "LN7",
}

# agfish.hunterorders has oversized/corrupt rows in older months (observed in
# 2024-10 and 2024-12) that trip Athena's hard 32MB per-row limit on ANY scan
# touching those files -- even projecting only productid -- aborting the whole
# daily report. Scan a rolling recent window instead of full history: 12 months
# stays clear of the known bad months, and since the window only moves forward it
# can never reach back into 2024 again. This also matches the "currently unopened
# PID" intent of the report (a productid active only years ago is stale anyway).
FISHHUNTER_ATHENA_LOOKBACK_DAYS = 365
_FISHHUNTER_ATHENA_CUTOFF = (datetime.now(timezone.utc) - timedelta(days=FISHHUNTER_ATHENA_LOOKBACK_DAYS)).strftime(
    "%Y-%m-%d %H:%M:%S"
)

# Shared Athena query for ss01 and ss03 (same slotorders table)
SLOT_ATHENA_QUERY = dedent(
    f"""
    SELECT distinct t.productid AS product_id FROM ag_share_data.slotorders AS t
    WHERE t.currency IN {ETL_CURRENCY_CODES} AND gametype = 'SB28' AND flag != -8.0
"""
)

QUERIES: dict[str, dict[str, Any]] = {
    "fish_hunter": {
        "redshift_query": dedent(
            f"""
            SELECT distinct t.op_code AS product_id FROM public.bullet AS t
            WHERE t.currency_type IN {ETL_CURRENCY_CODES} AND t.game_id = 'FM01'
            AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
        """
        ),
        "athena_query": dedent(
            f"""
            SELECT distinct t.productid AS product_id FROM agfish.hunterorders AS t
            WHERE t.currency IN {ETL_CURRENCY_CODES} AND t.gametype = 'HM3D' AND t.account != 0 AND t.fishcost != 0
            AND t.ordertype = 1 AND t.remark = t.gametype AND t.weaponid IS NULL
            AND t.billtime > TIMESTAMP '{_FISHHUNTER_ATHENA_CUTOFF}'
        """
        ),
        "athena_database": "agfish",
        "athena_output": f"s3://{S3_BUCKET}/ds-data-fish_hunter/product_id_pa",
        "redshift_database": "transform-agfish-game",
        "prefix": "fishhunter",
        "ctas_approach": True,
    },
    "ss01": {
        "redshift_query": dedent(
            f"""
            SELECT distinct t.op_code AS product_id FROM public.fct_bet_orders AS t
            WHERE t.currency_type IN {ETL_CURRENCY_CODES} AND t.status = 'COMPLETED' AND t.game_id = 'SS01'
            AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
        """
        ),
        "redshift_database": "slot-machine",
        "prefix": "ss01",
    },
    "ss03": {
        "redshift_query": dedent(
            f"""
            SELECT distinct t.op_code AS product_id FROM public.fct_bet_orders AS t
            WHERE t.currency_type IN {ETL_CURRENCY_CODES} AND t.status = 'COMPLETED' AND t.game_id = 'SS03'
            AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
        """
        ),
        "redshift_database": "slot-machine",
        "prefix": "ss03",
    },
}


def _output_dir() -> Path:
    return Path(__file__).resolve().parent / "output"


def run_pid_diff_check(
    athena_database,
    athena_output_location,
    athena_query,
    redshift_database,
    redshift_query,
    output_path_redshift,
    output_path_athena,
    pids_to_ignore,
    bastion_ip=DEFAULT_BASTION_IP,
    ctas_approach=False,
    reload=False,
):
    data_loader = DataLoader(
        backend=AthenaBackend(
            database=athena_database,
            output_location=athena_output_location,
            ctas_approach=ctas_approach,
        )
    )
    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database=redshift_database,
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=bastion_ip,
        )
    )
    try:
        df = redshift_loader.query_to_df(query=redshift_query, local_cache=output_path_redshift, reload=True)
        print(df)
    finally:
        redshift_loader.close()
    try:
        df_pa = data_loader.query_to_df(query=athena_query, local_cache=output_path_athena, reload=reload)
        print(df_pa)
    finally:
        data_loader.close()
    diff_product_ids = set(df_pa["product_id"]) - set(df["product_id"]) - pids_to_ignore
    return sorted(diff_product_ids)


def run_slot_pid_diff(
    redshift_query, redshift_database, prefix, output_path_redshift, output_path_athena, pids_to_ignore, bastion_ip
):
    """Run PID diff for ss01 or ss03 using shared Athena result."""
    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database=redshift_database,
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=bastion_ip,
        )
    )
    try:
        df = redshift_loader.query_to_df(query=redshift_query, local_cache=output_path_redshift, reload=True)
        print(df)
    finally:
        redshift_loader.close()

    import pandas as pd

    df_pa = pd.read_parquet(output_path_athena)
    diff_product_ids = set(df_pa["product_id"]) - set(df["product_id"]) - pids_to_ignore
    return sorted(diff_product_ids)


def main(
    bastion_ip: str = DEFAULT_BASTION_IP, output_dir: Path | None = None, reload_athena: bool = False
) -> dict[str, list[str]]:
    if output_dir is None:
        output_dir = _output_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[str]] = {}

    # Fish hunter (independent Athena + Redshift). Isolated so a failure here --
    # e.g. corrupt rows in agfish.hunterorders -- doesn't abort the ss01/ss03 checks.
    fishhunter_pa = output_dir / "fishhunter_product_id_pa.parquet"
    try:
        results["fishhunter"] = run_pid_diff_check(
            athena_database=QUERIES["fish_hunter"]["athena_database"],
            athena_output_location=QUERIES["fish_hunter"]["athena_output"],
            athena_query=QUERIES["fish_hunter"]["athena_query"],
            redshift_database=QUERIES["fish_hunter"]["redshift_database"],
            redshift_query=QUERIES["fish_hunter"]["redshift_query"],
            output_path_redshift=str(output_dir / "fishhunter_product_id.parquet"),
            output_path_athena=str(fishhunter_pa),
            pids_to_ignore=pids_to_ignore,
            bastion_ip=bastion_ip,
            ctas_approach=QUERIES["fish_hunter"]["ctas_approach"],
            reload=reload_athena,
        )
    except Exception as e:
        logger.exception("fishhunter PID check failed")
        results["fishhunter"] = _failed_marker(e)

    # Slot (ss01/ss03): run Athena once, reuse for both. Cached at output/slot_product_id_pa.parquet.
    # The shared Athena scan is a single point of failure for both games; isolate it
    # once, then isolate each game's Redshift diff so ss01 and ss03 stay independent.
    slot_pa_path = output_dir / "slot_product_id_pa.parquet"
    slot_athena_error: list[str] | None = None
    try:
        data_loader = DataLoader(
            backend=AthenaBackend(
                database="ag_share_data",
                output_location=f"s3://{S3_BUCKET}/ds-data-slot/product_id_pa",
                ctas_approach=False,
            )
        )
        try:
            df_slot_pa = data_loader.query_to_df(
                query=SLOT_ATHENA_QUERY, local_cache=str(slot_pa_path), reload=reload_athena
            )
            print(df_slot_pa)
        finally:
            data_loader.close()
    except Exception as e:
        logger.exception("slot (ss01/ss03) shared Athena query failed")
        slot_athena_error = _failed_marker(e)

    for game in ("ss01", "ss03"):
        if slot_athena_error is not None:
            results[game] = slot_athena_error
            continue
        try:
            results[game] = run_slot_pid_diff(
                redshift_query=QUERIES[game]["redshift_query"],
                redshift_database=QUERIES[game]["redshift_database"],
                prefix=game,
                output_path_redshift=str(output_dir / f"{game}_product_id.parquet"),
                output_path_athena=str(slot_pa_path),
                pids_to_ignore=pids_to_ignore,
                bastion_ip=bastion_ip,
            )
        except Exception as e:
            logger.exception("%s PID check failed", game)
            results[game] = _failed_marker(e)

    return results


def _render_section(results: dict[str, list[str]], key: str) -> str:
    """Render one game's PID list, or its inline failure marker if the check raised."""
    val = results.get(key)
    if val is None:
        return f"{_CHECK_FAILED_PREFIX} not run"
    if len(val) == 1 and val[0].startswith(_CHECK_FAILED_PREFIX):
        return val[0]
    return str(val)


def format_pid_report(results: dict[str, list[str]]) -> str:
    """Return PID diff report as string."""
    lines = [
        "\nss01_wucaishen 未开通PID：",
        "--------------------------------",
        _render_section(results, "ss01"),
        "--------------------------------",
        "ss03_majiang_streak 未开通PID：",
        "--------------------------------",
        _render_section(results, "ss03"),
        "--------------------------------",
        "捕鱼未开通PID：",
        "--------------------------------",
        _render_section(results, "fishhunter"),
        "--------------------------------",
    ]
    return "\n".join(lines)


def print_report(results: dict[str, list[str]]) -> None:
    """Print PID diff report for submission."""
    print(format_pid_report(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bastion-ip", type=str, default=DEFAULT_BASTION_IP)
    args = parser.parse_args()

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    results = main(bastion_ip=args.bastion_ip)
    print_report(results)
