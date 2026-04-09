"""
One-shot ETL: extract raw ``public.bullet`` events for a fixed set of usernames
from ``dim_user_latest`` and write Parquet to S3.

**Output columns**

``event_timestamp``, ``data_date``, ``user_id``, ``user_name``, ``game_id``,
``ip``, ``currency_type``, ``op_code``, ``strategy_name``, ``bullet_level``,
``killed``, ``bet``, ``payout``, ``profit``, ``curr_balance``, ``fish_value``,
``multiplier``, ``device_type``, then ``_processed_at`` (added in Python before
upload).

Filters: ``op_code`` not in B26/TST/TSB/TSO, ``currency_type = 'CNY'``,
``event_timestamp > '2025-01-01'``.
"""

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

S3_OUTPUT_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/risk_user_stats"

# Usernames resolved via ``dim_user_latest`` for the bullet aggregate below.
RISK_USER_NAMES = [
    "A5802j9xfrf0002",
    "A5802j9ffg656565",
    "A5802j9xdftgdrt988",
    "A5802j9sftydry877",
    "A5802j9kkk6666",
    "A5802j9dsds223335",
    "A5802j9fghfgfg6232",
    "A5802j9xfghdfghd25",
    "A5802j9dfgyrty88",
    "A5802j904048259",
    "A5802j9nhn6554695",
    "A5802j9dfgg78948",
    "A5802j9dwsd6896898",
    "A5802paluke",
    "A5802j9fghjftg0251",
    "A5802j9ccc001011",
    "A5802j9fef635656",
    "A5802j9ssrer01215",
    "A5802j9dffdg2142",
    "A5802j9dfdf3223",
    "A5802j9edwin",
    "A5802j9fasegtr888",
    "A5802j9sdtdrtet777",
    "A5802pa1013091",
    "A5802j9xfgdfg999",
    "A5802pakenneyn8140",
    "A5802pajake",
    "A5802pamartham7",
    "A5802pajohnnyk428",
    "A5802j9xdfgdfg010",
    "A5802pabidelachq22",
    "A5802pa10142039",
    "A5802j9jhywu010",
    "A5802j9dsre5445454",
    "A5802paasfsa321",
    "A5802pamuirp183",
    "A5802pasdrs0120",
    "A5802j9xfgdfgf0110",
    "A5802j904039841",
    "A5802padsafe320",
    "A5802padzfs3000",
    "A5802pa1014625",
    "A5802j9vcbvc564",
    "A5802j9fxghdf9871",
    "A5802parose",
    "A5802pahill",
    "A5802j9hggh526356",
    "A5802j9tytyyy11121",
    "A5802j9xcvhdfh887",
    "A5802j9zxfgdfg012",
    "A5802j9xfhfdty888",
    "A5802j9sds3032",
    "A5802pamcin",
    "A5802j9fhgfg546546",
    "A5802j9cxgufgy988",
    "A5802pasdfs698",
    "A5802j9cvvdf325656",
    "A5802j9xfhdfgh88",
    "A5802j9ds6898921",
    "A5802pahint",
    "A5802j9xfhdtgr888",
    "A5802j9xhgdfh0120",
    "A5802j9xzdgdrg889",
    "A5802j9gfg232323",
    "A5802j9fghjfghj89",
    "A5802j9xcfhfgh321",
    "A5802j9xchfgf888",
    "A5802j9xcyhftg985",
    "A5802j9xdfgdf988",
    "A5802j9xfhfth2548",
    "A5802j9mckenz",
    "A5802j9dfgdfgdfg88",
    "A5802j9dgdfgrd888",
    "A5802j9xfgdfg6987",
    "A5802pazdfs320",
    "A5802pazdfsd47",
    "A5802j9fgfg8889999",
    "A5802j9xcgxfgd999",
    "A5802pasdad0120",
    "A5802j9xfdgdfg878",
    "A5802pawiggi",
    "A5802j9dfb4552",
    "A5802pabressettf73",
    "A5802pakyla",
    "A5802j9xfhdgfhfdg6",
    "A5802j9fghfgh999",
    "A5802j904047776",
    "A5802pamacldr622",
    "A5802pa2510141473",
    "A5802padevanw281",
]


def _sql_in_string_literals(names: list[str]) -> str:
    """Comma-separated ``'...'`` lines for a SQL ``IN`` list (Redshift string escape)."""
    lines: list[str] = []
    for n in names:
        escaped = n.replace("'", "''")
        lines.append(f"                '{escaped}'")
    return ",\n".join(lines)


def build_query() -> str:
    """Return the Redshift SQL for raw risk-user bullet events."""
    if not RISK_USER_NAMES:
        raise ValueError("RISK_USER_NAMES must not be empty")

    in_list = _sql_in_string_literals(RISK_USER_NAMES)
    return dedent(
        f"""
        WITH user_id AS (
            SELECT
                user_id,
                user_name
            FROM public.dim_user_latest
            WHERE user_name IN (
                {in_list}
            )
        )

        SELECT
            t.event_timestamp,
            TRUNC(CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.event_timestamp)) AS data_date,
            t.user_id,
            t2.user_name,
            t.game_id,
            t.ip,
            t.currency_type,
            t.op_code,
            t.strategy_name,
            t.bullet_level,
            t.killed,
            t.bet,
            t.payout,
            t.profit,
            t.curr_balance,
            t.fish_value,
            t.multiplier,
            t.device_type

        FROM public.bullet AS t
        INNER JOIN user_id AS t2 ON t.user_id = t2.user_id
        WHERE
            t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
            AND t.currency_type = 'CNY'
            AND t.event_timestamp > '2025-01-01'
        ORDER BY t2.user_name, t.event_timestamp
        """
    ).strip()


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    sql = build_query()

    loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="transform-agfish-game",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=DEFAULT_BASTION_IP,
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

    skip_numeric = {
        "event_timestamp",
        "data_date",
        "user_name",
        "ip",
        "currency_type",
        "op_code",
        "strategy_name",
        "device_type",
        "_processed_at",
    }
    for col in df.columns:
        if col in skip_numeric or df[col].dtype != "object":
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception:
            pass

    out_uri = f"{S3_OUTPUT_PREFIX}/risk_user_stats.parquet"
    wr.s3.to_parquet(df=df, path=out_uri, index=False)
    logger.info("Wrote %s rows to %s", len(df), out_uri)


if __name__ == "__main__":
    main()
