"""
One-shot ETL: extract raw ``public.bullet`` events for a fixed set of usernames
from ``dim_user_latest`` and write Parquet to S3.

**Output columns**

``event_timestamp``, ``data_date``, ``user_id``, ``user_name``, ``risk_user_group``, ``game_id``,
``ip``, ``currency_type``, ``op_code``, ``strategy_name``, ``bullet_level``,
``killed``, ``bet``, ``payout``, ``profit``, ``curr_balance``, ``fish_value``,
``multiplier``, ``device_type``, then ``_processed_at`` (added in Python before
upload).

Filters: ``op_code`` / ``currency_type`` use ``ETL_EXCLUDED_OP_CODES`` and ``ETL_CURRENCY_CODES`` (SQL IN-list strings from config),
``event_timestamp > '2025-01-01'``.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import awswrangler as wr
import pandas as pd

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    ETL_CURRENCY_CODES,
    ETL_EXCLUDED_OP_CODES,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    TIMEZONE_SHANGHAI,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, RedshiftBackend

logger = logging.getLogger(__name__)

S3_OUTPUT_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/risk_user_stats"

RISK_USERS_JSON = Path(__file__).with_name("risk_users.json")


def _load_risk_user_groups() -> list[tuple[str, str]]:
    """Load username groups from JSON and return (user_name, group_label) rows."""
    payload = json.loads(RISK_USERS_JSON.read_text(encoding="utf-8"))
    group1 = list(dict.fromkeys([str(x) for x in payload.get("group1", [])]))
    raw_group2 = list(dict.fromkeys([str(x) for x in payload.get("group2", [])]))
    overlap = sorted(set(group1) & set(raw_group2))
    if overlap:
        logger.warning("Found %s overlapping usernames between group1/group2", len(overlap))
    group2 = [name for name in raw_group2 if name not in set(group1)]
    return [(name, "group1") for name in group1] + [(name, "group2") for name in group2]


# Usernames resolved via ``dim_user_latest`` for the bullet aggregate below.
RISK_USER_GROUPS = _load_risk_user_groups()
RISK_USER_NAMES = [name for name, _ in RISK_USER_GROUPS]


def _sql_user_group_union(rows: list[tuple[str, str]]) -> str:
    """Redshift-safe UNION ALL SELECT rows for (user_name, risk_user_group)."""
    lines: list[str] = []
    for idx, (user_name, group_label) in enumerate(rows):
        user_escaped = user_name.replace("'", "''")
        group_escaped = group_label.replace("'", "''")
        prefix = "SELECT" if idx == 0 else "UNION ALL SELECT"
        lines.append(f"                {prefix} '{user_escaped}' AS user_name, '{group_escaped}' AS risk_user_group")
    return "\n".join(lines)


def build_query() -> str:
    """Return the Redshift SQL for raw risk-user bullet events."""
    if not RISK_USER_GROUPS:
        raise ValueError("RISK_USER_GROUPS must not be empty")

    union_list = _sql_user_group_union(RISK_USER_GROUPS)
    return dedent(
        f"""
        WITH risk_user_group_map AS (
{union_list}
        ),
        user_id AS (
            SELECT
                d.user_id,
                d.user_name,
                m.risk_user_group
            FROM public.dim_user_latest AS d
            INNER JOIN risk_user_group_map AS m ON d.user_name = m.user_name
        )

        SELECT
            t.event_timestamp,
            TRUNC(CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', t.event_timestamp)) AS data_date,
            t.user_id,
            t2.user_name,
            t2.risk_user_group,
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
            t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
            AND t.currency_type IN {ETL_CURRENCY_CODES}
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
