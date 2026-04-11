"""
One-shot ETL: build random control-user groups and extract raw bullet events.

Behavior:
1) Read ``control_users.json``.
2) If ``group1`` is empty, randomly sample 100 usernames into group1.
3) Else if ``group2`` is empty, randomly sample 100 usernames into group2.
4) Save ``control_users.json``.
5) Query bullet events for all control users and write parquet to S3.
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
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, RedshiftBackend

logger = logging.getLogger(__name__)

S3_OUTPUT_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/control_user_stats"
CONTROL_USERS_JSON = Path(__file__).with_name("control_users.json")
RISK_USERS_JSON = Path(__file__).with_name("risk_users.json")
GROUP_SIZE = 100


def _load_risk_user_names() -> set[str]:
    payload = json.loads(RISK_USERS_JSON.read_text(encoding="utf-8"))
    group1 = [str(x) for x in payload.get("group1", [])]
    group2 = [str(x) for x in payload.get("group2", [])]
    return set(group1) | set(group2)


def _load_control_payload() -> dict[str, list[str]]:
    if not CONTROL_USERS_JSON.exists():
        return {"group1": [], "group2": []}
    payload = json.loads(CONTROL_USERS_JSON.read_text(encoding="utf-8"))
    group1 = list(dict.fromkeys([str(x) for x in payload.get("group1", [])]))
    group2 = list(dict.fromkeys([str(x) for x in payload.get("group2", [])]))
    return {"group1": group1, "group2": group2}


def _sql_in_string_literals(names: list[str], indent: str = "                ") -> str:
    lines: list[str] = []
    for n in names:
        escaped = n.replace("'", "''")
        lines.append(f"{indent}'{escaped}'")
    return ",\n".join(lines)


def _sample_user_names(loader: DataLoader, exclude_names: set[str], limit: int) -> list[str]:
    where_exclusions = ""
    if exclude_names:
        in_list = _sql_in_string_literals(sorted(exclude_names))
        where_exclusions = f"\n            AND user_name NOT IN (\n{in_list}\n            )"
    sql = dedent(
        f"""
        SELECT user_name
        FROM public.dim_user_latest
        WHERE user_name IS NOT NULL
            AND TRIM(user_name) <> ''{where_exclusions}
        ORDER BY RANDOM()
        LIMIT {int(limit)}
        """
    ).strip()
    df = loader.query_to_df(query=sql)
    if df is None or df.empty:
        return []
    return [str(x) for x in df["user_name"].dropna().astype(str).tolist()]


def _ensure_control_groups(loader: DataLoader) -> list[tuple[str, str]]:
    payload = _load_control_payload()
    group1 = payload["group1"]
    group2 = payload["group2"]
    target_group: str | None = None
    if not group1:
        target_group = "group1"
    elif not group2:
        target_group = "group2"

    if target_group is not None:
        exclude = _load_risk_user_names() | set(group1) | set(group2)
        sampled = _sample_user_names(loader, exclude_names=exclude, limit=GROUP_SIZE)
        if len(sampled) < GROUP_SIZE:
            logger.warning("Only sampled %s users for %s (target=%s)", len(sampled), target_group, GROUP_SIZE)
        payload[target_group] = sampled
        CONTROL_USERS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Updated %s with %s users in %s", CONTROL_USERS_JSON.name, len(sampled), target_group)
    else:
        logger.info("%s already has both groups, no new sampling.", CONTROL_USERS_JSON.name)

    rows = [(name, "group1") for name in payload["group1"]] + [(name, "group2") for name in payload["group2"]]
    return rows


def _sql_user_group_union(rows: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for idx, (user_name, group_label) in enumerate(rows):
        user_escaped = user_name.replace("'", "''")
        group_escaped = group_label.replace("'", "''")
        prefix = "SELECT" if idx == 0 else "UNION ALL SELECT"
        lines.append(f"                {prefix} '{user_escaped}' AS user_name, '{group_escaped}' AS control_user_group")
    return "\n".join(lines)


def build_query(rows: list[tuple[str, str]]) -> str:
    if not rows:
        raise ValueError("No control users available; please populate control groups first.")
    union_list = _sql_user_group_union(rows)
    return dedent(
        f"""
        WITH control_user_group_map AS (
{union_list}
        ),
        user_id AS (
            SELECT
                d.user_id,
                d.user_name,
                m.control_user_group
            FROM public.dim_user_latest AS d
            INNER JOIN control_user_group_map AS m ON d.user_name = m.user_name
        )

        SELECT
            t.event_timestamp,
            TRUNC(CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.event_timestamp)) AS data_date,
            t.user_id,
            t2.user_name,
            t2.control_user_group,
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
        control_rows = _ensure_control_groups(loader)
        sql = build_query(control_rows)
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
        "control_user_group",
    }
    for col in df.columns:
        if col in skip_numeric or df[col].dtype != "object":
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception:
            pass

    out_uri = f"{S3_OUTPUT_PREFIX}/control_user_stats.parquet"
    wr.s3.to_parquet(df=df, path=out_uri, index=False)
    logger.info("Wrote %s rows to %s", len(df), out_uri)


if __name__ == "__main__":
    main()
