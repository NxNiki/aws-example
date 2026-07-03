"""
One-shot ETL: build random control-user groups and aggregate their
``public.bullet`` events server-side in Redshift (same aggregate contract as
``etl_get_risk_user_stats.py`` — see ``aggregate_queries.py``).

Behavior:
1) Read ``control_users.json``.
2) If ``group1`` is empty, randomly sample 100 usernames into group1.
3) Else if ``group2`` is empty, randomly sample 100 usernames into group2.
4) Save ``control_users.json``.
5) Aggregate bullet events for all control users and write two parquet
   datasets under ``S3_OUTPUT_PREFIX`` (``user_summary/`` and
   ``category_counts/``), partitioned by ``control_user_group`` with
   ``overwrite_partitions``.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
from textwrap import dedent
from types import ModuleType

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

S3_OUTPUT_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/control_user_agg"
CONTROL_USERS_JSON = Path(__file__).with_name("control_users.json")
RISK_USERS_JSON = Path(__file__).with_name("risk_users.json")
GROUP_SIZE = 100

EVENT_START = "2026-04-01"
GROUP_COL = "control_user_group"


def _load_sibling(name: str) -> ModuleType:
    """Side-load a sibling module (``jobs/risk_control`` is not a package)."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_AGG = _load_sibling("aggregate_queries")


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


def sql_resolved_users(rows: list[tuple[str, str]]) -> str:
    """SELECT resolving control usernames to (user_id, user_name, control_user_group).

    Control identifiers are always user_names (sampled from dim_user_latest),
    so digit-only entries are NOT treated as user_ids.
    """
    return _AGG.sql_resolved_users(rows, GROUP_COL, treat_digits_as_ids=False)


def build_user_summary_query(
    rows: list[tuple[str, str]],
    event_start: str | None = None,
    event_end: str | None = None,
) -> str:
    """Render the per-user summary SQL for control users."""
    if not rows:
        raise ValueError("No control users available; please populate control groups first.")
    return _AGG.build_user_summary_query(sql_resolved_users(rows), GROUP_COL, event_start or EVENT_START, event_end)


def build_category_counts_query(
    rows: list[tuple[str, str]],
    event_start: str | None = None,
    event_end: str | None = None,
) -> str:
    """Render the per-user category-counts SQL for control users."""
    if not rows:
        raise ValueError("No control users available; please populate control groups first.")
    return _AGG.build_category_counts_query(sql_resolved_users(rows), GROUP_COL, event_start or EVENT_START, event_end)


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
        summary_df = loader.query_to_df(query=build_user_summary_query(control_rows))
        counts_df = loader.query_to_df(query=build_category_counts_query(control_rows))
    finally:
        loader.close()

    requested = sorted({g for _, g in control_rows})
    _AGG.write_aggregate_dataset(
        summary_df,
        "user_summary",
        output_prefix=S3_OUTPUT_PREFIX,
        group_col=GROUP_COL,
        requested_groups=requested,
        numeric_cols=_AGG.SUMMARY_NUMERIC_COLS,
        ts_cols=_AGG.SUMMARY_TS_COLS,
    )
    _AGG.write_aggregate_dataset(
        counts_df,
        "category_counts",
        output_prefix=S3_OUTPUT_PREFIX,
        group_col=GROUP_COL,
        requested_groups=requested,
        numeric_cols=_AGG.COUNTS_NUMERIC_COLS,
        ts_cols=[],
    )


if __name__ == "__main__":
    main()
