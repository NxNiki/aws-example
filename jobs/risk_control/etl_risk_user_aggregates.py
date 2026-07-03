"""
One-shot ETL: aggregate ``public.bullet`` events for configured risk-user
groups server-side in Redshift and write per-user aggregate parquet datasets
to S3.

This replaces the former raw-event pull: raw events do not scale (group4
alone is ~52M rows — a 22s server-side scan but 30+ minutes to stream through
the bastion tunnel), and the report only consumes per-user aggregates.

**Outputs** — two datasets under ``S3_OUTPUT_PREFIX``, partitioned by
``risk_user_group`` and written with ``overwrite_partitions`` (re-running a
subset of groups replaces only those groups' partitions):

- ``user_summary/`` — one row per user: ``n_orders``, ``total_bet``,
  ``total_payout``, ``total_profit``, ``first_event_ts``, ``last_event_ts``,
  ``avg_bet_interval_s``, ``median_bet_interval_s``, ``n_bet_sessions``.
- ``category_counts/`` — narrow rows ``(user, metric, category, n)`` for the
  report metrics: bullet_level, strategy_name, ip, fish_value,
  multiplier_x_bullet.

Run configuration: ``PROCESS_GROUPS`` selects which ``groupN`` keys from
``risk_users.json`` to process (None = all), ``EVENT_START`` bounds the event
window. Entries in ``risk_users.json`` are matched against
``dim_user_latest`` by ``user_name``, except digit-only entries which are
matched by ``user_id``.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
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

S3_OUTPUT_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/risk_user_agg"

RISK_USERS_JSON = Path(__file__).with_name("risk_users.json")

# --- run configuration ---
# Which groupN keys from risk_users.json to process (None = all groups), and
# the event_timestamp lower bound shared by all processed groups.
PROCESS_GROUPS: list[str] | None = None
EVENT_START = "2026-04-01"

GROUP_COL = "risk_user_group"


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


def _load_risk_user_groups() -> list[tuple[str, str]]:
    """Load user groups from JSON and return (identifier, group_label) rows.

    Identifiers are user_names, or user_ids when digit-only. Accepts any number
    of ``groupN`` keys (e.g. ``group1``, ``group2``, ``group3``).
    Earlier groups take precedence: an identifier appearing in multiple groups is
    only emitted under the first ``groupN`` it appears in (by ascending N).
    """
    payload = json.loads(RISK_USERS_JSON.read_text(encoding="utf-8"))
    group_keys = sorted(
        (k for k in payload.keys() if k.startswith("group") and k[len("group") :].isdigit()),
        key=lambda k: int(k[len("group") :]),
    )
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for group_key in group_keys:
        names = list(dict.fromkeys([str(x) for x in payload.get(group_key, [])]))
        overlap = sorted(set(names) & seen)
        if overlap:
            logger.warning(
                "Found %s usernames in %s already assigned to an earlier group; skipping in %s",
                len(overlap),
                group_key,
                group_key,
            )
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            rows.append((name, group_key))
    return rows


# Identifiers resolved via ``dim_user_latest`` for the bullet aggregates below.
RISK_USER_GROUPS = [
    (name, group) for name, group in _load_risk_user_groups() if PROCESS_GROUPS is None or group in PROCESS_GROUPS
]


def sql_resolved_users(rows: list[tuple[str, str]]) -> str:
    """SELECT resolving identifiers to (user_id, user_name, risk_user_group)."""
    return _AGG.sql_resolved_users(rows, GROUP_COL)


def build_user_summary_query(
    rows: list[tuple[str, str]] | None = None,
    event_start: str | None = None,
    event_end: str | None = None,
) -> str:
    """Render the per-user summary SQL (defaults to module run configuration)."""
    rows = RISK_USER_GROUPS if rows is None else rows
    if not rows:
        raise ValueError("No risk user groups to process")
    return _AGG.build_user_summary_query(sql_resolved_users(rows), GROUP_COL, event_start or EVENT_START, event_end)


def build_category_counts_query(
    rows: list[tuple[str, str]] | None = None,
    event_start: str | None = None,
    event_end: str | None = None,
) -> str:
    """Render the per-user category-counts SQL (defaults to module run configuration)."""
    rows = RISK_USER_GROUPS if rows is None else rows
    if not rows:
        raise ValueError("No risk user groups to process")
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
        summary_df = loader.query_to_df(query=build_user_summary_query())
        counts_df = loader.query_to_df(query=build_category_counts_query())
    finally:
        loader.close()

    requested = sorted({g for _, g in RISK_USER_GROUPS})
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
