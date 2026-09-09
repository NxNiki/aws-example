"""Shared helpers for the risk_control aggregate ETLs.

Both risk and control ETLs aggregate ``public.bullet`` server-side and
transfer only per-user aggregates (the raw-event pull did not scale: group4
alone is ~52M rows). This module holds the shared SQL builders and the
partitioned-dataset writer; the ETLs side-load it (``jobs/`` is not a
package) and differ only in group source and group column name.

Queries:
- ``build_user_summary_query``: one row per user — n_orders, total
  bet/payout/profit, first/last event timestamps, avg/median bet interval
  (seconds, gaps in ``[0, SESSION_GAP_SECONDS]`` only), n_bet_sessions
  (1 + count of gaps > SESSION_GAP_SECONDS).
- ``build_category_counts_query``: narrow (user, metric, category, n) counts
  for the report's five distribution metrics (``METRICS``).

Pandas-parity rules (the report must stay identical to the old raw-event
implementation in risk_user_analysis.py):
- gap filter is inclusive of 0 (pandas used ``diffs >= 0``);
- gaps use microsecond DATEDIFF (integer-second DATEDIFF truncates);
- PERCENTILE_CONT lives in its own CTE joined back (Redshift cannot mix
  WITHIN GROUP aggregates into the main GROUP BY; same pattern as
  jobs/etl/redshift/fish_hunter/etl_game_stats_by_bet.py);
- TRUNC before int casts (pandas ``.astype(int)`` truncates, CAST rounds);
- strategy/ip NULLs become 'UNKNOWN' (= fillna), bullet/fish/combo branches
  drop NULLs (= dropna);
- the combo label reproduces pandas float repr: ``"1.0x_L1"``.
"""

import logging
from datetime import datetime

import awswrangler as wr
import pandas as pd

from bituslabs_ds.config import ETL_CURRENCY_CODES, ETL_EXCLUDED_OP_CODES

logger = logging.getLogger(__name__)

SESSION_GAP_SECONDS = 1800
METRICS = ("bullet_level", "strategy_name", "ip", "fish_value", "multiplier_x_bullet")

SUMMARY_NUMERIC_COLS = [
    "user_id",
    "n_orders",
    "total_bet",
    "total_payout",
    "total_profit",
    "avg_bet_interval_s",
    "median_bet_interval_s",
    "n_bet_sessions",
]
SUMMARY_TS_COLS = ["first_event_ts", "last_event_ts"]
COUNTS_NUMERIC_COLS = ["user_id", "n"]

_COMBO_LABEL_EXPR = (
    "CASE WHEN multiplier = TRUNC(multiplier) "
    "THEN CAST(CAST(multiplier AS BIGINT) AS VARCHAR) || '.0' "
    "ELSE CAST(multiplier AS VARCHAR) END "
    "|| 'x_L' || CAST(CAST(TRUNC(bullet_level) AS BIGINT) AS VARCHAR)"
)


def _sql_quoted_list(values: list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def sql_resolved_users(rows: list[tuple[str, str]], group_col: str, *, treat_digits_as_ids: bool = True) -> str:
    """SELECT resolving (identifier, group) rows to (user_id, user_name, group_col).

    With ``treat_digits_as_ids`` (default), digit-only identifiers match
    ``dim_user_latest.user_id``; all others match ``user_name``. Pass False
    when identifiers are always user_names (e.g. sampled control users, where
    an all-digit user_name must not be misread as an id). IN-lists + CASE keep
    Redshift compile time flat (a UNION ALL map with hundreds of branches
    takes minutes just to compile). CASE order implements earlier-group-wins
    for users matched by multiple entries.
    """
    group_order: list[str] = []
    names_by_group: dict[str, list[str]] = {}
    ids_by_group: dict[str, list[int]] = {}
    for identifier, group in rows:
        if group not in group_order:
            group_order.append(group)
        if treat_digits_as_ids and identifier.isdigit():
            ids_by_group.setdefault(group, []).append(int(identifier))
        else:
            names_by_group.setdefault(group, []).append(identifier)

    case_whens: list[str] = []
    for group in group_order:
        conds: list[str] = []
        if names_by_group.get(group):
            conds.append(f"d.user_name IN ({_sql_quoted_list(names_by_group[group])})")
        if ids_by_group.get(group):
            conds.append(f"d.user_id IN ({', '.join(str(i) for i in ids_by_group[group])})")
        group_escaped = group.replace("'", "''")
        case_whens.append(f"            WHEN {' OR '.join(conds)} THEN '{group_escaped}'")
    case_block = "\n".join(case_whens)

    all_names = [n for g in group_order for n in names_by_group.get(g, [])]
    all_ids = [i for g in group_order for i in ids_by_group.get(g, [])]
    where_parts: list[str] = []
    if all_names:
        where_parts.append(f"d.user_name IN ({_sql_quoted_list(all_names)})")
    if all_ids:
        where_parts.append(f"d.user_id IN ({', '.join(str(i) for i in all_ids)})")
    where_clause = "\n        OR ".join(where_parts)

    return (
        "    SELECT\n"
        "        d.user_id,\n"
        "        d.user_name,\n"
        "        CASE\n"
        f"{case_block}\n"
        f"        END AS {group_col}\n"
        "    FROM public.dim_user_latest AS d\n"
        f"    WHERE {where_clause}"
    )


def _events_cte(
    resolved_users_sql: str,
    group_col: str,
    event_start: str,
    event_end: str | None,
    event_columns: str,
) -> str:
    end_filter = f"\n      AND t.event_timestamp <= '{event_end}'" if event_end else ""
    return f"""WITH user_map AS (
{resolved_users_sql}
),
events AS (
    SELECT
        t.user_id,
        u.user_name,
        u.{group_col},
{event_columns}
    FROM public.bullet AS t
    INNER JOIN user_map AS u ON t.user_id = u.user_id
    WHERE t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
      AND t.currency_type IN {ETL_CURRENCY_CODES}
      AND t.event_timestamp > '{event_start}'{end_filter}
)"""


def build_user_summary_query(
    resolved_users_sql: str,
    group_col: str,
    event_start: str,
    event_end: str | None = None,
) -> str:
    """Per-user summary aggregates over the filtered bullet events."""
    events = _events_cte(
        resolved_users_sql,
        group_col,
        event_start,
        event_end,
        "        t.event_timestamp,\n        t.bet,\n        t.payout,\n        t.profit",
    )
    gap = SESSION_GAP_SECONDS
    return f"""{events},
ordered AS (
    SELECT
        user_id,
        user_name,
        {group_col},
        event_timestamp,
        bet,
        payout,
        profit,
        DATEDIFF(
            microsecond,
            LAG(event_timestamp) OVER (PARTITION BY user_id ORDER BY event_timestamp),
            event_timestamp
        ) / 1000000.0 AS gap_s
    FROM events
),
base AS (
    SELECT
        user_id,
        user_name,
        {group_col},
        COUNT(*) AS n_orders,
        SUM(bet) AS total_bet,
        SUM(payout) AS total_payout,
        SUM(profit) AS total_profit,
        MIN(event_timestamp) AS first_event_ts,
        MAX(event_timestamp) AS last_event_ts,
        AVG(CASE WHEN gap_s >= 0 AND gap_s <= {gap} THEN gap_s END) AS avg_bet_interval_s,
        1 + COALESCE(SUM(CASE WHEN gap_s > {gap} THEN 1 ELSE 0 END), 0) AS n_bet_sessions
    FROM ordered
    GROUP BY 1, 2, 3
),
med AS (
    SELECT
        user_id,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_s) AS median_bet_interval_s
    FROM ordered
    WHERE gap_s >= 0 AND gap_s <= {gap}
    GROUP BY user_id
)
SELECT
    b.user_id,
    b.user_name,
    b.{group_col},
    b.n_orders,
    b.total_bet,
    b.total_payout,
    b.total_profit,
    b.first_event_ts,
    b.last_event_ts,
    b.avg_bet_interval_s,
    m.median_bet_interval_s,
    b.n_bet_sessions
FROM base AS b
LEFT JOIN med AS m ON b.user_id = m.user_id"""


def build_category_counts_query(
    resolved_users_sql: str,
    group_col: str,
    event_start: str,
    event_end: str | None = None,
) -> str:
    """Narrow per-user (metric, category, n) counts for the report's charts."""
    events = _events_cte(
        resolved_users_sql,
        group_col,
        event_start,
        event_end,
        "        t.strategy_name,\n        t.ip,\n        t.bullet_level,\n        t.fish_value,\n        t.multiplier",
    )
    ident = f"user_id,\n    user_name,\n    {group_col}"
    return f"""{events}
SELECT
    {ident},
    'bullet_level' AS metric,
    CAST(CAST(TRUNC(bullet_level) AS BIGINT) AS VARCHAR) AS category,
    COUNT(*) AS n
FROM events
WHERE bullet_level IS NOT NULL
GROUP BY 1, 2, 3, 5

UNION ALL

SELECT
    {ident},
    'strategy_name' AS metric,
    NVL(strategy_name, 'UNKNOWN') AS category,
    COUNT(*) AS n
FROM events
GROUP BY 1, 2, 3, 5

UNION ALL

SELECT
    {ident},
    'ip' AS metric,
    NVL(ip, 'UNKNOWN') AS category,
    COUNT(*) AS n
FROM events
GROUP BY 1, 2, 3, 5

UNION ALL

SELECT
    {ident},
    'fish_value' AS metric,
    CAST(CAST(TRUNC(fish_value) AS BIGINT) AS VARCHAR) AS category,
    COUNT(*) AS n
FROM events
WHERE fish_value IS NOT NULL
GROUP BY 1, 2, 3, 5

UNION ALL

SELECT
    {ident},
    'multiplier_x_bullet' AS metric,
    {_COMBO_LABEL_EXPR} AS category,
    COUNT(*) AS n
FROM events
WHERE multiplier IS NOT NULL AND bullet_level IS NOT NULL
GROUP BY 1, 2, 3, 5"""


def write_aggregate_dataset(
    df: pd.DataFrame | None,
    dataset_name: str,
    *,
    output_prefix: str,
    group_col: str,
    requested_groups: list[str],
    numeric_cols: list[str],
    ts_cols: list[str],
) -> None:
    """Type-normalize an aggregate frame and write it as a partitioned dataset.

    ``overwrite_partitions`` replaces only the group partitions present in
    this run, so re-pulling a subset of groups updates in place.
    """
    if df is None or df.empty:
        logger.warning("%s query returned no rows; skipping S3 write.", dataset_name)
        return
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["_processed_at"] = datetime.now()

    present = sorted(str(g) for g in df[group_col].unique())
    missing = [g for g in requested_groups if g not in present]
    if missing:
        logger.warning("%s: no data for requested groups %s", dataset_name, missing)

    out_uri = f"{output_prefix}/{dataset_name}/"
    wr.s3.to_parquet(
        df=df,
        path=out_uri,
        dataset=True,
        mode="overwrite_partitions",
        partition_cols=[group_col],
        index=False,
    )
    logger.info("Wrote %s rows to %s (groups %s)", len(df), out_uri, present)
