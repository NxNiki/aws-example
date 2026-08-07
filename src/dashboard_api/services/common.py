"""Shared data-access helpers for the /api/data/* services.

The Stats-by-Date, Stats-by-Group, and Deep Dive endpoints all read the same
one-row-per-user parquet, filter to the same cohort cross-product, and reuse the
``DataMetrics`` engine. The cohort/loading plumbing lives here so the three
services stay thin and behave identically (matching the legacy Dash tabs).
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import polars as pl

from bituslabs_ds.metrics.user_stats_aggregates import PERIODS_SINCE_FIRST_BET_COL
from bituslabs_ds.s3_utils import expand_paths_to_files, is_s3_path

logger = logging.getLogger(__name__)

# Collected-window cache, single-flight. Maps a request window to the
# MATERIALIZED user rows collected for it — key (config_id, granularity,
# start, end, projected-columns-or-None), value (monotonic timestamp,
# collected DataFrame). Example entry:
#   ("ss03", "day", "2026-07-01 00:00:00", "2026-07-28 00:00:00",
#    ("activity_date", "ab_group", "user_id", "user_rtp"))
#     -> (ts, DataFrame of every per-user daily row in that window, ~80 MB)
# Each tab fires one /api/data/* request PER PANEL in parallel, all for the
# same (config, granularity, window) — without this, each request collects
# its own copy of the user rows
# simultaneously, which OOM-killed 2 GB and 4 GB tasks in production. One lock
# serializes collects (a concurrent miss waits, then hits the cache). Eviction
# is budgeted by ESTIMATED BYTES, not entry count: a Stats-by-Group span can be
# months of per-user rows, and pinning a few of those by count is exactly how
# the 4 GB task died. The newest frame always stays (it's what the in-flight
# burst shares); a TTL picks up the daily ETL refresh without a restart.
# Sized against the 8 GB task (TASK_MEMORY in infra/dashboard_api/deploy_ecs.py):
# budget + the largest in-flight collects must stay under it with margin.
_WINDOW_CACHE_MAX_BYTES = 2_500_000_000
_WINDOW_CACHE_TTL_S = 900
_window_cache: "OrderedDict[tuple[Any, ...], tuple[float, pl.DataFrame]]" = OrderedDict()
_window_lock = threading.Lock()


def _evict_over_budget() -> None:
    while len(_window_cache) > 1:
        total = sum(df.estimated_size() for _, df in _window_cache.values())
        if total <= _WINDOW_CACHE_MAX_BYTES:
            return
        key, (_, df) = next(iter(_window_cache.items()))
        _window_cache.pop(key)
        logger.info("window cache: evicted %s (%.0f MB) over budget", key, df.estimated_size() / 1e6)


def _as_window_dt(value: Any) -> Optional[datetime]:
    """Normalize a cache-key bound (datetime, date, or its str form) to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _slice_covering_entry(
    config_id: Any, granularity: str, date_col: str, start_dt: Any, end_dt: Any, columns: Optional[Sequence[str]]
) -> Optional[pl.DataFrame]:
    """Serve [start_dt, end_dt] by slicing a fresh cached frame that covers it.

    One user loading a long range makes every narrower request (same config /
    granularity, columns available in the cached frame) an in-memory filter
    instead of a new S3 collect. Runs under ``_window_lock``. The slice is not
    re-cached — re-slicing a hit is cheap and caching it would duplicate rows
    the covering entry already holds.
    """
    req_start, req_end = _as_window_dt(start_dt), _as_window_dt(end_dt)
    if req_start is None or req_end is None:
        return None
    now = time.monotonic()
    for key, (ts, df) in _window_cache.items():
        if key[0] != config_id or key[1] != granularity or now - ts >= _WINDOW_CACHE_TTL_S:
            continue
        cov_start, cov_end = _as_window_dt(key[2]), _as_window_dt(key[3])
        if cov_start is None or cov_end is None or cov_start > req_start or cov_end < req_end:
            continue
        if date_col not in df.columns:
            continue
        if columns is not None and not set(columns) <= set(df.columns):
            continue
        if columns is None and key[4] is not None:
            # A projected frame can't serve a full-column request.
            continue
        out = df.filter((pl.col(date_col) >= req_start) & (pl.col(date_col) <= req_end))
        if columns is not None:
            out = out.select(list(columns))
        if out.is_empty():
            # Window keys extend past the data edge (retention load margin), so
            # a pre-ETL frame can "cover" dates it has no rows for. An empty
            # slice of a populated frame means collect fresh instead.
            continue
        _window_cache.move_to_end(key)  # covering entries that serve traffic stay warm
        logger.info("window cache: served %s..%s from covering entry %s (rows=%d)", start_dt, end_dt, key, out.height)
        return out
    return None


# Rows with date_col >= window start can live in a period that STARTS up to
# one period-length earlier, so pruning widens one period to the left.
_PERIOD_PRUNE_MARGIN_DAYS = {"day": 0, "week": 7, "month": 31}


def _prune_periods(lf: pl.LazyFrame, granularity: str, start_dt: Any, end_dt: Any) -> pl.LazyFrame:
    """Restrict the hive ``period`` partition column to the window so polars
    skips the other partitions' files by PATH — no footers or pages read.
    Purely an I/O optimization: the caller's date_col filter still bounds the
    rows, and flat layouts (no period column) pass through untouched."""
    if "period" not in lf.collect_schema().names():
        return lf
    start, end = _as_window_dt(start_dt), _as_window_dt(end_dt)
    if start is None or end is None:
        return lf
    lo = start.date() - timedelta(days=_PERIOD_PRUNE_MARGIN_DAYS.get(granularity, 31))
    return lf.filter(pl.col("period").cast(pl.Date, strict=False).is_between(lo, end.date()))


def collect_window(
    cfg: dict[str, Any],
    granularity: str,
    lf: pl.LazyFrame,
    date_col: str,
    start_dt: Any,
    end_dt: Any,
    columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """Collect ``lf`` filtered to [start_dt, end_dt], shared across requests.

    With ``columns`` the collect is projected to just those columns (parquet
    projection pushdown — only their chunks are read from S3) and cached per
    column set, so requests for different metrics don't evict each other's
    (much smaller) frames. A request whose window/columns are contained in a
    fresh cached entry is served by slicing that entry in memory — no collect.
    """
    config_id = cfg.get("id")
    if not config_id:
        # A None id would collide across every config and serve the wrong
        # game's rows from cache (load_raw_config stamps it — see configs.py).
        raise SeriesError("config dict is missing 'id'; cannot safely cache its window")
    key = (config_id, granularity, str(start_dt), str(end_dt), tuple(columns) if columns is not None else None)
    with _window_lock:
        hit = _window_cache.get(key)
        if hit and time.monotonic() - hit[0] < _WINDOW_CACHE_TTL_S:
            _window_cache.move_to_end(key)
            return hit[1]
        _window_cache.pop(key, None)
        sliced = _slice_covering_entry(config_id, granularity, date_col, start_dt, end_dt, columns)
        if sliced is not None:
            return sliced
        lf_win = _prune_periods(lf, granularity, start_dt, end_dt).filter(
            (pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt)
        )
        if columns is not None:
            lf_win = lf_win.select(list(columns))
        df = lf_win.collect()
        logger.info("window cache: collected %s rows=%d est=%.0f MB", key, df.height, df.estimated_size() / 1e6)
        # Never cache an EMPTY frame: a read racing the ETL's dynamic
        # partition overwrite (files deleted, then rewritten) can collect
        # nothing for a window that has data — caching it would pin "no data"
        # on every panel for the whole TTL. Genuinely-empty windows just
        # re-collect, which is cheap.
        if df.height > 0:
            _window_cache[key] = (time.monotonic(), df)
            _evict_over_budget()
        return df


class SeriesError(Exception):
    """Raised when a request can't be served (bad granularity / missing date col)."""


# Computed-response cache, single-flight. The window cache dedups I/O, but every
# request still re-ran cohort filtering + DataMetrics + CI computation — with
# several users on the same default views that recompute pinned the CPU. Keyed
# by the endpoint's full request signature; TTL stays well under the window
# cache's 900s so it never extends data staleness. Errors are never cached.
_RESPONSE_CACHE_TTL_S = 150
_RESPONSE_CACHE_MAX_ENTRIES = 512
_response_cache: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
_response_loading: dict[str, threading.Event] = {}
_response_lock = threading.Lock()


def cached_response(key: str, compute: Callable[[], Any], should_cache: Callable[[Any], bool] = lambda v: True) -> Any:
    """Return ``compute()`` cached under ``key`` with single-flight semantics.

    Identical concurrent requests run ``compute`` once and share the result;
    an exception in the loader propagates to it, and one waiter retries as the
    new loader. Callers must not mutate the returned value — it is shared.
    ``should_cache`` lets callers keep suspicious results (e.g. empty series
    computed while the ETL rewrites partitions) out of the cache: the value is
    still returned, just recomputed on the next request.
    """
    while True:
        with _response_lock:
            hit = _response_cache.get(key)
            if hit and time.monotonic() - hit[0] < _RESPONSE_CACHE_TTL_S:
                _response_cache.move_to_end(key)
                return hit[1]
            if hit:
                _response_cache.pop(key, None)
            loading = _response_loading.get(key)
            if loading is None:
                loading = threading.Event()
                _response_loading[key] = loading
                break
        loading.wait()
    try:
        value = compute()
        if should_cache(value):
            with _response_lock:
                _response_cache[key] = (time.monotonic(), value)
                while len(_response_cache) > _RESPONSE_CACHE_MAX_ENTRIES:
                    _response_cache.popitem(last=False)
        return value
    finally:
        with _response_lock:
            _response_loading.pop(key, None)
        loading.set()


def stats_by_date_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("stats_by_date") or {}


def user_group_cols(cfg: dict[str, Any]) -> list[str]:
    """Configured user-group (cohort) columns, normalized to a list — the legacy
    config allows either a scalar or a list."""
    raw = stats_by_date_cfg(cfg).get("user_group_cols")
    if not raw:
        return []
    cols = raw if isinstance(raw, list) else [raw]
    return [str(c) for c in cols]


def availability_cols(cfg: dict[str, Any]) -> list[str]:
    """Cohort columns whose availability actually varies with the date range.

    Dashboard feature: the pickers' grayed-out entries. Most cohort vocabularies
    (user_group, bet_level, …) are static — only dimensions that rotate over
    time (mathtable for the slot games, daily_group for fishhunter) need the
    per-range availability scan. ``stats_by_date.availability_cols`` names
    them; columns left out are simply absent from the ``available`` map and the
    frontend enables their values unconditionally (CohortSelect treats a
    missing entry as available). Default: every user_group column.
    """
    raw = stats_by_date_cfg(cfg).get("availability_cols")
    if not raw:
        return user_group_cols(cfg)
    cols = raw if isinstance(raw, list) else [raw]
    allowed = set(user_group_cols(cfg))
    return [str(c) for c in cols if str(c) in allowed]


def projection_columns(
    cfg: dict[str, Any], metrics: Iterable[str], date_col: str, available: set[str]
) -> Optional[list[str]]:
    """Columns a series request actually needs, or None to load everything.

    The user-row parquet carries every ``user_*`` metric column, but a panel
    plots a handful — projecting the window collect to just the needed columns
    (parquet pushdown reads only their chunks from S3) shrinks load time and
    cache size regardless of the date span. Includes the cohort/grain
    dimensions, each requested metric's transitive ``user_*`` deps, and the
    components collapse_user_rows recombines (a ratio without its numerator/
    denominator would be silently dropped; a weighted mean without its weight
    would pass through as first()). Falls back to None if any computed metric's
    deps can't be introspected.
    """
    from bituslabs_ds.metrics.user_stats_aggregates import DataMetrics

    cols: set[str] = {date_col, "user_id"}
    cols.update(user_group_cols(cfg))
    cols.update(user_row_grain(cfg))
    lc = lifecycle_col(cfg)
    if lc:
        cols.add(lc)
    rcol, _, _ = range_group_cfg(cfg)
    if rcol:
        cols.add(rcol)
    for m in metrics:
        if m in DataMetrics.METRICS:
            deps = DataMetrics.metric_user_col_deps(m)
            if not deps:
                return None
            cols.update(deps)
        else:
            cols.add(str(m))
    for name, num, den in _USER_ROW_RATIOS:
        if name in cols:
            cols.update((num, den))
    for mean_col, weight_col in _USER_ROW_WEIGHTED_MEANS:
        if mean_col in cols:
            cols.add(weight_col)
    return sorted(c for c in cols if c in available)


def effective_cohort_cols(cfg: dict[str, Any]) -> list[str]:
    """All configured user-group columns, de-duplicated, order preserved."""
    seen: set[str] = set()
    cols: list[str] = []
    for c in user_group_cols(cfg):
        if c not in seen:
            seen.add(c)
            cols.append(c)
    return cols


def user_row_grain(cfg: dict[str, Any]) -> list[str]:
    """Columns (beyond date × user) that define the stored user-row grain.

    Configs whose parquet is one row per (period, user, ab_group, mathtable)
    set ``user_row_grain: [ab_group, mathtable]``: each bet lives in exactly
    one row, so group-level sums are correct at any selection, but per-user
    stats must first collapse the grain back to one row per user whenever a
    grain column is unselected (see ``collapse_user_rows``)."""
    sd = stats_by_date_cfg(cfg)
    return [str(c) for c in (sd.get("user_row_grain") or [])]


# Virtual cohort dimension the "Lifecycle groups" picker defines (cohorts are
# derived at API time from first-bet dates, never read from a stored column).
# Configs opt in by listing this name in user_group_cols.
LIFECYCLE_COL = "life_cycle_group"
# Helper column carrying each row's period offset (days / weeks / months,
# matching the request granularity) from the user's first bet. Canonical name
# lives with DataMetrics, which reads it for num_new_users.
PERIODS_COL = PERIODS_SINCE_FIRST_BET_COL


def range_group_cfg(cfg: dict[str, Any]) -> tuple[Optional[str], str, list[dict[str, Any]]]:
    """(column, display name, default groups) of the config's value-range
    dimension (``stats_by_date.range_group``, e.g. fish_value buckets), or
    (None, "", []) when not configured. The column must also be listed in
    user_group_cols (picker order) and user_row_grain (collapse)."""
    rg = stats_by_date_cfg(cfg).get("range_group") or {}
    col = str(rg.get("column") or "") or None
    name = str(rg.get("name") or (col or ""))
    defaults = [dict(d) for d in (rg.get("defaults") or [])]
    return col, name, defaults


def range_group_values(cfg: dict[str, Any]) -> list[float]:
    """Config-declared selectable bounds for the range column (the game's
    stake/value ladder, ``stats_by_date.range_group.values``). The ladder is a
    stable product property, so declaring it beats deriving it from whatever
    values happen to exist in the loaded window; when empty the frontend falls
    back to the distinct data values from /group-values."""
    rg = stats_by_date_cfg(cfg).get("range_group") or {}
    return [float(v) for v in (rg.get("values") or [])]


def lifecycle_col(cfg: dict[str, Any]) -> Optional[str]:
    """The user-group column custom lifecycle groups redefine, or None when the
    config has no lifecycle cohort dimension."""
    return LIFECYCLE_COL if LIFECYCLE_COL in user_group_cols(cfg) else None


def derived_group_cols(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Virtual cohort columns rolled up from a stored column at load time.

    Dashboard feature: extra cohort pickers that combine stored subgroups
    without an ETL column or backfill — e.g. ss03's "ai_group" picker (AI vs
    non-AI) rolled up from the stored ``ab_group`` labels. Configured as
    ``stats_by_date.derived_group_cols``::

        derived_group_cols:
          ai_group:
            source: ab_group
            groups:
              AI: [AI]
            default: non-AI

    Behavior: each ``groups`` entry maps a derived label to the source values
    it combines; ``default`` labels every unmatched value (omit it to leave
    them null, i.e. only in 'all'). The column is attached in ``load_lazy``,
    so every endpoint — pickers, series, deep-dive — sees it as if stored.
    List the name in ``user_group_cols`` to surface the picker.
    """
    raw = stats_by_date_cfg(cfg).get("derived_group_cols") or {}
    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        source = str((spec or {}).get("source") or "")
        raw_groups = (spec or {}).get("groups") or {}
        groups = {str(label): [str(v) for v in (vals or [])] for label, vals in raw_groups.items()}
        if not source or not groups:
            continue
        default = (spec or {}).get("default")
        out[str(name)] = {"source": source, "groups": groups, "default": None if default is None else str(default)}
    return out


def attach_derived_group_cols(cfg: dict[str, Any], lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add each configured derived group column to the lazy frame (see
    ``derived_group_cols``). Skips a column whose source is missing, and never
    overwrites one that already exists in the data — so an ETL can later
    materialize the same column and the config keeps working unchanged."""
    derived = derived_group_cols(cfg)
    if not derived:
        return lf
    available = set(lf.collect_schema().names())
    exprs = []
    for name, spec in derived.items():
        if name in available or spec["source"] not in available:
            continue
        expr = pl.lit(spec["default"], dtype=pl.String)
        source = pl.col(spec["source"]).cast(pl.String)
        for label, values in spec["groups"].items():
            expr = pl.when(source.is_in(values)).then(pl.lit(label)).otherwise(expr)
        exprs.append(expr.alias(name))
    return lf.with_columns(exprs) if exprs else lf


def combo_group_cols(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Virtual cohort columns labeling each (period, user) by the COMBINATION
    of a grain column's values the user actually hit that period.

    Dashboard feature: ss03-AI's "mathtable_combo" picker — each user-day is
    labeled with the mathtables the user played that day, ordered by the
    user's bet count on each (dominant first, ties alphabetical), e.g. a day
    on normal_zero + normal_shi + normal_ichi → ``ai_zero_shi_ichi``.
    Configured as ``stats_by_date.combo_group_cols``::

        combo_group_cols:
          mathtable_combo:
            source: mathtable        # grain column whose values combine
            weight: user_num_bets    # per-row count that orders the label
            strip_prefix: "normal_"  # optional: shorten each value
            label_prefix: "ai"       # optional: prepended as "<prefix>_"
            order: user_first_spin_id  # optional: tie-break = first played
            max_tables: 4            # optional: cap label length, '+' suffix

    Behavior: attached in ``load_lazy`` AFTER ``filters``, so only the
    config's slice counts toward a user's combination. Equal-weight ties
    order by the ``order`` column's per-value minimum (first appearance in
    the period) when configured and stored, else alphabetically. With
    ``max_tables`` only the first N values name the label and any longer
    combination gets a trailing ``+``. Every row of a user's period carries
    the same label, so selecting a combo cohort keeps whole user-periods and
    the grain collapse combines the member mathtables' stats per user. At
    week/month granularity the combination spans the whole period, so labels
    are generally longer than the daily ones.
    """
    raw = stats_by_date_cfg(cfg).get("combo_group_cols") or {}
    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        source = str((spec or {}).get("source") or "")
        weight = str((spec or {}).get("weight") or "")
        if not source or not weight:
            continue
        out[str(name)] = {
            "source": source,
            "weight": weight,
            "strip_prefix": str((spec or {}).get("strip_prefix") or ""),
            "label_prefix": str((spec or {}).get("label_prefix") or ""),
            "order": str((spec or {}).get("order") or ""),
            "max_tables": int((spec or {}).get("max_tables") or 0),
        }
    return out


def attach_combo_group_cols(cfg: dict[str, Any], lf: pl.LazyFrame, date_col: str) -> pl.LazyFrame:
    """Add each configured combination group column (see ``combo_group_cols``).
    Skips a column whose inputs are missing and never overwrites a stored one.

    The window partitions include the hive ``period`` column (redundant with
    date_col — one period per date value) because polars only pushes the
    window-cache period/date predicates down to the parquet scan when they
    reference partition keys; without it every request would re-read the
    full history instead of the pruned window."""
    specs = combo_group_cols(cfg)
    if not specs:
        return lf
    available = set(lf.collect_schema().names())
    if "user_id" not in available or date_col not in available:
        return lf
    keys = (["period"] if "period" in available else []) + [date_col, "user_id"]
    for name, spec in specs.items():
        if name in available or spec["source"] not in available or spec["weight"] not in available:
            continue
        val = pl.col(spec["source"]).cast(pl.String)
        if spec["strip_prefix"]:
            val = val.str.strip_prefix(spec["strip_prefix"])
        vcol, wcol, ocol = f"_combo_val_{name}", f"_combo_w_{name}", f"_combo_o_{name}"
        weight = pl.col(spec["weight"]).sum().over(keys + [spec["source"]])
        # Tie-break: the value's first appearance in the period when the order
        # column is configured AND stored (older datasets lack it -> falls
        # back to the alphabetical tie via vcol below).
        use_order = bool(spec["order"]) and spec["order"] in available
        pos = pl.col(spec["order"]).min().over(keys + [spec["source"]]) if use_order else pl.lit(0)
        lf = lf.with_columns(val.alias(vcol), weight.alias(wcol), pos.alias(ocol))
        ordered = pl.col(vcol).sort_by([wcol, ocol, vcol], descending=[True, False, False]).unique(maintain_order=True)
        max_tables = spec["max_tables"]
        if max_tables > 0:
            label = ordered.head(max_tables).str.join("_").over(keys) + pl.when(
                pl.col(vcol).n_unique().over(keys) > max_tables
            ).then(pl.lit("+")).otherwise(pl.lit(""))
        else:
            label = ordered.str.join("_").over(keys)
        if spec["label_prefix"]:
            label = pl.lit(spec["label_prefix"] + "_") + label
        lf = lf.with_columns(label.alias(name)).drop([vcol, wcol, ocol])
    return lf


def row_filters(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Config-declared row filters restricting a whole dashboard to a slice of
    the stored parquet.

    Dashboard feature: cohort-scoped configs — e.g. the "SS03 AI" dashboard
    reads the same daily_stats parquet as the full ss03 config but keeps only
    ``ab_group = AI`` rows, so every picker, series, deep-dive and the 'all'
    cohort describe just that slice. Configured as ``stats_by_date.filters``::

        filters:
          ab_group: [AI]

    Behavior: each entry keeps rows whose column (stored or derived, compared
    as strings) matches one of the listed values; multiple entries AND
    together. Applied in ``load_lazy``, before any caching or aggregation.
    """
    raw = stats_by_date_cfg(cfg).get("filters") or {}
    out: dict[str, list[str]] = {}
    for col, values in raw.items():
        vals = values if isinstance(values, list) else [values]
        cleaned = [str(v) for v in vals if v is not None]
        if cleaned:
            out[str(col)] = cleaned
    return out


def apply_row_filters(cfg: dict[str, Any], lf: pl.LazyFrame) -> pl.LazyFrame:
    """Apply each configured row filter (see ``row_filters``) to the lazy
    frame. A filter column missing from the data is an error, not a skip —
    silently dropping the filter would serve the whole game's rows under a
    config that promises a slice of them."""
    filters = row_filters(cfg)
    if not filters:
        return lf
    available = set(lf.collect_schema().names())
    for col, values in filters.items():
        if col not in available:
            raise SeriesError(f"stats_by_date.filters column '{col}' not found in the data")
        lf = lf.filter(pl.col(col).cast(pl.String).is_in(values))
    return lf


# Per-config first-bet map. Derived, not stored: the daily parquet keeps the
# game's full per-user-per-day history, so min(activity_date | user bet that
# day) reproduces the ETL's Redshift-side first_bet_date (validated ≥99.97%
# per game, 100% on ss02/ss03/ss06). Cached because it scans the full daily
# history; the TTL picks up the daily ETL refresh.
_FIRST_BET_TTL_S = 900
_first_bet_cache: dict[str, tuple[float, pl.DataFrame]] = {}
_first_bet_lock = threading.Lock()


def load_first_bet_dates(cfg: dict[str, Any]) -> pl.DataFrame:
    """Per-user first bet date (columns: user_id, first_bet_date) derived from
    the config's DAILY user rows, shared across granularities and requests."""
    config_id = cfg.get("id")
    if not config_id:
        raise SeriesError("config dict is missing 'id'; cannot safely cache first-bet dates")
    with _first_bet_lock:
        hit = _first_bet_cache.get(config_id)
        if hit and time.monotonic() - hit[0] < _FIRST_BET_TTL_S:
            return hit[1]
        lf, date_col = load_lazy(cfg, "day")
        cols = set(lf.collect_schema().names())
        if "user_id" not in cols:
            raise SeriesError("Lifecycle groups need per-user daily rows ('user_id' column)")
        # Rows can predate the first bet (active days with zero bets), so gate
        # the min on the user actually betting when the column is available.
        if "user_num_bets" in cols:
            lf = lf.filter(pl.col("user_num_bets") > 0)
        df = (
            lf.select("user_id", pl.col(date_col).cast(pl.Datetime, strict=False).alias("first_bet_date"))
            .group_by("user_id")
            .agg(pl.col("first_bet_date").min())
            .collect()
        )
        logger.info("first-bet cache: derived %s users=%d", config_id, df.height)
        _first_bet_cache[config_id] = (time.monotonic(), df)
        return df


def attach_lifecycle_periods(cfg: dict[str, Any], df: pl.DataFrame, date_col: str, granularity: str) -> pl.DataFrame:
    """Add PERIODS_COL = the row's offset from the user's first bet, in units of
    the granularity (null for users never seen betting in the daily data).

    day   → whole days between first bet and the row's date.
    week  → calendar-week index: both dates truncate to their Monday week start
            (matching the ETL's Redshift ``DATE_TRUNC('week', …)``), so 0 is the
            week the user first bet, 1 the next week, … A weekly row aggregates
            the whole week, so units finer than a week would slice users by
            start weekday, not by age.
    month → calendar-month index, same reasoning.

    Negatives clamp to 0 (a rare pre-first-bet activity day — the ETL's
    ``DATEDIFF <= 3`` CASE also labels those 'new'). Nulls stay null, so
    never-bet users match only 'all'."""
    joined = df.join(load_first_bet_dates(cfg), on="user_id", how="left")
    d = pl.col(date_col).cast(pl.Datetime, strict=False)
    fb = pl.col("first_bet_date")
    if granularity == "week":
        periods = (d.dt.truncate("1w") - fb.dt.truncate("1w")).dt.total_days() // 7
    elif granularity == "month":
        periods = (d.dt.year() - fb.dt.year()) * 12 + (d.dt.month().cast(pl.Int32) - fb.dt.month().cast(pl.Int32))
    else:
        periods = (d - fb).dt.total_days()
    periods = pl.when(periods < 0).then(pl.lit(0)).otherwise(periods)
    return joined.with_columns(periods.alias(PERIODS_COL)).drop("first_bet_date")


def _scan_source(path: str) -> pl.LazyFrame:
    """One lazy scan per configured source path.

    A single ``scan_parquet`` over the expanded file list (hive-aware when the
    layout has ``period=`` directories) reads ONE footer for the schema and
    lets polars prune partitions by path. The previous per-file scan resolved
    each file's schema eagerly — one blocking S3 round-trip per file, which
    made request latency scale with a game's history length (ss01's ~250
    daily periods) rather than its data size.
    """
    if is_s3_path(path):
        files = expand_paths_to_files([path])
    elif Path(path).is_dir():
        files = sorted(str(p) for p in Path(path).rglob("*.parquet"))
    else:
        return pl.scan_parquet(path)
    if not files:
        return pl.DataFrame().lazy()
    hive = any("period=" in f for f in files)
    return pl.scan_parquet(files, hive_partitioning=hive)


def load_lazy(cfg: dict[str, Any], granularity: str) -> tuple[pl.LazyFrame, str]:
    """Lazily open the user-level parquet for a granularity; return (lf, date_col)."""
    sd = stats_by_date_cfg(cfg)
    date_col = str(sd.get("date_col", "activity_date"))
    files = (sd.get("files") or {}).get(granularity)
    if not files:
        raise SeriesError(f"No stats_by_date files configured for granularity '{granularity}'")
    scans = [_scan_source(str(p)) for p in (files if isinstance(files, list) else [files])]
    lf = scans[0] if len(scans) == 1 else pl.concat(scans, how="diagonal_relaxed")
    lf = apply_row_filters(cfg, attach_derived_group_cols(cfg, lf))
    return attach_combo_group_cols(cfg, lf, date_col), date_col


def to_iso_date(value: Any) -> str:
    """Render a polars date/datetime cell as an ISO date string (YYYY-MM-DD)."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def clean_floats(values: Any) -> list[Optional[float]]:
    """numpy array / iterable → JSON-safe list, mapping NaN/inf to None."""
    out: list[Optional[float]] = []
    for v in values:
        f = float(v)
        out.append(None if math.isnan(f) or math.isinf(f) else f)
    return out


def cohort_values(group_values: dict[str, list[str]], col: Optional[str]) -> list[str]:
    """Selected values for a cohort column, de-duped.

    An ABSENT column means ['all'] — callers need not enumerate every cohort
    dimension. A column PRESENT with an empty selection means no cohorts at
    all: the caller explicitly unselected everything, so nothing is plotted
    (the cohort cross-product becomes empty). Matches the pickers' semantics.
    """
    if not col:
        return ["all"]
    if col not in group_values or group_values[col] is None:
        return ["all"]
    seen: set[str] = set()
    vals: list[str] = []
    for v in group_values[col]:
        s = str(v)
        if s and s not in seen:
            seen.add(s)
            vals.append(s)
    return vals


def cohort_label(values: list[str]) -> str:
    """Display label for one cohort combination: the non-'all' dimension values
    joined with ' | ' ('all' when every dimension is unselected)."""
    parts = [v for v in values if v != "all"]
    return " | ".join(parts) if parts else "all"


# Additive user-row components: SUM when collapsing grain rows to one row per
# user. Formulas mirror the ETL's user_stats SELECT.
_USER_ROW_SUMS = [
    "user_num_bets",
    "user_num_bets_bg",
    "user_num_bets_fg",
    "user_total_bet",
    "user_total_bet_bg",
    "user_total_bet_fg",
    "user_total_payout",
    "user_total_payout_bg",
    "user_total_payout_fg",
    "user_num_bets_with_payout",
    "user_num_bets_bg_with_payout",
    "user_num_bets_fg_with_payout",
    "user_mathtable_change",
    "user_accu_pos_delta_bet",
    "user_accu_neg_delta_bet",
    "user_accu_delta_bet",
    "user_pos_delta_bet_num",
    "user_neg_delta_bet_num",
    "user_num_delta_bet",
    "user_num_delta_t_bg",
    "user_num_delta_t",
    "user_num_killed_bullets",
    "user_total_profit",
]

# Columns combined with MAX when collapsing (peak values / boolean flags).
_USER_ROW_MAXES = [
    "user_max_profit",
    "user_killed_fish",
]

# Mean columns recombined as weighted means: (column, weight column). The
# weight is the exact denominator the ETL used for the AVG.
_USER_ROW_WEIGHTED_MEANS = [
    ("user_avg_delta_t_seconds_bg", "user_num_delta_t_bg"),
    ("user_avg_delta_t_seconds", "user_num_delta_t"),
    ("user_avg_fish_value", "user_num_bets"),
    ("user_avg_killed_fish_value", "user_num_killed_bullets"),
    ("user_bullet_kill_avg_profit", "user_num_killed_bullets"),
]

# Ratio columns recomputed from the summed components: (name, numerator,
# denominator), null when the denominator is 0.
_USER_ROW_RATIOS = [
    ("user_avg_bet_amount", "user_total_bet", "user_num_bets"),
    ("user_rtp", "user_total_payout", "user_total_bet"),
    ("user_rtp_bg", "user_total_payout_bg", "user_total_bet"),
    ("user_rtp_fg", "user_total_payout_fg", "user_total_bet_fg"),
    ("user_hit_rate", "user_num_bets_with_payout", "user_num_bets"),
    ("user_hit_rate_bg", "user_num_bets_bg_with_payout", "user_num_bets_bg"),
    ("user_hit_rate_fg", "user_num_bets_fg_with_payout", "user_num_bets_fg"),
    ("user_fg_ratio", "user_num_bets_fg", "user_num_bets"),
    ("user_accu_pos_delta_bet_avg", "user_accu_pos_delta_bet", "user_pos_delta_bet_num"),
    ("user_accu_neg_delta_bet_avg", "user_accu_neg_delta_bet", "user_neg_delta_bet_num"),
    ("user_accu_delta_bet_avg", "user_accu_delta_bet", "user_num_delta_bet"),
    ("user_bullet_avg_profit", "user_total_profit", "user_num_bets"),
    ("user_bullet_kill_ratio", "user_num_killed_bullets", "user_num_bets"),
]


def collapse_user_rows(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Re-aggregate grain rows (one per user × grain combination) to one row
    per (period, user): additive components are summed and the derived ratios
    recomputed from the sums — the same math the ETL uses — so per-user
    means/CIs/histograms see each user exactly once. Columns outside the
    mapping keep their first value; grain columns should be dropped upstream.
    """
    if df.is_empty() or "user_id" not in df.columns:
        return df
    cols = set(df.columns)

    # Weighted means: pre-multiply mean × weight so the sums recombine exactly.
    wmeans = [(m, w) for m, w in _USER_ROW_WEIGHTED_MEANS if m in cols and w in cols]
    if wmeans:
        df = df.with_columns([(pl.col(m) * pl.col(w)).alias(f"_wm_{m}") for m, w in wmeans])
        cols = set(df.columns)

    keys = [date_col, "user_id"]
    sums = [c for c in _USER_ROW_SUMS if c in cols] + [f"_wm_{m}" for m, _ in wmeans]
    maxes = [c for c in _USER_ROW_MAXES if c in cols]
    ratio_names = {name for name, _, _ in _USER_ROW_RATIOS}
    combined = set(keys) | set(sums) | set(maxes) | ratio_names
    passthrough = [c for c in df.columns if c not in combined]

    # SQL semantics for the sums: all-null stays null (a user with no BASE
    # bets keeps user_total_bet_bg = null, excluded from per-user means, not 0).
    out = df.group_by(keys).agg(
        [pl.when(pl.col(c).is_not_null().any()).then(pl.col(c).sum()).otherwise(None).alias(c) for c in sums]
        + [pl.col(c).max() for c in maxes]
        + [pl.col(c).first() for c in passthrough]
    )
    derived = [
        pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(name)
        for name, num, den in _USER_ROW_RATIOS
        if num in out.columns and den in out.columns
    ] + [pl.when(pl.col(w) > 0).then(pl.col(f"_wm_{m}") / pl.col(w)).otherwise(None).alias(m) for m, w in wmeans]
    if derived:
        out = out.with_columns(derived)
    return out.drop([f"_wm_{m}" for m, _ in wmeans])


def apply_cohort(df: pl.DataFrame, col: Optional[str], value: str) -> pl.DataFrame:
    """Filter to rows where col == value; 'all' (or missing col) means no filter."""
    if not col or value == "all" or col not in df.columns:
        return df
    return df.filter(pl.col(col) == value)


def iter_cohorts(
    cfg: dict[str, Any],
    df: pl.DataFrame,
    group_values: dict[str, list[str]],
    lifecycle: Optional[list[dict[str, Any]]] = None,
    date_col: Optional[str] = None,
    granularity: str = "day",
    range_groups: Optional[list[dict[str, Any]]] = None,
) -> Iterator[tuple[str, pl.DataFrame]]:
    """Yield (cohort_label, cohort_df) over the cross-product of selected
    user-group values — the same cohorts the legacy tabs overlay. Filtering
    happens before any DataMetrics aggregation, so each cohort's metrics are
    computed on just its rows.

    Dashboard feature: the global "Lifecycle groups" picker above the tab bar.
    When ``lifecycle`` is given (dicts with label/start/end forming the
    half-open range [start, end), end None = open-ended, units =
    ``granularity`` periods since first bet), it redefines the LIFECYCLE_COL
    dimension: its labels become the selection for that column and each cohort
    filters rows by period range instead of the stored ETL label. A group
    labeled "all" means no filter. Ranges may overlap — each group is an
    independent filter, not a partition.
    """
    cols = effective_cohort_cols(cfg)
    lc = lifecycle_col(cfg) if lifecycle else None
    by_label = {str(g["label"]): g for g in lifecycle or []}
    grain = user_row_grain(cfg)
    rcol, _, _ = range_group_cfg(cfg)
    range_by_label = {str(g["label"]): g for g in range_groups or []} if rcol else {}
    # Attach the derived period offsets whenever the config has a lifecycle
    # dimension (DataMetrics needs them for num_new_users even when no
    # lifecycle groups are selected); lc only gates the lifecycle FILTERS.
    if lifecycle_col(cfg) is not None and date_col and "user_id" in df.columns:
        df = attach_lifecycle_periods(cfg, df, date_col, granularity)
    else:
        lc = None
    if lc is not None and lc not in cols:
        lc = None

    def values(col: str) -> list[str]:
        if col == lc:
            return list(by_label) or ["all"]
        if col == rcol and range_by_label:
            return list(range_by_label)
        return cohort_values(group_values, col)

    def apply(df_: pl.DataFrame, col: str, value: str) -> pl.DataFrame:
        if col == lc and value != "all":
            g = by_label[value]
            cond = pl.col(PERIODS_COL) >= int(g["start"])
            if g.get("end") is not None:
                cond = cond & (pl.col(PERIODS_COL) < int(g["end"]))
            return df_.filter(cond)
        if col == rcol and value in range_by_label and value != "all":
            g = range_by_label[value]
            cond = pl.col(col) >= float(g["min"])
            if g.get("max") is not None:
                cond = cond & (pl.col(col) <= float(g["max"]))
            return df_.filter(cond)
        return apply_cohort(df_, col, value)

    if not cols:
        yield "all", df
        return

    seen: set[str] = set()
    for combo in itertools.product(*(values(c) for c in cols)):
        label = cohort_label(list(combo))
        if label in seen:
            continue
        seen.add(label)
        df_c = df
        for col, value in zip(cols, combo):
            df_c = apply(df_c, col, value)
        # Grain rows (one per user × grain combination) collapse back to one
        # row per user whenever a grain dimension is unselected, so per-user
        # stats see each user once; with every grain column pinned the rows
        # are already per-user.
        # The range column never pins to a single stored value (a range spans
        # multiple grain rows per user), so it always requires the collapse.
        selected = dict(zip(cols, combo))
        if grain and date_col and any(selected.get(g, "all") == "all" or g == rcol for g in grain):
            df_c = collapse_user_rows(df_c.drop([g for g in grain if g in df_c.columns]), date_col)
        yield label, df_c
