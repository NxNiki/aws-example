"""Compute metric time-series for the Stats-by-Date tab from the S3 parquet cache.

Parity with the legacy Dash tab (``dashboards/game_stats_monitor.py`` data flow,
~L4436-4530): the source parquet is one-row-per-user, and group-level metrics
(active users, RTP, retention, …) are recomputed from those rows via the shared
``DataMetrics`` engine — NOT read pre-aggregated. We return data series only;
all display concerns (dual-axis, hybrid-log, CI bands, weekend stripes, colors)
live in the frontend option-builder.

Per (requested metric × selected cohort) we emit one series with its own dates,
values, and CI bounds. Cohort filtering happens before ``DataMetrics`` so each
cohort's metrics are computed on just its rows (matching the legacy tab).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional, cast

import polars as pl

from bituslabs_ds.metrics.user_stats_aggregates import RETENTION_LOAD_EXTRA_DAYS, DataMetrics
from dashboard_api.services.common import (
    SeriesError,
    availability_cols,
    clean_floats,
    collect_window,
    iter_cohorts,
    load_lazy,
    projection_columns,
    stats_by_date_cfg,
    to_iso_date,
    user_group_cols,
)

logger = logging.getLogger(__name__)

__all__ = ["SeriesError", "load_series", "load_date_bounds", "load_group_values", "load_group_values_in_range"]


def _metric_kind(metric: str) -> str:
    if metric in DataMetrics.METRICS:
        return "computed"
    if metric.startswith("user_"):
        return "user"
    return "raw"


def _window_series(
    cfg: dict[str, Any],
    granularity: str,
    metrics: list[str],
    lf: pl.LazyFrame,
    date_col: str,
    start_dt: datetime,
    end_dt: datetime,
    group_values: dict[str, list[str]],
    lifecycle: Optional[list[dict[str, Any]]],
    range_groups: Optional[list[dict[str, Any]]],
    produced: set[str],
) -> list[dict[str, Any]]:
    """One window's series dicts (per metric × cohort); adds to ``produced``."""
    sd = stats_by_date_cfg(cfg)
    # Retention metrics need follow-up days beyond the window; extend the load
    # (not the displayed range) when user-row aggregation is enabled.
    aggregate_from_rows = bool(sd.get("aggregate_stats_from_user_rows", False))
    load_end = end_dt + timedelta(days=RETENTION_LOAD_EXTRA_DAYS) if aggregate_from_rows else end_dt

    # Shared + single-flight: the per-panel requests of one tab render all hit
    # the same window and must not each collect their own copy (OOM). The
    # collect is projected to the columns these metrics need — a few user_*
    # columns instead of the whole row — so frames stay small at any span.
    columns = projection_columns(cfg, metrics, date_col, set(lf.collect_schema().names()))
    df_raw = collect_window(cfg, granularity, lf, date_col, start_dt, load_end, columns=columns)
    if df_raw.is_empty():
        return []

    series: list[dict[str, Any]] = []
    for label, df_c in iter_cohorts(
        cfg,
        df_raw,
        group_values,
        lifecycle=lifecycle,
        date_col=date_col,
        granularity=granularity,
        range_groups=range_groups,
    ):
        df_c_date = df_c.filter((pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt))
        if df_c_date.is_empty():
            continue
        # presence_df=df_raw (full window, all groups) so any-group retention
        # counts returns even when a user's group changed across days.
        dm = DataMetrics(
            df_c, start_dt=start_dt, end_dt=end_dt, key_cols=[date_col], granularity=granularity, presence_df=df_raw
        )
        for metric in metrics:
            x_dates, y, lo, hi = dm.plot_values(metric, df_c_date)
            if len(x_dates) == 0:
                continue
            produced.add(metric)
            series.append(
                {
                    "metric": metric,
                    "cohort": label,
                    "kind": _metric_kind(metric),
                    "additive": metric in DataMetrics.ADDITIVE_METRICS,
                    "x": [to_iso_date(d) for d in x_dates],
                    "y": clean_floats(y),
                    "lower": clean_floats(lo),
                    "upper": clean_floats(hi),
                }
            )
    return series


def load_series(
    cfg: dict[str, Any],
    granularity: str,
    metrics: list[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    group_values: Optional[dict[str, list[str]]] = None,
    lifecycle: Optional[list[dict[str, Any]]] = None,
    range_groups: Optional[list[dict[str, Any]]] = None,
    ranges: Optional[list[tuple[Optional[str], Optional[str]]]] = None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Return (date_col, series, missing).

    ``series`` is one dict per (metric, cohort) with x/y/lower/upper + metadata;
    ``missing`` lists requested metrics that produced no data in any cohort.
    With ``ranges`` (the global Date-groups picker) one series is emitted per
    metric × cohort × range, tagged with range_index/range_label; each range
    collects its own window so far-apart ranges don't load the span between.
    """
    group_values = group_values or {}
    lf, date_col = load_lazy(cfg, granularity)

    available = set(lf.collect_schema().names())
    if date_col not in available:
        raise SeriesError(f"Date column '{date_col}' not found in source data")

    lf = lf.with_columns(pl.col(date_col).cast(pl.Datetime, strict=False))
    produced: set[str] = set()

    if ranges is not None:
        series: list[dict[str, Any]] = []
        for i, (start, end) in enumerate(ranges):
            if not (start and end):
                continue
            start_dt, end_dt = datetime.fromisoformat(start), datetime.fromisoformat(end)
            for s in _window_series(
                cfg,
                granularity,
                metrics,
                lf,
                date_col,
                start_dt,
                end_dt,
                group_values,
                lifecycle,
                range_groups,
                produced,
            ):
                s["range_index"] = i
                s["range_label"] = f"{start} → {end}"
                series.append(s)
        return date_col, series, [m for m in metrics if m not in produced]

    win_start: Optional[datetime] = datetime.fromisoformat(date_from) if date_from else None
    win_end: Optional[datetime] = datetime.fromisoformat(date_to) if date_to else None
    if win_start is None or win_end is None:
        bounds = lf.select(pl.col(date_col).min().alias("lo"), pl.col(date_col).max().alias("hi")).collect()
        win_start = win_start or bounds["lo"][0]
        win_end = win_end or bounds["hi"][0]
    if win_start is None or win_end is None:
        return date_col, [], list(metrics)

    series = _window_series(
        cfg, granularity, metrics, lf, date_col, win_start, win_end, group_values, lifecycle, range_groups, produced
    )
    return date_col, series, [m for m in metrics if m not in produced]


def load_date_bounds(cfg: dict[str, Any], granularity: str) -> tuple[Optional[str], Optional[str]]:
    """Min/max date present in the source data (ISO strings), for seeding the
    UI's default date window. Cheap: a single min/max scan, lazily."""
    lf, date_col = load_lazy(cfg, granularity)
    if date_col not in set(lf.collect_schema().names()):
        raise SeriesError(f"Date column '{date_col}' not found in source data")
    lf = lf.with_columns(pl.col(date_col).cast(pl.Datetime, strict=False))
    bounds = lf.select(pl.col(date_col).min().alias("lo"), pl.col(date_col).max().alias("hi")).collect()
    lo, hi = bounds["lo"][0], bounds["hi"][0]
    return (to_iso_date(lo) if lo is not None else None, to_iso_date(hi) if hi is not None else None)


def load_group_values(cfg: dict[str, Any], granularity: str) -> dict[str, list[str]]:
    """Distinct selectable values for each configured user_group column, for the
    cohort pickers. Reads the same parquet as the series endpoint."""
    cols = user_group_cols(cfg)
    if not cols:
        return {}
    lf, _ = load_lazy(cfg, granularity)
    available = set(lf.collect_schema().names())
    present = [c for c in cols if c in available]
    if not present:
        return {}
    df = lf.select(present).unique().collect()
    return {c: sorted(str(v) for v in df[c].drop_nulls().unique().to_list()) for c in present}


_PERIOD_RE = re.compile(r"period=(\d{4}-\d{2}-\d{2})")
# Availability scans widen left by one period so week/month rows whose period
# START precedes the range still count as present inside it.
_PERIOD_MARGIN_DAYS = {"day": 0, "week": 7, "month": 31}


def load_group_values_in_range(cfg: dict[str, Any], granularity: str, start: str, end: str) -> dict[str, list[str]]:
    """Distinct cohort values PRESENT in [start, end] — the availability set
    behind the pickers' grayed-out entries (full vocabulary comes from
    ``load_group_values``). Reads only the ``period=`` files whose partition
    date falls inside the (margin-widened) range, so it stays fast on datasets
    with hundreds of daily partitions; paths without a ``period=`` segment
    (flat layouts) are always read. Scans just the ``availability_cols``
    dimensions — the ones that rotate over time (e.g. mathtable, daily_group);
    static vocabularies don't need a per-range scan and the frontend enables
    values with no availability entry."""
    from bituslabs_ds.s3_utils import expand_paths_to_files, read_files

    cols = availability_cols(cfg)
    if not cols:
        return {}
    sd = stats_by_date_cfg(cfg)
    files = (sd.get("files") or {}).get(granularity)
    if not files:
        raise SeriesError(f"No stats_by_date files configured for granularity '{granularity}'")
    lo = date.fromisoformat(start) - timedelta(days=_PERIOD_MARGIN_DAYS.get(granularity, 31))
    hi = date.fromisoformat(end)
    keep = []
    for p in expand_paths_to_files(list(files)):
        m = _PERIOD_RE.search(p)
        if not m or lo <= date.fromisoformat(m.group(1)) <= hi:
            keep.append(p)
    if not keep:
        return {c: [] for c in cols}
    lf = cast(pl.LazyFrame, read_files(keep, lazy_load=True, expand_s3_prefixes=False))
    schema_names = set(lf.collect_schema().names())
    present = [c for c in cols if c in schema_names]
    if not present:
        return {}
    # Row-level date filter: path pruning only narrows hive (period=) layouts;
    # flat single-file datasets (e.g. fish_hunter daily_stats) need the rows
    # themselves bounded or availability degenerates to the full vocabulary.
    date_col = str(sd.get("date_col", "activity_date"))
    if date_col in schema_names:
        lf = lf.with_columns(pl.col(date_col).cast(pl.Datetime, strict=False)).filter(
            (pl.col(date_col) >= datetime.combine(lo, datetime.min.time()))
            & (pl.col(date_col) <= datetime.combine(hi, datetime.min.time()))
        )
    df = lf.select(present).unique().collect()
    return {c: sorted(str(v) for v in df[c].drop_nulls().unique().to_list()) for c in present}
