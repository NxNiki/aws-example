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
from datetime import datetime, timedelta
from typing import Any, Optional

import polars as pl

from bituslabs_ds.metrics.user_stats_aggregates import RETENTION_LOAD_EXTRA_DAYS, DataMetrics
from dashboard_api.services.common import (
    SeriesError,
    clean_floats,
    collect_window,
    iter_cohorts,
    load_lazy,
    stats_by_date_cfg,
    to_iso_date,
    user_group_cols,
)

logger = logging.getLogger(__name__)

__all__ = ["SeriesError", "load_series", "load_date_bounds", "load_group_values"]


def _metric_kind(metric: str) -> str:
    if metric in DataMetrics.METRICS:
        return "computed"
    if metric.startswith("user_"):
        return "user"
    return "raw"


def load_series(
    cfg: dict[str, Any],
    granularity: str,
    metrics: list[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    group_values: Optional[dict[str, list[str]]] = None,
    lifecycle: Optional[list[dict[str, Any]]] = None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Return (date_col, series, missing).

    ``series`` is one dict per (metric, cohort) with x/y/lower/upper + metadata;
    ``missing`` lists requested metrics that produced no data in any cohort.
    """
    group_values = group_values or {}
    sd = stats_by_date_cfg(cfg)
    lf, date_col = load_lazy(cfg, granularity)

    available = set(lf.collect_schema().names())
    if date_col not in available:
        raise SeriesError(f"Date column '{date_col}' not found in source data")

    lf = lf.with_columns(pl.col(date_col).cast(pl.Datetime, strict=False))

    start_dt = datetime.fromisoformat(date_from) if date_from else None
    end_dt = datetime.fromisoformat(date_to) if date_to else None
    if start_dt is None or end_dt is None:
        bounds = lf.select(pl.col(date_col).min().alias("lo"), pl.col(date_col).max().alias("hi")).collect()
        start_dt = start_dt or bounds["lo"][0]
        end_dt = end_dt or bounds["hi"][0]
    if start_dt is None or end_dt is None:
        return date_col, [], list(metrics)

    # Retention metrics need follow-up days beyond the window; extend the load
    # (not the displayed range) when user-row aggregation is enabled.
    aggregate_from_rows = bool(sd.get("aggregate_stats_from_user_rows", False))
    load_end = end_dt + timedelta(days=RETENTION_LOAD_EXTRA_DAYS) if aggregate_from_rows else end_dt

    # Shared + single-flight: the per-panel requests of one tab render all hit
    # the same window and must not each collect their own copy (OOM).
    df_raw = collect_window(cfg, granularity, lf, date_col, start_dt, load_end)
    if df_raw.is_empty():
        return date_col, [], list(metrics)

    series: list[dict[str, Any]] = []
    produced: set[str] = set()

    for label, df_c in iter_cohorts(
        cfg, df_raw, group_values, lifecycle=lifecycle, date_col=date_col, granularity=granularity
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

    missing = [m for m in metrics if m not in produced]
    return date_col, series, missing


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
