"""Distribution summaries for the Stats-by-Group tab.

Parity with the legacy Dash tab (``game_stats_monitor.py`` ~L4885-5079): for a
single metric, compare its distribution across cohorts and up to three date
ranges. We return per (cohort × range) a 5-number summary plus mean / bootstrap
CI, so the frontend can render either a box plot or a bar (mean ± 95% CI)
without a refetch. Values are read from the same ``DataMetrics`` engine as
Stats-by-Date (group-level for computed metrics, per-user rows for ``user_*``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import polars as pl

from bituslabs_ds.metrics.user_stats_aggregates import RETENTION_LOAD_EXTRA_DAYS, DataMetrics, _bootstrap_ci
from dashboard_api.services.common import (
    SeriesError,
    collect_window,
    iter_cohorts,
    load_lazy,
    projection_columns,
    stats_by_date_cfg,
)
from dashboard_api.services.reshape import clip_values, filter_values

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 500


def _opt(v: float) -> Optional[float]:
    return None if (np.isnan(v) or np.isinf(v)) else float(v)


def _summary(cohort: str, range_label: str, range_index: int, vals: np.ndarray) -> dict[str, Any]:
    lo, hi = _bootstrap_ci(vals, n_boot=N_BOOTSTRAP) if len(vals) >= 2 else (float("nan"), float("nan"))
    return {
        "cohort": cohort,
        "range_label": range_label,
        "range_index": range_index,
        "n": int(len(vals)),
        "min": _opt(float(np.min(vals))),
        "q1": _opt(float(np.percentile(vals, 25))),
        "median": _opt(float(np.median(vals))),
        "q3": _opt(float(np.percentile(vals, 75))),
        "max": _opt(float(np.max(vals))),
        "mean": _opt(float(np.mean(vals))),
        "std": _opt(float(np.std(vals))),
        "ci_lower": _opt(lo),
        "ci_upper": _opt(hi),
    }


def load_group_distribution(
    cfg: dict[str, Any],
    granularity: str,
    metric: str,
    ranges: list[tuple[Optional[str], Optional[str]]],
    group_values: Optional[dict[str, list[str]]] = None,
    lifecycle: Optional[list[dict[str, Any]]] = None,
    range_groups: Optional[list[dict[str, Any]]] = None,
    clip_enable: bool = False,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    clip_percentile: bool = False,
    filter_enable: bool = False,
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    filter_percentile: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (stats, missing). One stat dict per (cohort × non-empty range)."""
    group_values = group_values or {}
    sd = stats_by_date_cfg(cfg)
    lf, date_col = load_lazy(cfg, granularity)
    if date_col not in set(lf.collect_schema().names()):
        raise SeriesError(f"Date column '{date_col}' not found in source data")
    lf = lf.with_columns(pl.col(date_col).cast(pl.Datetime, strict=False))

    # Keep only ranges with both endpoints set; remember their original index.
    parsed: list[tuple[int, datetime, datetime, str]] = []
    for i, (start, end) in enumerate(ranges):
        if start and end:
            s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
            parsed.append((i, s, e, f"{start} → {end}"))
    if not parsed:
        return [], True

    overall_start = min(s for _, s, _, _ in parsed)
    overall_end = max(e for _, _, e, _ in parsed)
    aggregate_from_rows = bool(sd.get("aggregate_stats_from_user_rows", False))
    load_end = overall_end + timedelta(days=RETENTION_LOAD_EXTRA_DAYS) if aggregate_from_rows else overall_end

    # Shared + single-flight with the other tabs' requests (see common.py —
    # parallel per-panel collects OOM-killed the task in production). Projected
    # to this metric's columns, like the series path.
    columns = projection_columns(cfg, [metric], date_col, set(lf.collect_schema().names()))
    df_raw = collect_window(cfg, granularity, lf, date_col, overall_start, load_end, columns=columns)
    if df_raw.is_empty():
        return [], True

    stats: list[dict[str, Any]] = []
    produced = False
    for label, df_c in iter_cohorts(
        cfg,
        df_raw,
        group_values,
        lifecycle=lifecycle,
        date_col=date_col,
        granularity=granularity,
        range_groups=range_groups,
    ):
        dm = DataMetrics(
            df_c,
            start_dt=overall_start,
            end_dt=overall_end,
            key_cols=[date_col],
            granularity=granularity,
            presence_df=df_raw,  # full population for any-group retention
        )
        metric_df = dm[metric]
        if metric_df is None:
            continue
        for range_index, start_dt, end_dt, range_label in parsed:
            window = metric_df.filter((pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt))
            vals = window.get_column(metric).drop_nulls().to_numpy().astype(float)
            vals = filter_values(vals, filter_enable, filter_min, filter_max, filter_percentile)
            vals = clip_values(vals, clip_enable, clip_min, clip_max, clip_percentile)
            if len(vals) == 0:
                continue
            produced = True
            stats.append(_summary(label, range_label, range_index, vals))

    return stats, not produced
