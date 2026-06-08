"""Histogram + correlation data for the Deep Dive tab.

Parity with the legacy Dash tab (``game_stats_monitor.py`` ~L5105-5684), scoped
to the histogram and heatmap modes (scatter deferred). Two panels: ``derived``
reads group-level ``DataMetrics`` values (one per date), ``user`` reads raw
``user_*`` per-user rows. We compute aggregates server-side so payloads stay
small: histograms are pre-binned (``np.histogram``) and the heatmap is a Pearson
correlation matrix (``np.corrcoef``).

Deferred vs legacy (need scipy, intentionally absent from the light image):
p-value significance stars and the Yeo-Johnson per-metric transform.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import polars as pl

from dashboard_api.services.common import SeriesError, iter_cohorts, load_lazy, stats_by_date_cfg
from dashboards.user_stats_aggregates import RETENTION_LOAD_EXTRA_DAYS, DataMetrics

logger = logging.getLogger(__name__)


def _clip(vals: np.ndarray, enable: bool, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    if enable and (lo is not None or hi is not None):
        return np.clip(vals, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
    return vals


def _metric_values(
    panel: str,
    metric: str,
    dm: Optional[DataMetrics],
    df_win: pl.DataFrame,
    date_col: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Optional[np.ndarray]:
    """Values of one metric within a (cohort, range) window. Derived metrics come
    from DataMetrics (group-level); user metrics are raw columns of df_win."""
    if panel == "derived":
        if dm is None:
            return None
        mdf = dm[metric]
        if mdf is None:
            return None
        w = mdf.filter((pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt))
        return w.get_column(metric).drop_nulls().to_numpy().astype(float)
    if metric not in df_win.columns:
        return None
    return df_win.get_column(metric).drop_nulls().to_numpy().astype(float)


def load_deepdive(
    cfg: dict[str, Any],
    granularity: str,
    panel: str,
    mode: str,
    metrics: list[str],
    ranges: list[tuple[Optional[str], Optional[str]]],
    group_values: Optional[dict[str, list[str]]] = None,
    clip_enable: bool = False,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    nbins: int = 50,
    normalize: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return (histograms, heatmaps, missing)."""
    group_values = group_values or {}
    sd = stats_by_date_cfg(cfg)
    lf, date_col = load_lazy(cfg, granularity)
    available = set(lf.collect_schema().names())
    if date_col not in available:
        raise SeriesError(f"Date column '{date_col}' not found in source data")
    lf = lf.with_columns(pl.col(date_col).cast(pl.Datetime, strict=False))

    parsed: list[tuple[int, datetime, datetime, str]] = []
    for i, (start, end) in enumerate(ranges):
        if start and end:
            parsed.append((i, datetime.fromisoformat(start), datetime.fromisoformat(end), f"{start} → {end}"))
    if not parsed:
        return [], [], list(metrics)

    overall_start = min(s for _, s, _, _ in parsed)
    overall_end = max(e for _, _, e, _ in parsed)
    aggregate_from_rows = bool(sd.get("aggregate_stats_from_user_rows", False))
    load_end = overall_end + timedelta(days=RETENTION_LOAD_EXTRA_DAYS) if aggregate_from_rows else overall_end

    df_raw = lf.filter((pl.col(date_col) >= overall_start) & (pl.col(date_col) <= load_end)).collect()
    if df_raw.is_empty():
        return [], [], list(metrics)

    histograms: list[dict[str, Any]] = []
    heatmaps: list[dict[str, Any]] = []
    produced: set[str] = set()

    for label, df_c in iter_cohorts(cfg, df_raw, group_values):
        dm = (
            DataMetrics(df_c, start_dt=overall_start, end_dt=overall_end, key_cols=[date_col], granularity=granularity)
            if panel == "derived"
            else None
        )
        for range_index, start_dt, end_dt, range_label in parsed:
            df_win = df_c.filter((pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt))
            if df_win.is_empty():
                continue

            # Collect each metric's values once; reused by both modes.
            per_metric: dict[str, np.ndarray] = {}
            for m in metrics:
                vals = _metric_values(panel, m, dm, df_win, date_col, start_dt, end_dt)
                if vals is None or len(vals) == 0:
                    continue
                per_metric[m] = _clip(vals, clip_enable, clip_min, clip_max)
                produced.add(m)

            if mode == "histogram":
                for m, vals in per_metric.items():
                    counts, edges = np.histogram(vals, bins=nbins, density=normalize)
                    histograms.append(
                        {
                            "metric": m,
                            "cohort": label,
                            "range_label": range_label,
                            "range_index": range_index,
                            "bin_edges": [float(x) for x in edges.tolist()],
                            "counts": [float(x) for x in counts.tolist()],
                        }
                    )
            else:  # heatmap: Pearson correlation across metrics with aligned observations
                cols = [m for m in metrics if m in per_metric]
                matrix = _correlation_matrix(panel, cols, dm, df_win, date_col, start_dt, end_dt)
                if matrix is not None:
                    heatmaps.append(
                        {
                            "cohort": label,
                            "range_label": range_label,
                            "range_index": range_index,
                            "metrics": cols,
                            "corr": matrix,
                        }
                    )

    return histograms, heatmaps, [m for m in metrics if m not in produced]


def _correlation_matrix(
    panel: str,
    cols: list[str],
    dm: Optional[DataMetrics],
    df_win: pl.DataFrame,
    date_col: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Optional[list[list[Optional[float]]]]:
    """Pearson correlation matrix across `cols`, on observations aligned by the
    natural row key (date for derived, user row for user). None if < 2 metrics."""
    if len(cols) < 2:
        return None
    if panel == "derived":
        frame: Optional[pl.DataFrame] = None
        for m in cols:
            mdf = dm[m] if dm is not None else None
            if mdf is None:
                continue
            w = mdf.filter((pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt)).select([date_col, m])
            frame = w if frame is None else frame.join(w, on=date_col, how="inner")
        if frame is None:
            return None
        obs = frame.select(cols)
    else:
        obs = df_win.select([c for c in cols if c in df_win.columns])

    obs = obs.drop_nulls()
    if obs.height < 2 or obs.width < 2:
        return None
    arr = obs.to_numpy().astype(float)
    corr = np.corrcoef(arr, rowvar=False)
    return [[None if np.isnan(v) or np.isinf(v) else float(v) for v in row] for row in corr]


def load_deepdive_metrics(cfg: dict[str, Any], granularity: str) -> tuple[list[str], list[str]]:
    """Return (derived, user) metric names for the Deep Dive pickers, scoped to
    THIS game: the union of the config's ``group{1,2,3}_columns`` (the curated
    metric lists), split into computed DataMetrics (derived) vs the rest
    (user-level / raw). This keeps the pickers game-specific rather than listing
    every metric the engine could compute."""
    sd = stats_by_date_cfg(cfg)
    columns: list[str] = []
    for key in ("group1_columns", "group2_columns", "group3_columns"):
        for c in sd.get(key) or []:
            c = str(c)
            if c not in columns:
                columns.append(c)
    derived = [c for c in columns if c in DataMetrics.METRICS]
    user = [c for c in columns if c not in DataMetrics.METRICS]
    return derived, user
