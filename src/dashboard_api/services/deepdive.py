"""Histogram, correlation, and scatter data for the Deep Dive tab.

Parity with the legacy Dash tab (``game_stats_monitor.py`` ~L5105-5920). Two
panels: ``derived`` reads group-level ``DataMetrics`` values (one per date),
``user`` reads raw ``user_*`` per-user rows. Aggregates are computed server-side
so payloads stay small: histograms are pre-binned (``np.histogram``), the heatmap
is a Pearson correlation matrix (``np.corrcoef``), and scatter points are
outlier-filtered and down-sampled to a cap.

Deferred vs legacy (need scipy/sklearn, intentionally absent from the light
image): heatmap p-value stars, the Yeo-Johnson per-metric transform, and symlog
axes. Scatter supports z-score outlier removal and log axes (frontend).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import polars as pl

from bituslabs_ds.metrics.user_stats_aggregates import RETENTION_LOAD_EXTRA_DAYS, DataMetrics
from dashboard_api.services.common import SeriesError, collect_window, iter_cohorts, load_lazy, stats_by_date_cfg
from dashboard_api.services.reshape import clip_columns, clip_values, filter_row_mask, filter_values

logger = logging.getLogger(__name__)

# Cap points shipped per scatter series; sampled deterministically so the plot
# is stable across refetches. Legacy ships all points (client-side Plotly).
MAX_SCATTER_POINTS = 5000


def _metric_values(
    panel: str,
    metric: str,
    dm: Optional[DataMetrics],
    df_win: pl.DataFrame,
    date_col: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Optional[np.ndarray]:
    """Values of one metric within a (cohort, range) window — for histograms.
    Derived metrics come from DataMetrics (group-level); user metrics are raw
    columns of df_win."""
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


def _observation_frame(
    panel: str,
    cols: list[str],
    dm: Optional[DataMetrics],
    df_win: pl.DataFrame,
    date_col: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Optional[pl.DataFrame]:
    """A frame with one column per metric and one row per aligned observation
    (date for derived, user row for user). Shared by heatmap + scatter so both
    see the same rows. Columns absent from the data are dropped."""
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
        return frame.select([c for c in cols if c in frame.columns])
    present = [c for c in cols if c in df_win.columns]
    return df_win.select(present) if present else None


def _outlier_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Keep rows within `threshold` std of the mean on EVERY column (columns with
    zero/NaN std are ignored). Mirrors the legacy _viz_drop_outliers."""
    mask = np.ones(arr.shape[0], dtype=bool)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    for k in range(arr.shape[1]):
        if not np.isfinite(std[k]) or std[k] == 0:
            continue
        mask &= np.abs(arr[:, k] - mean[k]) / std[k] <= threshold
    return mask


def load_deepdive(
    cfg: dict[str, Any],
    granularity: str,
    panel: str,
    mode: str,
    metrics: list[str],
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
    nbins: int = 50,
    normalize: bool = False,
    outliers_std: Optional[float] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return (histograms, heatmaps, scatters, missing)."""
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
        return [], [], [], list(metrics)

    overall_start = min(s for _, s, _, _ in parsed)
    overall_end = max(e for _, _, e, _ in parsed)
    aggregate_from_rows = bool(sd.get("aggregate_stats_from_user_rows", False))
    load_end = overall_end + timedelta(days=RETENTION_LOAD_EXTRA_DAYS) if aggregate_from_rows else overall_end

    # Shared + single-flight with the other tabs' requests (see common.py —
    # parallel per-panel collects OOM-killed the task in production).
    df_raw = collect_window(cfg, granularity, lf, date_col, overall_start, load_end)
    if df_raw.is_empty():
        return [], [], [], list(metrics)

    histograms: list[dict[str, Any]] = []
    heatmaps: list[dict[str, Any]] = []
    scatters: list[dict[str, Any]] = []
    produced: set[str] = set()

    for label, df_c in iter_cohorts(
        cfg,
        df_raw,
        group_values,
        lifecycle=lifecycle,
        date_col=date_col,
        granularity=granularity,
        range_groups=range_groups,
    ):
        dm = (
            DataMetrics(
                df_c,
                start_dt=overall_start,
                end_dt=overall_end,
                key_cols=[date_col],
                granularity=granularity,
                presence_df=df_raw,  # full population for any-group retention
            )
            if panel == "derived"
            else None
        )
        for range_index, start_dt, end_dt, range_label in parsed:
            df_win = df_c.filter((pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt))
            if df_win.is_empty():
                continue

            if mode == "histogram":
                for m in metrics:
                    vals = _metric_values(panel, m, dm, df_win, date_col, start_dt, end_dt)
                    if vals is None or len(vals) == 0:
                        continue
                    produced.add(m)
                    vals = filter_values(vals, filter_enable, filter_min, filter_max, filter_percentile)
                    if len(vals) == 0:
                        continue
                    vals = clip_values(vals, clip_enable, clip_min, clip_max, clip_percentile)
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
                continue

            # heatmap + scatter both need an aligned observation frame.
            obs = _observation_frame(panel, metrics, dm, df_win, date_col, start_dt, end_dt)
            if obs is None:
                continue
            obs = obs.drop_nulls()
            cols = obs.columns
            if obs.height < 2 or len(cols) < 2:
                continue
            produced.update(cols)
            arr = obs.to_numpy().astype(float)
            if filter_enable and arr.shape[0] > 0 and (filter_min is not None or filter_max is not None):
                arr = arr[filter_row_mask(arr, filter_min, filter_max, filter_percentile)]
            arr = clip_columns(arr, clip_enable, clip_min, clip_max, clip_percentile)

            if mode == "heatmap":
                if arr.shape[0] < 2:
                    continue
                corr = np.corrcoef(arr, rowvar=False)
                heatmaps.append(
                    {
                        "cohort": label,
                        "range_label": range_label,
                        "range_index": range_index,
                        "metrics": cols,
                        "corr": [[None if not np.isfinite(v) else float(v) for v in row] for row in corr],
                    }
                )
            else:  # scatter
                if outliers_std is not None:
                    arr = arr[_outlier_mask(arr, outliers_std)]
                arr = arr[np.isfinite(arr).all(axis=1)]
                if arr.shape[0] == 0:
                    continue
                if arr.shape[0] > MAX_SCATTER_POINTS:
                    idx = np.sort(np.random.default_rng(0).choice(arr.shape[0], MAX_SCATTER_POINTS, replace=False))
                    arr = arr[idx]
                scatters.append(
                    {
                        "cohort": label,
                        "range_label": range_label,
                        "range_index": range_index,
                        "metrics": cols,
                        "points": [[float(v) for v in row] for row in arr],
                    }
                )

    return histograms, heatmaps, scatters, [m for m in metrics if m not in produced]


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
