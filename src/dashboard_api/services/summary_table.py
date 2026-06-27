"""All-metrics summary grid for the Summary-table tab.

Where Stats-by-Group shows ONE metric's distribution as a chart, this builds a
single table: every metric (across all three configured groups) as a row, and
every selected (cohort × date-range) as a column. Each cell carries the metric's
mean / median / quartiles for that column; the frontend picks one column as the
reference and shows the others as ``value (±%)``. When requested, a per-metric
significance test (Welch t-test for 2 columns, one-way ANOVA for 3+) is computed
on the same per-row value arrays.

Reads through the same ``DataMetrics`` engine + shared window cache as the other
tabs (see ``group_distribution.py`` / ``common.py``), so the numbers match.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np
import polars as pl

from bituslabs_ds.metrics.stat_tests import compare_groups
from bituslabs_ds.metrics.user_stats_aggregates import RETENTION_LOAD_EXTRA_DAYS, DataMetrics
from dashboard_api.services.common import SeriesError, collect_window, iter_cohorts, load_lazy, stats_by_date_cfg

logger = logging.getLogger(__name__)


def _opt(v: float) -> Optional[float]:
    return None if (np.isnan(v) or np.isinf(v)) else float(v)


def _cell(vals: np.ndarray) -> dict[str, Any]:
    """Per-column value summary for one metric (no CI — the table shows point stats)."""
    return {
        "n": int(len(vals)),
        "mean": _opt(float(np.mean(vals))),
        "median": _opt(float(np.median(vals))),
        "q1": _opt(float(np.percentile(vals, 25))),
        "q3": _opt(float(np.percentile(vals, 75))),
        "std": _opt(float(np.std(vals))),
        "min": _opt(float(np.min(vals))),
        "max": _opt(float(np.max(vals))),
    }


def _reshape(vals: np.ndarray, opt: Optional[dict[str, Any]]) -> np.ndarray:
    """Apply a metric's clip then signed-log1p (per-metric display reshaping)."""
    if not opt:
        return vals
    if opt.get("clip_enable") and (opt.get("clip_min") is not None or opt.get("clip_max") is not None):
        lo = opt["clip_min"] if opt.get("clip_min") is not None else -np.inf
        hi = opt["clip_max"] if opt.get("clip_max") is not None else np.inf
        vals = np.clip(vals, lo, hi)
    if opt.get("log"):
        vals = np.sign(vals) * np.log1p(np.abs(vals))
    return vals


def _groups_from_cfg(sd: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """(group_id, group_label, metrics) for each configured metric group."""
    groups: list[tuple[str, str, list[str]]] = []
    for i in (1, 2, 3):
        cols = sd.get(f"group{i}_columns")
        if cols:
            groups.append((f"group{i}", str(sd.get(f"group{i}_label", f"Group {i}")), [str(c) for c in cols]))
    return groups


def load_summary_table(
    cfg: dict[str, Any],
    granularity: str,
    ranges: list[tuple[Optional[str], Optional[str]]],
    group_values: Optional[dict[str, list[str]]] = None,
    metric_options: Optional[dict[str, dict[str, Any]]] = None,
    pvalues: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (columns, rows). Columns are cohort × non-empty range; rows are metrics."""
    group_values = group_values or {}
    metric_options = metric_options or {}
    sd = stats_by_date_cfg(cfg)
    groups = _groups_from_cfg(sd)
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
    if not parsed or not groups:
        return [], []

    overall_start = min(s for _, s, _, _ in parsed)
    overall_end = max(e for _, _, e, _ in parsed)
    aggregate_from_rows = bool(sd.get("aggregate_stats_from_user_rows", False))
    load_end = overall_end + timedelta(days=RETENTION_LOAD_EXTRA_DAYS) if aggregate_from_rows else overall_end

    df_raw = collect_window(cfg, granularity, lf, date_col, overall_start, load_end)
    cohorts = list(iter_cohorts(cfg, df_raw, group_values))
    n_ranges = len(parsed)
    n_cols = len(cohorts) * n_ranges

    # Columns, in cohort-major order (A·R1, A·R2, B·R1, …).
    columns: list[dict[str, Any]] = []
    for label, _ in cohorts:
        for range_index, _s, _e, range_label in parsed:
            columns.append(
                {
                    "key": f"{label}@@{range_index}",
                    "cohort": label,
                    "range_index": range_index,
                    "range_label": range_label,
                }
            )

    # Pre-build one row per metric (cells default to None → "no data").
    rows: list[dict[str, Any]] = []
    row_pos: dict[tuple[str, str], int] = {}
    for gid, glabel, metrics in groups:
        for m in metrics:
            row_pos[(gid, m)] = len(rows)
            rows.append(
                {"metric": m, "group_id": gid, "group_label": glabel, "cells": [None] * n_cols, "missing": True}
            )
    # Per-row value arrays per column, accumulated only when a p-value is wanted.
    pval_arrays: Optional[List[List[Optional[np.ndarray]]]] = [[None] * n_cols for _ in rows] if pvalues else None

    for cohort_pos, (_label, df_c) in enumerate(cohorts):
        dm = DataMetrics(
            df_c,
            start_dt=overall_start,
            end_dt=overall_end,
            key_cols=[date_col],
            granularity=granularity,
            presence_df=df_raw,  # full population for any-group retention
        )
        for gid, _glabel, metrics in groups:
            for m in metrics:
                metric_df = dm[m]
                if metric_df is None:
                    continue
                ridx = row_pos[(gid, m)]
                opt = metric_options.get(m)
                for range_pos, (_ri, start_dt, end_dt, _rl) in enumerate(parsed):
                    window = metric_df.filter((pl.col(date_col) >= start_dt) & (pl.col(date_col) <= end_dt))
                    vals = window.get_column(m).drop_nulls().to_numpy().astype(float)
                    if len(vals) == 0:
                        continue
                    vals = _reshape(vals, opt)
                    col_idx = cohort_pos * n_ranges + range_pos
                    rows[ridx]["cells"][col_idx] = _cell(vals)
                    rows[ridx]["missing"] = False
                    if pval_arrays is not None:
                        pval_arrays[ridx][col_idx] = vals

    if pval_arrays is not None:
        for ridx, row in enumerate(rows):
            p, test = compare_groups(pval_arrays[ridx])
            row["pvalue"] = p
            row["test"] = test

    return columns, rows
