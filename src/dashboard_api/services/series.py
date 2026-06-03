"""Read metric time-series from the S3 parquet cache for the data API.

Phase 0 scope: read the pre-aggregated `stats_by_date` parquet, select the
requested metric columns that exist, and return them aggregated per date.
The per-date aggregation here is a mean across source rows/shards — a
placeholder. Full parity with the legacy dashboard's user-row enrichment
(`dashboards/user_stats_aggregates.py`) lands in Phase 1.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import polars as pl

from bituslabs_ds.s3_utils import read_files

logger = logging.getLogger(__name__)


class SeriesError(Exception):
    """Raised when a series request can't be served (bad granularity / missing date col)."""


def load_series(
    cfg: dict[str, Any],
    granularity: str,
    metrics: list[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> tuple[str, list[Optional[str]], list[dict[str, Any]], list[str]]:
    """Return (date_col, x, series, missing) for the requested metrics."""
    stats_by_date = cfg.get("stats_by_date") or {}
    date_col = str(stats_by_date.get("date_col", "activity_date"))
    files = (stats_by_date.get("files") or {}).get(granularity)
    if not files:
        raise SeriesError(f"No stats_by_date files configured for granularity '{granularity}'")

    lf = cast(pl.LazyFrame, read_files(files, lazy_load=True, expand_s3_prefixes=True))

    available = set(lf.collect_schema().names())
    if date_col not in available:
        raise SeriesError(f"Date column '{date_col}' not found in source data")

    present = [m for m in metrics if m in available]
    missing = [m for m in metrics if m not in available]

    lf = lf.select([date_col, *present]).with_columns(pl.col(date_col).cast(pl.Date, strict=False))
    if date_from:
        lf = lf.filter(pl.col(date_col) >= pl.lit(date_from).str.to_date())
    if date_to:
        lf = lf.filter(pl.col(date_col) <= pl.lit(date_to).str.to_date())
    if present:
        lf = lf.group_by(date_col).agg([pl.col(m).mean().alias(m) for m in present])
    df = lf.sort(date_col).collect()

    x: list[Optional[str]] = [d.isoformat() if d is not None else None for d in df[date_col].to_list()]
    series: list[dict[str, Any]] = []
    for metric in present:
        ys = df[metric].to_list()
        series.append({"name": metric, "y": [float(v) if v is not None else None for v in ys]})

    return date_col, x, series, missing
