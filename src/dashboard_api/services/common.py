"""Shared data-access helpers for the /api/data/* services.

The Stats-by-Date, Stats-by-Group, and Deep Dive endpoints all read the same
one-row-per-user parquet, filter to the same cohort cross-product, and reuse the
``DataMetrics`` engine. The cohort/loading plumbing lives here so the three
services stay thin and behave identically (matching the legacy Dash tabs).
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any, Iterator, Optional, cast

import polars as pl

from bituslabs_ds.s3_utils import read_files

logger = logging.getLogger(__name__)


class SeriesError(Exception):
    """Raised when a request can't be served (bad granularity / missing date col)."""


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


def effective_cohort_cols(cfg: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Return (col1, col2); col2 only when configured and distinct from col1."""
    cols = user_group_cols(cfg)
    col1 = cols[0] if cols else None
    col2 = cols[1] if len(cols) > 1 and cols[1] != col1 else None
    return col1, col2


def load_lazy(cfg: dict[str, Any], granularity: str) -> tuple[pl.LazyFrame, str]:
    """Lazily open the user-level parquet for a granularity; return (lf, date_col)."""
    sd = stats_by_date_cfg(cfg)
    date_col = str(sd.get("date_col", "activity_date"))
    files = (sd.get("files") or {}).get(granularity)
    if not files:
        raise SeriesError(f"No stats_by_date files configured for granularity '{granularity}'")
    lf = cast(pl.LazyFrame, read_files(files, lazy_load=True, expand_s3_prefixes=True))
    return lf, date_col


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
    """Selected values for a cohort column, de-duped; empty selection means ['all']."""
    if not col:
        return ["all"]
    seen: set[str] = set()
    vals: list[str] = []
    for v in group_values.get(col, []) or []:
        s = str(v)
        if s and s not in seen:
            seen.add(s)
            vals.append(s)
    return vals or ["all"]


def cohort_label(v1: str, v2: str, has_col2: bool) -> str:
    if not has_col2:
        return "all" if v1 == "all" else v1
    if v1 == "all" and v2 == "all":
        return "all"
    if v1 == "all":
        return v2
    if v2 == "all":
        return v1
    return f"{v1} | {v2}"


def apply_cohort(df: pl.DataFrame, col: Optional[str], value: str) -> pl.DataFrame:
    """Filter to rows where col == value; 'all' (or missing col) means no filter."""
    if not col or value == "all" or col not in df.columns:
        return df
    return df.filter(pl.col(col) == value)


def iter_cohorts(
    cfg: dict[str, Any], df: pl.DataFrame, group_values: dict[str, list[str]]
) -> Iterator[tuple[str, pl.DataFrame]]:
    """Yield (cohort_label, cohort_df) over the cross-product of selected
    user-group values — the same cohorts the legacy tabs overlay. Filtering
    happens before any DataMetrics aggregation, so each cohort's metrics are
    computed on just its rows."""
    col1, col2 = effective_cohort_cols(cfg)
    v1s = cohort_values(group_values, col1)
    v2s = cohort_values(group_values, col2) if col2 else ["all"]
    seen: set[str] = set()
    for v1 in v1s:
        for v2 in v2s:
            label = cohort_label(v1, v2, has_col2=col2 is not None)
            if label in seen:
                continue
            seen.add(label)
            df_c = apply_cohort(apply_cohort(df, col1, v1), col2, v2)
            yield label, df_c
