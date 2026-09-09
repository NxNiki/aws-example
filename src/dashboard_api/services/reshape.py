"""Value-array reshaping shared by the Stats-by-Group / Deep Dive / Summary
tabs: filter (drop samples) and clip (pin values), each with a value-based or
percentile-based range."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _bounds(vals: np.ndarray, lo: Optional[float], hi: Optional[float], percentile: bool) -> tuple[float, float]:
    """Absolute [lo, hi] bounds; percentile bounds are computed on ``vals`` itself."""
    if percentile:
        lo = float(np.percentile(vals, lo)) if lo is not None and len(vals) else None
        hi = float(np.percentile(vals, hi)) if hi is not None and len(vals) else None
    return (lo if lo is not None else -np.inf, hi if hi is not None else np.inf)


def filter_values(
    vals: np.ndarray,
    enable: bool,
    lo: Optional[float],
    hi: Optional[float],
    percentile: bool = True,
) -> np.ndarray:
    """DROP samples outside the [lo, hi] range.

    Dashboard feature: the "filter" / "filter %" rows of the clip-filter box on
    the Stats-by-Group and Deep Dive tabs. Value-based bounds are absolute;
    percentile bounds (0-100) are computed on each (metric x cohort x range)
    sample itself. Unlike clip (which pins values), filtered samples are
    REMOVED before any stats/binning.
    """
    if not enable or len(vals) == 0 or (lo is None and hi is None):
        return vals
    lo_b, hi_b = _bounds(vals, lo, hi, percentile)
    return vals[(vals >= lo_b) & (vals <= hi_b)]


def clip_values(
    vals: np.ndarray,
    enable: bool,
    lo: Optional[float],
    hi: Optional[float],
    percentile: bool = False,
) -> np.ndarray:
    """PIN values outside the [lo, hi] range to the bound (winsorize; keeps every sample).

    Dashboard feature: the "clip" / "clip %" rows of the clip-filter box on the
    Stats-by-Group, Deep Dive, and Summary tabs. Value-based bounds are
    absolute; percentile bounds (0-100) are computed on each
    (metric x cohort x range) sample itself.
    """
    if not enable or len(vals) == 0 or (lo is None and hi is None):
        return vals
    lo_b, hi_b = _bounds(vals, lo, hi, percentile)
    return np.clip(vals, lo_b, hi_b)


def filter_row_mask(
    arr: np.ndarray,
    lo: Optional[float],
    hi: Optional[float],
    percentile: bool = True,
) -> np.ndarray:
    """Row mask keeping rows inside the [lo, hi] range on EVERY column (for the
    Deep Dive heatmap/scatter observation frame, where rows must stay aligned
    across metrics). Percentile bounds are computed per column."""
    mask = np.ones(arr.shape[0], dtype=bool)
    for k in range(arr.shape[1]):
        col = arr[:, k]
        lo_b, hi_b = _bounds(col, lo, hi, percentile)
        mask &= (col >= lo_b) & (col <= hi_b)
    return mask


def clip_columns(
    arr: np.ndarray,
    enable: bool,
    lo: Optional[float],
    hi: Optional[float],
    percentile: bool = False,
) -> np.ndarray:
    """Column-wise clip of a 2-D observation frame (percentile bounds per column)."""
    if not enable or arr.shape[0] == 0 or (lo is None and hi is None):
        return arr
    if not percentile:
        return np.clip(arr, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
    out = arr.copy()
    for k in range(arr.shape[1]):
        lo_b, hi_b = _bounds(arr[:, k], lo, hi, True)
        out[:, k] = np.clip(arr[:, k], lo_b, hi_b)
    return out
