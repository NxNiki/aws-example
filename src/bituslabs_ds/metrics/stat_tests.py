"""Lightweight significance tests (Welch t-test + one-way ANOVA), numpy-only.

The dashboard image deliberately ships without scipy/sklearn (see
``dashboard_api/services/deepdive.py``), so the Summary-table tab's optional
p-value column can't lean on ``scipy.stats``. The t/F survival functions reduce
to the regularized incomplete beta function ``I_x(a, b)``, which we evaluate
directly with the Numerical-Recipes continued-fraction algorithm using only
``math`` + ``numpy``. Results match ``scipy.stats.ttest_ind(equal_var=False)``
and ``scipy.stats.f_oneway`` to ~1e-10.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np

__all__ = ["welch_ttest", "one_way_anova", "compare_groups"]

# Anything np.asarray(..., dtype=float) can consume (lists, tuples, ndarrays).
ArrayLike = Union[Sequence[float], np.ndarray]

_EPS = 1e-15
_FPMIN = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b), x in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    # Use the form with faster convergence on each side of the mode.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def welch_ttest(a: ArrayLike, b: ArrayLike) -> tuple[Optional[float], Optional[float]]:
    """Welch's two-sample t-test (unequal variances), two-sided.

    Returns ``(t_statistic, p_value)``; either is ``None`` when the test is
    undefined (fewer than 2 values per group, or zero pooled variance).
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return None, None
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se2 = vx / nx + vy / ny
    if se2 <= 0.0:  # both samples constant → no variance to test against
        return None, None
    t = float((x.mean() - y.mean()) / math.sqrt(se2))
    df = se2 * se2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return t, float(p)


def one_way_anova(groups: Sequence[ArrayLike]) -> tuple[Optional[float], Optional[float]]:
    """One-way ANOVA F-test across 2+ groups.

    Returns ``(f_statistic, p_value)``; either is ``None`` when undefined
    (fewer than 2 usable groups, total N ≤ #groups, or zero within-group
    variance).
    """
    arrs = [np.asarray(g, dtype=float) for g in groups if np.asarray(g, dtype=float).size > 0]
    k = len(arrs)
    if k < 2:
        return None, None
    n_total = sum(a.size for a in arrs)
    df_b, df_w = k - 1, n_total - k
    if df_w <= 0:
        return None, None
    grand = np.concatenate(arrs).mean()
    ss_between = sum(a.size * (a.mean() - grand) ** 2 for a in arrs)
    ss_within = sum(float(((a - a.mean()) ** 2).sum()) for a in arrs)
    if ss_within <= 0.0:  # perfectly separated / all-constant groups
        return None, None
    f = float((ss_between / df_b) / (ss_within / df_w))
    p = _betai(df_w / 2.0, df_b / 2.0, df_w / (df_w + df_b * f))
    return f, float(p)


def compare_groups(arrays: Sequence[Optional[ArrayLike]]) -> tuple[Optional[float], Optional[str]]:
    """Pick and run the right test for the columns of a Summary-table row.

    Two usable columns → Welch t-test; three or more → one-way ANOVA. A column
    needs ≥2 values to count. Returns ``(p_value, test_name)`` with both ``None``
    when fewer than two columns are usable or the test is undefined.
    """
    usable: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.size >= 2:
            usable.append(a)
    if len(usable) < 2:
        return None, None
    if len(usable) == 2:
        _, p = welch_ttest(usable[0], usable[1])
        return p, "t-test"
    _, p = one_way_anova(usable)
    return p, "anova"
