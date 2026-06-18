"""Unit tests for DataMetrics retention (any-group definition).

Day-N retention is ANY-GROUP: a day-0 cohort user counts as retained if they are
active on the follow-up date in ANY group, not only their day-0 group. See the
`Retention definition` section in ``bituslabs_ds.metrics.user_stats_aggregates``.
"""

from datetime import datetime

import polars as pl
import pytest

from bituslabs_ds.metrics.user_stats_aggregates import DataMetrics

KEY_COLS = ["activity_date", "ai_group"]


def _dm(df: pl.DataFrame, d0: datetime) -> DataMetrics:
    return DataMetrics(df, start_dt=d0, end_dt=d0, key_cols=KEY_COLS)


def test_day1_retention_counts_cross_group_return():
    """A user who moves to a different group on D+1 still counts as retained for their D group."""
    d0, d1 = datetime(2026, 6, 1), datetime(2026, 6, 2)
    # Group A cohort on D0 = {u1, u2}. On D1: u1 stays in A, u2 switches to B, u3 is new in A.
    df = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u1", "u2", "u3"],
            "activity_date": [d0, d0, d1, d1, d1],
            "ai_group": ["A", "A", "A", "B", "A"],
        }
    )
    res = _dm(df, d0)["retention_rate_day1"]
    rate = res.filter((pl.col("ai_group") == "A") & (pl.col("activity_date") == d0))["retention_rate_day1"][0]
    # any-group: both u1 (A) and u2 (B) returned on D1 -> 2/2 = 1.0 (same-group would be 0.5)
    assert rate == pytest.approx(1.0)


def test_day1_retention_zero_when_no_return():
    """Nobody from the D0 cohort is active on D+1 -> retention 0."""
    d0, d1 = datetime(2026, 6, 1), datetime(2026, 6, 2)
    df = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "activity_date": [d0, d0, d1],  # u3 on D1 was not in the D0 cohort
            "ai_group": ["A", "A", "A"],
        }
    )
    res = _dm(df, d0)["retention_rate_day1"]
    rate = res.filter(pl.col("ai_group") == "A")["retention_rate_day1"][0]
    assert rate == pytest.approx(0.0)


def test_day0_cohort_is_per_group():
    """The denominator (cohort) is still per-group, even though the return test is any-group."""
    d0 = datetime(2026, 6, 1)
    df = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "activity_date": [d0, d0, d0],
            "ai_group": ["A", "A", "B"],
        }
    )
    res = _dm(df, d0)["day0_num_users"].sort("ai_group")
    assert res.filter(pl.col("ai_group") == "A")["day0_num_users"][0] == 2
    assert res.filter(pl.col("ai_group") == "B")["day0_num_users"][0] == 1
