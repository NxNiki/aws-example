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


def test_renamed_group_across_days_via_presence_df():
    """Dashboard flow: df is pre-filtered to one group, key_cols=[date] only.

    Reproduces the production bug — a group present on D0 but renamed/replaced on
    D1 (DYNAMIC_RTP_V2 → _V3). Without the full population the return test reads
    only the filtered slice (no D1 rows) → 0 retention; presence_df=full fixes it.
    """
    d0, d1 = datetime(2026, 6, 1), datetime(2026, 6, 2)
    full = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u1", "u2"],
            "activity_date": [d0, d0, d1, d1],
            "daily_group": ["DYNAMIC_RTP_V2", "DYNAMIC_RTP_V2", "DYNAMIC_RTP_V3", "DYNAMIC_RTP_V3"],
        }
    )
    df_v2 = full.filter(pl.col("daily_group") == "DYNAMIC_RTP_V2")  # what iter_cohorts hands DataMetrics

    buggy = DataMetrics(df_v2, start_dt=d0, end_dt=d0, key_cols=["activity_date"])["retention_rate_day1"]
    assert buggy["retention_rate_day1"][0] == pytest.approx(0.0)  # the symptom

    fixed = DataMetrics(df_v2, start_dt=d0, end_dt=d0, key_cols=["activity_date"], presence_df=full)[
        "retention_rate_day1"
    ]
    assert fixed["retention_rate_day1"][0] == pytest.approx(1.0)  # both users returned under V3


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
