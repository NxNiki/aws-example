"""Semantic tests for the fish_hunter group_tag policy.

The row-level CASE and the user-day collapse use only portable SQL, so
sqlite executes the exact composed expressions against stub tables — no JVM
required (the SageMaker container is the only place the full Spark query
runs).
"""

import importlib.util
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, name: str):
    sys.path.insert(0, str(REPO_ROOT / "jobs/etl/sagemaker"))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _load_common():
    sys.path.insert(0, str(REPO_ROOT / "jobs/etl/sagemaker"))
    try:
        import spark_etl_common

        return spark_etl_common
    finally:
        sys.path.pop(0)


COMMON = _load_common()
BULLETS_JOB = _load_module(
    REPO_ROOT / "jobs/etl/sagemaker/fish_hunter/etl_fish_bullets_group_tag.py", "_fish_bullets_job"
)
STATS_JOB = _load_module(
    REPO_ROOT / "jobs/etl/sagemaker/fish_hunter/etl_game_stats_daily_by_user_cold_data.py", "_fish_stats_job"
)

PRE = "2026-07-30 23:59:59"
POST = "2026-08-01 12:00:00"


def _group_tag_rows(rows):
    """Run the row-level policy CASE against sqlite (TIMESTAMP literals
    stripped: sqlite compares ISO datetime strings directly)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bullets (strategy_name, user_id, event_ts)")
    con.executemany("INSERT INTO bullets VALUES (?, ?, ?)", rows)
    case = COMMON.fish_group_tag_sql("strategy_name", "user_id", "event_ts")
    sql = f"SELECT {case} FROM bullets ORDER BY rowid".replace("TIMESTAMP '", "'")
    return [r[0] for r in con.execute(sql)]


def test_strategy_branches_apply_to_all_history():
    rows = [
        # legacy BOOST_POOL keeps its own label (no longer exists in new data).
        ("BOOST_POOL", 5, POST),
        # the whole DYNAMIC_RTP family shares one label across its date ranges.
        ("DYNAMIC_RTP", 5, PRE),
        ("DYNAMIC_RTP_V2", 5, POST),
        ("DYNAMIC_RTP_V3", 5, PRE),
        ("RC_FISHING_20260601", 5, PRE),
        # CR_FISHING is the retention treatment itself: retention regardless
        # of the launch gate (pre-launch canary) or the user digit.
        ("CR_FISHING_V1:T3:P1", 5, PRE),
        ("CR_FISHING_V1:T1:P2", 25, POST),
        ("AB_RISK_CONTROL_V2", 9, PRE),
        # legacy RISK_CONTROLLED contains the RISK_CONTROL substring.
        ("RISK_CONTROLLED", 3, PRE),
        # the RC_FISHING match requires the literal underscore (escaped LIKE).
        ("RC_FISHINGX", 5, PRE),
        ("DEFAULT_FALLBACK", 5, PRE),
        (None, 5, PRE),
    ]
    assert _group_tag_rows(rows) == [
        "boost_pool",
        "dynamic_rtp",
        "dynamic_rtp",
        "dynamic_rtp",
        "risk_control",
        "retention",
        "retention",
        "risk_control",
        "risk_control",
        "default",
        "default",
        "default",
    ]


def test_retention_is_timestamp_gated_on_user_digit():
    rows = [
        # before the launch the digit is ignored.
        ("DEFAULT_FALLBACK", 10, PRE),
        (None, 21, PRE),
        # from the launch (inclusive) digits 0/1 label retention...
        ("DEFAULT_FALLBACK", 10, "2026-07-31 00:00:00"),
        ("DEFAULT_FALLBACK", 21, POST),
        (None, 30, POST),
        # ...unless a strategy branch claims the row first.
        ("DYNAMIC_RTP_V3", 10, POST),
        ("RC_FISHING_X_1", 21, POST),
        # other digits stay default.
        ("DEFAULT_FALLBACK", 12, POST),
        ("BOOST_POOL", 29, POST),
    ]
    assert _group_tag_rows(rows) == [
        "default",
        "default",
        "retention",
        "retention",
        "retention",
        "dynamic_rtp",
        "risk_control",
        "default",
        "boost_pool",
    ]


def test_user_day_collapse_priority():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bullets (user_id, activity_date, group_tag)")
    con.executemany(
        "INSERT INTO bullets VALUES (?, ?, ?)",
        [
            # a single higher-tier bullet claims the whole user-day;
            # boost_pool outranks everything for legacy data.
            ("u0", "d1", "dynamic_rtp"),
            ("u0", "d1", "boost_pool"),
            ("u1", "d1", "default"),
            ("u1", "d1", "risk_control"),
            ("u1", "d1", "dynamic_rtp"),
            ("u2", "d1", "retention"),
            ("u2", "d1", "risk_control"),
            ("u3", "d1", "default"),
            ("u3", "d1", "retention"),
            ("u4", "d1", "default"),
            # days do not bleed into each other.
            ("u1", "d2", "retention"),
        ],
    )
    case = COMMON.fish_group_tag_day_case("group_tag")
    sql = (
        f"SELECT user_id, activity_date, {case} FROM bullets"
        " GROUP BY user_id, activity_date ORDER BY user_id, activity_date"
    )
    assert list(con.execute(sql)) == [
        ("u0", "d1", "boost_pool"),
        ("u1", "d1", "dynamic_rtp"),
        ("u1", "d2", "retention"),
        ("u2", "d1", "risk_control"),
        ("u3", "d1", "retention"),
        ("u4", "d1", "default"),
    ]


def test_bullets_query_composition():
    """The Athena-backing bullets copy: selected raw columns plus the policy
    group_tag. Test bets drop at the source; currency filtering stays
    downstream (a selection, not junk)."""
    sql = BULLETS_JOB.generate_query(date(2026, 8, 1), date(2026, 8, 5))
    assert "AS group_tag" in sql and f"TIMESTAMP '{COMMON.FISH_RETENTION_POLICY_START_UTC}'" in sql
    assert "t.strategy_name" in sql  # kept for auditing the tag
    assert "op_code NOT IN" in sql and "currency_type =" not in sql
    # group_tag reads the bullet's UTC event_timestamp, not the BJ-shifted one.
    assert "% 10 IN (0, 1)" in sql


def test_stats_query_composition():
    sql = STATS_JOB.generate_query(
        stats_agg_col="activity_date",
        effective_start=date(2026, 8, 1),
        output_end=date(2026, 8, 5),
        scan_start_utc=datetime(2026, 7, 31, 16),
        scan_end_utc=datetime(2026, 8, 4, 16),
        game_id="FM01",
        currency="CNY",
    )
    # the policy is derived once, upstream in the bullets dataset; test
    # op-codes are already dropped there.
    assert "strategy_name" not in sql and "op_code" not in sql
    assert "b.group_tag" in sql and "user_group_tag" in sql
    assert "WHEN MAX(CASE WHEN group_tag = 'boost_pool' THEN 1 ELSE 0 END) > 0 THEN 'boost_pool'" in sql
    assert sql.index("'boost_pool'") < sql.index("group_tag = 'dynamic_rtp'")
    assert "GROUP BY b.user_id, u.group_tag, b.fish_value" in sql
    assert "ab_test_group" not in sql
