"""Semantic tests for the slot cold-data ETL's --min-mathtable-run filter.

The run-filter CTE chain uses only portable window SQL (LAG, SUM ... ROWS
BETWEEN, COUNT() OVER PARTITION), so sqlite executes the exact composed SQL
against a stub ``bets`` table — no JVM required (the SageMaker container is
the only place the full Spark query runs).
"""

import importlib.util
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_PATH = REPO_ROOT / "jobs/etl/sagemaker/slot_machine/etl_game_stats_daily_by_user_group_cold_data.py"


def _load_job_module():
    sys.path.insert(0, str(REPO_ROOT / "jobs/etl/sagemaker"))
    try:
        spec = importlib.util.spec_from_file_location("_slot_cold_data_job", JOB_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


JOB = _load_job_module()


def _load_common():
    sys.path.insert(0, str(REPO_ROOT / "jobs/etl/sagemaker"))
    try:
        import spark_etl_common

        return spark_etl_common
    finally:
        sys.path.pop(0)


COMMON = _load_common()

DAY_WINDOW = "PARTITION BY t.user_id, t.activity_date ORDER BY t.spin_id, t.created_at"


def _kept_spins(rows, min_run):
    """Run the composed run-filter CTEs over (user_id, activity_date,
    mathtable, spin_id, created_at) rows; return the surviving spin_ids."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bets (user_id, activity_date, mathtable, spin_id, created_at)")
    con.executemany("INSERT INTO bets VALUES (?, ?, ?, ?, ?)", rows)
    sql = f"WITH {JOB._mathtable_run_ctes(DAY_WINDOW, min_run)} SELECT spin_id FROM long_run_bets ORDER BY spin_id"
    return [r[0] for r in con.execute(sql)]


def _row(user, day, mathtable, spin):
    return (user, day, mathtable, spin, spin)


def test_short_runs_dropped_within_day():
    # u1's day: A x4, B x2, A x3 — with min 3 the B dabble drops and both A
    # stretches survive as separate runs (they are NOT merged: 4 and 3, not 7).
    rows = (
        [_row("u1", "d1", "A", i) for i in (1, 2, 3, 4)]
        + [_row("u1", "d1", "B", i) for i in (5, 6)]
        + [_row("u1", "d1", "A", i) for i in (7, 8, 9)]
    )
    assert _kept_spins(rows, 3) == [1, 2, 3, 4, 7, 8, 9]
    # A run of exactly N bets survives (threshold is inclusive: >= N).
    assert _kept_spins(rows, 4) == [1, 2, 3, 4]
    # min 5 kills everything: the interrupted A stretches never re-join.
    assert _kept_spins(rows, 5) == []
    assert _kept_spins(rows, 1) == list(range(1, 10))


def test_runs_never_bridge_users_or_days():
    rows = (
        # u1 plays A across two days, 2 bets each: day-partitioned runs of 2.
        [_row("u1", "d1", "A", 1), _row("u1", "d1", "A", 2)]
        + [_row("u1", "d2", "A", 3), _row("u1", "d2", "A", 4)]
        # u2's same-day A bets interleave by spin_id with u1's but form their
        # own run of 2.
        + [_row("u2", "d1", "A", 5), _row("u2", "d1", "A", 6)]
    )
    assert _kept_spins(rows, 3) == []  # no cross-day (or cross-user) 4-bet run
    assert _kept_spins(rows, 2) == [1, 2, 3, 4, 5, 6]


def test_generate_query_wires_the_filter():
    kwargs = dict(
        stats_agg_col="activity_date",
        game_id="SS03",
        effective_start=date(2026, 8, 1),
        output_end=date(2026, 8, 5),
        scan_start_utc=datetime(2026, 7, 31, 16),
        scan_end_utc=datetime(2026, 8, 4, 16),
        currency="CNY",
    )
    base = JOB.generate_query(**kwargs)
    assert "long_run_bets" not in base and "FROM bets AS t" in base
    # Combination-label tie-break input for the dashboards.
    assert "MIN(t.spin_id) AS user_first_spin_id" in base and "us.user_first_spin_id" in base

    filtered = JOB.generate_query(**kwargs, min_mathtable_run=30)
    assert "mathtable_run_len >= 30" in filtered
    # user_bets (and every sequence metric) reads the filtered stream.
    assert "FROM long_run_bets AS t" in filtered


def test_generate_query_ab_group_scan_filter():
    kwargs = dict(
        stats_agg_col="activity_date",
        game_id="SS03",
        effective_start=date(2026, 8, 1),
        output_end=date(2026, 8, 5),
        scan_start_utc=datetime(2026, 7, 31, 16),
        scan_end_utc=datetime(2026, 8, 4, 16),
        currency="CNY",
    )
    base = JOB.generate_query(**kwargs)
    assert "AND t.ab_group =" not in base  # no scan filter by default

    only_ai = JOB.generate_query(**kwargs, ab_group="AI")
    # bet_events' WHERE keeps only bets stored with the AI label (the policy
    # is derived once, upstream in slot_orders_ab_group), so with the run
    # filter runs are computed over the AI stream alone.
    where = only_ai.split("bet_events")[1].split("bets AS")[0]
    assert "AND t.ab_group = 'AI'" in where

    # Games without AB test groups collapse the stored test labels into
    # Default; games with them pass the stored label through.
    assert "WHEN t.ab_group IN ('AB_TEST_A', 'AB_TEST_B') THEN 'Default'" in JOB.generate_query(
        **{**kwargs, "game_id": "SS01"}
    )
    assert "WHEN t.ab_group IN" not in base  # SS03 keeps the full vocabulary


def _ab_group_rows(rows, include_ab_tests=True):
    """Run the policy CASE against sqlite (TIMESTAMP literals stripped:
    sqlite compares ISO datetime strings directly)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bets (ab_label, user_id, bj_ts)")
    con.executemany("INSERT INTO bets VALUES (?, ?, ?)", rows)
    case = COMMON.ab_group_sql("ab_label", "user_id", "bj_ts", include_ab_tests=include_ab_tests)
    sql = f"SELECT {case} FROM bets ORDER BY rowid".replace("TIMESTAMP '", "'")
    return [r[0] for r in con.execute(sql)]


def test_slot_orders_query_composition():
    """The Athena-backing orders copy: rows plus the policy ab_group.
    Incomplete/test bets drop at the source; currency filtering stays
    downstream (a selection, not junk)."""
    orders_path = REPO_ROOT / "jobs/etl/sagemaker/slot_machine/etl_slot_orders_ab_group.py"
    sys.path.insert(0, str(REPO_ROOT / "jobs/etl/sagemaker"))
    try:
        spec = importlib.util.spec_from_file_location("_slot_orders_job", orders_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    sql = mod.generate_query(["SS03", "SS06"], date(2026, 8, 1), date(2026, 8, 5))
    assert "AS ab_group" in sql and "TIMESTAMP '2026-08-04 05:30:00'" in sql and "% 10" in sql
    assert "t.game_id IN ('SS03', 'SS06')" in sql
    assert "t.status = 'COMPLETED'" in sql and "op_code NOT IN" in sql
    assert "currency_type =" not in sql


def test_ab_group_policy_timestamp_gate():
    """AB groups: partition_ab ids before the empirically-located cutover
    (2026-08-04 05:30 Beijing — the deploy hour where mathtable serving
    flips), user-id last digit (0-3 Default, 4-5 A, 6-7 B, 8-9 AI) after."""
    rows = [
        # Before the cutover the digit is ignored — only the id counts
        # (including the early hours of 08-04, which a date gate mislabels).
        (COMMON.AI_GROUP_ID, 13, "2026-08-03 12:00:00"),
        (COMMON.AB_TEST_GROUP_A, 18, "2026-08-04 03:00:00"),
        ("some-other-id", 19, "2026-08-04 05:29:59"),
        (None, 15, "2026-08-03 23:00:00"),
        # From the cutover the id is ignored — only the digit counts.
        (COMMON.AI_GROUP_ID, 13, "2026-08-04 05:30:00"),
        ("whatever", 24, "2026-08-04 06:00:00"),
        (None, 37, "2026-08-05 00:00:00"),
        ("whatever", 58, "2026-08-04 23:59:59"),
    ]
    assert _ab_group_rows(rows) == [
        "AI",
        "AB_TEST_A",
        "Default",
        "Default",
        "Default",  # digit 3
        "AB_TEST_A",  # digit 4
        "AB_TEST_B",  # digit 7
        "AI",  # digit 8
    ]
    # Games without AB test groups collapse the test digits/ids into Default.
    no_tests = _ab_group_rows(
        [
            (COMMON.AB_TEST_GROUP_A, 11, "2026-08-03 12:00:00"),
            ("x", 25, "2026-08-04 06:00:00"),
            ("x", 39, "2026-08-04 06:00:00"),
        ],
        include_ab_tests=False,
    )
    assert no_tests == ["Default", "Default", "AI"]
