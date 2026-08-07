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
    assert JOB.AI_GROUP_ID not in JOB.generate_query(**kwargs).split("CASE")[0]  # no scan filter by default

    only_ai = JOB.generate_query(**kwargs, ab_group="AI")
    # bet_events' WHERE keeps only the AI partition (raw id equality), so
    # with the run filter runs are computed over the AI stream alone.
    assert f"AND get_json_object(CAST(t.partition_ab AS STRING), '$[0]') = '{JOB.AI_GROUP_ID}'" in only_ai
