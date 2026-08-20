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


def _load_policy():
    sys.path.insert(0, str(REPO_ROOT / "jobs/etl/sagemaker"))
    try:
        import group_policy
        import group_policy_sql

        return group_policy, group_policy_sql
    finally:
        sys.path.pop(0)


POLICY, POLICY_SQL = _load_policy()

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


AI_ID = "jojpin-9mokha-rexQug"
A_ID = "4f1a46ca-7baa-4452-9a40-ef21d9b33b57"
B_ID = "4a04df21-c749-4808-8e55-3a0b74c084d2"


def _ab_group_rows(rows):
    """Run the SLOT_ORDERS policy CASE against sqlite (TIMESTAMP literals
    stripped: sqlite compares ISO datetime strings directly)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bets (ab_label, user_id, created_utc)")
    con.executemany("INSERT INTO bets VALUES (?, ?, ?)", rows)
    case = POLICY_SQL.group_label_sql(
        "SLOT_ORDERS", {"partition_ab_label": "ab_label", "user_id": "user_id"}, "created_utc"
    )
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
    assert "AS ab_group" in sql and f"TIMESTAMP '{POLICY.AB_GROUP_DIGIT_POLICY_START_UTC}'" in sql and "% 10" in sql
    assert "t.game_id IN ('SS03', 'SS06')" in sql
    assert "t.status = 'COMPLETED'" in sql and "op_code NOT IN" in sql
    assert "currency_type =" not in sql


def test_ab_group_policy_timestamp_gate():
    """AB groups: partition_ab ids before the empirically-located cutover
    (2026-08-04 05:30 Beijing = 2026-08-03 21:30 UTC — the deploy hour where
    mathtable serving flips), user-id last digit (0-3 Default, 4-5 A, 6-7 B,
    8-9 AI) after. Rows here carry UTC timestamps."""
    rows = [
        # Before the cutover the digit is ignored — only the id counts
        # (including the early BJ hours of 08-04, which a date gate mislabels).
        (AI_ID, 13, "2026-08-03 04:00:00"),
        (A_ID, 18, "2026-08-03 19:00:00"),
        ("some-other-id", 19, "2026-08-03 21:29:59"),
        (None, 15, "2026-08-03 15:00:00"),
        # From the cutover the id is ignored — only the digit counts.
        (AI_ID, 13, "2026-08-03 21:30:00"),
        ("whatever", 24, "2026-08-03 22:00:00"),
        (None, 37, "2026-08-04 16:00:00"),
        ("whatever", 58, "2026-08-04 15:59:59"),
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
    # Games without their own AB arms fold the test labels into Default via
    # the stats policies' fold config (per-bet labels stay 4-way upstream).
    fold_case = POLICY_SQL._fold_expr(POLICY.GROUP_POLICY["SS01"], "g")
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE labels (g)")
    con.executemany("INSERT INTO labels VALUES (?)", [("AB_TEST_A",), ("AB_TEST_B",), ("AI",), ("Default",)])
    assert [r[0] for r in con.execute(f"SELECT {fold_case} FROM labels ORDER BY rowid")] == [
        "Default",
        "Default",
        "AI",
        "Default",
    ]


def _ss03_label_rows(rows):
    """Run the SS03 announced-cutover CASE against sqlite (TIMESTAMP literals
    stripped; ISO strings compare correctly)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bets (ab_label, user_id, created_at)")
    con.executemany("INSERT INTO bets VALUES (?, ?, ?)", rows)
    case = POLICY_SQL.group_label_sql("SS03", {"partition_ab_label": "ab_label", "user_id": "user_id"}, "created_at")
    sql = f"SELECT {case} FROM bets ORDER BY rowid".replace("TIMESTAMP '", "'")
    return [r[0] for r in con.execute(sql)]


def test_ss03_announced_cutover_labels():
    """Old era by partition_ab id (4f1a...=A continues digit-4/5's tables,
    4a04...=B), new era by digit from the ANNOUNCED 2026-08-03 23:00 UTC."""
    rows = [
        (AI_ID, 15, "2026-08-03 22:59:59"),
        (A_ID, 18, "2026-08-01 00:00:00"),
        ("4a04df21-c749-4808-8e55-3a0b74c084d2", 14, "2026-08-03 22:00:00"),
        ("wytsuj-fothap-5Qixda", 19, "2026-07-01 00:00:00"),
        (None, 18, "2026-08-03 20:00:00"),
        # from the announced moment (inclusive) only the digit counts.
        (AI_ID, 13, "2026-08-03 23:00:00"),
        ("whatever", 24, "2026-08-04 06:00:00"),
        (None, 37, "2026-08-05 00:00:00"),
        ("whatever", 58, "2026-08-04 23:59:59"),
    ]
    assert _ss03_label_rows(rows) == [
        "AI",
        "AB_TEST_A",
        "AB_TEST_B",
        "Default",
        "Default",
        "Default",  # digit 3
        "AB_TEST_A",  # digit 4
        "AB_TEST_B",  # digit 7
        "AI",  # digit 8
    ]


def test_session_day_attribution():
    """Sessions break on gaps > the threshold; every bet carries its
    session's start timestamp (numeric epochs stand in for timestamps)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bet_events (user_id, spin_id, created_at)")
    con.executemany(
        "INSERT INTO bet_events VALUES (?, ?, ?)",
        [
            # u1: bets at 0, 100, 250 (one session), then 1000 (new session).
            ("u1", 1, 0),
            ("u1", 2, 100),
            ("u1", 3, 250),
            ("u1", 4, 1000),
            # u2 interleaves but sessions never bridge users.
            ("u2", 5, 90),
        ],
    )
    sql = f"WITH {COMMON.session_day_ctes('bet_events', 'spin_id', 180)} SELECT user_id, spin_id, session_start_ts FROM session_days ORDER BY spin_id"
    assert list(con.execute(sql)) == [
        ("u1", 1, 0),
        ("u1", 2, 0),
        ("u1", 3, 0),
        ("u1", 4, 1000),
        ("u2", 5, 90),
    ]


def test_day_group_collapse_priority():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE bets_labeled (user_id, activity_date, bet_ab_group)")
    con.executemany(
        "INSERT INTO bets_labeled VALUES (?, ?, ?)",
        [
            # one AI bet claims the whole user-day.
            ("u1", "d1", "Default"),
            ("u1", "d1", "AI"),
            ("u1", "d1", "AB_TEST_B"),
            ("u2", "d1", "AB_TEST_B"),
            ("u2", "d1", "AB_TEST_A"),
            ("u3", "d1", "Default"),
            # days do not bleed into each other.
            ("u1", "d2", "AB_TEST_B"),
        ],
    )
    case = POLICY_SQL.collapse_case_over("SS03", "t.bet_ab_group", "t.activity_date")
    sql = (
        f"SELECT DISTINCT t.user_id, t.activity_date, {case} AS g FROM bets_labeled AS t"
        " ORDER BY t.user_id, t.activity_date"
    )
    assert list(con.execute(sql)) == [
        ("u1", "d1", "AI"),
        ("u1", "d2", "AB_TEST_B"),
        ("u2", "d1", "AB_TEST_A"),
        ("u3", "d1", "Default"),
    ]


def test_generate_query_group_grain():
    kwargs = dict(
        stats_agg_col="activity_date",
        game_id="SS03",
        effective_start=date(2026, 8, 1),
        output_end=date(2026, 8, 5),
        scan_start_utc=datetime(2026, 7, 31, 16),
        scan_end_utc=datetime(2026, 8, 4, 16),
        currency="CNY",
    )
    # session-start day attribution applies everywhere; group policy varies:
    # SS06 reads the stored per-bet label, and so does the SS03 run-filter
    # variant (opt-out) — neither collapses.
    base = JOB.generate_query(**{**kwargs, "game_id": "SS06"})
    assert "t.session_start_ts + INTERVAL '8' HOUR" in base
    assert "MAX(CASE WHEN" not in base
    variant = JOB.generate_query(**kwargs, min_mathtable_run=30)
    assert "MAX(CASE WHEN" not in variant.split("mathtable_run")[0]

    # the SS03 base query compiles its declared policy: label re-derived from
    # partition_ab_label under the announced cutover, collapsed per user-day.
    sess = JOB.generate_query(**kwargs)
    assert f"TIMESTAMP '{POLICY.SS03_AB_GROUP_ANNOUNCED_START_UTC}'" in sess
    assert "t.partition_ab_label IN" in sess
    assert "MAX(CASE WHEN CASE" in sess.replace("\n", " ") or "MAX(CASE WHEN" in sess
    assert "AS ab_group" in sess
