"""ETLScheduler data-integrity guardrails: compaction/dedup failures must not
stay log-only (a silently skipped compaction leaves each run's lookback
re-pull as duplicate rows that downstream consumers sum twice)."""

from datetime import date

import pandas as pd
import pytest


@pytest.fixture()
def scheduler(tmp_path, monkeypatch):
    from bituslabs_ds.etl import ETLScheduler

    for var in ("SLACK_USER_TOKEN", "SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID"):
        monkeypatch.delenv(var, raising=False)
    return ETLScheduler(data_loader=None, storage_root=tmp_path, lookback_days=3)


def _write(scheduler, job_name: str, df: pd.DataFrame, filename: str) -> None:
    job_dir = scheduler._job_path(job_name)
    job_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(job_dir / filename, index=False)


def test_alert_records_and_survives_missing_slack(scheduler):
    scheduler._alert("daily_stats", "Compaction FAILED: boom")
    assert scheduler.alerts == ["[daily_stats] Compaction FAILED: boom"]


def test_verify_unique_keys_flags_duplicates(scheduler):
    keys = ["activity_date", "user_id", "ai_group"]
    clean = pd.DataFrame(
        {"activity_date": ["2026-07-15", "2026-07-15"], "user_id": ["u1", "u2"], "ai_group": ["AI", "AI"], "v": [1, 2]}
    )
    _write(scheduler, "clean_job", clean, "a.parquet")
    scheduler._verify_unique_keys("clean_job", keys, "none", date(2026, 7, 15))
    assert scheduler.alerts == []

    # The same key landing in two files (a failed compaction) must alert.
    _write(scheduler, "dup_job", clean, "a.parquet")
    _write(scheduler, "dup_job", clean.assign(v=[3, 4]), "b.parquet")
    scheduler._verify_unique_keys("dup_job", keys, "none", date(2026, 7, 15))
    assert len(scheduler.alerts) == 1
    assert "2 duplicate" in scheduler.alerts[0] and "dup_job" in scheduler.alerts[0]


def test_verify_unique_keys_alerts_when_dataset_unreadable(scheduler, monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(scheduler, "_alert", lambda job, msg: called.append(msg))
    job_dir = scheduler._job_path("broken_job")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "corrupt.parquet").write_bytes(b"not parquet")
    scheduler._verify_unique_keys("broken_job", ["user_id"], "none", date(2026, 7, 15))
    assert called and "could not read" in called[0]


def test_chunk_windows_period_aligned_and_contiguous(scheduler):
    from datetime import datetime, timedelta

    for chunk_days, lookback in [(30, 3), (28, 7), (62, 31)]:
        windows = scheduler._chunk_windows("2026-01-05", chunk_days, lookback)
        assert windows[-1][1] is None
        for (s1, e1), (s2, _) in zip(windows, windows[1:]):
            assert e1 == s2  # contiguous, no gap or overlap
        for _, end in windows[:-1]:
            d = datetime.strptime(end, "%Y-%m-%d").date()
            if lookback >= 30:
                assert d.day == 1
            elif lookback >= 7:
                assert d.weekday() == 0

    # A short catch-up stays a single open-ended window (plain incremental).
    start = (datetime.now().date() - timedelta(days=3)).strftime("%Y-%m-%d")
    assert scheduler._chunk_windows(start, 30, 3) == [(start, None)]
