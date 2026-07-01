"""Snapshot tests for dashboard-feeding ETL SQL.

The per-user-group ``generate_query`` in the fish_hunter and ss03 dashboard ETL
jobs builds the daily/weekly/monthly stats that back their dashboard tabs. Each
granularity is compared character-for-character to a checked-in golden file
under ``tests/unit/etl_snapshots/`` so an accidental change to the SQL (e.g. a
stale AB-test id mapping) fails CI and the snapshot diff is exactly what a
reviewer inspects.

If you intentionally change the SQL, regenerate the snapshots:

    REGENERATE_SNAPSHOTS=1 poetry run pytest tests/unit/test_etl_sql_snapshots.py

and commit the updated ``.sql`` files alongside the code change.
"""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

SNAPSHOTS_DIR = Path(__file__).parent / "etl_snapshots"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Full-history start date the schedulers use for an --overwrite reload
# (ETLScheduler.default_start_date); the snapshots are rendered from it.
_FULL_HISTORY_START = "2025-01-01"

_GRANULARITIES = [("daily", "activity_date"), ("weekly", "activity_week"), ("monthly", "activity_month")]


def _load_job_module(relative_path: str) -> ModuleType:
    """Side-load a job script as a module (``jobs/<game>/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_jobs_{relative_path.replace('/', '_')}", REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FISH_HUNTER = _load_job_module("jobs/fish_hunter/etl_game_stats_daily_by_user.py")
_SS03 = _load_job_module("jobs/ss03_mahjiang_streak/etl_game_stats_daily_by_user_group.py")
_SS06 = _load_job_module("jobs/ss06_pocket_soccer/etl_game_stats_daily_by_user_group.py")

# (job module, snapshot stem) — one snapshot file per (stem, granularity).
_JOBS = [
    (_FISH_HUNTER, "fish_hunter"),
    (_SS03, "ss03_mahjiang_streak"),
    (_SS06, "ss06_pocket_soccer"),
]
_CASES = [(module, stem, label, agg_col) for module, stem in _JOBS for label, agg_col in _GRANULARITIES]


def _check_snapshot(actual: str, snapshot_path: Path, label: str) -> None:
    if os.environ.get("REGENERATE_SNAPSHOTS") == "1":
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual)
        pytest.skip(f"Regenerated {snapshot_path.relative_to(REPO_ROOT)}")

    if not snapshot_path.exists():
        pytest.fail(
            f"Snapshot {snapshot_path} does not exist. "
            "Run REGENERATE_SNAPSHOTS=1 pytest tests/unit/test_etl_sql_snapshots.py to create it."
        )

    assert actual == snapshot_path.read_text(), (
        f"SQL for {label} drifted from snapshot {snapshot_path}. "
        "If this is intentional, regenerate with REGENERATE_SNAPSHOTS=1."
    )


@pytest.mark.parametrize("module,stem,label,agg_col", _CASES, ids=[f"{c[1]}-{c[2]}" for c in _CASES])
def test_etl_stats_snapshot(module: ModuleType, stem: str, label: str, agg_col: str) -> None:
    actual = module.generate_query(agg_col, _FULL_HISTORY_START).strip() + "\n"
    _check_snapshot(actual, SNAPSHOTS_DIR / f"{stem}_{label}.sql", f"{stem} {label}")
