"""Snapshot tests for dashboard-feeding ETL SQL.

``generate_query`` in ``jobs/fish_hunter/etl_game_stats_daily_by_user.py`` builds
the daily/weekly/monthly per-user stats that back the dashboard fish_hunter tab.
Each granularity is compared character-for-character to a checked-in golden file
under ``tests/unit/etl_snapshots/`` so an accidental change to the SQL fails CI
and the snapshot diff is exactly what a reviewer inspects.

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


def _load_job_module(relative_path: str) -> ModuleType:
    """Side-load a job script as a module (``jobs/<game>/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_jobs_{relative_path.replace('/', '_')}", REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FISH_HUNTER = _load_job_module("jobs/fish_hunter/etl_game_stats_daily_by_user.py")

# (snapshot label, stats_agg_col) for the three granularities the scheduler runs.
_FISH_HUNTER_CASES = [
    ("daily", "activity_date"),
    ("weekly", "activity_week"),
    ("monthly", "activity_month"),
]


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


@pytest.mark.parametrize("label,agg_col", _FISH_HUNTER_CASES, ids=[c[0] for c in _FISH_HUNTER_CASES])
def test_fish_hunter_stats_snapshot(label: str, agg_col: str) -> None:
    actual = _FISH_HUNTER.generate_query(agg_col, _FISH_HUNTER.DEFAULT_DATE_START).strip() + "\n"
    _check_snapshot(actual, SNAPSHOTS_DIR / f"fish_hunter_{label}.sql", f"fish_hunter {label}")
