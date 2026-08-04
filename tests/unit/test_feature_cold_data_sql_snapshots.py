"""Snapshot tests for the SageMaker cold-data feature-engineering Spark SQL.

For each ai_group slice we render the enriched query, and per bin size the
grouped query, with FIXED dates, and compare character-for-character to a
checked-in golden file under ``tests/unit/feature_cold_data_snapshots/`` —
the Spark-side twin of ``test_features_sql_snapshots.py`` (Redshift).

If you intentionally change the SQL composition, regenerate the snapshots:

    REGENERATE_SNAPSHOTS=1 poetry run pytest tests/unit/test_feature_cold_data_sql_snapshots.py

and commit the updated files alongside the code change. The diff in the
snapshot files is exactly what a reviewer needs to inspect to validate the
SQL change.
"""

import importlib.util
import os
import sys
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

SNAPSHOTS_DIR = Path(__file__).parent / "feature_cold_data_snapshots"
REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_PATH = REPO_ROOT / "jobs/etl/sagemaker/slot_machine/etl_feature_engineer_cold_data.py"

# Fixed window so snapshots are deterministic (the runtime window is the only
# templated piece of the SQL).
SCAN_START = date(2026, 6, 17)
SCAN_END = date(2026, 8, 1)


def _load_job_module():
    sys.path.insert(0, str(REPO_ROOT / "jobs/etl/sagemaker"))
    try:
        spec = importlib.util.spec_from_file_location("_feature_cold_data_job", JOB_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


JOB = _load_job_module()


def _check_or_regenerate(name: str, sql: str) -> None:
    path = SNAPSHOTS_DIR / f"{name}.sql"
    if os.environ.get("REGENERATE_SNAPSHOTS"):
        SNAPSHOTS_DIR.mkdir(exist_ok=True)
        path.write_text(sql)
        pytest.skip(f"regenerated {path.name}")
    assert path.exists(), f"missing snapshot {path}; run with REGENERATE_SNAPSHOTS=1 to create it"
    assert sql == path.read_text(), (
        f"{path.name} drifted from the composed SQL. If the change is intentional, regenerate with "
        "REGENERATE_SNAPSHOTS=1 and commit the diff."
    )


@pytest.mark.parametrize("group", sorted(JOB.GROUPS))
def test_enriched_sql_snapshot(group):
    _, group_filter, _, _ = JOB.GROUPS[group]
    _check_or_regenerate(f"{group}_features_enriched", JOB.enriched_sql(group_filter, SCAN_START, SCAN_END))


@pytest.mark.parametrize("group,bin_size", [(g, n) for g in sorted(JOB.GROUPS) for n in JOB.GROUPS[g][2]])
def test_grouped_sql_snapshot(group, bin_size):
    _check_or_regenerate(f"{group}_features_grouped_binsize_{bin_size}", JOB.grouped_sql(f"enriched_{group}", bin_size))
