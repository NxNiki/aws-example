"""Snapshot tests for the unified feature engineering SQL.

For each per-game ``GameFeatureConfig`` we compose the two queries
(``features_enriched`` + ``features_grouped``) and compare the result
character-for-character to a checked-in golden file under
``tests/unit/features_snapshots/``.

If you intentionally change the SQL composition, regenerate the snapshots:

    REGENERATE_SNAPSHOTS=1 poetry run pytest tests/unit/test_features_sql_snapshots.py

and commit the updated files alongside the code change. The diff in the
snapshot files is exactly what a reviewer needs to inspect to validate the
SQL change.
"""

import importlib.util
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bituslabs_ds.features import GameFeatureConfig, compose_enriched_query, compose_grouped_query

SNAPSHOTS_DIR = Path(__file__).parent / "features_snapshots"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_jobs_config(relative_path: str, attr: str = "CONFIG") -> GameFeatureConfig:
    """Import a ``CONFIG`` constant from a job script under ``jobs/`` by file path.

    ``jobs/<game>/`` is not a Python package, so we side-load the module via
    importlib instead of a normal import.
    """
    spec = importlib.util.spec_from_file_location(f"_jobs_{relative_path.replace('/', '_')}", REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


# Configs match what the per-game job scripts under jobs/ss0x/etl_feature_engineer.py
# will use after Phases 2-4. Keep these in sync if you change the job-side configs.
SS01 = GameFeatureConfig(
    game_id="SS01",
    output_prefix="output_ss01_feature_engineer",
    date_start="2025-11-24",
    date_end="2026-11-24",
    ai_groups=("AI", "Default"),
    selected_groups=("Default",),
    partition_cols=(),
    bin_size=40,
    session_break_threshold_seconds=60 * 60 * 12,
    drop_incomplete_tail_groups=True,
    extra_where_clauses=("t.script_id = 'giftShop'",),
)

SS02 = GameFeatureConfig(
    game_id="SS02",
    output_prefix="output_ss02_feature_engineer",
    date_start="2026-04-01",
    date_end="2026-06-01",
    ai_groups=("AI", "Default"),
    selected_groups=("AI",),
    partition_cols=("math_table_id",),
    bin_size=30,
    session_break_threshold_seconds=60 * 60 * 12,
)

# SS03 fans out per ai_group slice (GROUPS/build_config in the job script);
# the configs are loaded from the job so the snapshots cannot drift from it.
_SS03_JOB = "jobs/etl/redshift/ss03_mahjiang_streak/etl_feature_engineer.py"
_ss03_build_config: Any = _load_jobs_config(_SS03_JOB, attr="build_config")
_ss03_groups: Any = _load_jobs_config(_SS03_JOB, attr="GROUPS")
SS03_CONFIGS: dict[str, GameFeatureConfig] = {group: _ss03_build_config(group) for group in sorted(_ss03_groups)}


def _check_snapshot(actual: str, snapshot_path: Path, label: str) -> None:
    if os.environ.get("REGENERATE_SNAPSHOTS") == "1":
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual)
        pytest.skip(f"Regenerated {snapshot_path.relative_to(Path(__file__).parent.parent.parent)}")

    if not snapshot_path.exists():
        pytest.fail(
            f"Snapshot {snapshot_path} does not exist. "
            "Run REGENERATE_SNAPSHOTS=1 pytest tests/unit/test_features_sql_snapshots.py to create it."
        )

    assert actual == snapshot_path.read_text(), (
        f"SQL for {label} drifted from snapshot {snapshot_path}. "
        "If this is intentional, regenerate with REGENERATE_SNAPSHOTS=1."
    )


# Enriched is bin-independent -> one snapshot per game (no bin in the name).
def _snapshot_stem(config: GameFeatureConfig) -> str:
    return config.output_prefix.removeprefix("output_").replace("_feature_engineer", "")


@pytest.mark.parametrize("config", [SS01, SS02, *SS03_CONFIGS.values()], ids=lambda c: _snapshot_stem(c))
def test_enriched_snapshot(config: GameFeatureConfig) -> None:
    actual = compose_enriched_query(config, config.date_start)
    _check_snapshot(actual, SNAPSHOTS_DIR / f"{_snapshot_stem(config)}_enriched.sql", f"{config.game_id} enriched")


# Grouped depends on bin_size -> one snapshot per (game, bin_size), composed from
# a scalar-bin_size config (mirroring how FeaturePipelineRunner fans out).
_GROUPED_CASES = [(cfg, n) for cfg in (SS01, SS02, *SS03_CONFIGS.values()) for n in cfg.bin_sizes()]


@pytest.mark.parametrize(
    "config,bin_size",
    _GROUPED_CASES,
    ids=lambda v: _snapshot_stem(v) if isinstance(v, GameFeatureConfig) else str(v),
)
def test_grouped_snapshot(config: GameFeatureConfig, bin_size: int) -> None:
    actual = compose_grouped_query(replace(config, bin_size=bin_size), config.date_start)
    snapshot_path = SNAPSHOTS_DIR / f"{_snapshot_stem(config)}_grouped_binsize_{bin_size}.sql"
    _check_snapshot(actual, snapshot_path, f"{config.game_id} grouped binsize={bin_size}")


# ---------------------------------------------------------------------------
# Per-game job-config parity checks. Each migrated jobs/ss0x/etl_feature_engineer.py
# must declare a CONFIG identical to the one used to generate the snapshot.
# ---------------------------------------------------------------------------
def test_ss01_jobs_config_matches_snapshot_config() -> None:
    jobs_config = _load_jobs_config("jobs/etl/redshift/ss01_wucaishen/etl_feature_engineer.py")
    assert jobs_config == SS01, (
        "jobs/etl/redshift/ss01_wucaishen/etl_feature_engineer.py CONFIG drifted from the snapshot's SS01 config. "
        "If the change is intentional, update both and regenerate the snapshot."
    )


def test_ss02_jobs_config_matches_snapshot_config() -> None:
    jobs_config = _load_jobs_config("jobs/etl/redshift/ss02_deepdive/etl_feature_engineer.py")
    assert jobs_config == SS02, (
        "jobs/etl/redshift/ss02_deepdive/etl_feature_engineer.py CONFIG drifted from the snapshot's SS02 config. "
        "If the change is intentional, update both and regenerate the snapshot."
    )


# (No ss03 parity test: SS03_CONFIGS are loaded from the job script itself,
# so job-vs-snapshot drift shows up directly as a snapshot diff.)
