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

import os
from pathlib import Path

import pytest

from bituslabs_ds.features import GameFeatureConfig, compose_enriched_query, compose_grouped_query

SNAPSHOTS_DIR = Path(__file__).parent / "features_snapshots"

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
    session_length=40,
    drop_incomplete_tail_groups=True,
    extra_where_clauses=("t.script_id = 'giftShop'",),
)

SS02 = GameFeatureConfig(
    game_id="SS02",
    output_prefix="output_ss02_feature_engineer",
    date_start="2026-01-01",
    date_end="2026-05-01",
    ai_groups=("AI", "Default"),
    selected_groups=("AI",),
    partition_cols=("math_table_id",),
    session_length=100,
)

SS03 = GameFeatureConfig(
    game_id="SS03",
    output_prefix="output_ss03_feature_engineer",
    date_start="2026-01-01",
    date_end="2026-05-01",
    ai_groups=("AI", "AB_TEST_A", "AB_TEST_B", "Default"),
    selected_groups=("Default",),
    partition_cols=("math_table_id",),
    session_length=100,
)


@pytest.mark.parametrize(
    "config,kind",
    [
        (SS01, "enriched"),
        (SS01, "grouped"),
        (SS02, "enriched"),
        (SS02, "grouped"),
        (SS03, "enriched"),
        (SS03, "grouped"),
    ],
    ids=lambda v: v if isinstance(v, str) else v.game_id.lower(),
)
def test_sql_matches_snapshot(config: GameFeatureConfig, kind: str) -> None:
    composer = compose_enriched_query if kind == "enriched" else compose_grouped_query
    actual = composer(config, config.date_start)

    snapshot_path = SNAPSHOTS_DIR / f"{config.game_id.lower()}_{kind}.sql"

    if os.environ.get("REGENERATE_SNAPSHOTS") == "1":
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual)
        pytest.skip(f"Regenerated {snapshot_path.relative_to(Path(__file__).parent.parent.parent)}")

    if not snapshot_path.exists():
        pytest.fail(
            f"Snapshot {snapshot_path} does not exist. "
            "Run REGENERATE_SNAPSHOTS=1 pytest tests/unit/test_features_sql_snapshots.py to create it."
        )

    expected = snapshot_path.read_text()
    assert actual == expected, (
        f"SQL for {config.game_id} {kind} drifted from snapshot {snapshot_path}. "
        "If this is intentional, regenerate with REGENERATE_SNAPSHOTS=1."
    )
