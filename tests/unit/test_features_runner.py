"""Unit tests for FeaturePipelineRunner's bin_size fan-out + config validation.

Covers the bin_size validation, the ``bin_sizes()`` normalization that drives
the fan-out, and (with ETLScheduler mocked) the run plan: one shared enriched
job + one grouped job per bin size. No S3/Redshift.
"""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from bituslabs_ds.features import FeaturePipelineRunner, GameFeatureConfig

_BASE = GameFeatureConfig(
    game_id="SS03",
    output_prefix="output_ss03_feature_engineer",
    date_start="2026-01-01",
    date_end="2026-05-01",
    ai_groups=("AI", "Default"),
    selected_groups=("Default",),
    partition_cols=("math_table_id",),
)


def _cfg(**overrides) -> GameFeatureConfig:
    # replace() re-runs __post_init__, so invalid overrides raise as expected.
    return replace(_BASE, **overrides)


def test_bin_sizes_normalizes_scalar_and_list():
    assert _cfg(bin_size=100).bin_sizes() == [100]
    assert _cfg(bin_size=[30, 50, 70, 100]).bin_sizes() == [30, 50, 70, 100]


@pytest.mark.parametrize(
    "bin_size,expected_jobs",
    [
        (100, ["features_enriched", "features_grouped_binsize_100"]),
        (
            [30, 50, 70, 100],
            [
                "features_enriched",
                "features_grouped_binsize_30",
                "features_grouped_binsize_50",
                "features_grouped_binsize_70",
                "features_grouped_binsize_100",
            ],
        ),
    ],
)
def test_run_emits_one_enriched_plus_grouped_per_bin(bin_size, expected_jobs):
    """Enriched is run once (shared); grouped once per bin under the same root."""
    runner = FeaturePipelineRunner(_cfg(bin_size=bin_size))
    with (
        patch("bituslabs_ds.features.runner.ETLScheduler") as MockSched,
        patch.object(FeaturePipelineRunner, "_check_config_drift"),
        patch.object(FeaturePipelineRunner, "_write_config_sidecar"),
    ):
        scheduler = MockSched.return_value
        runner.run(MagicMock())
        job_names = [call.kwargs["job_name"] for call in scheduler.run_incremental_job.call_args_list]
    assert job_names == expected_jobs


@pytest.mark.parametrize("bad", [0, -5, [], [50, 0], [50, -1], 1.5, [50, 1.5], True, [True]])
def test_invalid_bin_size_rejected(bad):
    with pytest.raises(ValueError, match="bin_size"):
        _cfg(bin_size=bad)
