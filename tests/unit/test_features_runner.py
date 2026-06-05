"""Unit tests for FeaturePipelineRunner's bin_size fan-out + config validation.

These cover the pure planning logic (``_per_bin_configs`` + ``bin_size``
validation) without touching S3/Redshift.
"""

from dataclasses import replace

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


def test_scalar_bin_size_is_suffixed():
    """A scalar bin_size still gets the _binsize_{N} suffix (single run)."""
    runner = FeaturePipelineRunner(_cfg(bin_size=100))
    configs = runner._per_bin_configs()
    assert [c.bin_size for c in configs] == [100]
    assert [c.output_prefix for c in configs] == ["output_ss03_feature_engineer_binsize_100"]


def test_list_bin_size_fans_out():
    """A list bin_size produces one scalar-bin_size config per size, each suffixed."""
    runner = FeaturePipelineRunner(_cfg(bin_size=[50, 100]))
    configs = runner._per_bin_configs()
    assert [c.bin_size for c in configs] == [50, 100]
    assert [c.output_prefix for c in configs] == [
        "output_ss03_feature_engineer_binsize_50",
        "output_ss03_feature_engineer_binsize_100",
    ]
    # every derived config carries a scalar bin_size (the SQL builders require it)
    assert all(isinstance(c.bin_size, int) for c in configs)


@pytest.mark.parametrize("bad", [0, -5, [], [50, 0], [50, -1], 1.5, [50, 1.5], True, [True]])
def test_invalid_bin_size_rejected(bad):
    with pytest.raises(ValueError, match="bin_size"):
        _cfg(bin_size=bad)
