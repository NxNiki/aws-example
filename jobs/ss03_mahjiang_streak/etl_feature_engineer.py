"""ETL Feature Engineer for SS03 (mahjiang streak).

Per-bet feature engineering for downstream bet segmentation / clustering.
The SQL pipeline + ETLScheduler wiring live in ``bituslabs_ds.features``;
this file only declares the SS03 configuration and invokes the runner.

Output (written via ETLScheduler), one dataset root per ai_group slice:
    s3://bituslabs-team-ai/etl-results/jobs/output_ss03_feature_engineer/      (Default)
    s3://bituslabs-team-ai/etl-results/jobs/output_ss03_feature_engineer_ai/   (AI)
each containing:
        features_enriched/              (per-bet raw stats; shared across bin sizes)
        features_grouped_binsize_{N}/   (per agg_group stats, one dataset per bin_size)

SS03 has four partition_ab buckets (AI / AB_TEST_A / AB_TEST_B / Default).
``--group`` selects which slice to materialise: ``default`` (the training group
for cluster analysis) or ``ai`` (the inference group). Run once per group:

    poetry run python jobs/ss03_mahjiang_streak/etl_feature_engineer.py --group default --overwrite
    poetry run python jobs/ss03_mahjiang_streak/etl_feature_engineer.py --group ai --overwrite

Runs incrementally by default. ``session_start_date`` / ``session_group``
are stable across runs, so the lookback-window merge is safe. Default
lookback is derived from ``session_break_threshold_seconds`` via
``GameFeatureConfig.effective_lookback_days()`` (2 days for the 12-hour
threshold); pass ``--overwrite`` only when the SQL semantics change (which
includes flipping the bin_size set or selected_groups, as done here).
"""

import argparse

from bituslabs_ds.features import FeaturePipelineRunner, GameFeatureConfig

# ai_group slice -> (output_prefix, selected_groups). Separate prefixes keep the
# Default and AI datasets isolated; the cluster pipeline reads each via its own group.
GROUPS = {
    "default": ("output_ss03_feature_engineer", ("Default",)),
    "ai": ("output_ss03_feature_engineer_ai", ("AI",)),
}


def build_config(group: str) -> GameFeatureConfig:
    output_prefix, selected_groups = GROUPS[group]
    return GameFeatureConfig(
        game_id="SS03",
        output_prefix=output_prefix,
        date_start="2026-01-01",
        date_end="2026-07-01",
        ai_groups=("AI", "AB_TEST_A", "AB_TEST_B", "Default"),
        selected_groups=selected_groups,
        partition_cols=("math_table_id",),
        bin_size=[30, 50, 70],
        session_break_threshold_seconds=60 * 60 * 12,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS03 feature engineering ETL", add_help=False)
    parser.add_argument(
        "--group",
        choices=sorted(GROUPS),
        default="default",
        help="ai_group slice to materialise (default: 'default').",
    )
    args, remaining = parser.parse_known_args()
    FeaturePipelineRunner(build_config(args.group)).run_from_cli(remaining)
