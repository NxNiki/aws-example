"""ETL Feature Engineer for SS03 (mahjiang streak).

Per-bet feature engineering for downstream bet segmentation / clustering.
The SQL pipeline + ETLScheduler wiring live in ``bituslabs_ds.features``;
this file only declares the SS03 configuration and invokes the runner.

Output (written via ETLScheduler):
    s3://bituslabs-team-ai/etl-results/jobs/output_ss03_feature_engineer/
        features_enriched/   (per-bet raw stats)
        features_grouped/    (per agg_group stats)

SS03 has four partition_ab buckets (AI / AB_TEST_A / AB_TEST_B / Default).
The script is configured to pull Default-only by default, matching the
historical SS03 cluster-analysis runs. Flip ``selected_groups`` to a tuple
containing other labels (e.g. ``("AI",)`` or ``("AB_TEST_A", "AB_TEST_B")``)
to materialise other slices.

Runs incrementally by default. ``session_start_date`` / ``session_group``
are stable across runs, so the lookback-window merge is safe. Default
lookback is derived from MAX_SESSION_INTERVAL via
``GameFeatureConfig.effective_lookback_days()``; pass ``--overwrite``
only when the SQL semantics change.
"""

from bituslabs_ds.features import FeaturePipelineRunner, GameFeatureConfig

CONFIG = GameFeatureConfig(
    game_id="SS03",
    output_prefix="output_ss03_feature_engineer",
    date_start="2026-01-01",
    date_end="2026-05-01",
    ai_groups=("AI", "AB_TEST_A", "AB_TEST_B", "Default"),
    selected_groups=("Default",),
    partition_cols=("math_table_id",),
    session_length=100,
)


if __name__ == "__main__":
    FeaturePipelineRunner(CONFIG).run_from_cli()
