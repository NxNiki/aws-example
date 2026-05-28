"""ETL Feature Engineer for SS02.

Per-bet feature engineering for downstream bet segmentation / clustering.
The SQL pipeline + ETLScheduler wiring live in ``bituslabs_ds.features``;
this file only declares the SS02 configuration and invokes the runner.

Output (written via ETLScheduler):
    s3://bituslabs-team-ai/etl-results/jobs/output_ss02_feature_engineer/
        features_enriched/   (per-bet raw stats)
        features_grouped/    (per agg_group stats)

Window functions (session_group, streak_group, ...) reach back across the
watermark, so the canonical run is ``--overwrite``; incremental runs near
the leading edge may have session / streak values that ignore prior history.
"""

from bituslabs_ds.features import FeaturePipelineRunner, GameFeatureConfig

CONFIG = GameFeatureConfig(
    game_id="SS02",
    output_prefix="output_ss02_feature_engineer",
    date_start="2026-01-01",
    date_end="2026-05-01",
    # SS02 has no AB_TEST partitions. Keep "AI" in ai_groups so the CASE +
    # WHERE filter properly bucket / exclude AI users.
    ai_groups=("AI", "Default"),
    selected_groups=("AI",),
    partition_cols=("math_table_id",),
    session_length=100,
)


if __name__ == "__main__":
    FeaturePipelineRunner(CONFIG).run_from_cli()
