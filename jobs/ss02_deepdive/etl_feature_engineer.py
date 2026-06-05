"""ETL Feature Engineer for SS02.

Per-bet feature engineering for downstream bet segmentation / clustering.
The SQL pipeline + ETLScheduler wiring live in ``bituslabs_ds.features``;
this file only declares the SS02 configuration and invokes the runner.

Output (written via ETLScheduler):
    s3://bituslabs-team-ai/etl-results/jobs/output_ss02_feature_engineer/
        features_enriched/              (per-bet raw stats; shared across bin sizes)
        features_grouped_binsize_{N}/   (per agg_group stats, one dataset per bin_size)

Runs incrementally by default. ETLScheduler reads the existing S3 watermark
and re-queries from ``max(activity_date) - effective_lookback_days()``; the
dataclass derives a 2-day lookback from the 12-hour session-break threshold,
which is the shortest window that guarantees any active session's first bet is
visible. ``session_start_date`` / ``session_group`` are stable across runs,
so the new rows merge cleanly with existing rows on the lookback boundary.
Pass ``--overwrite`` only when the SQL semantics change.
"""

from bituslabs_ds.features import FeaturePipelineRunner, GameFeatureConfig

CONFIG = GameFeatureConfig(
    game_id="SS02",
    output_prefix="output_ss02_feature_engineer",
    date_start="2026-04-01",
    date_end="2026-06-01",
    # SS02 has no AB_TEST partitions. Keep "AI" in ai_groups so the CASE +
    # WHERE filter properly bucket / exclude AI users.
    ai_groups=("AI", "Default"),
    selected_groups=("AI",),
    partition_cols=("math_table_id",),
    bin_size=30,
    session_break_threshold_seconds=60 * 60 * 12,
)


if __name__ == "__main__":
    FeaturePipelineRunner(CONFIG).run_from_cli()
