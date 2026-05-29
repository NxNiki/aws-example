"""ETL Feature Engineer for SS01 (wucaishen).

Per-bet feature engineering for downstream bet segmentation / clustering.
The SQL pipeline + ETLScheduler wiring live in ``bituslabs_ds.features``;
this file only declares the SS01 configuration and invokes the runner.

Output (written via ETLScheduler):
    s3://bituslabs-team-ai/etl-results/jobs/output_ss01_feature_engineer/
        features_enriched/   (per-bet raw stats)
        features_grouped/    (per agg_group stats; tail-incomplete groups dropped)

SS01 specifics vs SS02 / SS03:
* No ``math_table_id`` in the aggregation key (``partition_cols=()``).
* Smaller ``session_length`` (40 vs 100) — every agg_group is a window of 40 bets.
* ``drop_incomplete_tail_groups=True`` — the final partial agg_group per session
  is dropped via ``HAVING COUNT = session_length``.
* SS01 raw data has both AI and Default partitions; this job selects Default
  only (matching the historical SS01 cluster-analysis runs). The output's
  ``ai_group`` column will therefore be the constant ``'Default'``.
* ``script_id = 'giftShop'`` filter is appended via ``extra_where_clauses``.

Runs incrementally by default. ``session_start_date`` / ``session_group``
are stable across runs, so the lookback-window merge is safe. Default
lookback is derived from MAX_SESSION_INTERVAL via
``GameFeatureConfig.effective_lookback_days()`` (8 days here, since
SS01 keeps the 7-day MAX_SESSION_INTERVAL); pass ``--overwrite`` only
when the SQL semantics change.
"""

from bituslabs_ds.features import FeaturePipelineRunner, GameFeatureConfig

CONFIG = GameFeatureConfig(
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


if __name__ == "__main__":
    FeaturePipelineRunner(CONFIG).run_from_cli()
