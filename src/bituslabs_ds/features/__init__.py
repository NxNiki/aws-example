"""Unified per-bet feature engineering pipeline.

See ``GameFeatureConfig`` for the per-game knobs and
``FeaturePipelineRunner`` for wiring a game's config to Redshift +
ETLScheduler. The SQL itself is composed from CTE builders under
``bituslabs_ds.features.sql``.
"""

from bituslabs_ds.features.config import AI_GROUP_PARTITION_IDS, GameFeatureConfig
from bituslabs_ds.features.runner import FeaturePipelineRunner
from bituslabs_ds.features.sql.pipeline import compose_enriched_query, compose_grouped_query

__all__ = [
    "AI_GROUP_PARTITION_IDS",
    "FeaturePipelineRunner",
    "GameFeatureConfig",
    "compose_enriched_query",
    "compose_grouped_query",
]
