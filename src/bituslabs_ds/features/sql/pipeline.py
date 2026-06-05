"""Compose the two final SQL queries from the per-CTE builders.

* ``compose_enriched_query``: the per-bet ``raw_stats`` output (one row per
  aggregated bet). Used by ``features_enriched``.
* ``compose_grouped_query``: per-agg-group rollup (one row per
  ``(user_id, ai_group, [partition_cols], session_group, agg_group)``). Used by
  ``features_grouped``.

Both queries share the same six "pre-aggregation" CTEs (``user_bets`` →
``raw_stats``); the grouped query appends the percentile CTEs, ``stats_base``,
and the final assembly SELECT.
"""

from bituslabs_ds.features.config import GameFeatureConfig
from bituslabs_ds.features.sql.binned import build_binned_cte
from bituslabs_ds.features.sql.delta_stats import build_delta_stats_cte
from bituslabs_ds.features.sql.final_select import build_final_select
from bituslabs_ds.features.sql.free_game import build_agg_free_game_cte, build_free_game_group_cte
from bituslabs_ds.features.sql.percentiles import build_percentile_ctes
from bituslabs_ds.features.sql.raw_stats import build_raw_stats_cte
from bituslabs_ds.features.sql.sessions import build_user_group_cte
from bituslabs_ds.features.sql.stats_base import build_stats_base_cte
from bituslabs_ds.features.sql.user_bets import build_user_bets_cte


def _preaggregation_ctes(cfg: GameFeatureConfig, start_date: str) -> list[str]:
    return [
        build_user_bets_cte(cfg, start_date),
        build_free_game_group_cte(cfg),
        build_agg_free_game_cte(cfg),
        build_delta_stats_cte(cfg),
        build_user_group_cte(cfg),
        build_raw_stats_cte(cfg),
    ]


def _join_ctes_with(prefix_cte: str, ctes: list[str]) -> str:
    """Join CTEs with ``,\\n\\n`` and prepend the first one with ``prefix_cte``."""
    body = ",\n\n".join(ctes)
    return f"{prefix_cte}{body}"


def compose_enriched_query(cfg: GameFeatureConfig, start_date: str) -> str:
    """Per-bet enriched output. Selects every column of ``raw_stats``."""
    ctes = _preaggregation_ctes(cfg, start_date)
    return _join_ctes_with("WITH ", ctes) + "\nSELECT * FROM raw_stats;\n"


def compose_grouped_query(cfg: GameFeatureConfig, start_date: str) -> str:
    """Per-agg-group rollup with percentiles + derived ratios.

    Requires a scalar ``cfg.bin_size`` (the runner composes this once per bin via
    ``replace(cfg, bin_size=n)``); the ``binned`` CTE derives ``agg_group`` from the
    bin-independent ``session_bet_index`` in ``raw_stats``.
    """
    if isinstance(cfg.bin_size, list):
        raise ValueError(
            f"compose_grouped_query needs a scalar bin_size, got list {cfg.bin_size!r}; "
            "the runner should pass one bin per call."
        )
    ctes = _preaggregation_ctes(cfg, start_date)
    binned = build_binned_cte(cfg, cfg.bin_size)
    # build_percentile_ctes returns 12 already-comma-separated CTEs.
    percentile_block = build_percentile_ctes(cfg)
    stats_base = build_stats_base_cte(cfg)
    final = build_final_select(cfg)
    return (
        _join_ctes_with("WITH ", ctes)
        + ",\n\n"
        + binned
        + ",\n\n"
        + percentile_block
        + ",\n\n"
        + stats_base
        + "\n"
        + final
        + "\n;\n"
    )
