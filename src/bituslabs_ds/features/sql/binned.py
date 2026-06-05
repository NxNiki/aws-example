"""binned CTE: add the per-bin ``agg_group`` on top of ``raw_stats``.

``raw_stats`` is bin-independent (it carries ``session_bet_index``, the 1-based
row number of a bet within its session). The grouped query buckets those bets
into ``agg_group``s of ``bin_size`` consecutive bets:

    agg_group = floor((session_bet_index - 1) / bin_size)

This is the only CTE that depends on ``bin_size``, which is why the enriched
output (``SELECT * FROM raw_stats``) can be shared across all bin sizes while
the grouped output is produced once per size.
"""

from bituslabs_ds.features.config import GameFeatureConfig


def build_binned_cte(cfg: GameFeatureConfig, bin_size: int) -> str:
    return "\n".join(
        [
            "binned AS (",
            "    SELECT",
            "        t.*,",
            f"        ROUND((t.session_bet_index - 1) / {bin_size}) AS agg_group",
            "    FROM raw_stats AS t",
            ")",
        ]
    )
