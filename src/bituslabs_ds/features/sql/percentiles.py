"""Percentile CTEs (p25 / median / p75) for the grouped output.

Redshift's ``PERCENTILE_CONT`` requires a single ``WITHIN GROUP (ORDER BY ...)``
clause per CTE, so we emit 12 small CTEs — one per metric or per
sort-key family — and join them all back into the final SELECT. The metric
expression usually equals the column name; when it's a derived expression
(rtp = payout / NULLIF(bet_amount, 0)), pass ``expr``.

The list of percentile families is held centrally as ``PERCENTILE_FAMILIES``
so the final SELECT (in ``final_select.py``) can iterate the same data when
emitting projections and JOINs.
"""

from dataclasses import dataclass

from bituslabs_ds.features.config import GameFeatureConfig
from bituslabs_ds.features.sql._helpers import all_partition_cols


@dataclass(frozen=True)
class PercentileFamily:
    """One percentile CTE: name + base metric + (optionally) the SQL expression."""

    cte_alias: str  # CTE name without the 'perc_' prefix, e.g. 'delta_t'
    output_prefix: str  # output column prefix, e.g. 'delta_t_seconds'
    expr: str  # SQL expression that gets percentiled

    @property
    def cte_name(self) -> str:
        return f"perc_{self.cte_alias}"


# Order matches the original handcrafted SQL so the final SELECT lines up
# (and old/new outputs are byte-comparable column-by-column).
PERCENTILE_FAMILIES: tuple[PercentileFamily, ...] = (
    PercentileFamily("delta_t", "delta_t_seconds", "delta_t_seconds"),
    PercentileFamily("delta_t_ng", "delta_t_seconds_nogap", "delta_t_seconds_nogap"),
    PercentileFamily("bet_amount", "bet_amount", "bet_amount"),
    PercentileFamily("delta_bet", "delta_bet_amount", "delta_bet_amount"),
    PercentileFamily("payout", "payout", "payout"),
    PercentileFamily("rtp", "rtp", "payout * 1.0 / NULLIF(bet_amount, 0)"),
    PercentileFamily("profit", "profit", "profit"),
    PercentileFamily("delta_payout", "delta_payout", "delta_payout"),
    PercentileFamily("balance", "balance_after_bet", "balance_after_bet"),
    PercentileFamily("streak", "streak", "streak"),
    PercentileFamily("win_streak", "win_streak", "win_streak"),
    PercentileFamily("lose_streak", "lose_streak", "lose_streak"),
)


def _build_one_percentile_cte(family: PercentileFamily, cfg: GameFeatureConfig) -> str:
    key = ", ".join(all_partition_cols(cfg))
    p = family.output_prefix
    e = family.expr
    return (
        f"{family.cte_name} AS (\n"
        f"    SELECT\n"
        f"        {key},\n"
        f"        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {e}) AS {p}_p25,\n"
        f"        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {e}) AS {p}_median,\n"
        f"        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {e}) AS {p}_p75\n"
        f"    FROM raw_stats\n"
        f"    GROUP BY {key}\n"
        f")"
    )


def build_percentile_ctes(cfg: GameFeatureConfig) -> str:
    """Emit all 12 percentile CTEs, comma-separated."""
    return ",\n".join(_build_one_percentile_cte(f, cfg) for f in PERCENTILE_FAMILIES)
