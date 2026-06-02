"""Final SELECT: joins stats_base to all percentile CTEs + emits derived ratios.

The projection has three classes of columns:

1. *Aggregation key + identifiers* — user_id, ai_group, partition_cols,
   session_group, agg_group, activity_date, min/max_created_at, bet_rounds,
   fg_rounds. Carried straight from ``stats_base``.

2. *Raw aggregates + percentiles* — e.g. ``bet_amount_avg`` from stats_base
   paired with ``bet_amount_p25/median/p75`` from ``perc_bet_amount``. Emitted
   in the same order as the original handcrafted SQL so columns line up
   visually with the legacy outputs.

3. *Derived ratios* — coefficient of variation (``_cv``), spike ratio,
   drawdown ratio, accumulator-to-avg ratios. These divide an aggregate by
   ``bet_amount_avg`` (or ``balance_after_bet_avg`` for the balance family),
   guarded with ``NULLIF(..., 0)``. Streak metrics are additionally wrapped in
   ``COALESCE(..., 0)`` because users with no streak rows produce NULL
   aggregates in Redshift.
"""

from bituslabs_ds.features.config import GameFeatureConfig
from bituslabs_ds.features.sql._helpers import all_partition_cols, equi_join_on
from bituslabs_ds.features.sql.percentiles import PERCENTILE_FAMILIES


def _key_select_lines(cfg: GameFeatureConfig) -> list[str]:
    return [f"    b.{c}," for c in all_partition_cols(cfg)] + [
        "    b.activity_date,",
        "    b.min_created_at,",
        "    b.max_created_at,",
        "    b.bet_rounds,",
        "    b.fg_rounds,",
    ]


def _metric_block(
    metric: str,
    perc_alias: str,
    *,
    norm_by: str = "bet_amount_avg",
    include_extremes: bool = True,
    include_payout_rate: bool = False,
    label: str | None = None,
    coalesce_derived: bool = False,
) -> list[str]:
    """Aggregate + percentile + derived ratios for one metric family.

    Args:
        metric: column-prefix in stats_base (e.g. ``"bet_amount"``).
        perc_alias: short alias used in the JOIN (``p3`` for ``perc_bet_amount``).
        norm_by: denominator for ``_cv`` / drawdown / spike ratios.
        include_extremes: when False, skip ``_min/_drawdown/_max/_spike`` (e.g. delta_t doesn't get them).
        include_payout_rate: only True for ``payout`` (extra ``payout_rate`` line).
        label: section comment text. Defaults to ``metric`` with underscores spaced.
        coalesce_derived: wrap derived ratios in COALESCE(..., 0) (used by delta_*).
    """
    cmt = label or metric.replace("_", " ")
    lines = [f"    -- {cmt}", f"    b.{metric}_avg,"]
    lines.append(f"    {perc_alias}.{metric}_p25,")
    lines.append(f"    {perc_alias}.{metric}_median,")
    lines.append(f"    {perc_alias}.{metric}_p75,")
    lines.append(f"    b.{metric}_std,")

    def maybe_coalesce(expr: str) -> str:
        return f"COALESCE({expr}, 0)" if coalesce_derived else expr

    lines.append(f"    {maybe_coalesce(f'b.{metric}_std * 1.0 / NULLIF(b.{norm_by}, 0)')} AS {metric}_cv,")
    if include_extremes:
        lines.append(f"    b.{metric}_min,")
        lines.append(
            f"    {maybe_coalesce(f'b.{metric}_min * 1.0 / NULLIF(b.{norm_by}, 0)')} AS {metric}_drawdown_ratio,"
        )
        lines.append(f"    b.{metric}_max,")
        lines.append(f"    {maybe_coalesce(f'b.{metric}_max * 1.0 / NULLIF(b.{norm_by}, 0)')} AS {metric}_spike_ratio,")
    if include_payout_rate:
        lines.append("    b.payout_rate,")
    return lines


def _profit_block() -> list[str]:
    # profit is special: min_profit_ratio / max_profit_ratio (NOT *_drawdown / *_spike)
    # plus accum_pos_profit / accum_neg_profit / profit_rate.
    return [
        "    -- profit",
        "    b.profit_avg,",
        "    p7.profit_p25,",
        "    p7.profit_median,",
        "    p7.profit_p75,",
        "    b.profit_std,",
        "    b.profit_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS profit_cv,",
        "    b.profit_min,",
        "    b.profit_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS min_profit_ratio,",
        "    b.profit_max,",
        "    b.profit_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS max_profit_ratio,",
        "    b.accum_pos_profit,",
        "    b.accum_pos_profit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_pos_profit_ratio,",
        "    b.accum_neg_profit,",
        "    b.accum_neg_profit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_neg_profit_ratio,",
        "    b.profit_rate,",
    ]


def _accum_delta_bet_block() -> list[str]:
    return [
        "    -- accumulative delta bet",
        "    b.accum_pos_delta_bet_amount,",
        "    b.accum_neg_delta_bet_amount,",
        "    b.accum_pos_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_pos_delta_bet_amount_ratio,",
        "    b.accum_neg_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_neg_delta_bet_amount_ratio,",
    ]


def _rtp_block() -> list[str]:
    return [
        "    -- rtp",
        "    b.rtp_mean,",
        "    b.rtp_max,",
        "    b.rtp_min,",
        "    p6.rtp_p25,",
        "    p6.rtp_median,",
        "    p6.rtp_p75,",
    ]


def _deposit_withdraw_block() -> list[str]:
    return [
        "    -- deposit/withdraw",
        "    b.num_deposit,",
        "    b.accum_deposit,",
        "    b.accum_deposit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_deposit_ratio,",
        "    b.num_withdraw,",
        "    b.accum_withdraw,",
        "    b.accum_withdraw * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_withdraw_ratio,",
    ]


def _streak_block(metric: str, perc_alias: str) -> list[str]:
    return [
        f"    COALESCE(b.{metric}_avg, 0) AS {metric}_avg,",
        f"    COALESCE({perc_alias}.{metric}_p25, 0) AS {metric}_p25,",
        f"    COALESCE({perc_alias}.{metric}_median, 0) AS {metric}_median,",
        f"    COALESCE({perc_alias}.{metric}_p75, 0) AS {metric}_p75,",
        f"    COALESCE(b.{metric}_std, 0) AS {metric}_std,",
        f"    COALESCE(b.{metric}_min, 0) AS {metric}_min,",
        f"    COALESCE(b.{metric}_max, 0) AS {metric}_max,",
    ]


def _build_select_lines(cfg: GameFeatureConfig) -> list[str]:
    lines = ["SELECT"]
    lines.extend(_key_select_lines(cfg))
    # delta bet time + delta bet time nogap — no _cv / drawdown / spike at all.
    for metric, alias, label in (
        ("delta_t_seconds", "p1", "delta bet time"),
        ("delta_t_seconds_nogap", "p2", "delta bet time nogap"),
    ):
        lines.extend(
            [
                f"    -- {label}",
                f"    b.{metric}_avg,",
                f"    {alias}.{metric}_p25,",
                f"    {alias}.{metric}_median,",
                f"    {alias}.{metric}_p75,",
                f"    b.{metric}_std,",
                f"    b.{metric}_min,",
                f"    b.{metric}_max,",
            ]
        )
    # bet amount: derived ratios w/o COALESCE.
    lines.extend(_metric_block("bet_amount", "p3"))
    # delta_bet_amount: derived ratios wrapped in COALESCE.
    lines.extend(_metric_block("delta_bet_amount", "p4", coalesce_derived=True))
    lines.extend(_accum_delta_bet_block())
    # payout: extra payout_rate line.
    lines.extend(_metric_block("payout", "p5", include_payout_rate=True))
    lines.extend(_rtp_block())
    lines.extend(_profit_block())
    # delta_payout: COALESCE-wrapped derived ratios, no payout_rate.
    lines.extend(_metric_block("delta_payout", "p8", coalesce_derived=True))
    # balance_after_bet: normalised by its own avg, not bet_amount_avg.
    lines.extend(_metric_block("balance_after_bet", "p9", norm_by="balance_after_bet_avg"))
    lines.extend(_deposit_withdraw_block())
    # Streaks: every column wrapped in COALESCE(..., 0).
    lines.append("    -- streaks")
    lines.extend(_streak_block("streak", "p10"))
    lines.extend(_streak_block("win_streak", "p11"))
    lines.extend(_streak_block("lose_streak", "p12"))

    # The last streak block ends with a trailing comma; drop it.
    if lines[-1].endswith(","):
        lines[-1] = lines[-1].rstrip(",")
    return lines


def _build_join_lines(cfg: GameFeatureConfig) -> list[str]:
    key = all_partition_cols(cfg)
    out: list[str] = ["FROM stats_base AS b"]
    for i, family in enumerate(PERCENTILE_FAMILIES, start=1):
        out.append(f"    JOIN {family.cte_name} AS p{i}")
        out.append(equi_join_on("b", f"p{i}", key, indent="        "))
    return out


def build_final_select(cfg: GameFeatureConfig) -> str:
    lines = _build_select_lines(cfg)
    lines.extend(_build_join_lines(cfg))
    lines.append("ORDER BY")
    lines.append(",\n".join(f"    b.{c}" for c in all_partition_cols(cfg)))
    return "\n".join(lines)
