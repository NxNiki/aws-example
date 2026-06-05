"""stats_base CTE: GROUP BY aggregation over raw_stats.

One row per ``(user_id, ai_group, [partition_cols], session_group, agg_group)``
with AVG / STDDEV / MIN / MAX for each per-bet metric, plus the SUM-with-
condition accumulators (accum_pos_*, accum_neg_*) and the COUNT-based rate
metrics (payout_rate, profit_rate).

The optional ``HAVING COUNT(user_id) = bin_size`` drops the final
incomplete agg_group per session when ``cfg.drop_incomplete_tail_groups``
is True.
"""

from bituslabs_ds.features.config import GameFeatureConfig
from bituslabs_ds.features.sql._helpers import all_partition_cols

# Plain (metric_column) tuples that get AVG/STDDEV/MIN/MAX.
_BASIC_METRICS: tuple[str, ...] = (
    "delta_t_seconds",
    "delta_t_seconds_nogap",
    "bet_amount",
    "delta_bet_amount",
    "payout",
    "profit",
    "delta_payout",
    "balance_after_bet",
    "streak",
    "win_streak",
    "lose_streak",
)


def _basic_agg_lines(metric: str) -> list[str]:
    return [
        f"        AVG(t.{metric}) AS {metric}_avg,",
        f"        STDDEV(t.{metric}) AS {metric}_std,",
        f"        MIN(t.{metric}) AS {metric}_min,",
        f"        MAX(t.{metric}) AS {metric}_max,",
    ]


def build_stats_base_cte(cfg: GameFeatureConfig) -> str:
    key_cols = all_partition_cols(cfg)
    group_by = ",\n".join(f"        t.{c}" for c in key_cols)

    lines: list[str] = ["stats_base AS (", "    SELECT"]
    for c in key_cols:
        lines.append(f"        t.{c},")
    lines.extend(
        [
            "        MIN(t.activity_date) AS activity_date,",
            "        MIN(t.min_created_at) AS min_created_at,",
            "        MAX(t.max_created_at) AS max_created_at,",
            "        COUNT(t.user_id) AS bet_rounds,",
            "        SUM(t.fg_rounds) AS fg_rounds,",
        ]
    )

    # AVG/STDDEV/MIN/MAX for each basic metric, with the accumulator additions
    # interleaved at the same positions as the original handcrafted SQL.
    for metric in _BASIC_METRICS:
        lines.extend(_basic_agg_lines(metric))
        if metric == "delta_bet_amount":
            lines.extend(
                [
                    "        SUM(CASE WHEN t.delta_bet_amount > 0 THEN t.delta_bet_amount ELSE 0 END)"
                    " AS accum_pos_delta_bet_amount,",
                    "        SUM(CASE WHEN t.delta_bet_amount < 0 THEN t.delta_bet_amount ELSE 0 END)"
                    " AS accum_neg_delta_bet_amount,",
                ]
            )
        if metric == "payout":
            lines.extend(
                [
                    "        SUM(CASE WHEN t.payout > 0 THEN 1 END) * 1.0 / NULLIF(COUNT(t.user_id), 0)"
                    " AS payout_rate,",
                    "        AVG(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_mean,",
                    "        MAX(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_max,",
                    "        MIN(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_min,",
                ]
            )
        if metric == "profit":
            lines.extend(
                [
                    "        SUM(CASE WHEN t.profit > 0 THEN t.profit ELSE 0 END) AS accum_pos_profit,",
                    "        SUM(CASE WHEN t.profit < 0 THEN t.profit ELSE 0 END) AS accum_neg_profit,",
                    "        SUM(CASE WHEN t.profit > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(t.user_id), 0)"
                    " AS profit_rate,",
                ]
            )
        if metric == "balance_after_bet":
            lines.extend(
                [
                    "        SUM(CASE WHEN t.deposit > 0 THEN 1 ELSE 0 END) AS num_deposit,",
                    "        SUM(CASE WHEN t.deposit > 0 THEN t.deposit ELSE 0 END) AS accum_deposit,",
                    "        SUM(CASE WHEN t.withdraw > 0 THEN 1 ELSE 0 END) AS num_withdraw,",
                    "        SUM(CASE WHEN t.withdraw > 0 THEN t.withdraw ELSE 0 END) AS accum_withdraw,",
                ]
            )

    # Strip trailing comma off the final aggregate line.
    if lines[-1].endswith(","):
        lines[-1] = lines[-1].rstrip(",")

    lines.extend(["    FROM binned AS t", "    GROUP BY", group_by])
    if cfg.drop_incomplete_tail_groups:
        lines.append(f"    HAVING COUNT(t.user_id) = {cfg.bin_size}")
    lines.append(")")
    return "\n".join(lines)
