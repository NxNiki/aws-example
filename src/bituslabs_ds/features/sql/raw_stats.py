"""raw_stats CTE: per-bet derived metrics + agg_group bucketing.

Adds the per-bet derived metrics that the grouped stats roll up:

* ``delta_t_seconds_nogap``: ``delta_t_seconds`` clamped (NULL above gap threshold)
* ``profit``: ``payout - bet_amount``
* ``deposit`` / ``withdraw``: split of ``balance_transaction``
* ``streak`` / ``win_streak`` / ``lose_streak``: ROW_NUMBER within each *_group
* ``agg_group``: floor((row - 1) / session_length) bucket id per session
* ``activity_date``: ``CAST(min_created_at AS DATE)`` — used by ETLScheduler as
  the watermark / partition column
"""

from bituslabs_ds.features.config import GameFeatureConfig

_ORDER = "ORDER BY t.spin_id, t.min_created_at"


def build_raw_stats_cte(cfg: GameFeatureConfig) -> str:
    n = cfg.session_length
    gap = cfg.max_session_gap_seconds

    lines: list[str] = [
        "raw_stats AS (",
        "    SELECT",
        "        t.user_id,",
        "        t.ai_group,",
    ]
    for col in cfg.partition_cols:
        lines.append(f"        t.{col},")
    lines.extend(
        [
            "        t.session_group,",
            "        t.min_created_at,",
            "        t.max_created_at,",
            "        CAST(t.min_created_at AS DATE) AS activity_date,",
            "        t.spin_id,",
            "        t.fg_rounds,",
            "        t.bet_amount,",
            "        t.delta_bet_amount,",
            "        t.payout,",
            "        t.delta_payout,",
            "        t.balance_after_bet,",
            "        t.delta_t_seconds,",
            "        -- ignore delta t larger than the configured gap threshold (default 1 hour).",
            f"        CASE WHEN t.delta_t_seconds <= {gap} THEN t.delta_t_seconds END AS delta_t_seconds_nogap,",
            "        t.payout - t.bet_amount AS profit,",
            "        CASE WHEN t.balance_transaction > .1 THEN t.balance_transaction END AS deposit,",
            "        CASE WHEN t.balance_transaction < -.1 THEN -t.balance_transaction END AS withdraw,",
            f"        ROW_NUMBER() OVER (PARTITION BY t.user_id, t.streak_group {_ORDER}) AS streak,",
            "        CASE",
            "            WHEN t.payout > t.bet_amount",
            f"                THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.win_streak_group {_ORDER})",
            "        END AS win_streak,",
            "        CASE",
            "            WHEN t.payout < t.bet_amount",
            f"                THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.lose_streak_group {_ORDER})",
            "        END AS lose_streak,",
            f"        ROUND((ROW_NUMBER() OVER (PARTITION BY t.user_id, t.session_group {_ORDER}) - 1) / {n})",
            "            AS agg_group",
            "    FROM user_group AS t",
            ")",
        ]
    )
    return "\n".join(lines)
