"""raw_stats CTE: per-bet derived metrics + agg_group bucketing.

Adds the per-bet derived metrics that the grouped stats roll up:

* ``session_start_date``: ``CAST(session_start_ts AS DATE)`` -- the date the
  session began. Together with ``session_group`` it uniquely identifies a
  session for a user (stable across runs).
* ``session_group``: within-(user, session_start_date) ordinal starting at 0.
  Mostly 0; only >=1 when a user has multiple sessions on the same calendar
  date (rare with 7-day MAX_SESSION_INTERVAL; possible with the future 12-hour
  threshold).
* ``delta_t_seconds_nogap``: ``delta_t_seconds`` clamped (NULL above gap threshold)
* ``profit``: ``payout - bet_amount``
* ``deposit`` / ``withdraw``: split of ``balance_transaction``
* ``streak`` / ``win_streak`` / ``lose_streak``: ROW_NUMBER within each *_group
* ``session_bet_index``: 1-based row number of the bet within its session
  (partitioned by ``session_start_ts`` so it's stable across runs). This is
  bin-independent; the grouped query derives ``agg_group = floor((idx-1)/bin_size)``
  from it (see ``binned.py``), so the enriched output is shared across bin sizes.
* ``activity_date``: ``CAST(min_created_at AS DATE)`` -- ETLScheduler's
  watermark / partition column. Note this is the bet's date, not the
  session's start date.
"""

from bituslabs_ds.features.config import GameFeatureConfig

_ORDER = "ORDER BY t.spin_id, t.min_created_at"


def build_raw_stats_cte(cfg: GameFeatureConfig) -> str:
    gap = cfg.max_delta_t_gap_seconds

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
            "        CAST(t.session_start_ts AS DATE) AS session_start_date,",
            "        DENSE_RANK() OVER (",
            "            PARTITION BY t.user_id, CAST(t.session_start_ts AS DATE)",
            "            ORDER BY t.session_start_ts",
            "        ) - 1 AS session_group,",
            "        t.session_start_ts,",
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
            f"        ROW_NUMBER() OVER (PARTITION BY t.user_id, t.session_start_ts {_ORDER})",
            "            AS session_bet_index",
            "    FROM user_group AS t",
            ")",
        ]
    )
    return "\n".join(lines)
