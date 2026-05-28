"""user_group CTE: session_group + streak_group + win/lose_streak_group.

All four are running counts that *increment* every time the gap-or-outcome
condition fails. They're computed as cumulative SUMs of a boolean (0 = stay in
the same group, 1 = start a new one), giving each run a stable group id you
can ROW_NUMBER() within downstream.

* ``session_group``: new whenever ``delta_t_seconds > max_session_interval_seconds``.
* ``streak_group``: new whenever ``delta_t_seconds > streak_threshold_seconds``.
* ``win_streak_group``: new whenever the previous bet wasn't a win, or the
  current bet isn't a win, or the gap exceeds the streak threshold. Same shape
  for ``lose_streak_group``.
"""

from bituslabs_ds.features.config import GameFeatureConfig

_OVER = (
    "OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at "
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
)


def build_user_group_cte(cfg: GameFeatureConfig) -> str:
    interval = cfg.max_session_interval_seconds
    streak = cfg.streak_threshold_seconds

    lines: list[str] = [
        "user_group AS (",
        "    SELECT",
        "        t.user_id,",
        "        t.ai_group,",
    ]
    for col in cfg.partition_cols:
        lines.append(f"        t.{col},")
    lines.extend(
        [
            "        t.spin_id,",
            "        t.min_created_at,",
            "        t.max_created_at,",
            "        t.fg_rounds,",
            "        t.delta_t_seconds,",
            "        t.bet_amount,",
            "        t.delta_bet_amount,",
            "        t.payout,",
            "        t.delta_payout,",
            "        t.balance_after_bet,",
            "        t.balance_transaction,",
            "        t.is_win,",
            "        t.is_lose,",
            "        t.prev_win,",
            "        t.prev_lose,",
            f"        SUM(CASE WHEN t.delta_t_seconds <= {interval} THEN 0 ELSE 1 END)",
            f"            {_OVER}",
            "            AS session_group,",
            f"        SUM(CASE WHEN t.delta_t_seconds <= {streak} THEN 0 ELSE 1 END)",
            f"            {_OVER}",
            "            AS streak_group,",
            "        SUM(",
            "            CASE",
            f"                WHEN t.delta_t_seconds <= {streak} AND t.prev_win = 1 AND t.is_win = 1 THEN 0",
            "                ELSE 1",
            "            END",
            "        )",
            f"            {_OVER}",
            "            AS win_streak_group,",
            "        SUM(",
            "            CASE",
            f"                WHEN t.delta_t_seconds <= {streak} AND t.prev_lose = 1 AND t.is_lose = 1 THEN 0",
            "                ELSE 1",
            "            END",
            "        )",
            f"            {_OVER}",
            "            AS lose_streak_group",
            "    FROM delta_stats AS t",
            ")",
        ]
    )
    return "\n".join(lines)
