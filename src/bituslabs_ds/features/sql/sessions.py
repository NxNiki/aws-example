"""user_group CTE: session_start_ts + streak_group + win/lose_streak_group.

``session_start_ts`` is the timestamp of the session's first bet, carried
forward to every row in the session via ``LAST_VALUE(... IGNORE NULLS)``. It
is *stable across runs*: the same logical session always has the same
session_start_ts regardless of how much history the SQL was given. Downstream
(``raw_stats``) derives ``session_start_date = CAST(session_start_ts AS DATE)``
and a within-(user, session_start_date) ``session_group`` ordinal.

The three streak counters stay as running cumulative SUMs. They're never
emitted in the output (only used as ``PARTITION BY`` for ``ROW_NUMBER`` in
``raw_stats`` to compute the ``streak`` / ``win_streak`` / ``lose_streak``
columns), so their lack of cross-run stability does not affect downstream
data. As long as the lookback exceeds ``streak_threshold_seconds`` (200s) the
streak boundaries are correctly captured -- which is trivial for any sane
incremental window.

* ``session_start_ts``: ``min_created_at`` of the session's first bet.
  A bet starts a new session when ``delta_t_seconds > max_session_interval_seconds``
  or is NULL (the user's very first bet).
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
            "        LAST_VALUE(",
            "            CASE",
            f"                WHEN t.delta_t_seconds > {interval} OR t.delta_t_seconds IS NULL",
            "                    THEN t.min_created_at",
            "            END",
            "            IGNORE NULLS",
            "        )",
            f"            {_OVER}",
            "            AS session_start_ts,",
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
