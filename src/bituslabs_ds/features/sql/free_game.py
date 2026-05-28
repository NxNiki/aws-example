"""free_game_group + agg_free_game CTEs.

A "free game" is a sequence of FREE bets triggered by a preceding BASE bet.
We collapse each (BASE-then-FREE*) run into one aggregated row so downstream
deltas (delta_t_seconds, delta_bet_amount) treat the free spins as a single
bet attached to the triggering BASE bet — matching the GAIL model's semantics.

* ``free_game_group``: assigns a monotonically increasing ``fg_group`` id per
  user using a cumulative SUM of ``is_new_game_group`` (1 on each BASE row).
* ``agg_free_game``: collapses each ``fg_group`` to one row, summing bet/payout
  and taking the earliest spin_id / timestamps.
"""

from bituslabs_ds.features.config import GameFeatureConfig


def build_free_game_group_cte(cfg: GameFeatureConfig) -> str:
    lines: list[str] = [
        "free_game_group AS (",
        "    SELECT",
        "        t.spin_id,",
        "        t.user_id,",
        "        t.ai_group,",
    ]
    for col in cfg.partition_cols:
        lines.append(f"        t.{col},")
    lines.extend(
        [
            "        t.created_at,",
            "        t.bet_type,",
            "        t.bet_amount,",
            "        t.payout,",
            "        t.balance_after_bet,",
            "        t.balance_after_payout,",
            "        t.prev_bet_time,",
            "        SUM(t.is_new_game_group) OVER (",
            "            PARTITION BY t.user_id",
            "            ORDER BY t.spin_id, t.created_at",
            "            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            "        ) AS fg_group",
            "    FROM user_bets AS t",
            ")",
        ]
    )
    return "\n".join(lines)


def build_agg_free_game_cte(cfg: GameFeatureConfig) -> str:
    group_by_cols = ["t.user_id", "t.ai_group", *(f"t.{c}" for c in cfg.partition_cols), "t.fg_group"]
    lines: list[str] = [
        "agg_free_game AS (",
        "    SELECT",
        "        t.user_id,",
        "        t.ai_group,",
    ]
    for col in cfg.partition_cols:
        lines.append(f"        t.{col},")
    lines.extend(
        [
            "        MIN(t.spin_id) AS spin_id,",
            "        MIN(t.created_at) AS min_created_at,",
            "        MAX(t.created_at) AS max_created_at,",
            "        SUM(CASE WHEN t.bet_type = 'FREE' THEN 1 ELSE 0 END) AS fg_rounds,",
            "        MIN(t.prev_bet_time) AS prev_bet_time,",
            "        SUM(t.bet_amount) AS bet_amount,",
            "        SUM(t.payout) AS payout,",
            "        MIN(t.balance_after_bet) AS balance_after_bet,",
            "        MAX(t.balance_after_payout) AS balance_after_payout",
            "    FROM free_game_group AS t",
            f"    GROUP BY {', '.join(group_by_cols)}",
            ")",
        ]
    )
    return "\n".join(lines)
