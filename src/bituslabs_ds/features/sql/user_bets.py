"""user_bets CTE: base filter + ai_group labeling + prev_bet_time LAG.

Selects per-bet rows from ``public.fct_bet_orders`` for the configured game,
date window, currency, ai_group selection, and any extra game-specific WHERE
clauses. Adds:

* ``bet_amount`` (NULL for FREE rounds so they don't pollute base-game stats)
* ``prev_bet_time`` (LAG of created_at per user)
* ``is_new_game_group`` (1 for BASE, 0 for FREE — used by free_game_group)
* ``ai_group`` (CASE label or constant)
"""

from bituslabs_ds.config import ETL_CURRENCY_CODES, ETL_EXCLUDED_OP_CODES
from bituslabs_ds.features.config import GameFeatureConfig
from bituslabs_ds.features.sql._helpers import ai_group_case, ai_group_filter


def build_user_bets_cte(cfg: GameFeatureConfig, start_date: str) -> str:
    lines: list[str] = [
        "user_bets AS (",
        "    SELECT",
        "        t.spin_id,",
        "        t.user_id,",
    ]
    for col in cfg.partition_cols:
        lines.append(f"        t.{col},")
    lines.extend(
        [
            "        t.created_at,",
            "        t.bet_type,",
            "        -- ignore bet amount in free game (0). This will influence avg and std stats",
            "        CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END AS bet_amount,",
            "        t.actual_payout AS payout,",
            "        t.balance_after_bet,",
            "        t.balance_after_payout,",
            "        LAG(t.created_at, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at)"
            " AS prev_bet_time,",
            "        CASE",
            "            WHEN t.bet_type = 'BASE' THEN 1",
            "            ELSE 0",
            "        END AS is_new_game_group,",
            f"        {ai_group_case(cfg, alias='t', indent='        ')}",
            "    FROM public.fct_bet_orders AS t",
            "    WHERE",
            f"        t.created_at >= '{start_date}'",
            f"        AND t.created_at < '{cfg.date_end}'",
            f"        AND t.currency_type IN {ETL_CURRENCY_CODES}",
            "        AND t.status = 'COMPLETED'",
            f"        AND t.game_id = '{cfg.game_id}'",
            f"        AND t.op_code NOT IN {ETL_EXCLUDED_OP_CODES}",
        ]
    )

    ai_filter_sql = ai_group_filter(cfg)
    if ai_filter_sql:
        lines.append(f"        {ai_filter_sql}")
    for clause in cfg.extra_where_clauses:
        lines.append(f"        AND ({clause})")
    lines.append(")")
    return "\n".join(lines)
