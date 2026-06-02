"""delta_stats CTE: bet-to-bet LAG-based deltas + win/lose flags.

For each aggregated bet (from ``agg_free_game``) computes:

* ``delta_t_seconds``: interval to the previous bet's start time
* ``delta_bet_amount`` / ``delta_payout``: simple LAG diffs
* ``balance_transaction``: balance change between bets, used downstream to
  derive ``deposit`` (positive) / ``withdraw`` (negative)
* ``is_win`` / ``is_lose``: current-row outcome flags
* ``prev_win`` / ``prev_lose``: previous-row outcome flags (drive streak resets)
"""

from bituslabs_ds.features.config import GameFeatureConfig

_ORDER_BY = "ORDER BY t.spin_id, t.min_created_at"


def build_delta_stats_cte(cfg: GameFeatureConfig) -> str:
    over = f"OVER (PARTITION BY t.user_id {_ORDER_BY})"
    lines: list[str] = [
        "delta_stats AS (",
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
            "        t.bet_amount,",
            "        t.payout,",
            "        t.balance_after_bet,",
            "        DATEDIFF(SECONDS, t.prev_bet_time, t.min_created_at) AS delta_t_seconds,",
            f"        t.bet_amount - LAG(t.bet_amount, 1) {over} AS delta_bet_amount,",
            f"        t.payout - LAG(t.payout, 1) {over} AS delta_payout,",
            "        t.balance_after_bet",
            "            + t.bet_amount",
            f"            - LAG(t.balance_after_payout, 1) {over} AS balance_transaction,",
            "        CASE WHEN t.payout > t.bet_amount THEN 1 ELSE 0 END AS is_win,",
            "        CASE WHEN t.payout < t.bet_amount THEN 1 ELSE 0 END AS is_lose,",
            "        CASE",
            f"            WHEN LAG(t.payout, 1) {over} > LAG(t.bet_amount, 1) {over}",
            "                THEN 1",
            "            ELSE 0",
            "        END AS prev_win,",
            "        CASE",
            f"            WHEN LAG(t.payout, 1) {over} < LAG(t.bet_amount, 1) {over}",
            "                THEN 1",
            "            ELSE 0",
            "        END AS prev_lose",
            "    FROM agg_free_game AS t",
            ")",
        ]
    )
    return "\n".join(lines)
