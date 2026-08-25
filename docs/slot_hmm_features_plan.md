# Slot HMM lifecycle features — migration plan (PARKED, not implemented)

Apply the fish_hunter HMM feature-engineering process
(`jobs/fish_hunter/feature_engineer_life_cycle.py`) to the slot games.
Status: feasibility checked, implementation parked — see the draft PR for
the open decisions.

## Source: `slot_orders_ab_group`

The orders dataset (Athena `bituslabs_ds.slot_orders_ab_group`) is the
input — already in our bucket, pre-filtered (no test/incomplete bets),
`game_id`-partitioned, `period=` pruning. Column mapping vs the fish job:

| fish input | slot equivalent |
| --- | --- |
| `bullet_id` / `event_id` (ordering tiebreak) | `spin_id` |
| `bet` / `payout` / `profit` | `bet_amount` / `actual_payout` / difference |
| `curr_balance` (→ `current_balance_day_max`) | `balance_after_payout` |
| `bullet_level` changes | `bet_amount` changes (proposed) |
| `multiplier` changes | `math_table_id` changes (proposed) |
| `fish_value` tier ratios + `target_selection_entropy` | OPEN — bet-amount tier entropy / mathtable entropy / drop |
| `created_at`, `user_id`, `currency_type` | identical |

## Planned shape

- Job: `jobs/etl/sagemaker/slot_machine/etl_hmm_features.py` (+ submit),
  parameterized by `--game-id`, same monthly tiling / dedup / per-user
  history-window stages as the fish job. No CSV exports (the Athena table
  replaces that access path).
- Output, partitioned by game FIRST:
  `s3://bituslabs-team-ai/etl-results/jobs/output_slot_order_hmm_features/`
  `{user_day_base,user_day_base_dedup,selected_hmm_features}/game_id=<G>/month=YYYY-MM/`
- Catalog: `infra/etl/register_slot_hmm_features_catalog.py` →
  **`bituslabs_ds.slot_order_hmm_features`** over `selected_hmm_features/`
  (partition projection: `game_id` enum × `month` date `yyyy-MM`).

## Open decisions (blockers before implementing)

1. Analog for the fish `target_selection_entropy` + 4 tier ratios:
   bet-amount tier entropy (per-game stake tiers) vs mathtable entropy vs
   dropping those features.
2. Session gap for `bet_date` attribution: 180 s `hmm_session` (matches the
   fish HMM semantics) vs the 30-min `bet_session` (matches the game-stats
   datasets). If 180 s, update the session-vocabulary table (`hmm_session`
   is currently documented as fish-only).
