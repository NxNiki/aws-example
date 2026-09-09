# Feature Engineering — Feature Reference (SS01 / SS02 / SS03)

Feature dictionary for the unified bet-segmentation feature engineering
pipeline in `src/bituslabs_ds/features/`. The SQL is generated per game from a
`GameFeatureConfig` (`features/config.py`) by the CTE builders in
`features/sql/`; the exact SQL emitted for each game is snapshotted in
`tests/unit/features_snapshots/`.

Each game produces two datasets under
`s3://bituslabs-team-ai/etl-results/jobs/<output_prefix>/`:

| Dataset | Grain | Query |
|---|---|---|
| `features_enriched/` | one row per **aggregated bet** (free-game runs merged into the preceding base game) | `compose_enriched_query` → `raw_stats` |
| `features_grouped_binsize_{N}/` | one row per **agg_group** (a window of N consecutive aggregated bets within a session) | `compose_grouped_query` → `stats_base` + percentiles |

## Pipeline key steps

The logic follows the original SQL-script design (v1 doc), with updated
thresholds:

1. **Merge free games into the previous base game.** A base-game bet plus its
   following run of free-game rounds is treated as a single aggregated bet.
   Bet amounts of free-game rounds are set to NULL (only base-game
   `bet_amount` enters the stats); payouts are summed; the timestamp and
   `delta_t_seconds` of the aggregated bet come from the base game. The
   `delta_t_seconds` of the bet *after* a free-game run is measured from the
   last free-game round.
2. **Sessionize.** A new `session_group` starts when the gap to the previous
   bet exceeds `session_break_threshold_seconds` — **12 hours** for all three
   games today (v1 used 7 days; the proposed 1-hour split is still pending
   sync with the MLE team). `session_group` is a 0-based rank within
   `(user_id, session_start_date)`, so `(user_id, session_start_date,
   session_group)` identifies a session stably across incremental runs.
3. **Streaks.** Defined on a **200-second** gap threshold, starting from 1:
   `streak` counts consecutive fast bets; `win_streak` counts consecutive wins
   (`payout > bet_amount`), NULL on non-win rows; `lose_streak` mirrors it for
   losses (`payout < bet_amount`).
4. **Bin into agg_groups.** Within each session, every `bin_size` consecutive
   aggregated bets form one `agg_group`
   (`ROUND((session_bet_index - 1) / bin_size)`). Bin size is per game — see
   the per-game section.
5. **Aggregate + normalize.** Per agg_group: avg / std / min / max /
   p25 / median / p75, accumulative positives and negatives, and ratios
   normalized by `bet_amount_avg` (cv, spike ratio, drawdown ratio,
   accumulative ratios). `balance_after_bet` ratios are normalized by its own
   average instead.

## Common features across games

All feature columns below are produced identically for SS01, SS02 and SS03
(verified by diffing the SQL snapshots — only WHERE filters, grouping keys and
bin sizes differ).

### `features_enriched` (per aggregated bet)

**Keys / metadata**

| Column | Description |
|---|---|
| `user_id` | Player id. |
| `ai_group` | Partition label derived from `partition_ab[0]`: `AI`, `AB_TEST_A`, `AB_TEST_B` or `Default` (catch-all for NULL/unmapped). Which labels exist per game is config-driven. |
| `session_start_date` | `session_start_ts` cast to date; partition-friendly session key part. |
| `session_group` | 0-based rank of the session within `(user_id, session_start_date)`. New session when the gap to the previous bet > 12 h. |
| `session_start_ts` | Timestamp of the first bet of the session. |
| `min_created_at` / `max_created_at` | Start / end timestamp of the aggregated bet (equal for a pure base-game bet; span the free-game run otherwise). |
| `activity_date` | `min_created_at` cast to date. |
| `spin_id` | `spin_id` of the base-game bet (MIN over the merged free-game group). |
| `session_bet_index` | 1-based position of the aggregated bet within its session; the grouped query derives `agg_group` from it. |

**Per-bet features**

| Column | Description |
|---|---|
| `fg_rounds` | Number of free-game rounds merged into this bet (0 for a plain base-game bet). |
| `bet_amount` | Base-game bet amount. NULL for free-game rounds by construction, so free spins never dilute bet-amount stats. |
| `payout` | Total actual payout of the aggregated bet (base + merged free-game payouts). |
| `profit` | `payout - bet_amount`. |
| `delta_bet_amount` | Change in bet amount vs the user's previous aggregated bet. |
| `delta_payout` | Change in payout vs the previous aggregated bet. |
| `delta_t_seconds` | Seconds between this bet and the user's previous bet (base-game timestamp for merged free-game runs, per step 1 above). |
| `delta_t_seconds_nogap` | `delta_t_seconds`, NULLed when above `max_delta_t_gap_seconds` (1 hour) — the workaround for long idle gaps inflating time stats. |
| `balance_after_bet` | Wallet balance right after the bet was placed (MIN over a merged group). |
| `deposit` | Money added to the balance before this bet: `balance_after_bet + bet_amount - previous balance_after_payout`, kept only when > 0.1; NULL otherwise. |
| `withdraw` | Negated balance transaction when < −0.1 (money removed before this bet); NULL otherwise. |
| `streak` | 1-based count of consecutive bets with gaps ≤ 200 s. Resets to 1 after a slow gap. |
| `win_streak` | 1-based count of consecutive wins within a ≤ 200 s cadence; NULL on non-win bets. |
| `lose_streak` | Same for consecutive losses; NULL on non-lose bets. |

### `features_grouped_binsize_{N}` (per agg_group)

**Keys / metadata**

| Column | Description |
|---|---|
| `user_id`, `ai_group`, `session_start_date`, `session_group` | As above. |
| `agg_group` | 0-based window index within the session (`ROUND((session_bet_index - 1) / N)`). |
| `activity_date`, `min_created_at`, `max_created_at` | Date / time span covered by the window. |
| `bet_rounds` | Number of aggregated bets in the window (≤ N; see SS01 note on incomplete tails). |
| `fg_rounds` | Total free-game rounds merged into the window's bets. |

**Distribution stats.** Each base metric below gets the same stat suite —
`_avg`, `_p25`, `_median`, `_p75`, `_std`, `_min`, `_max`:

| Base metric | Notes |
|---|---|
| `delta_t_seconds` | Inter-bet gap. |
| `delta_t_seconds_nogap` | Gap stats excluding > 1 h gaps. |
| `bet_amount` | Base-game bets only. |
| `delta_bet_amount` | Bet-size changes. |
| `payout` | |
| `delta_payout` | |
| `profit` | |
| `balance_after_bet` | |
| `streak`, `win_streak`, `lose_streak` | All streak stats are COALESCEd to 0 when the window has no qualifying rows. |
| `rtp` | Per-bet `payout / bet_amount`; exposed as `rtp_mean`, `rtp_min`, `rtp_max`, `rtp_p25`, `rtp_median`, `rtp_p75` (no std). |

**Normalized ratios** (v1 "normalized by avg(bet_amount)" — still in place;
denominator is `bet_amount_avg` unless noted):

| Column(s) | Description |
|---|---|
| `bet_amount_cv`, `payout_cv`, `profit_cv`, `delta_bet_amount_cv`, `delta_payout_cv` | Coefficient of variation: `std / bet_amount_avg`. Delta variants COALESCE to 0. |
| `bet_amount_spike_ratio`, `payout_spike_ratio`, `delta_bet_amount_spike_ratio`, `delta_payout_spike_ratio` | `max / bet_amount_avg`. |
| `bet_amount_drawdown_ratio`, `payout_drawdown_ratio`, `delta_bet_amount_drawdown_ratio`, `delta_payout_drawdown_ratio` | `min / bet_amount_avg`. |
| `max_profit_ratio`, `min_profit_ratio` | `profit_max` / `profit_min` over `bet_amount_avg`. |
| `balance_after_bet_cv`, `balance_after_bet_spike_ratio`, `balance_after_bet_drawdown_ratio` | Normalized by **`balance_after_bet_avg`**, not by bet amount. |

**Accumulative + rate features**

| Column | Description |
|---|---|
| `accum_pos_delta_bet_amount` / `accum_neg_delta_bet_amount` | Sum of positive / negative bet-size changes in the window. |
| `accum_pos_delta_bet_amount_ratio` / `accum_neg_delta_bet_amount_ratio` | Same, over `bet_amount_avg`. |
| `accum_pos_profit` / `accum_neg_profit` | Sum of winning / losing profits. |
| `accum_pos_profit_ratio` / `accum_neg_profit_ratio` | Same, over `bet_amount_avg`. |
| `payout_rate` | Share of bets with `payout > 0`. |
| `profit_rate` | Share of bets with `profit > 0` (win rate). |
| `num_deposit` / `accum_deposit` / `accum_deposit_ratio` | Count / sum of deposits in the window; ratio over `bet_amount_avg`. |
| `num_withdraw` / `accum_withdraw` / `accum_withdraw_ratio` | Same for withdrawals. |

## Per-game configuration

The feature *columns* are common; what differs per game is the input filter,
the grouping keys and the bin sizes. Configs live in each game's
`jobs/<game>/etl_feature_engineer.py`.

| Knob | SS01 (wucaishen) | SS02 (deepdive) | SS03 (mahjiang streak) |
|---|---|---|---|
| Extra grouping key | — | `math_table_id` | `math_table_id` |
| ai_group labels | AI, Default | AI, Default | AI, AB_TEST_A, AB_TEST_B, Default |
| Selected group(s) | Default | AI | one slice per run: default / ai / ab_test_a / ab_test_b (separate output prefixes) |
| Bin size(s) | 40 | 30 | default & AB arms: 50, 70; ai: 30, 50, 70, 100 (clustering consumes binsize_50) |
| Incomplete tail agg_groups | **dropped** (`HAVING COUNT = 40`) | kept | kept |
| Extra WHERE | `script_id = 'giftShop'` | — | — |
| Session break / streak / nogap thresholds | 12 h / 200 s / 1 h | same | same |

Shared base filter for all games: `public.fct_bet_orders`, `currency_type =
'CNY'`, `status = 'COMPLETED'`, `op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')`,
plus the per-game `game_id` and partition_ab selection.

Notes:

- **`math_table_id` (SS02/SS03 only)** is the only per-game *column*: it rides
  along every CTE and joins as an aggregation key, so windows never mix bets
  from different math tables. SS01 has a single math configuration and omits
  it.
- SS03's `--group` flag materialises one dataset root per ai_group slice
  (`output_ss03_feature_engineer[_ai|_ab_test_a|_ab_test_b]`).

## Game-specific features

> **Placeholder.** All engineered features are currently common across games —
> there are no game-specific feature columns yet (beyond the `math_table_id`
> grouping key noted above). When a game gains features tied to its own
> mechanics (e.g. gift-shop script events in SS01, per-math-table derived
> metrics in SS03), document them here per game.

### SS01 (wucaishen)

_None yet._

### SS02 (deepdive)

_None yet._

### SS03 (mahjiang streak)

_None yet._
