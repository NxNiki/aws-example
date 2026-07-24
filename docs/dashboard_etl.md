# Game-Stats ETL & Dashboard — Metric Reference

Metric dictionary for the per-user game-stats pipelines
(`jobs/<game>/etl_game_stats_daily_by_user_group.py`, fish_hunter:
`etl_game_stats_daily_by_user.py`) and the dashboard metrics computed from
them by `bituslabs_ds.metrics.user_stats_aggregates.DataMetrics`. The exact
SQL per game is snapshotted in `tests/unit/etl_snapshots/`.

## Datasets

Each game writes three datasets to
`s3://bituslabs-team-ai/etl-results/jobs/<output_prefix>/`:

| Dataset | Period grain (`activity_date`) |
|---|---|
| `daily_stats/` | Beijing calendar day (`DATE_START_HOUR` = 0 → midnight boundary) |
| `weekly_stats/` | Monday-aligned calendar week |
| `monthly_stats/` | calendar month |

**Row grain (slot games, `*_v2` prefixes)**: one row per
`(activity_date, user_id, ab_group, mathtable)` — every bet lives in exactly
one row, so sums are correct under any selection and no label re-partitions
the same bets.

- `ab_group` — the bet's AB-test arm from `partition_ab[0]`: `AI`,
  `AB_TEST_A` / `AB_TEST_B` (ss03/ss06 only), else `Default`. Assigned per
  bet, so a user re-bucketed mid-period splits across arms.
- `mathtable` — the bet's `math_table_id` (`(none)` when empty; ss02
  attributes `FourScatter` free-game buy-ins to the following spin's
  mathtable).

**fish_hunter** (`output_fish_hunter_v2`) has its own grain: one row per
`(activity_date, user_id, daily_group, fish_value)`. `daily_group` assigns
each whole user-day to one strategy tier (RISK_CONTROLLED > BOOST_POOL >
DYNAMIC_RTP* > DEFAULT_FALLBACK); `fish_value` is the bullet's target fish
value, so the dashboard can bucket by user-defined "fish level" ranges.
Metrics that only exist at the user-day level (`user_num_rooms`,
`user_profit_coef_var`, streaks, session stats) are computed once per
user-day and repeated on each of the user's fish_value rows — the collapse
takes their first value, never sums them.

## How the dashboard aggregates

- `ab_group` and `mathtable` are two independent cohort pickers; lifecycle
  groups (periods since first bet, derived at query time from the daily data)
  are a third dimension. Cohorts are the cross product of the selections.
- **Range groups** (fish_hunter's "Fish level"): user-defined buckets over a
  numeric grain column (`stats_by_date.range_group` in the game's config),
  with INCLUSIVE `[min, max]` bounds (unlike lifecycle's half-open ranges;
  `max` empty = open-ended). Definitions (name + bounds) are global to the
  game; which buckets are shown is chosen per tab, next to the other cohort
  pickers. Buckets may overlap — each is filtered independently, then the
  matching rows are collapsed per user, so an overlapping value counts in
  every bucket that contains it (by design; the buckets are lenses, not a
  partition).
- When a grain dimension is unselected ("all"), the API collapses rows back
  to **one row per user** before computing per-user stats
  (`dashboard_api.services.common.collapse_user_rows`): additive components
  are summed (all-null stays null), ratio columns are recomputed from the
  sums using the same formulas as the ETL. Group-level metrics need no
  collapse — the grain is disjoint, sums are always right.

## Per-user row columns (`user_*`)

All aggregates are over the row's bets — the user's bets in that period on
that `(ab_group, mathtable)`. BG = `bet_type = 'BASE'`, FG = `'FREE'`.

| Column | Definition |
|---|---|
| `user_num_bets` / `_bg` / `_fg` | bet counts (all / base game / free game) |
| `user_total_bet` / `_bg` | Σ `bet_amount` (all / base) |
| `user_total_bet_fg` | Σ of the base-game bet that *triggered* each free-game run (`prev_bet_type = 'BASE'` and current row FREE) — the FG "stake" |
| `user_total_payout` / `_bg` / `_fg` | Σ `actual_payout` |
| `user_avg_bet_amount` | `total_bet / num_bets` |
| `user_total_profit` | `total_payout − total_bet` |
| `user_rtp` | `total_payout / total_bet` |
| `user_rtp_bg` | `total_payout_bg / total_bet` (denominator is **all** bets, by design) |
| `user_rtp_fg` | `total_payout_fg / total_bet_fg` |
| `user_num_bets*_with_payout` | counts of bets with `payout > 0` |
| `user_hit_rate` / `_bg` / `_fg` | `with_payout / num_bets` of the segment |
| `user_fg_ratio` | `num_bets_fg / num_bets` |
| `user_mathtable_change` | # of bets whose mathtable differs from the previous bet (within the day, see below) |
| `user_avg_delta_t_seconds_bg` | mean seconds between consecutive BASE bets (both BASE, gap ≤ `ETL_DELTA_T_MAX_SECONDS`, floored at `ETL_DELTA_T_MIN_SECONDS`) |
| `user_num_delta_t_bg` | exact count of gaps entering that mean (its recombination weight) |
| `user_accu_pos_delta_bet` / `_neg` | Σ of positive / negative bet-to-bet amount changes |
| `user_accu_pos_delta_bet_avg` / `_neg` | the corresponding means (`accu / *_delta_bet_num`) |
| `user_accu_delta_bet` / `_avg` | Σ and mean of all bet-to-bet changes |
| `user_pos_delta_bet_num` / `_neg` | counts of positive / negative changes |
| `user_num_delta_bet` | count of bets with a previous bet (denominator of `accu_delta_bet_avg`) |

### Sequence metrics are day-partitioned (since 2026-07-23)

`delta_t`, `delta_bet*`, `mathtable_change` and the FG trigger are computed
with `LAG … PARTITION BY (user_id, activity_date)`: **each calendar day is
self-contained** and the day's first bet has no predecessor by definition. A
session crossing midnight contributes no delta to the new day.

Why: these metrics were previously computed over the user's full bet stream,
which made their values depend on *how the data was loaded* — a full-history
reload saw cross-day predecessors, while each incremental run's 3-day pull
window cut the chain at its start. Stored history was therefore a mix of two
semantics (stream-based for the initial backfill era, day-truncated for
incrementally written dates), and numbers could not be reproduced by ad-hoc
SQL over a date range. Day-partitioning makes the definition uniform and
load-order-independent. Consequence: pre-2026-07 sequence-metric values
changed slightly on dates that carried the old stream semantics; all
non-sequence columns were bit-identical through the migration.

## Group-level dashboard metrics (DataMetrics)

Computed per (period × cohort) from the user rows above.

| Metric | Definition |
|---|---|
| `num_active_users` | distinct users with `user_num_bets ≥ ACTIVE_USER_MIN_BETS` |
| `num_new_users` | distinct users whose **first-ever bet** falls in the period (derived period offset = 0; before 2026-07 this was the stored `user_group = 'new'` ≈ first bet ≤ 3 days ago) |
| `day0_num_users` | distinct users on the anchor date (the retention cohort, no bet threshold) |
| `day{N}_num_users` | members of the day-0 cohort who appear again N days later (N ∈ 1,2,3,5,7,10,15,30; presence in **any** group counts). NULL when the follow-up date is beyond the latest loaded date — the value is unknowable yet, and a 0/undercount would read as a retention collapse for recent cohorts |
| `retention_rate_day{N}` | `dayN_num_users / day0_num_users` (NULL propagates from the numerator) |
| `num_active_user_0_rtp` / `active_user_0_rtp_ratio` | users with zero total payout in the period, and their share |
| `active_user_rtp_less_0_{X}_ratio` | share of users with `user_rtp < 0.X` |
| `active_user_no_fg_ratio` | share of users with no free-game bets |
| `total_num_bets` / `_bg` / `_fg`, `total_bet`, `total_payout*`, `total_profit` | sums of the per-user columns |
| `rtp` / `rtp_bg` / `rtp_fg` | pooled Σ payout / Σ bet of the segment |
| `hit_rate` / `_bg` / `_fg` | pooled Σ with-payout / Σ bets |
| `fg_ratio` | pooled Σ fg bets / Σ bets |
| `user_rtp_median` | median of per-user RTPs |
| `user_rtp_ultilization_ratio` | `user_rtp_median / rtp` |

Per-user (`user_*`) metrics shown on the dashboard are the mean across users
with a bootstrap 95% CI; distributions (histograms, box plots, summary-table
cells) use the same per-user samples.

## Known limitations / future work

- **Calendar-day attribution.** Bets, deltas, DAU and the retention cohorts
  all bucket by the bet's own Beijing calendar day. A session spanning
  midnight is split across two days: its bets count in both days, its
  cross-midnight delta is dropped (see above), and a through-midnight player
  "returns on day 1" for retention purposes even though it was one sitting.
  If this ever matters, the candidate change is **session-start attribution**
  (all bets of a session bucket to the session's start date) — a deliberate,
  breaking semantics change: it redefines DAU / daily totals / retention and
  breaks reconciliation against plain calendar-day `fct_bet_orders` queries,
  so it needs its own migration and validation pass. A lighter alternative is
  shifting the day boundary via `DATE_START_HOUR` (e.g. 4 → days run
  4am–4am), which keeps most late-night sessions whole for every metric at
  once.
- ss01/ss02's `*_pa` variant jobs still produce the legacy single-column
  (`ai_group`) shape and are excluded from the dashboard configs until
  migrated.
