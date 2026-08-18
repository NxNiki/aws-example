# Slot AB grouping policy & the 2026-08 cutover

How slot-machine bets are assigned to `ab_group` (`Default` / `AB_TEST_A` /
`AB_TEST_B` / `AI`), where the single source of truth lives, and the evidence
behind the policy cutover timestamp.

## The two policies

| Era | Assignment rule |
| --- | --- |
| before 2026-08-04 05:30 Beijing | first element of the raw `partition_ab` JSON: the AI/A/B group ids (see `spark_etl_common.py`), anything else → `Default` |
| from 2026-08-04 05:30 Beijing | **last digit of `user_id`**: 0–3 `Default`, 4–5 `AB_TEST_A`, 6–7 `AB_TEST_B`, 8–9 `AI` |

The switch was announced on 2026-08-03 (LA time). After the cutover the
`partition_ab` column kept its old values and **stopped reflecting
assignment** — post-cutover it agrees with the digit rule only at chance
level (~30%), so it must never be used for dates past the cutover.

Both rules live in one place: `jobs/etl/sagemaker/spark_etl_common.py::
ab_group_sql` (a timestamp-gated SQL CASE) with the cutover constant
`AB_GROUP_DIGIT_POLICY_START_BJ`. Games without AB test groups (SS01, SS01A)
collapse the test digits / legacy test ids into `Default`.

## How the cutover timestamp was determined

Empirically, from the orders data. Under both eras, which mathtable a group
is SERVED identifies the group (SS03):

| Group | Tables served before the cutover | After |
| --- | --- | --- |
| `AB_TEST_A` | `normal_zero` (dominant), `normal_shi` | `normal_shi` only |
| `AB_TEST_B` | `normal_zero93kai`, `normal_Zero_BGTR95_Saitekika_BGadj_v3` | `..._BGadj_v3` only |
| `Default` | `normal_zero_95_kai` | `normal_zero_95_kai` only |
| `AI` | rotating 8-table set (`zero`, `zero_95_kai`, `shi`, `san`, `go`, `ni`, `roku`, `ichi`) | same |
| (any) | `normal_kakuteiB` / `normal_kakuteiC` are bonus tables, served to every group — no group signal | same |

For users whose old group differs from their digit group the two table sets
are disjoint, so hourly "which config is this user being served?" is
unambiguous. Across 2026-08-03 → 08-04 (Beijing) that evidence reads:

- through 08-04 04:59 — 100% old-config serving;
- 08-04 05:00–06:00 — bet volume collapses ~10× (the deploy window), mixed
  serving (~600 bets, mostly the group-agnostic kakutei tables);
- from 08-04 06:00 — ~99% digit-config serving.

`2026-08-04 05:30 Beijing` (≈ 08-03 14:30 LA) splits the deploy hour. Every
bet outside that hour gets a provably correct label; only the deploy-hour
bets carry a best-effort boundary.

## Single source of truth: `slot_orders_ab_group`

`jobs/etl/sagemaker/slot_machine/etl_slot_orders_ab_group.py` copies the raw
warehouse `bet_order` rows for the five slot games and adds the
policy-correct `ab_group` (plus `partition_ab_label` and the Beijing
`activity_date`). Rows are otherwise unfiltered — `status`, `op_code`,
`currency_type` stay as columns for downstream queries to filter.

- Dataset: `s3://bituslabs-team-ai/etl-results/jobs/output_slot_orders_ab_group/orders/game_id=<G>/period=YYYY-MM-DD/`
- Athena: **`bituslabs_ds.slot_orders_ab_group`** (external parquet,
  partition projection — no crawler; register/update with
  `infra/etl/register_slot_orders_catalog.py`)
- Refresh: FIRST step of the `slot-cold-data-daily` SageMaker pipeline
  (9:30 AM LA ≈ 00:30 next-day Beijing, so each Beijing day is complete
  before the copy). All game-stats steps are chained after it in the same
  pipeline — there is no separate schedule to race.

Downstream consumers — the per-game game-stats ETL
(`etl_game_stats_daily_by_user_group_cold_data.py`) and the SS03
feature-engineer (`etl_feature_engineer_cold_data.py`) — read this dataset
and use the stored `ab_group` directly. The policy is derived exactly once;
nothing downstream touches `partition_ab`.

## Day basis: session-start dates (all game-stats datasets)

Every game-stats dataset (all five slot games, both SS03 variants, and
fish_hunter) computes `activity_date` as the SESSION-START Beijing date,
using the platform's **bet_session** (30 minutes without a bet ends the
session) — the daily-stats midnight fix: cross-midnight play stays on the
day it started. This is pure day attribution, independent of group labeling
(a bet's label never depends on which day bucket it lands in).

Session vocabulary (four distinct notions — don't mix them):

| name | break rule | used for |
| --- | --- | --- |
| `bet_session` | 30 min without a bet | `activity_date` attribution in ALL game-stats datasets; delta-t caps (`DELTA_T_MAX_SECONDS`) |
| `agg_session` | consecutive 30/40/50-bet windows | AI aggregation features |
| `ai_session` | 12 h without a bet | AI modulation only (ss03 feature ETL `SESSION_BREAK_SECONDS`; multi-day sessions exist) |
| `hmm_session` | 180 s without a bet | fish_hunter feature engineering only: `bet_date` attribution for the HMM lifecycle features (`jobs/fish_hunter/feature_engineer_{daily,life_cycle}.py` `SESSION_BREAK_SECONDS`) |

## SS03 dashboard exception: announced cutover + user-day groups

The SS03 game-stats dataset (the dashboard's `ab_group` dimension) does NOT
use the stored row-level label. Its stats ETL applies the user-day group
policy (`GAME_CONFIG['SS03']['user_day_groups']` in
`etl_game_stats_daily_by_user_group_cold_data.py`; the run/cohort variants
opt out and keep the stored row-level label):

- **Cutover**: the game team's ANNOUNCED start, 2026-08-03 16:00 PDT =
  **2026-08-03 23:00 UTC** (`spark_etl_common.SS03_AB_GROUP_ANNOUNCED_START_UTC`),
  not the empirical 05:30-Beijing constant — bets in the ~1.5h between the
  serving flip and the announced time keep stale partition_ab labels by
  product decision. The label is re-derived from `partition_ab_label` /
  `user_id` / `created_at` (`ss03_bet_ab_group_sql`).
- **Old-era collapse**: each (user, session-day) gets ONE label, priority
  `AI > AB_TEST_A > AB_TEST_B > Default` — old-era assignment was per-bet,
  so a single AI bet claims the user's whole day. Digit-era labels are
  user-stable, so the collapse is a no-op there.

Everything else — the orders dataset, the other four games' stats, the ss03
feature ETL (`ai_group`) and the AI-run30 combo dataset — keeps the stored
row-level `ab_group` under the empirical cutover.

## Transition-day handling

- **Game stats / dashboards / Athena**: every bet keeps its timestamp-gated
  label — 08-03 is fully old-policy, 08-04 splits at 05:30 Beijing.
- **SS03 feature engineering**: sessions **starting** on 2026-08-03 or
  08-04 (Beijing) are excluded from all outputs
  (`EXCLUDED_SESSION_START_DATES_BJ`) — group membership mid-transition is
  ambiguous for behavioral features, and keying the exclusion on session
  START keeps cross-midnight sessions whole. The exclusion is recorded in
  each dataset's `_feature_config.json` sidecar; the sidecar's semantic-drift
  guard requires a one-time `--allow-semantic-drift` when a recorded
  semantic field changes (it exists so months computed under different
  semantics are never silently appended next to each other).

## How the schedule picks up code changes

The EventBridge schedule only calls ``StartPipelineExecution`` by pipeline
name; the code each run executes is the SNAPSHOT uploaded to S3 by the last
pipeline upsert. Editing an ETL script locally (or even merging it) changes
nothing for the scheduled runs until the deploy script is re-run:

    poetry run python infra/etl/deploy_slot_cold_data_pipeline.py
    poetry run python infra/etl/deploy_ss03_feature_engineer_pipeline.py

A stale snapshot silently rewrites its rolling window with old logic every
morning (that's how the 2026-08 post-cutover residue appeared), so re-run
the upsert as part of shipping ANY slot ETL change. Manual ``*_submit.py``
runs are different: they upload the current local file per job.

## Changing the policy again

1. Update `ab_group_sql` / add a new gate in `spark_etl_common.py`.
2. Backfill `slot_orders_ab_group` for the affected window
   (`etl_slot_orders_ab_group_submit.py --start ... --end ...`).
3. Backfill the game-stats datasets for the same window (the submit script's
   period alignment handles weekly/monthly recomputes).
4. Decide whether feature outputs need a transition exclusion, and rerun the
   feature pipeline if so.
