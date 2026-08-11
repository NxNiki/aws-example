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

## Changing the policy again

1. Update `ab_group_sql` / add a new gate in `spark_etl_common.py`.
2. Backfill `slot_orders_ab_group` for the affected window
   (`etl_slot_orders_ab_group_submit.py --start ... --end ...`).
3. Backfill the game-stats datasets for the same window (the submit script's
   period alignment handles weekly/monthly recomputes).
4. Decide whether feature outputs need a transition exclusion, and rerun the
   feature pipeline if so.
