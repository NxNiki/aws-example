# Fish Hunter (FM01) `group_tag` policy

How fish_hunter bullets are assigned to `group_tag` (`dynamic_rtp` /
`risk_control` / `retention` / `default`), where the single source of truth
lives, and how the user-day dashboard cohorts are derived from it.

## The policy

One row-level CASE, evaluated per bullet in this order (first match wins):

| # | Rule | Tag |
| --- | --- | --- |
| 1 | `strategy_name = 'BOOST_POOL'` | `boost_pool` |
| 2 | `strategy_name IN ('DYNAMIC_RTP', 'DYNAMIC_RTP_V2', 'DYNAMIC_RTP_V3')` | `dynamic_rtp` |
| 3 | `strategy_name LIKE 'RC_FISHING\_%'` (literal `_`) | `risk_control` |
| 4 | `strategy_name LIKE '%RISK_CONTROL%'` | `risk_control` |
| 5 | `strategy_name LIKE 'CR_FISHING\_%'` (literal `_`) | `retention` |
| 6 | `event_timestamp >= 2026-07-31 00:00 UTC` **and** `user_id % 10 IN (0, 1)` | `retention` |
| 7 | everything else (`DEFAULT_FALLBACK`, NULL, …) | `default` |

Key properties:

- **The strategy rules (1–5) apply to ALL history**, regardless of date or
  user digit. This matters because several strategies predate the retention
  launch: `RISK_CONTROLLED` (matched by rule 4) runs from 2026-04-03 and
  `RC_FISHING_V1:*` from 2026-07-14.
- **Legacy strategies keep their own labels**: `BOOST_POOL` → `boost_pool`
  (the strategy no longer exists in new data); `DYNAMIC_RTP` /
  `DYNAMIC_RTP_V2` share `dynamic_rtp` with V3 (the same program over
  different date ranges); `RISK_CONTROLLED` → `risk_control` (same as
  RC_*); `DEFAULT_FALLBACK` / NULL → `default`.
- **`CR_FISHING_*` is the personalized-retention treatment itself** —
  empirically it is served *exclusively* to digit-0/1 users (100.00% of its
  rows, both eras), including a small canary in the hour before the official
  launch (9.5k rows / 13 users from 2026-07-30 23:10 UTC). The explicit rule
  5 tags those canary bullets `retention` even though they precede the gate.
- **Only rule 6 is timestamp-gated.** The personalized-retention
  (个性化挽留) experiment went live 2026-07-30 16:00 PST = **2026-07-31
  00:00 UTC** (`FISH_RETENTION_POLICY_START_UTC`), assigning users whose id
  ends in 0 or 1. Apart from the CR_FISHING canary, bullets before that
  moment can never be tagged `retention`, so applying the CASE to full
  history preserves pre-launch tagging unchanged — there is no separate
  "old era" branch.
- The old dashboard ladder vocabulary (`ab_test_group`: literal RC_*/CR_*
  labels plus the `RC_ALL`/`CR_ALL` rollup) is retired.

The CASE lives in one place: `jobs/etl/sagemaker/spark_etl_common.py::
fish_group_tag_sql` (the `RC_/CR_FISHING_` prefixes are `substr` tests —
the escaped-underscore LIKE — and rule 4's unescaped `_` wildcard also
matches the literal underscore in `RISK_CONTROLLED`).

## Session conventions

The stats grain uses the platform **`bet_session`** for day attribution:
`activity_date` is the Beijing date the bullet's session STARTED (a session
breaks after 30 minutes without a bet), so cross-midnight play stays on the
day it started — same rule as every slot game-stats dataset. The
**fish_hunter HMM feature jobs** (`feature_engineer_{daily,life_cycle}.py`)
instead attribute each bet to the Beijing date its **`hmm_session`** started
(180 s without a bet ends the session, `SESSION_BREAK_SECONDS`). Note this
Beijing dating is specific to the fish HMM jobs — the slot/MAB feature ETL
(`etl_feature_engineer_cold_data.py`) keys `session_start_date` /
`activity_date` by **UTC** dates (its original design; they are stable
incremental-merge keys, so changing the timezone is a semantic rewrite).
The repo's full session vocabulary (`bet_session`
30 min, `agg_session` consecutive-N bets, `ai_session` 12 h, `hmm_session`
180 s) is tabled in [`ab_group_policy.md`](ab_group_policy.md).

## User-day collapse (dashboard cohorts)

The stats grain is one row per `(period, user, group_tag, fish_value)`, so
each user-day needs exactly ONE tag. `fish_group_tag_day_case` collapses a
user-day's row tags with the same priority as the CASE branch order
(boost_pool outranks everything for legacy data):

    boost_pool > dynamic_rtp > risk_control > retention > default

A single bullet in a higher tier claims the whole user-day (e.g. a
retention-cohort user with one `RC_FISHING_*` bullet that day counts as
`risk_control`, not `retention`).

## Single source of truth: `fish_bullets_group_tag`

`jobs/etl/sagemaker/fish_hunter/etl_fish_bullets_group_tag.py` copies the
oceanhunter warehouse `cold_data/bullet` rows and adds the policy-correct
`group_tag` (plus the Beijing `activity_date`). Test bets (op_code
B26/TST/TSB/TSO) are dropped at the source, so consumers and ad-hoc Athena
queries need no op-code filter; `currency_type`/`game_id` filtering stays
downstream (selections, not junk). The copy keeps only the columns the
daily-stats and feature-engineering ETLs consume, plus `strategy_name` for
auditing the tag. (The slot `slot_orders_ab_group` dataset applies the same
source-level hygiene, additionally dropping `status != 'COMPLETED'`; the
bullet table has no status column.)

- Dataset: `s3://bituslabs-team-ai/etl-results/jobs/output_fish_bullets_group_tag/bullets/period=YYYY-MM-DD/`
  (full history from 2025-05-21, the earliest fish cold data)
- Athena: **`bituslabs_ds.fish_bullets_group_tag`** (external parquet,
  partition projection — no crawler; register/update with
  `infra/etl/register_fish_bullets_catalog.py`)
- Refresh: the `etl-fish-bullets-group-tag` step of the
  `slot-cold-data-daily` SageMaker pipeline (9:30 AM LA), immediately before
  the `etl-fm01` stats step that reads it. The policy is derived exactly
  once; nothing downstream touches `strategy_name` for grouping.

Downstream consumers: the daily/weekly/monthly stats ETL
(`etl_game_stats_daily_by_user_cold_data.py`, output
`output_fish_hunter_v3_cold_data` — the dashboard's serving roots) and the
two feature-engineering jobs (`jobs/fish_hunter/feature_engineer_daily.py`,
`feature_engineer_life_cycle.py`).

## How the schedule picks up code changes

Same rule as the slot pipeline (see `ab_group_policy.md`): the EventBridge
schedule executes the code SNAPSHOT from the last pipeline upsert, so after
changing ANY of these ETLs re-run:

    poetry run python infra/etl/deploy_slot_cold_data_pipeline.py

Manual `*_submit.py` runs upload the current local file and are unaffected.

## Changing the policy again

1. Update `fish_group_tag_sql` (and, if the priority changes,
   `FISH_GROUP_TAG_PRIORITY`) in `spark_etl_common.py`; extend
   `tests/unit/test_fish_group_tag.py`.
2. Rewrite the bullets dataset for the affected window (full history:
   `etl_fish_bullets_group_tag_submit.py --start 2025-01-01`).
3. Rerun the stats for the same window
   (`etl_game_stats_daily_by_user_cold_data_submit.py`; its period
   alignment handles weekly/monthly recomputes). The dashboard picks the
   rewrite up automatically — same roots, no config change.
4. Re-run the pipeline upsert (previous section) so the nightly run ships
   the new policy.
