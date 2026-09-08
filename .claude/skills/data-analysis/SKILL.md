---
name: data-analysis
description: Conventions for ad-hoc game-data analyses (SS03/slot games) — data sources, metric definitions, extreme-value handling, dashboard reconciliation, and the Jira + Confluence publishing workflow. Use when running a retention / bet / RTP analysis or publishing its report.
---

# Game Data Analysis Conventions

Established on the SS03 AI-group mathtable analysis (AIP-575, Confluence page
1353973761). Reference implementation:
`jobs/ss03_mahjiang_streak/analysis_ai_mathtable_permutation.py` (extraction +
metrics) and `generate_ai_permutation_report.py` (Confluence publishing).

## Workflow

1. **Jira first** (only when asked): Task under epic **AIP-21 "Data Analysis"**,
   title `[Data Analysis][<game>] <outcome>`, ADF description. See CLAUDE.md
   "Jira Tickets".
2. **Extract** from Athena to local parquet (`--extract`), **analyze** from the
   cache (`--analyze`) — separate CLI flags so iteration never re-scans Athena.
   Outputs under `jobs/output_<game>*/<analysis_name>/`.
3. **Publish** to Confluence in **Simplified Chinese** (CLAUDE.md "Confluence
   Reports"), attach the CSVs, and reconcile headline numbers with the DS
   dashboard before sharing.

## Data sources

- **Order-level truth: Athena `bituslabs_ds.slot_orders_ab_group`**
  (S3: `s3://bituslabs-team-ai/etl-results/jobs/output_slot_orders_ab_group/orders/`,
  partitioned `game_id`/`period`=Beijing date). Carries the **policy-correct
  `ab_group`** (partition_ab before the 2026-08-04 05:30 BJ cutover, user-id
  digits after). status ≠ COMPLETED and test op_codes already removed.
  Columns include bet_amount, actual_payout, math_table_id, currency_type.
- It is **NOT currency-filtered**; the dashboard `*_stats` datasets and the
  feature-engineering datasets are **CNY-only**.
- **Grouping is game-specific.** The dataset's global `ab_group` applies the
  2026-08-04 digit-policy cutover to EVERY game, but that cutover is
  **SS03-only**: use `ab_group` for SS03; for **SS02 derive groups from the
  raw `partition_ab_label`** (AI id `jojpin-9mokha-rexQug` → AI, else
  Default — SS02 never had A/B groups; verified 2026-09-02 by table-mix:
  digit-"AI" converges to Default's medium3 post-cutover while partition-AI
  keeps the MAB rotation). For other games, verify which source reflects the
  real treatment (compare table mix by group around 08-04) before analyzing.
- Verify partition coverage (`MIN/MAX(period)`, per-day row counts) before
  analyzing: **today's partition is partial**; last complete Beijing day is
  yesterday.

## Metric conventions

- **Currency**: MAB/AI analyses are **CNY-only** (`currency_type='CNY'`) — the
  MAB policy only acts on CNY users. Never mix currencies in per-user averages;
  if a multi-currency view is ever needed, convert to CNY and label it clearly
  (converted means are LOWER than CNY-only, since non-CNY markets bet small).
- **Day attribution**: sessionize with a **30-min bet gap**; every bet belongs
  to the **session START's Beijing date** (midnight-crossing play stays on the
  day it started). Scan 2 extra days before the window so straddling sessions
  keep their true start. Retention presence uses the same dating.
- **Counts/amounts**: `user_num_bets` = BASE spins; `user_total_bet` = BASE
  bet_amount; `user_avg_bet_amount` = BASE total / BASE count per user-day;
  RTP = actual_payout (all rows, FREE included) / BASE bet_amount.
- **Extreme values**: WINSORIZE the means of user_total_bet / user_num_bets at
  the **99th percentile** — clip to the threshold, never drop (whales are real
  revenue). Estimate thresholds per date-range on the FULL population (linear
  interpolation), apply to every cell; never re-estimate inside small cells.
  Medians, retention, and **all RTP inputs stay unclipped** (clipping the RTP
  denominator inflates RTP).
- **Daily-spend tiers are game-specific**: pick round edges near ~p20 /
  median / ~p90 of the game's user-day total-bet distribution (SS03:
  10/100/1000 CNY; SS02: 10/50/500 — its day-bet median is 49 vs SS03's 110).
  Check the distribution before reusing another game's edges.
- **Betting rhythm**: per user-day, `user_med_interval` = median WITHIN-
  session gap (seconds) between consecutive BASE bets; `user_max_streak` =
  longest run of consecutive BASE bets with gaps < 200s. Cells show
  p99-clipped mean / raw median like the other activity metrics. (Null-safe
  clipping: guard `min_horizontal` so null stays null, not the threshold.)
- **User GGR**: one value per USER per range/population — Σ(bet − payout)
  over all the user's user-days in the condition (negative = user won). A
  user spanning several cells (tier/table/permutation) goes to their modal
  cell; ties to the larger cell. Being signed, its mean is winsorized on
  BOTH tails (p1/p99, full-population thresholds); median raw.
- **Retention D1**: cohort user-days on D; retained = any SS03 CNY bet in a
  session starting D+1; null past the last complete day. Exclude the AB-policy
  cutover days 2026-08-03/04 from cohorts.
- **Mathtable labeling**: `normal_kakuteiB/C` are single-spin bonus sub-tables
  (median 1 BASE spin) — never a "second table" and never a day's label; their
  bets still count in totals. For single-table cohorts, drop user-days with
  ≥2 REAL tables (~0.1%) instead of winner-take-all.
- **Consecutive-block cleaning** (drop same-table runs < 30 BASE bets) applies
  only where the question demands it (e.g. table-permutation ranking) and only
  to the LABEL: a cleaned user-day's bet/spin/payout metrics still cover the
  whole day (dropped short blocks included); user-days with no surviving block
  drop out. Never clean headline metrics.
- **Rankings**: require a minimum cell size (≥30 user-days) so one-off cells
  can't top a leaderboard; always show sample sizes and mean AND median.
- **Sequence/permutation labels use rotation-cadence segments ("|"-joined)
  and rank within length**: split each consecutive same-table run into
  segments matching the game's MAB rotation cadence — measure it from the
  run-length histogram first (SS03: 50 bets, remainder <30 dropped; SS02: 30
  bets, remainder <20 dropped). 100 bets of shi → "shi|shi", length 2; no
  adjacent collapsing; label length ≈ cleaned bets/segment. Engaged users bet
  more and get longer labels (reverse causality), so rank only within the
  same length (1-4, ≥20 user-days), never across.

## Dashboard reconciliation

The DS dashboard (`etl_game_stats_daily_by_user_group.py` → `*_stats`) differs
by design: CNY-only, no FX, `user_num_bets` counts FREE spins (bet_amount 0, so
totals match), day = the BET's calendar date (not session start), and its
percentile filter **drops** rows outside `np.percentile(vals, lo/hi)` (linear)
computing mean AND median on the filtered set. When numbers "disagree",
reproduce the dashboard's definition from `slot_orders_ab_group` first — every
past gap (785 vs 667 vs 596) was definitional, not data. Document the
reconciliation chain in the report.

## Publishing to Confluence

- Simplified Chinese; storage-format HTML built by a `generate_*_report.py`
  beside the analysis script; SS03 analyses go under folder **1267040280**
  ("SS03_MahjiangStreak_数学表分析").
- Use `bituslabs_ds.confluence.client` (`_v2_session` to create under a folder,
  `update_page_storage` to update). Pin the page id in the script after first
  publish so re-runs update in place.
- **Attachment corruption — the real story (learned the hard way 2026-09-04).**
  (1) `Confluence.attach_content` re-posts a CONSUMED stream when the filename
  already exists → silently replaces the attachment with a 0-byte version;
  `bituslabs_ds.confluence.client.attach_file` now uses raw multipart requests
  and verifies uploaded size — never call the atlassian lib for attachments.
  (2) Once a 0-byte upload happened, a USER-EDITED page's image node keeps the
  broken media binding in its `ac:local-id`d element: EVERY page save (even
  external API saves) re-materializes the broken media as a fresh 0-byte
  attachment version — re-uploading the file or renaming it via `ri:filename`
  alone does NOT break the cycle. Repair recipe: upload the figure under a NEW
  filename, then replace the ENTIRE `<ac:image ...local-id...>` element with a
  minimal fresh node (`<ac:image ac:align="center" ac:width="...">
  <ri:attachment ri:filename="NEW.png" /></ac:image>`), publish, verify the
  attachment size survives ~30s after the save. `ri:version-at-save` is diff
  metadata, not what the renderer uses — don't chase it.
- On user-edited pages never run a full `--publish` (it reverts their edits);
  use attachment updates + surgical storage-body patches via raw v2 PUT.

## Dashboard reconciliation

The DS dashboard (`etl_game_stats_daily_by_user_group.py` → `*_stats`) differs
by design: CNY-only, no FX, `user_num_bets` counts FREE spins (bet_amount 0, so
totals match), day = the BET's calendar date (not session start), and its
percentile filter **drops** rows outside `np.percentile(vals, lo/hi)` (linear)
computing mean AND median on the filtered set. When numbers "disagree",
reproduce the dashboard's definition from `slot_orders_ab_group` first — every
past gap (785 vs 667 vs 596) was definitional, not data. Document the
reconciliation chain in the report.

## Publishing to Confluence

- Simplified Chinese; storage-format HTML built by a `generate_*_report.py`
  beside the analysis script; SS03 analyses go under folder **1267040280**
  ("SS03_MahjiangStreak_数学表分析").
- Use `bituslabs_ds.confluence.client` (`_v2_session` to create under a folder,
  `update_page_storage` to update). Pin the page id in the script after first
  publish so re-runs update in place.
- **Attachments: update IN PLACE, never delete+re-attach, and always BEFORE
  the body save.** Pass the right content_type ("image/png" / "text/csv") —
  the historical 400s came from type mismatch. Rationale: a user-edited body
  pins images with `ri:version-at-save`; deleting an attachment (new id) or
  bumping its version after the body save leaves the pin dangling → the
  figure renders as invisible/UNKNOWN_ATTACHMENT. A body save re-pins every
  image to the attachment's current version, so the order attachments→body
  is self-healing. To repair a broken page without touching user edits:
  fetch storage, fix `ri:filename`/strip stale pins surgically, republish
  that exact body (the save round-trip re-pins).
- Report structure: 背景 / 数据来源 (table + S3 path + producing ETL) / 口径与
  数据清洗 (every convention above that applies, with the dashboard对照 bullet)
  / result tables (sample sizes + 均值/中位 pairs) / 结论 / 数据与脚本.
