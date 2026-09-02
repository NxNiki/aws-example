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
- Do NOT use the S3 copy's `partition_ab_label` for post-cutover grouping — it
  diverges from the policy `ab_group`; use `ab_group`.
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
  bet_amount; RTP = actual_payout (all rows, FREE included) / BASE bet_amount.
- **Extreme values**: WINSORIZE the means of user_total_bet / user_num_bets at
  the **99th percentile** — clip to the threshold, never drop (whales are real
  revenue). Estimate thresholds per date-range on the FULL population (linear
  interpolation), apply to every cell; never re-estimate inside small cells.
  Medians, retention, and **all RTP inputs stay unclipped** (clipping the RTP
  denominator inflates RTP).
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
- **Attachment gotcha**: updating an existing attachment's data 400s —
  delete-by-name (`confluence.delete_attachment`) then re-attach.
- Report structure: 背景 / 数据来源 (table + S3 path + producing ETL) / 口径与
  数据清洗 (every convention above that applies, with the dashboard对照 bullet)
  / result tables (sample sizes + 均值/中位 pairs) / 结论 / 数据与脚本.
