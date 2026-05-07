# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Dashboard config switch leaked UI state between games.** Switching the YAML
  config (e.g. ss02 → fish_hunter) used to leave the previous game's metric and
  group selections in the freshly built dropdowns: `restore_dashboard_state`
  fired on every layout rebuild and re-applied the persistent `dcc.Store`
  payloads. Worst-case symptom was a `polars.exceptions.ColumnNotFoundError`
  on slot-only columns like `user_num_bets_fg` when the Stats Deepdive heatmap
  ran on fish_hunter data. `update_config` now also resets all per-tab Stores
  and `dashboard-load-trigger` to `None`, short-circuiting the restore.
- **Stats Deepdive offered metrics the loaded game can't compute.** The
  `_viz_derived_metric_options` dropdown returned the union of
  `DataMetrics.METRICS` regardless of which `user_*` columns were actually in
  the parquet, so a fresh fish_hunter load could pre-fill heatmap rows with
  slot-only metrics and crash. The list is now intersected with the columns
  present in the schema, backed by a new
  `DataMetrics.metric_user_col_deps` classmethod that introspects each
  metric's body (and any helpers it transitively calls) for `pl.col("user_*")`
  references.
- `_reset_state` now also clears `self.sessions`, restoring symmetry with
  `self.lf_bet` and tightening the early-return guard inside
  `_load_bet_data`.

## [0.2.0] - 2026-05-06

This release establishes the documented release process (see `CONTRIBUTING.md`) and bundles all work since `0.1.5`. Future releases will track changes incrementally in `[Unreleased]`.

### Added

#### AI agent
- Variable-explanation chatbot deployed independently on ECS Fargate.
- Confluence search module to augment chatbot context.
- Metadata enrichment with ETL SQL snippets and `DataMetrics` aggregation; `read_etl_source` / `list_etl_sources` tools.
- Multiple LLM model options for chat performance.

#### Dashboards
- Per-product dashboards (`ss01`, `ss02`, `ss03`, `fish_hunter`) deployed to ECS Fargate.
- Visualization tab with histograms (configurable bins, y-log scale) and box/bar plots grouped by date.
- Weekly report tab.
- Save/load dashboard configuration to/from JSON.
- Dual user-group column selectors with per-(g1, g2) data pre-filtering, replacing the single `ai_group` filter.
- New metrics surfaced: `user_avg_bet_amount`, `user_avg_remaining_bet_amount`, `user_rtp_median`, `rtp_utilization_ratio`, `num_users_no_fg`, delta-bet metrics, num bets for base/free game.
- Cluster transition stats and ss03 cluster_stats view (no need to re-run cluster pipeline).
- IP geolocation lookup.

#### ETL
- ETL pipelines for `ss02` and `ss03` (incentive user stats, num_bets-before-fg, AB-test groups).
- ETL scheduler with partition-based S3 storage, file compaction, and HG-group rollup to `ai`/`default`/per-mathtable groups.
- Incremental ETL for `fish_hunter` (user-level stats).
- Risk-control daily aggregated data for `fish_hunter`.
- Configurable risk-user groups; control group support in risk-user analysis.
- New-user / beginner / old user-group classification in ETL and dashboard.
- ss03 feature-engineering ETL (ordered by spin_id).

#### Cluster analysis & modeling
- Hierarchical clustering model.
- Cluster transition analysis with MCMC.
- Model-inference pipeline (`test_model`, formerly `fit_cluster_model`).
- Elbow-method clip-threshold reporting.
- ss01 cluster-states transition analysis.

#### Operations
- Slack bot for daily reports (user OAuth token).
- Daily-report orchestrator (`run_daily_report.py`): ETL → display → PID-difference check → optional Slack send.
- Risk-user analysis report.
- Single-script orchestrator to run all specified ETL jobs (`run_scheduled_etl_jobs.py`).

#### Infrastructure
- ECS Fargate deployment for ETL jobs, dashboard, and AI agent (five Dockerfiles in `infra/`).
- ECR public mirror for Python base image.
- API keys retrieved from AWS Secrets Manager during deployment.
- `.env.example` template; auto-load env vars in `config.py`.

#### Documentation
- Commit guidelines in `CLAUDE.md`.
- `CONTRIBUTING.md` — branch model, three workflows, ASCII diagrams (incl. rebase before/after), PR template, semver tag scheme. Feature branches must be deleted after squash-merge to `dev`.
- `CHANGELOG.md` — this file.

### Changed

- Centralized configuration: A/B test group IDs, currency, op-code filters, time zone, activity-date offset, delta-t bounds — moved into `config.py`.
- Restricted `user_avg_delta_t_seconds_bg` to BASE game; added delta-bet metrics to ss01/ss02/ss03 ETLs.
- Performance: lazy-load via Polars (replacing pandas in data IO); WSGI server for dashboard; parallel ETL jobs; dashboard deployment tuning to avoid 503s; AI agent never scales to zero.
- Dashboard layout/CSS unified; bootstrap worker extracted to utils; `DEFAULT_VISIBLE_GROUPS = 2`.
- ETL data persistence moved to S3 with compact partitions; ETL function isolated from dashboard into modules.
- Master ETL script: removed `sys.path` manipulation; default lookback days = 1; `--overwrite` option.
- `jobs/operation_daily_report/run_daily_report.py`: removed unused `--bastion-ip`, `--slack-channel`, `--skip-hg-etl`, `--skip-pa-etl`/`--no-skip-pa-etl` flags; replaced with `--run-pa-etl` (default off). HG ETL always runs; downstream scripts use their own bastion-IP defaults; Slack channel comes from `SLACK_CHANNEL_ID`.
- Compute `delta_t_seconds` in the `user_bets` CTE; tightened max bound to 1800s; floored at 1s (slots) / 0.25s (fish_hunter).
- Cluster pipeline: removed highly correlated and payout-related features.
- Limit ss03 cluster-stats query to data from 2026-03-01 onward.

### Fixed

#### ETL
- ss01/ss02 ETL: filter default group for default mathtables.
- ss01: replace `script_id` with `math_table_id` for filtering.
- ss02: handle `FourScatter` (buy-free-game) — rename by mathtable id of next bet, separate buy-free-game indicator column.
- ss03: remove `script_id = giftShop` filter; remove bets in AB-test group in `default_*` mathtables; sql bug fix.
- fish_hunter ETL: `bullet_id` missing in CTE; bullet_id added to window ordering; ETL job name in Dockerfile.
- AI/default group classification based on partition ID (ss01, ss03).
- Avoid dropping NaNs in feature processing; log dropped data in cluster pipeline.
- Hierarchical model not correctly applied.
- Schema-checking bug.
- NaN-value issue in feature-engineering ETL for ss01.
- SSH tunnel connection not stopped properly after `check_pid` jobs.
- Retention calculation for fish_hunter: `-n days` → `+n days`.
- Avoid partial updates for weekly/monthly stats.
- Date filter for last-7-days in ss01 daily report.
- Group ordering in dashboard.

#### Dashboard
- Group-column bug in stats-by-date tab.
- Tab-date data not saved to JSON config; date-range save/restore bug; column-change callbacks no longer overwrite restored selections.
- Drop duplicates for non-user-level stats.
- Log-threshold bug.
- Dashboard service not scaling in (health-check counted as requests).
- `to_markdown()` error in `s3_utils` for weekly report.
- Allow duplicates in save-config output.
- ss01/ss02 g1 not-found error.

#### AI agent / chat
- AI agent replying with stale answer when message not correctly retrieved.
- Show AI message output when output is not valid.
- API key now sourced from Secrets Manager (replaces hard-coded).
- AI agent URL wired into dashboard deployment script.
- Prompt updated to avoid repeating previous answers; fuzzy column-name search enabled.
- Optimized chat-message display (show question instantly while waiting for answer).

#### Operations & infra
- Daily report: `poetry` not found in scheduled run.
- Daily report: AI-group check (null → default); date-start consistency; active-user-stats clarification.
- Write permission on dashboard ECS execution role (fix for config-not-saved issue).
- `dl` Docker image build.
- Lockfile updates.
- Bastion password storage.
- Pyright errors in `etl.py` and confluence client.

### Removed
- Single `ai_group` filter and "Group(s) to Show" checklist (replaced by dual selectors).
- Old per-group ETL jobs (superseded by ETL scheduler with HG rollup).
- `sys.path` manipulation in master ETL script.

---

Releases prior to this changelog (`0.1.0`–`0.1.5`) are recorded in git tags only.
