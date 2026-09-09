# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **SS03 feature engineering on SageMaker cold data + monthly schedule.**
  The per-bet feature pipeline (`features_enriched` +
  `features_grouped_binsize_{N}`, four ai_group slices) moved from manual
  Redshift-via-bastion runs to one PySpark job reading
  `partition_cold_data/bet_order`, scheduled monthly via EventBridge
  (`ss03-feature-engineer-monthly`, 1st of month 20:00 LA). SQL ported 1:1
  (binary-JSON `partition_ab`, truncating second-diffs, integer-division
  binning, Redshift's integer-AVG truncation reproduced); outputs stay
  byte-compatible with the awswrangler-era files (same roots, column order,
  parquet dtypes). Incremental model: whole-month recompute with dynamic
  partition overwrite — replaces the ETLScheduler key-dedup compaction and
  its mutable-key leak — with a self-healing window (a missed firing widens
  the next run), guards for the pre-cold-data Feb 2026 history and
  partial-month truncation, and the sidecar semantic-drift check. Validated
  against the Redshift-produced July output (100% key parity on enriched,
  exact agreement on all money columns) and re-validated refactor-neutral
  (all 14 datasets row-identical across the review refactor).
- **Reusable SageMaker ETL modules.** `bituslabs_ds.sagemaker_etl` (the
  trusted-role PySpark processor, EventBridge scheduler role/schedule
  helpers with admin-permission fallbacks and ClientRequestToken dedup, the
  shared upsert-with-schedule deploy tail) and
  `jobs/etl/sagemaker/spark_etl_common.py` (container-side helpers shipped
  via `submit_py_files`: AB-group ids + `partition_ab` extraction, common
  Spark session, schema check, partition pruning, window helpers) — used by
  the feature-engineering and daily-stats jobs and all four launchers. The
  monthly schedule reuses the daily pipeline's scheduler role via a
  per-pipeline inline policy.
- **SQL snapshot tests for the cold-data feature queries.** One golden file
  per ai_group slice (enriched) and per slice × bin size (grouped) under
  `tests/unit/feature_cold_data_snapshots/`, mirroring the Redshift suite
  (`REGENERATE_SNAPSHOTS=1` to update).
- **Dashboard response cache + window sharing.** Identical concurrent/repeated
  `/api/data/series` and `/api/data/group-distribution` requests compute once
  and share the result (single-flight, 150 s TTL, keyed by the full request
  JSON; errors and empty results are never cached). Requests whose date
  window and columns are contained in a fresh cached window are served by an
  in-memory slice of that frame instead of a new S3 collect.
- **Column-projected data loads.** Series and Stats-by-Group requests collect
  only the columns their metrics need (cohort/grain dimensions, each
  metric's transitive `user_*` dependencies, and the components
  `collapse_user_rows` recombines); parquet projection pushdown reads just
  those chunks from S3, so loaded frames stay small at any date span.
- **Hive-aware scans with `period=` path pruning.** Each configured source
  loads through one `scan_parquet` (single listing + one schema read) and
  `collect_window` restricts the hive `period` column to the request window,
  so partitions outside it are skipped by path. Request latency now scales
  with the window, not a game's history length (ss01: 248 files / 15 MB,
  cold 8.6 s → 4.4 s).
- **Scale-to-zero.** dashboard-api (min 0 / max 1) scales to zero after 60
  idle minutes — only `/api/*` requests count as activity, so public-ALB
  scanner noise can't keep it awake — and wakes on the ALB 5xx a visit
  produces (~2–3 min task start).
- **Fish hunter on the daily cold-data schedule.** The fish ETL gained the
  slot games' rolling 3-Beijing-day incremental window and an `etl-fm01`
  step in the `slot-cold-data-daily` pipeline; daily/weekly/monthly stats
  were backfilled (2026-01-24 →) and the dashboard config now reads the
  partitioned cold-data datasets for all three granularities.
- **Data-edge auto refresh.** The SPA re-checks date bounds on tab focus and
  hourly; if the data max advanced and the user hasn't moved the
  default window off the previous edge, the window slides forward
  (forward-only — week/month period-start labels can't drag it back) and the
  tab refetches.
- **Scoped availability scans.** New `stats_by_date.availability_cols`
  limits the per-date-range cohort availability scan to dimensions that
  rotate over time (`mathtable`, `ab_test_group`); static vocabularies are
  always selectable.
- `docs/dashboard_dataflow.md` — backend dataflow: request lifecycles,
  the read path, every cache layer with TTLs, ETL freshness guarantees.

- **Fish-level range groups (fish_hunter).** The fish_hunter ETL now stores
  one row per (period, user, daily_group, **fish_value**) in
  `output_fish_hunter_v2`, replacing the fixed per-fish-type wide columns
  (`user_num_hits_fish_low/…` and their `_20_200` variants) with
  dashboard-side bucketing: a "Fish level" picker with user-editable names
  and INCLUSIVE `[min, max]` bounds (defaults low [0, 10], medium [11, 130],
  high [131, 200], ultra [201, max]). Definitions are global per game;
  selection is per tab, next to the cohort pickers. Buckets may overlap and
  are collapsed per user at query time (weighted means recombined by their
  exact bet/kill counts). Also adds `user_num_delta_t` / `user_num_delta_bet`
  recombination weights and drops the pre-bucketed hit/kill ratio columns.
- **Two-column group grain: `ab_group` × `mathtable`.** The slot-game ETLs
  (ss01/ss01a/ss02/ss03/ss06) now store one row per (period, user, ab_group,
  mathtable) instead of UNION-ing every bet into overlapping `ai_group`
  labels: datasets shrink 34–46%, the `group_col_partition` workaround is
  obsolete, and AB-arm × mathtable combinations become directly selectable.
  The dashboard exposes the two columns as independent cohort pickers
  (crossable with lifecycle groups) and re-aggregates per-user stats when a
  dimension is unselected. Sequence metrics (delta_t / delta_bet /
  mathtable_change / FG trigger) are now DAY-partitioned — uniform and
  load-order-independent (previously a mix of full-stream and window-truncated
  semantics); `num_new_users` is derived from the first-bet map (it had
  silently gone missing with the stored `user_group` column) and now means
  "first-ever bet in the period". The flawed `user_avg_remaining_bet_amount`
  metric is removed. Full metric reference: `docs/dashboard_etl.md`.
- **Date-groups picker (dashboard-wide).** Granularity and up to three date
  windows moved from per-tab controls to a global bar above the tab selector
  (and above Lifecycle groups): defined once per game, read by every tab.
  Stats-by-Date plots the shown windows concatenated horizontally on each line
  chart (small gap between windows, one legend entry per cohort × metric,
  weekend stripes per window); the comparison tabs use them side by side as
  before. `/api/data/series` accepts an optional `ranges` list and tags each
  series with `range_index`/`range_label`; loading a saved view migrates old
  per-tab date settings into the global picker.

- **Lifecycle-group picker (dashboard-wide).** New bar above the tab selector:
  up to three user-defined cohorts as half-open ranges `[start, end)` of
  periods since the user's first bet, with an "all" overlay. Units follow the
  active tab's granularity (days / calendar weeks / calendar months, one
  definition per unit); defined once per game and shared by every tab, report
  figure, and saved view. Cohort labels carry the range
  (`new[0, 3)`, `old[7, max)`).

- **SS01A (Golden Goal) game-stats pipeline.** New ETL
  (`jobs/ss01a_golden_goal/`) producing daily/weekly/monthly per-user stats
  with AI/Default groups and per-mathtable variants (no AB arms, no `HG`
  copy), wired into the nightly scheduler, plus the matching
  `dashboard_config-ss01a.yaml` (lifecycle picker enabled,
  `group_col_partition: [AI, Default]`).
- **ETL data-integrity alerts.** Compaction failures and a new post-run
  invariant check (on-disk keys must be unique within the incremental scope)
  now send a Slack alert and are recorded on `ETLScheduler.alerts`, instead of
  being visible only in the job log — a silently skipped compaction previously
  let duplicate lookback rows accumulate unnoticed.

### Changed

- **Explicit, one-to-one picker semantics.** Every dimension picker (cohorts,
  fish/bet-level range groups, lifecycle bar) starts with an explicit `all`
  selected, and fully unselecting any of them shows *no data* on that tab —
  server-side too: a cohort column present with an empty selection yields no
  cohorts (an absent column still means `all` for API callers). Saved views
  were migrated to explicit selections (originals in
  `dashboard-views-backup/`); stale renamed columns were dropped.
- **First launch fetches one panel.** Stats-by-Date seeds only the first
  panel's first metric; Stats-by-Group defaults to `num_active_users` in the
  first panel with a `<None>` option on every panel. Selections persist
  across tab switches.
- **Large-sample CIs use the m-out-of-n bootstrap.** Samples above 10k values
  resample at m=10k and rescale deviations by √(m/n) — empirical shape kept,
  correct width, cost capped (~0.03 s at n=1M vs ~2 s); resampling is one
  integer-indexed draw at a time (~80 KB transient instead of a
  hundreds-of-MB matrix). Smaller samples keep the plain percentile
  bootstrap.
- **dashboard-api task 1 → 2 vCPU** (8 GB unchanged): after the software
  fixes, memory sat under 50% while CPU still pinned at 100% in busy
  windows; polars aggregation + CI computation is CPU-bound.
- **SPA cache headers.** `index.html` serves `Cache-Control: no-cache`
  (revalidates every load; ETag 304s) and hashed assets are immutable —
  browsers no longer run stale bundles after deploys.
- Cache TTLs tightened against compound staleness: response 300→150 s,
  first-bet map 3600→900 s, group-values vocabulary 3600→600 s.
- **Lifecycle cohorts are derived at query time.** `dashboard_api` computes
  each user's first bet date from the daily user rows (cached per config) and
  filters cohorts by period range on demand
  (`lifecycle_groups` on the data endpoints), instead of reading a stored
  label. Custom ranges therefore apply retroactively to all history.
- Loading a saved view now drops cohort selections for columns the config no
  longer defines, so stale snapshots can't silently filter the data; re-saving
  the view persists the cleaned state.

### Fixed

- **Redshift ss03 feature-SQL snapshot tests were silently red** since the
  job moved to its per-slice `GROUPS`/`build_config` structure: the test's
  inline config copy had drifted and the parity check hit an
  `AttributeError`. The configs are now loaded from the job script itself
  and fan out per slice, so drift shows up as a reviewable snapshot diff.
- **Empty results are never cached.** A read racing the ETL's dynamic
  partition overwrite (delete-then-rewrite) could collect an empty frame for
  a window that has data; caching it pinned "no data" on every panel for the
  TTL. Empty windows, empty covering slices, and empty responses are now
  returned but recomputed on the next request.
- **Group-values cache stampede.** The picker vocabulary cache had no lock
  across the request threadpool, warmup and refresh threads; concurrent cold
  misses each ran a full-parquet scan. Now lock-guarded with per-key
  single-flight (one scan, shared result).
- **Backward window slide.** The data-edge refresh compared week/month
  period-start labels against the day-granularity max and could pull an
  up-to-date window back two days; it now only advances.
- **Duplicate startup fetches.** `selectConfig` and the tab effect both fired
  `loadAllSeries`; the second aborted and re-issued an identical request.
  In-flight requests with the same signature are now left to land.
- **Fish hunter daily stats had no schedule.** The cold-data cutover left the
  fish ETL running only manually with hardcoded dates; its dashboard data
  froze at the last manual run until the pipeline step above landed.
- **Silent ETL compaction failure duplicated recent rows on every game.** The
  2026-07-17 in-place parquet rewrite produced Arrow `large_string` columns
  while the ETL writes `string`; `_compact_partitions`' dataset read refused
  to merge the two types and the error was only logged, so each incremental
  run appended its lookback re-pull as duplicate keys (daily rows for
  2026-07-15→07-18 doubled; weekly/monthly windows back to early July / May —
  SS03 total_bet read 2× its true value on the duplicated dates; distinct
  user counts, ratios, and per-user means were unaffected). Compaction now
  falls back to per-file reads on Arrow type mismatches, and all 18 S3
  prefixes were de-duplicated and type-normalized in place (~515k rows
  removed; originals under
  `s3://bituslabs-team-ai/etl-results/backup/dup_type_repair_20260720/`).
- **"all" cohort double-counted bets on the ss games.** The ss01/ss02/ss03/ss06
  ETLs UNION every bet into a combined AB-test label AND a per-mathtable
  re-partition of the same bets (ss01 adds a third full `HG` copy); the
  dashboard's "all" summed every row, inflating totals ~1.4–3× and polluting
  per-user averages/distributions (SS03 total_bet showed 147.8M for
  2026-06-01→07-20 where the true figure is 103.9M). Configs now declare the
  disjoint labels (`group_col_partition`) and "all" aggregates only those;
  individually selected groups are unchanged.

### Removed

- **Stored `user_group` / `user_group2` columns.** Dropped from the game-stats
  ETLs (ss01/ss02/ss03/ss06/fish_hunter) and removed in place from the
  existing dashboard parquet on S3 (originals backed up under
  `s3://bituslabs-team-ai/etl-results/backup/user_group_drop_20260717/`).
  The fixed new/beginner/old split is superseded by the lifecycle-group
  picker's default ranges.

## [0.5.0] - 2026-07-14

Major release. The legacy Dash dashboard is replaced by a FastAPI `dashboard_api`
service + a React/TypeScript SPA, alongside new metrics, two new game pipelines,
and the SS03 clustering / AB-test workflow.

### Added

- **New dashboard: `dashboard_api` (FastAPI) + React/TypeScript SPA (`frontend/`).**
  Replaces the Dash `game_stats_monitor`. Tabs: Stats-by-Date, Stats-by-Group,
  Deep Dive, Summary Table, and Report; plus a group-distribution endpoint and an
  OpenAPI-generated typed frontend client (`scripts/gen_openapi_client.sh`).
  Deployed to ECS Fargate via `infra/dashboard_api/`.
- **Dashboard configs served from S3 at runtime.** `dashboard_api` reads
  `dashboard_config-*.yaml` from an `s3://` `config_dir` (default
  `s3://<bucket>/dashboard-configs`) with a short TTL cache, so adding a config is
  an S3 upload — no image rebuild/redeploy. Added `read_yaml_from_s3` and
  `uri_basename` to `s3_utils`.
- **Summary Table tab** (dedicated Stat column, per-metric red→green colormap for
  ±%) with Confluence HTML export; Report-tab figure reordering and
  summary-above-figures.
- **Retention metrics day-5/7/10/15/30** (and `dayN_num_users`) in `DataMetrics`,
  exposed in dashboard configs + agent metadata.
- **numpy-only Welch t-test and one-way ANOVA** for group comparisons.
- **SS06 (pocket_soccer)** ETL pipeline + dashboard config.
- **risk_control user-aggregate ETLs** (`etl_{risk,control}_user_aggregates`,
  aggregated in Redshift), a `get_ip_locations` ip-api `/batch` helper, and
  anomaly-flagged user-id groups (group4/5) in `risk_users.json`.
- **Apply a pretrained KMeans model to new cohorts from S3 (no retraining).**
  `ClusterAnalysisPipeline` gains `pretrained_model_dir`,
  `ensure_pretrained_artifacts()` (fetch model + `feature_order.json` +
  `clip_bounds.json` from a prior S3 run into the current run) and
  `save_apply_run_info()`; apply-only groups (`default`/`ab_test_a`/`ab_test_b`) in
  the SS03 cluster config; run_id-keyed S3 upload so groups sharing a `--run-id`
  land in one folder.
- **SS03 user daily stats ETL** (`etl_user_daily_stats.py`): per
  `(user_id, session_start_date, cluster)` rows with a dominant `ab_group` label
  and nominal `"cluster N"` labels; feeds the "SS03 Cluster & AB Test" dashboard.
- **SS03 AB-test feature-engineering groups** (`ab_test_a` / `ab_test_b`) with
  per-group `date_start`.
- **fish_hunter FTUE report** enhancements: per-bin bet-behavior time series with
  95% CI bands, per-active-user means / raw totals / grouped metrics, bet
  increase/decrease counts and ratios, a strategy-comparison section, and a
  bet-behavior markdown export.
- **fish_hunter SQL query snapshots** in `tests/unit/etl_snapshots/` with a
  self-verifying `tests/unit/test_etl_sql_snapshots.py` (regenerate via
  `REGENERATE_SNAPSHOTS=1`).
- **Feature engineering feature reference** in `docs/feature_engineering.md`:
  feature dictionary for the SS01/SS02/SS03 bet-segmentation pipeline
  (`src/bituslabs_ds/features/`) covering the pipeline key steps, the common
  `features_enriched` / `features_grouped_binsize_{N}` columns, per-game
  configuration differences, and a placeholder section for future
  game-specific features.

### Changed

- **`ml` lazily imports matplotlib/seaborn/skl2onnx** so the module imports with
  just the `ml` dependency group (no viz/onnx stack needed for
  clustering/prediction).
- **`load_raw_data` no longer bakes `row_filters` into the local cache** — filters
  are applied at read time by `load_cluster_data`/`load_attach_data`, so changing a
  filter no longer needs a manual cache bust (only a source-data change needs
  `reload`).
- **risk_control stats are aggregated in Redshift** instead of pulling raw bullet
  events; the control ETL uses full history (short random windows sampled mostly
  inactive users).
- **`dashboard_api` ECS deploy is a steady-state deployer** (one-time cutover
  logic removed).
- Frontend polish: per-tab granularity/date/cohort controls, incremental
  per-panel loading with request cancellation, number precision/formatting, and
  chart sizing.

### Fixed

- **fish_hunter daily/weekly/monthly user stats only counted fish-killers**
  (`HAVING MAX(b.killed) > 0` in `stats_by_user_date`), skewing distinct-user
  metrics (retention, `num_active_users`, RTP). All betting users are now emitted;
  re-run with `--overwrite` to backfill.
- **SS03 `AB_TEST_A` / `AB_TEST_B` partition_ab ids were swapped**, so the ETL
  labeled the cohorts inversely; ids corrected and historical datasets relabeled
  with `--overwrite`.
- **`dashboard_api` memory / caching**: single-flight window cache shared across
  Stats-by-Group / Deep Dive, batched bootstrap-CI resampling, and 8 GB / 1 vCPU /
  single-worker task sizing to stop OOMs; the window cache no longer served the
  wrong game's data across config switches.
- **Any-group retention** read the cohort slice instead of the full population.
- **Deep-dive log axes** broke on non-positive values.
- Frontend: `crypto.randomUUID` crash on HTTP; chat sent on IME-composition Enter;
  tooltip CI bounds; assorted display fixes.
- Infra: associate target groups with the ALB before ECS attach; unblock the
  `dashboard_api` image build.

### Removed

- **Legacy Dash dashboard decommissioned:** `src/dashboards/game_stats_monitor.py`,
  `weekly_report.py`, `report_agent/`, and the old `infra/dashboard/` deploy —
  superseded by `dashboard_api` + `frontend/`.
- `jobs/risk_control/etl_get_risk_user_stats.py` (replaced by the aggregate ETLs).

## [0.4.0] - 2026-06-17

### Added

- **Two-phase clustering for SS03 (train one ai_group, score another).**
  `ClusterAnalysisPipeline` trains KMeans on the `train` group (Default) and
  applies the frozen model to the `inference` group (AI) from one config via a
  `groups:` mapping and `--group {train,inference}`. Training persists the model,
  ordered feature list (`feature_order.json`) and fitted clip bounds
  (`clip_bounds.json`); inference reuses them so preprocessing is a pure transform.
  Cross-group labels coexist in a shared `cluster_labels.parquet`.
- **Timestamped, bin-size-tagged clustering outputs.** Runs write to
  `result_<date>_binsize<N>/` (local + S3) so they don't overwrite; per-cluster
  attach files and processed caches carry the `_binsize<N>` tag.
- **SS03 feature-engineering `--group` switch** with separate Default/AI S3
  prefixes and per-group `bin_size`.
- **fish_hunter FTUE analytics:** first-session ETL, event-timing HTML report, and
  an interactive strategy-comparison report (multi-group select, ratio/abs + log-y,
  per-group palette) with a Confluence publisher; deposits/withdrawals, device/IP
  and per-strategy breakdowns; CNY transactions attached to the FTUE ETL.
- **rag_service:** daily scheduled reindex via Fargate task + EventBridge;
  `life_cycle_prediction` Confluence source.
- **SS03 daily-by-user-group** SQL query snapshots + ai_group double-counting docs.

### Changed

- `ETLScheduler` default start date is now configurable.
- rag_service: removed the unused `/reindex` endpoint.

### Fixed

- `clip_outliers` no longer emits a pandas FutureWarning on integer columns.
- `check_pid_difference`: rolling-window comparison with per-game isolation.

## [0.3.0] - 2026-06-05

### Added

- **Report tab in the dashboard.** New tab that turns rendered figures
  into a Confluence-published analytical report. Workflow: click
  *Add to report* under any chart to stage it; open the Report tab to
  generate per-figure descriptions and an overall summary via the LLM;
  click *Export to Doc* to push the snapshot to a Confluence page.
  Subsequent exports replace the `Dashboard Report` region in place.
  See `docs/report_agent.md` for the full contract.
  - **Editable descriptions and summary.** Both render as textareas;
    edits persist on blur and are sent to the LLM as a "prior draft to
    refine, preserving any user edits" on the next regenerate.
  - **`/prompt` instruction syntax + intent-driven Generate.** Lines
    starting with `/prompt` (legacy `/prompt:` also accepted) at the
    start of a textarea line are extracted as explicit instructions
    to the next LLM call. Generate is now safe to click — it never
    silently overwrites manual edits:
    * Empty textarea → first draft from the data summary.
    * Non-empty + no `/prompt` → **no-op**; a status hint tells the
      user nothing happened.
    * Non-empty + `/prompt` at the very start → full regenerate from
      instructions.
    * Non-empty + `/prompt` after some prose → prose before the first
      `/prompt` is preserved byte-for-byte; the LLM produces only
      additional paragraphs and the caller concatenates the two.
      Uses a dedicated `*_APPEND_SYSTEM_PROMPT` that forbids the
      model from echoing or rewriting the preserved region.
  - **Per-figure data summary.** The LLM receives a JSON summary of
    each figure's traces, axes, error bars, and heatmap z-matrix —
    not the rendered PNG — so descriptions cite exact values instead
    of guessing. Plotly 6.x binary array dicts (`{dtype, bdata,
    shape}`) are decoded back to Python lists transparently.
  - **User-curated references.** *Add reference* button attaches
    Confluence URLs (or external URLs) that ground the LLM prompts.
    Each row eagerly fetches on URL blur and shows a green ✓ + page
    title, red ✗ + error, or grey ↗ for external links. Reference
    bodies are inlined in subsequent Generate prompts; every URL is
    listed under `<h2>References</h2>` at the bottom of the exported
    Confluence doc.
  - **Languages.** Description and summary generation accept English,
    Simplified Chinese, or Traditional Chinese via a top-bar dropdown.
- **`Add to doc` button under each chart in the existing tabs**, which
  stages the figure into the Report tab in one click.
- **RAG service.** New FastAPI microservice (`src/rag_service/`) that
  indexes Confluence documentation declared in `rag_sources.yaml`,
  embeds it with Gemini `gemini-embedding-001` (768-dim via Matryoshka
  truncation), persists a Faiss + JSON metadata pair to S3, and serves
  a `POST /retrieve` endpoint backed by an in-memory Faiss index. The
  ai-chat-agent's `search_confluence_rag` tool is now wired to this
  service; documentation questions are answered from the curated
  corpus in ~3–10 s instead of via per-turn live Confluence calls.
  See `docs/rag_service.md` for the full design spec (storage backend,
  build cadence, env vars, and Phase 2 migration to managed
  OpenSearch). Includes:
  - Confluence **folder** URL support via the v2 API's
    `direct-children` endpoint — paste a `/folder/<id>` URL in
    `page_urls` and the loader recurses into nested subfolders.
  - `RAG_BACKEND` env switch so a future migration to OpenSearch
    requires only a config flip + one reindex.
  - Daily-rebuild Fargate task using the same image with CMD
    overridden to `python jobs/build_rag_index.py`.
  - Smoke-test queries (`docs/rag_service.md`) and two pytest
    modules (`tests/integration/test_rag_service.py` and
    `test_chat_agent_rag.py`) — the latter inspects message
    `tool_calls` so the "agent isn't actually calling the RAG" class
    of bug fails the suite before it ships.
- **Deploy script for the RAG service**
  (`infra/rag_service/deploy_ecs.py`) — public ALB on the shared
  ECS cluster, IAM scoped to S3 read of the Faiss artifact plus
  Secrets Manager read of `GOOGLE_API_KEY`.
- **RAG source: `ai_summary_pages`.** New entry in
  `src/rag_service/config/rag_sources.yaml` pointing at the
  `spaces/hub/pages/627179541/AI` Confluence page (no recursion).
  Picked up by the next daily index rebuild.
- **`infra/shared/ecs_helpers.py`** consolidates the helpers that were
  duplicated across the three deploy scripts (account ID lookup,
  default VPC/subnet discovery, ECR repo idempotent create, ECS task
  execution role ensure) plus `ECS_CLUSTER_NAME`. Removes ≈210 lines
  of duplication.
- **`dashboards/secrets.py`** — small env → AWS Secrets Manager
  lookup module extracted from `chat_agent.py`. Lets the slim
  rag-service Docker image use `_get_secret` without dragging
  `langchain_core` into its dependency closure.
- **Report-agent LLM moved to the ai_agent service.** Two new endpoints
  on the ai_agent FastAPI app — `POST /api/report/description` and
  `POST /api/report/summary` — wrap the existing `generate_description`
  / `generate_summary` functions. The dashboard's Report tab now calls
  these over HTTP instead of running LangChain in-process. The four-case
  generate semantics (empty / no-`/prompt` no-op / `/prompt` at start /
  preserved + append) are preserved verbatim and reported as a typed
  status string in the response body. Reference fetching stays in the
  dashboard — pre-fetched bodies ship in the request payload, so the
  ai_agent never does Confluence I/O on the hot path. Effect: prompt
  and behavior tweaks redeploy only the ai_agent; dashboard rebuilds
  drop from ~5–10 min to never-needed for report-agent iteration.
  Poetry's `confluence` group splits out of `llm` so the dashboard
  image can install Confluence integration without the LangChain stack.
- **Slack bot integration for the AI agent.** New `POST /slack/events`
  endpoint on the ai_agent FastAPI service that responds to Slack
  `app_mention` events (DMs and untagged channel messages are ignored
  on purpose — heavyweight LLM calls only fire when the bot is
  explicitly tagged). Replies post in-thread, with the prior 20 thread
  messages pulled back as history so follow-up @-mentions have
  context. The handler dispatches the LLM call as an `asyncio` task
  so the Bolt adapter can ack inside Slack's 3-second deadline, and
  dedupes on `event_id` to swallow Slack's cold-start retries. The
  endpoint is opt-in — mounted only when `SLACK_BOT_TOKEN` +
  `SLACK_SIGNING_SECRET` are present in env or Secrets Manager, so
  local dev without Slack credentials still boots cleanly. Fronted by
  a CloudFront distribution that exposes the HTTP-only ALB over
  `https://*.cloudfront.net` (Slack requires HTTPS for Event
  Subscriptions URLs). See `docs/ai_agent.md` for the design spec,
  request flow, CloudFront configuration reference (for recovery),
  Slack app setup steps, and a troubleshooting table covering the
  scope-mismatch and missing-`aiohttp` gotchas hit while wiring this
  up.
- **Unified per-bet feature-engineering pipeline (`bituslabs_ds.features`).**
  A single SQL-composition engine driven by a per-game `GameFeatureConfig`
  replaces the previously copy-pasted per-game ETL scripts. CTE builders
  (`user_bets` → `delta_stats` → `user_group` → `raw_stats` → percentiles →
  `stats_base` → final select) consume the config; each game
  (`jobs/ss0{1,2,3}/etl_feature_engineer.py`) just constructs a
  `GameFeatureConfig` and hands it to `FeaturePipelineRunner`. Ships
  character-for-character SQL **snapshot tests** with per-game job-config
  parity checks (`tests/unit/test_features_sql_snapshots.py`).
  - **Stable session IDs.** `session_start_ts` is forward-filled from each
    session's first bet via `LAST_VALUE(... IGNORE NULLS)` over a local gap
    marker, so the same logical session keeps the same id no matter how much
    history a run scans — making incremental appends safe. Downstream
    `session_start_date` + a within-date `session_group` ordinal identify a
    session.
  - **Config-driven incremental lookback.** `effective_lookback_days()`
    derives the ETLScheduler lookback from `session_break_threshold_seconds`
    (`ceil(threshold/86400)+1`), so each in-window session's first bet is
    always captured.
  - **Config sidecar + drift guard.** Each run writes a `_feature_config.json`
    sidecar recording the semantic config; the next run hard-fails (unless
    `--overwrite`) if a semantic field changed, so rows built under
    incompatible semantics (e.g. a different `bin_size`) can't be silently
    appended.
  - **Multi-bin-size runs with a shared enriched dataset.** `bin_size` accepts an
    `int` or a `list[int]`. `raw_stats` now emits a bin-independent
    `session_bet_index` (the bet's row number within its session) instead of
    `agg_group`, so `features_enriched/` is written **once** and shared across all
    bin sizes; the runner derives `agg_group = floor((session_bet_index-1)/bin_size)`
    in a per-bin `binned` CTE and writes one `features_grouped_binsize_{N}/` per size.
    The cluster pipeline (`ml.py` + `cluster_config-*.yaml`) reads the shared enriched
    and recomputes `agg_group` from `session_bet_index` for the label merge.
- **ss03 clustering enhancements.** DBSCAN and subsampled-hierarchical model
  options alongside k-means in `ClusterAnalysisPipeline`; a shared
  `cluster_labels.parquet` (one column per model/feature/k run); config-driven
  removal of incomplete bins (`remove_short_sessions` + `session_count_column`);
  and `spin_id` carried through `attach_cluster_label`'s per-cluster output so
  cluster labels can be joined back to spin-level events. See
  `jobs/cluster_analysis/README.md`.

### Changed

- **Production cutover: the React dashboard replaced the legacy Dash app**
  (frontend redesign Phase 5). `dashboard_api` (FastAPI: SPA + `/api/data/*` +
  `/api/report/*`) now serves the same production URL the Dash app did,
  behind the existing `game-stats-dashboard-alb` — no new ALB. The listener
  default was flipped from the Dash target group to `dashboard-api-tg`
  (atomic `modify_listener`), and a priority-10 `/api/agent/*` rule routes
  browser chat traffic to the `ai-chat-agent` service via a second target
  group on that ALB (a TG belongs to one ALB, so the agent service carries two
  — note: recreating that service from scratch silently drops the extra TG).
  ALB idle timeout raised 60 → 300 s for SSE chat streams. The fleet stays at
  three ECS services (dashboard-api, ai-chat-agent, rag-service); the legacy
  `game-stats-dashboard` service is parked at desired-count 0 for a rollback
  bake, with its scale-to-zero alarms/policies removed (the scale-out alarm
  watched the shared ALB's 503 count and would otherwise resurrect Dash).
  `infra/dashboard_api/deploy_ecs.py` performs the whole sequence idempotently
  and prints a rollback runbook. The Weekly Report placeholder tab was removed
  from the SPA (its nightly ETL job is kept). `dashboard_api` also emits the
  `Dashboard/UserRequestCount` metric (scale-to-zero idle signal), gated on
  `DASHBOARD_SERVICE_NAME`.
- **Repository reorganization (moves-only, behavior-preserving).** Shared
  infrastructure that lived inside the legacy `dashboards` package moved into
  the `bituslabs_ds` library so the Dash package can be deleted after the bake:
  `metrics/user_stats_aggregates.py` (DataMetrics + `_bootstrap_ci`),
  `confluence/{client,export_html,references}.py`, and `aws_secrets.py`.
  One-line re-export shims remain at the old `dashboards.*` paths so the frozen
  legacy image stays rebuildable until deletion; `dashboard_api`, `ai_agent`,
  `rag_service`, and tests import from the new paths, and the three service
  Dockerfiles drop their per-file `src/dashboards` COPY lines (the modules ride
  the existing `COPY src/bituslabs_ds`). The six `dashboard_config-*.yaml`
  moved out of `src/dashboards/` to a top-level `configs/dashboard/`, resolved
  via `DASHBOARD_CONFIG_DIR` everywhere — which also fixed a latent bug where
  `ai_agent`'s `chat_api` resolved configs from a directory that never held
  any YAMLs, silently dropping every `dashboard_config` request. Eleven
  root-level one-off scripts moved to `jobs/analyses/` (scheduled job paths are
  frozen — EventBridge bakes them into rule targets — and stayed put); see the
  new `jobs/README.md`.
- Renamed `src/dashboards/secrets.py` → `src/dashboards/aws_secrets.py`
  to avoid shadowing the Python stdlib ``secrets`` module. Running any
  module under ``src/dashboards/`` as a path (e.g. ``python
  src/dashboards/game_stats_monitor.py``) used to put the directory on
  ``sys.path[0]`` and break numpy's ``bit_generator`` import
  (``cannot import name randbits``). All importers updated:
  ``ai_agent.chat_agent``, ``ai_agent.slack_handler``,
  ``dashboards.confluence_client``, ``rag_service.embeddings``, plus
  the ``COPY`` lines in ``infra/ai_agent/Dockerfile`` and
  ``infra/rag_service/Dockerfile``.
- `chat_agent._build_llm` defaults `CHAT_PROVIDER` to `auto`, picking
  OpenAI if `OPENAI_API_KEY` is set or Gemini if
  `GOOGLE_API_KEY` / `GEMINI_API_KEY` is. Previously the default
  `openai` raised `ValueError` whenever only Gemini credentials were
  available. Backwards-compatible: explicit `CHAT_PROVIDER=openai`
  keeps prior behavior.
- `confluence_client.extract_page_id_from_url` now resolves both the
  legacy 6-char base64 tinyurl format (`/wiki/x/Z4AfOw`) offline and
  newer Cloud short codes via authenticated HTTP redirect, with a
  graceful fallback when neither path matches.
- Pinned `kaleido` to `0.2.1` (briefly tried `>=1.0` for arm64 wheel
  availability, then reverted): 1.x rewrote its renderer to drive an
  *external* Chrome via DevTools Protocol and fails with
  `Kaleido requires Google Chrome to be installed` on the slim ECS
  image used for the dashboard. 0.2.1 ships a self-contained
  Chromium wheel (~70 MB) that works in slim containers — much
  smaller than installing system Chrome (~150–250 MB). Affects the
  Report tab's PNG export of figures into Confluence.
- `_viz_derived_metric_options` is now schema-aware (see Fixed below);
  this changes the **Stats Deepdive** dropdown contents per-game.
- `chat_agent` tool docstrings and `_SYSTEM_PROMPT` rewritten to make
  `search_confluence_rag` clearly **PRIMARY** and `search_confluence` /
  `read_confluence_page` clearly **FALLBACK ONLY**. The prompt now
  forbids calling the live tools unless the RAG tool returned
  literally "No Confluence passages found" or "RAG service
  unavailable". Gemini 2.5 Flash was previously prone to picking the
  live keyword tool first.
- `infra/dashboard/deploy_ecs.py` drops the `--cluster-name`
  CLI argument; cluster name comes from `infra/shared/ecs_helpers.py` so
  all three deploy scripts share one source of truth.
- `infra/ai_agent/deploy_ecs.py` auto-detects the rag-service ALB at
  deploy time and injects `RAG_SERVICE_URL` into the task environment
  (mirrors how the dashboard deploy auto-detects the ai-chat-agent's
  ALB for `CHAT_API_URL`).
- **Promoted ai_agent service modules out of `src/dashboards/` into a
  new `src/ai_agent/` package.** Moved: `chat_agent.py`, `chat_api.py`,
  `slack_handler.py`, `metadata_builder.py`, `column_metadata.yaml`,
  `report_agent/description.py`. Stays in `dashboards/` (shared or
  dashboard-only): `confluence_client.py`, `secrets.py`,
  `user_stats_aggregates.py`, `report_agent/{references,exporter,
  data_summary}.py`, dashboard configs. uvicorn entrypoint:
  `dashboards.chat_api:app` → `ai_agent.chat_api:app`. `Dockerfile.ai_agent`
  swaps 6 per-file COPYs for one `COPY src/ai_agent ./src/ai_agent`.
  External code referencing `dashboards.chat_agent` /
  `dashboards.chat_api` / `dashboards.slack_handler` /
  `dashboards.metadata_builder` / `dashboards.report_agent.description`
  must update to `ai_agent.X`; all in-tree call sites + 6 doc files
  + the dashboard's "AI agent unreachable" UI message updated together.
- **Migrated chat agent off the deprecated `langgraph.prebuilt.create_react_agent`**
  to `langchain.agents.create_agent`. Same first two positional args,
  so the call site is unchanged (import is aliased to
  `_create_langchain_agent` to avoid colliding with the local
  `create_agent()` helper). Removes the `LangGraphDeprecatedSinceV10`
  warning. The new `CompiledStateGraph.invoke` / `ainvoke` type their
  input as a private `_InputAgentState` TypedDict; the plain
  `{"messages": […]}` dict gets a targeted `# type: ignore[arg-type]`
  at the three call sites — runtime unaffected, just silences pyright
  without coupling to a private symbol.
- **Feature-engineering config field renames** (clarity; pure renames — values
  and generated SQL unchanged): `session_length` → `bin_size` (consecutive bets
  per `agg_group`, not a session length); `max_session_gap_seconds` →
  `max_delta_t_gap_seconds` (only clamps `delta_t` for the `*_nogap` metric — not
  a session boundary); `max_session_interval_seconds` →
  `session_break_threshold_seconds` (the gap that starts a new session,
  paralleling `streak_threshold_seconds`). The cluster-analysis YAMLs + `ml.py`'s
  `bin_size` property were renamed to match. Because the config sidecar is keyed
  by field name, the first run after this against an existing dataset needs
  `--overwrite` (or a one-time key rename in the sidecar).
- `GameFeatureConfig.requires_full_history` now defaults to `False` — the
  stable-session-id incremental run is canonical; set it only when SQL semantics
  change.
- **`session_break_threshold_seconds` default is now 12 hours** (was 7 days);
  ss01/ss02/ss03 also set it explicitly. Sessions split on any gap > 12h and the
  derived incremental lookback drops to 2 days — a session-boundary (semantic)
  change, so the first run against existing data needs `--overwrite`.
- **ss03 feature engineering now runs across `bin_size=[30, 50, 70, 100]`**
  (was a single `100`): one shared `output_ss03_feature_engineer/features_enriched/`
  plus four `features_grouped_binsize_{30,50,70,100}/` datasets under the same root.

### Fixed

- **dashboard_api served the wrong game's data after a config switch.** The
  in-process window cache (`services/common.py`, added with the OOM fixes
  below) keyed each cached user-row frame on `cfg.get("id")` — but the raw
  config YAML has no `id` (it's derived from the filename), so the key was
  `(None, granularity, start, end)` for *every* config. Switching games with
  the same granularity and date window (e.g. ss02 → ss03) hit the previous
  game's cached frame and rendered its numbers under the new config. Affected
  all three data tabs (series, deep dive, group distribution).
  - Surfaced two ways: (1) the freshly loaded config showed the previous
    game's values; (2) **retention appeared to "change" when the date range
    changed** — not a retention-calculation problem, but because the range is
    part of the cache key, so changing it forced a cache *miss* that finally
    fetched the correct config's data. With the cache keyed correctly, both
    go away.
  - Fix: `load_raw_config` now stamps `cfg["id"] = config_id` (the single load
    chokepoint every data route uses), and `collect_window` raises rather than
    caching under a `None` id, so a future caller that forgets fails loudly
    instead of silently serving another game's rows. Regression tests cover
    the no-collision keying and the missing-id guard. Verified in production:
    ss02 vs ss03 `user_total_bet` now return distinct means (1041.80 vs
    1082.78), stable across a round-trip.
- **dashboard_api task OOM-killed under real dashboard load (502s).** Three
  compounding causes, fixed in sequence: (1) the per-config parquet cache is
  held in-process, so two uvicorn workers doubled it — pinned to a single
  worker; (2) a tab render fires one `/api/data/*` request per panel in
  parallel, and each collected its own copy of the same user-row window —
  added the single-flight, byte-budgeted `collect_window` cache so one render
  collects once; (3) `_bootstrap_ci` allocated the full `(n_boot × len(arr))`
  resample matrix at once (~800 MB for a 200k-row per-user metric, several
  landing concurrently) — now resampled in batches (~80 MB transient,
  statistically identical CIs). Task sized to 8 GB / 1 vCPU.

- **Chat agent silently fell back to live Confluence instead of RAG.**
  `search_confluence_rag` raised `ImportError` inside the slim
  ai-agent Docker image because `rag_service/client.py` wasn't copied
  into it. The tool caught the import and returned its
  "RAG service client is not installed" string; the LLM read that as
  "no docs found" and switched to `search_confluence`. Symptom: chat
  answers cited Confluence pages correctly but took 15–30 s, and the
  rag-service log showed zero `POST /retrieve` entries. Fix is two
  extra `COPY` lines in `Dockerfile.ai_agent`
  (`rag_service/__init__.py` + `client.py` — no Faiss / OpenSearch /
  embeddings deps pulled in). Verified end-to-end: production chat
  call now produces a `POST /retrieve` and the answer cites the
  indexed passage in ≈5–10 s. (The new
  `tests/integration/test_chat_agent_rag.py` would have caught this
  before deploy — it inspects message `tool_calls` for the
  `search_confluence_rag` invocation.)


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
- **`read_files` silently returned an empty DataFrame when every parallel read
  failed**, which surfaced far downstream as a confusing `KeyError` on an
  expected column. It now raises `RuntimeError` with the first underlying error
  when all reads fail; partial failures are still tolerated. Covered by new tests.
- **`load_attach_data` raised `KeyError: 'billtime'` when ss03 attach was
  enabled.** `merge_date` (derived from `billtime`) is a legacy-schema artifact
  used only by the wucaishen/deepdive `merge_on`; it is now built only when
  `merge_date` is actually a merge key, so ss03 (which has no `billtime` column)
  is unaffected.
- **Resilient local parquet cache.** A crash mid-write (e.g. an interrupted
  incremental cache build) left a footer-less parquet that wedged every later
  run; the reader now drops an unreadable cache and reloads from source.
- **ETL partition columns are built in a single concat** rather than inserted
  one at a time, avoiding pandas DataFrame-fragmentation warnings.

### Removed

- **Legacy Dash dashboard decommissioned (frontend redesign Phase 5
  complete).** After a clean one-week production bake of the React dashboard
  (steady traffic, zero errors, legacy service idle at desired 0), the old Dash
  app and its infra were deleted: `src/dashboards/` (the ~7,200-LOC
  `game_stats_monitor.py`, `weekly_report.py`, the legacy `report_agent`
  exporter/data_summary, and the back-compat shims left by the reorg) and
  `infra/dashboard/`. The `dashboards` package entry and the `dashboard` poetry
  group (dash, plotly, kaleido, gunicorn, matplotlib) were dropped from
  `pyproject.toml` — every shared dep in that group remains available via the
  `ds`/`dev`/`dashboard_api` groups. AWS teardown removed the
  `game-stats-dashboard` ECS service, its task-definition family,
  `game-stats-dashboard-tg`, the legacy task security group, the
  `/ecs/game-stats-dashboard` log group, and the `bituslabs-ds-dashboard` ECR
  repository. **Kept:** the production ALB (`game-stats-dashboard-alb`, now
  fronting `dashboard-api`) and its security group; the weekly-report ETL job
  and its scheduled-jobs registry entry. Rollback to Dash is no longer a
  one-line ALB flip — it requires rebuilding from a pre-deletion git SHA.

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
