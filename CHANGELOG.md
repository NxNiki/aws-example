# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  (`infra/deploy_rag_service_ecs.py`) — public ALB on the shared
  ECS cluster, IAM scoped to S3 read of the Faiss artifact plus
  Secrets Manager read of `GOOGLE_API_KEY`.
- **`infra/ecs_helpers.py`** consolidates the helpers that were
  duplicated across the three deploy scripts (account ID lookup,
  default VPC/subnet discovery, ECR repo idempotent create, ECS task
  execution role ensure) plus `ECS_CLUSTER_NAME`. Removes ≈210 lines
  of duplication.
- **`dashboards/secrets.py`** — small env → AWS Secrets Manager
  lookup module extracted from `chat_agent.py`. Lets the slim
  rag-service Docker image use `_get_secret` without dragging
  `langchain_core` into its dependency closure.
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

### Changed

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
- `infra/deploy_dashboard_ecs.py` drops the `--cluster-name`
  CLI argument; cluster name comes from `infra/ecs_helpers.py` so
  all three deploy scripts share one source of truth.
- `infra/deploy_ai_agent_ecs.py` auto-detects the rag-service ALB at
  deploy time and injects `RAG_SERVICE_URL` into the task environment
  (mirrors how the dashboard deploy auto-detects the ai-chat-agent's
  ALB for `CHAT_API_URL`).

### Fixed

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
