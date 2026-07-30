# Dashboard backend dataflow

How data moves from the ETL pipelines to a rendered plot, what happens on
each user action (page refresh, date-range change, picker change), where
every cache sits, and how fresh data reaches an open browser tab.

Code anchors: `src/dashboard_api/` (FastAPI service), `frontend/src/store/
dashboardStore.ts` (SPA state + fetch orchestration), `bituslabs_ds/
s3_utils.py` (S3 helpers), `infra/etl/deploy_slot_cold_data_pipeline.py`
(daily ETL schedule).

## 1. Big picture

```
Game warehouses (slotmachine / oceanhunter buckets, cross-account)
        │  SageMaker pipeline "slot-cold-data-daily", 9:00 America/Los_Angeles
        │  one PySpark step per game: ss01, ss01a, ss02, ss03, ss06, fm01(fish)
        │  rolling window: recompute the last 3 Beijing days, dynamic
        │  period= partition overwrite (history untouched)
        ▼
S3 parquet, hive layout: .../output_<game>_v2_cold_data/{daily,weekly,monthly}_stats/period=YYYY-MM-DD/*.parquet
        │  one row per (period, user, ab_group, mathtable[, bet_level]) for
        │  slots; (period, user, daily_group, fish_value) for fish hunter
        ▼
dashboard_api (ECS Fargate, 2 vCPU / 8 GB, scale-to-zero after 1h idle)
        │  polars lazy scans + in-process caches (see §5)
        │  configs read from s3://…/dashboard-configs/*.yaml (60s TTL —
        │  editing a config is an S3 upload, no redeploy)
        ▼
React SPA (served by the same container; index.html no-cache, hashed assets immutable)
```

The parquet is **one row per user per period per grain combination** — the
API recomputes group-level metrics (RTP, active users, retention, CIs) from
user rows on every request; nothing group-level is precomputed. That is what
makes arbitrary cohort cross-products, lifecycle groups, and value-range
buckets possible, and it is why the caching layers below exist.

## 2. What happens on a page refresh

1. **Bundle load.** The browser revalidates `index.html` (served with
   `Cache-Control: no-cache`; the ETag makes an unchanged page a 304).
   Hashed `/assets/*` bundles are immutable-cached for a year — a new deploy
   produces new hashes, so refresh always runs the latest code.
2. **`loadConfigs()`** → `GET /api/data/configs` (yaml list from S3, 60s
   in-process TTL) → auto-selects the first config → **`selectConfig(id)`**,
   which runs strictly in order:
   1. `GET /api/data/config/{id}` — resolved config: tabs, metric groups,
      cohort columns, range-group ladder. The store seeds per-tab
      `rangeSelection = ["all"]` (when the config has a range dimension) and
      `lifecycleAll = true`.
   2. **`ensureGroupValues`** → `GET /api/data/group-values?config=…` —
      the full cohort vocabulary (server-cached 10 min, stale-while-
      revalidate, single-flight) plus, when a window is known, the
      range-scoped `available` subset for the columns listed in
      `stats_by_date.availability_cols` (mathtable / daily_group — the
      dimensions that rotate over time). When values arrive, every visible
      picker that has no selection yet is seeded with an explicit `"all"`.
   3. `loadDeepdiveMetrics`, then **`loadDateBounds`** →
      `GET /api/data/date-bounds` — min/max of the date column, computed
      fresh per request. The store records `dataMax` and seeds Date-groups
      R1 to the last 30 days ending at max.
   4. **`loadAllSeries`** — one `POST /api/data/series` per panel *that has
      metrics selected*. On first launch only the first panel does (its
      first metric), so a fresh session costs one series request.
      Stats-by-Group behaves the same way (first panel = `num_active_users`
      when offered, later panels `<None>`).
3. The Stats-by-Date tab's effect watches (config, granularity, ranges,
   cohorts, metrics, …) and refetches on change. Duplicate invocations with
   an identical in-flight request signature are dropped rather than
   abort-and-reissued (`_panelInflight`).

## 3. What happens on each control change

| Action | Frontend | Requests fired |
| --- | --- | --- |
| Date range edit | R1/R2/R3 in the global Date-groups bar; context signature changes | `POST /series` per panel-with-metrics (new window), availability re-scan for the union window |
| Granularity day/week/month | window kept as-is; week/month labels are **period starts** (a "2026-07-27" week label is the Monday of the current week) | series refetch at the new granularity (different parquet dataset) |
| Cohort / range-group / lifecycle change | per-tab selection state; **explicit semantics** — `"all"` selected by default; emptying any picker means *show no data* and skips the fetch entirely | series (or group-distribution / summary / deepdive) refetch |
| Metric added to a panel | incremental — only the newly added metric is fetched and appended; removed metrics stay cached client-side for instant re-add | `POST /series` with just the new metric |
| Tab switch | pure state flip; selections persist per tab | none (the entering tab refetches only if its context changed) |
| Saved view load | snapshot sanitized: stale columns dropped, missing pickers seeded `"all"`, empty selections read as `"all"` | full refetch for the restored state |
| Idle tab | on tab-focus and every 10 min the store re-checks `date-bounds`; **forward-only** — if the data edge advanced and the user hasn't moved R1's end off the old edge, the window slides forward, a toast announces it, and the tab refetches | `GET /date-bounds`, then series on change |

Selection semantics are one-to-one by design: *absent* column (API callers
only) = "all"; *explicit `"all"`* = one aggregate cohort; *explicit values*
= one cohort per value; *explicitly emptied* = no cohorts, nothing plotted —
enforced in the UI (fetch skipped) and the API (`cohort_values` yields an
empty cross-product).

## 4. Anatomy of one series request (server side)

`POST /api/data/series` → `routers/data.py::post_series`:

1. **Response cache** (`cached_response`): key = the full request JSON.
   Hit → return in ~0.1s. Miss → compute single-flight (identical concurrent
   requests wait for one computation). Responses with zero series are
   returned but never cached (see §6).
2. **`load_lazy`** — one hive-aware `pl.scan_parquet` per configured source
   path: a single S3 LIST expands the prefix, one footer read resolves the
   schema, and `period=` directories become a hive partition column.
   (Historically this was one scan + eager schema fetch *per file* — one
   blocking S3 round-trip each, so latency scaled with history length;
   ss01's 248 small files made the smallest game the slowest.)
3. **Column projection** (`projection_columns`): the request's metrics are
   resolved to the exact columns needed — cohort/grain/partition dimensions,
   each metric's transitive `user_*` dependencies
   (`DataMetrics.metric_user_col_deps`), and the numerator/denominator/
   weight columns `collapse_user_rows` recombines. Parquet projection
   pushdown then reads only those column chunks from S3.
4. **Period pruning** (`_prune_periods`): the hive `period` column is
   restricted to the request window (widened one period left for week/month
   labels), so polars skips all other partitions **by path** — a 24-day
   window on ss01 touches ~24 files, not 248.
5. **Window cache** (`collect_window`): the collected frame is cached under
   (config, granularity, window, column-set) for 15 min, single-flight under
   one lock (parallel per-panel collects of the same window OOM-killed
   earlier task sizes). Two extra rules:
   - **Covering slice**: a request contained in a *fresh* cached entry with
     a column superset is served by an in-memory filter of that frame — one
     user loading a long range makes narrower requests nearly free. Empty
     slices are never served (a pre-ETL frame can "cover" dates it has no
     rows for).
   - **Empty frames are never cached** (§6).
6. **Cohorts** (`iter_cohorts`): the frame is filtered per cohort
   cross-product; grain rows (per user × ab_group × mathtable × bet_level)
   collapse back to one row per (period, user) when a grain dimension is
   unselected, recombining ratios/weighted means from their exact
   components. Lifecycle groups join a per-config first-bet map (derived
   from the daily parquet, cached 15 min).
7. **Metrics** (`DataMetrics`): computed metrics aggregate group-level;
   `user_*` metrics get a mean ± 95% CI per date. CIs use a percentile
   bootstrap; samples above 10k values use the m-out-of-n form (resample
   m=10k, deviations rescaled by √(m/n)) so cost is capped while the
   empirical (asymmetric-capable) shape is kept. Resampling is one
   integer-indexed draw at a time (~80 KB transient).
8. The response (one series per metric × cohort × range with x/y/CI) is
   cached (if non-empty) and returned.

`POST /api/data/group-distribution` (Stats-by-Group) follows the same path
with a single metric and per-(cohort × range) five-number summaries + CIs.

## 5. Cache reference

| Layer | Where | Key | TTL | Size limit | Notes |
| --- | --- | --- | --- | --- | --- |
| Browser: `index.html` | client | — | `no-cache` (ETag 304) | — | refresh always gets the current bundle |
| Browser: `/assets/*` | client | content hash | 1 year, immutable | browser-managed | safe by construction |
| Config yamls | server | path | 60 s | one per config | S3-served; edit = upload, no redeploy |
| Response cache | server | full request JSON | 150 s | 512 entries (LRU) | single-flight; errors and empty results never cached |
| Window cache | server | config+gran+window+columns | 900 s | **2.5 GB estimated bytes (LRU)** | single-flight lock; covering-slice reuse; empty frames never cached |
| Group-values vocabulary | server | config+granularity | 600 s | one per (config, gran) | stale-while-revalidate (background refresh); single-flight cold misses |
| First-bet map (lifecycle) | server | config | 900 s | one per config | full-history min-date scan per user |
| Date bounds | server | — | none | — | computed fresh per request (this is what lets an open tab discover new data) |
| Frontend `_panelCtx` / series | client memory | panel + context signature | session | per open tab | incremental metric fetches; cleared on context change |

### Storage limits in detail

- **Window cache — 2.5 GB of estimated frame bytes**
  (`_WINDOW_CACHE_MAX_BYTES`, `services/common.py`), enforced by
  `_evict_over_budget()` on every insert. Budgeted in *bytes*
  (`DataFrame.estimated_size()`), not entry count, because entries vary from
  a few MB (one projected metric, short window) to GBs (a months-long
  Stats-by-Group span) — counting entries is exactly how an earlier 4 GB
  task OOM'd. Eviction is LRU (hits `move_to_end`; eviction pops the
  front), so frames that keep serving traffic — including via covering
  slices — survive. The `len > 1` guard means the **newest frame is never
  evicted**, even if it alone exceeds the budget: it is what the in-flight
  burst of panel requests is sharing. The 2.5 GB figure is sized against
  the 8 GB task — budget + the largest concurrent collects must fit with
  margin (sizing history in the comment above the constant).
- **Response cache — 512 entries** (`_RESPONSE_CACHE_MAX_ENTRIES`), LRU by
  insertion. Entries are small JSON-serializable results, so a count bound
  is enough.
- **Vocabulary / first-bet / config caches** hold one small entry per
  (config[, granularity]) — bounded by the number of games, no explicit
  limit needed.
- **`DataMetrics` instances** memoize computed metrics via
  `cached_property`, but each instance lives only for one request × cohort
  and is garbage-collected with it — request-scoped, not a persistent
  cache.

Worst-case staleness after new data lands in S3 (no restarts needed):
window cache (≤15 min) + response cache (≤2.5 min) ≈ **~17 min** for series
content; the pickers' vocabulary ≤10 min; lifecycle first-bet ≤15 min. An
open tab's *window* advances within ≤10 min of the bounds check (or on the
next tab focus).

## 6. Cache sharing across users

Every server-side cache is process-global and keyed by *what* was requested,
never by *who*: there is no per-user state, so one user's work is reused by
everyone. The layers form a funnel:

1. **Response cache** — hit when a second user's request JSON is byte-identical.
   Deterministic defaults (explicit `all` pickers, last-30-days window, first
   panel's first metric) make fresh sessions on the same game byte-identical
   on purpose. Concurrent identical requests single-flight: one computes, the
   rest wait and share.
2. **Window cache / covering slice** — hit when the data window overlaps: a
   different cohort selection reuses the same frame (cohort filtering re-runs
   on it); a narrower range is sliced from a fresh covering frame in memory;
   a different metric set collects its own (small) projected entry.
3. **S3** — only when nobody loaded that window recently.

Practical consequence: the first person each morning pays the cold collect;
everyone else inside the window TTL is fast, and identical views inside the
response TTL are near-instant. Caches live in one process on one ECS task —
scaling out would warm one copy per task, and a scale-to-zero wake starts
cold except for the vocabulary pre-warm.

## 7. ETL freshness and the empty-result rule

The daily pipeline overwrites the last 3 Beijing days of `period=`
partitions with Spark **dynamic partition overwrite** — files are deleted,
then rewritten. A dashboard read racing that rewrite can list files that
vanish before collection and produce an empty (or partial) frame for a
window that has data. Two protections:

- **Never cache empty**: empty windows, empty covering slices, and empty
  responses are returned to the requester (the panel may show a blank for
  that one request) but are never pinned in a cache — the next request
  re-reads and, once the rewrite finishes, succeeds.
- TTLs bound how long any partial read that *did* return rows can persist
  (≤15 min).

A stronger guarantee (atomic `_SUCCESS` markers or temp-dir-and-rename in
the ETL) is a known future hardening; today the exposure is the few minutes
around the 9am LA pipeline run.

## 8. Operational notes

- **Scale-to-zero**: the ECS service (min 0 / max 1) scales to zero after 60
  minutes with no `Dashboard/UserRequestCount` datapoint. Only `/api/*`
  requests count (the public ALB's scanner traffic on the SPA/static paths
  is ignored). A visit while asleep 503s at the ALB, which trips the wake
  alarm; the task starts in ~2–3 min. Deploys reset desired count to 1.
- **Deploy** (`infra/dashboard_api/deploy_ecs.py`): syncs repo configs to
  S3, rolls a new task definition, and (re)applies the autoscaling policies.
  Rollback = re-tag a previous ECR digest and re-run.
- **Task sizing**: 2 vCPU / 8 GB. Memory sizing history and the window-cache
  byte budget are documented at the constants in `deploy_ecs.py` and
  `services/common.py`.
