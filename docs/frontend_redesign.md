# Frontend Redesign: Agent-Native Dashboard (React + Vite + TypeScript + ECharts)

**Status:** Complete — Phases 0–5 shipped on `feat/frontend-redesign-react`; the legacy Dash app is decommissioned (see CHANGELOG → Removed).
**Author:** michael.niu@bituslabs.com
**Date:** 2026-06-03
**Scope:** Replace the monolithic Dash app (`src/dashboards/game_stats_monitor.py`, ~7,200 LOC) with a React + Vite + TypeScript SPA (charts via **Apache ECharts**), backed by a consolidated FastAPI backend that **returns data, not figures**. The goal is a dashboard where LLM/agentic capabilities are first-class — the agent can answer data questions by driving the UI, and can regenerate whole reports from a declarative spec on a one-line prompt.

---

## 1. Motivation

The current dashboard is a single 7,200-line `GameStatsDashboard` class with 34 Dash callbacks, inline CSS dicts, and HTML built in Python. It works, but it has hit structural limits:

- **Agentic features are bolted on.** The chat panel and Report-tab generation already POST to the `ai_agent` service over `urllib` with a 120s blocking timeout — no streaming, no live tool-call visibility, and the agent cannot *act on* the dashboard (change a date range, add a figure, switch tabs, switch games). Every new agent capability means threading more state through the callback graph.
- **The callback graph is the bottleneck.** State lives in ~a dozen `dcc.Store`s wired through pattern-matching callbacks with `allow_duplicate=True`. Adding behavior means reasoning about the whole graph; there is no type safety on callback signatures (they rely on argument order).
- **The UI and the data/agent logic are fused.** Layout, styling, data fetching, and figure construction all happen in the same Python methods, so the UI can't be iterated independently or tested in isolation.
- **Reports aren't reproducible.** A report is assembled by hand (click "add to report" per figure, type descriptions). There is no machine-readable definition of *what a report is*, so "rebuild last month's report with this month's data" is fully manual.

We want to **rebuild the frontend so the agent is a first-class actor**: it reads dashboard state, drives the UI through a documented, typed action API, streams its reasoning and tool calls live, asks clarifying questions, and regenerates reports from a saved spec. That requires a clean client/server split, a real frontend framework, and a declarative report model.

### Target use cases (what the agentic dashboard must do)

These drive the architecture. Each is mapped to the mechanism that delivers it.

1. **Schema-driven report regeneration.** A prompt like *"update the report with new data from 2026-05-01 to 2026-06-01"* takes an existing **Report Spec** (a serializable definition — "the data that can recover the report"), shifts its period, re-fetches data, re-renders the figures, regenerates the per-figure descriptions and the summary, and presents them in the Report tab ready to **export to a Confluence doc**.
   → Mechanism: **Report Spec** (§6) + `/api/report/*` (§5) + the `update_report` **skill** (§6).
2. **Natural-language data lookup that drives the UI.** *"Show me the number of active users in fishhunter / ss01 / ss02"* → the agent picks the metric, **switches the game config file**, **chooses the correct tab**, sets the date range/breakdown, and renders the chart — by emitting structured **actions** the frontend executes, not by screen-scraping.
   → Mechanism: **DashboardAction** contract (§4) + `/api/agent/chat` streaming (§5) + the `show_metric` skill.
3. **Clarifying follow-up questions.** When a request is ambiguous (*"show active users"* — *"do you want it by date or by group?"*), the agent asks a follow-up, the UI shows quick-reply chips, and the answer resolves a **pending intent** held in the store.
   → Mechanism: `clarify` stream event + pending-intent state (§4, §7).
4. **Per-chart contextual actions.** "Explain this", "find anomalies", "compare to last week" on any chart, given that chart's data context.
   → Mechanism: `/api/agent/explain` (§5).

### Goals

1. **Agent-native UX** — streaming chat with live tool-call rendering; the agent reads and *drives* dashboard state and regenerates reports.
2. **Clean client/server boundary** — the dashboard becomes a documented JSON/REST + SSE API. The same API serves the UI **and** the agent.
3. **Declarative, reproducible reports** — a Report Spec is the source of truth a report renders from and the agent edits.
4. **Maintainable, testable frontend** — typed components, isolated rendering, a real state layer.
5. **Keep the Python data + agent stack intact** — polars/S3/Redshift data access and LangGraph agents stay in Python.
6. **No regression in deployment story** — still ECS Fargate behind an ALB; no new server runtime.

### Non-goals

- Rewriting `bituslabs_ds` (ETL, S3, polars) or the LangGraph agent core logic.
- Introducing a Node server runtime (see §3 — the backend stays FastAPI).
- Changing the data pipeline / parquet cache format.
- Adding end-user authentication (still assumed behind SSO/VPN at the ALB).

---

## 2. Why React + Vite + TypeScript + ECharts over the current Dash/Plotly frontend

The wins are not cosmetic — they are what make deep agentic integration tractable.

| Area | Today (Dash/Plotly in Python) | New stack (React + Vite + TS + ECharts) | Why it matters for an agent-native app |
|---|---|---|---|
| **Type safety** | None — callbacks rely on positional `Input`/`Output`/`State`; a renamed store ID fails silently at runtime | End-to-end TS; the API client is generated from FastAPI's OpenAPI schema, so the contract is compiler-checked | The agent's **action contract** and **Report Spec** are type-checked on both ends. A malformed agent action fails at build, not in production. Discriminated unions give exhaustiveness checks on every action handler. |
| **State management** | ~12 `dcc.Store`s through 34 callbacks with `allow_duplicate=True`; data flow is implicit | Centralized store (Zustand) — one serializable source of truth | "Agent drives the dashboard" only works if state is centralized and serializable: the agent reads a snapshot, emits actions, the store applies them. A callback graph can't offer that. |
| **Streaming / real-time** | `urllib` request/response, 120s blocking, no partial output | Native `fetch`/`EventSource`; React re-renders as tokens, tool events, and actions arrive | Agent UX *is* streaming — tokens, tool calls, clarifying questions. The current model shows none of it. |
| **Tool-call & generative UI** | Agent output is a finished string in a div | Mature ecosystem (Vercel AI SDK, `assistant-ui`) renders tool calls, citations, quick-reply chips, agent-generated widgets | Clarifying questions become chips; agent results become interactive ("apply this filter?"), not just text. |
| **Charts** | Plotly figures built **server-side**, serialized, rendered by `dcc.Graph` — backend locked to Plotly forever | **ECharts** rendered client-side from **data series** the backend returns; `dataset`+`encode` separates data from visual encoding | Decoupling the backend from the renderer is what lets the **agent treat a chart as data** (a spec it can generate/mutate). ECharts is performant on large series, actively maintained, and declarative. |
| **Component model** | HTML trees built in Python; no isolation/preview | JSX components, HMR, Storybook-able, unit-testable | Iterating on agent surfaces (chat, inline actions, report editor) is fast and safe. |
| **Dev loop** | Restart Python, full reload | Vite HMR — sub-second updates preserving state | Tightens the loop for UX-heavy agent work. |
| **Styling** | 700+ lines of inline CSS dicts | Tailwind / CSS modules + a theme; dark mode trivial | Polished, consistent UI at low friction. |
| **Performance** | Every interaction round-trips to Python + server-side re-render | Client-side rendering; only data fetches hit the server | Snappier UI; FastAPI does data + agents, not view diffing. |
| **Testing** | Callback testing is awkward; mostly manual | Vitest + React Testing Library; Playwright E2E; backend is a plain API | Agent behaviors (action dispatch, streaming, spec regeneration) become testable units. |
| **Ecosystem / hiring** | Dash-specific, small pool | Mainstream React/TS + ECharts; large ecosystem | Easier to staff and to pull in best-in-class agent-UI components. |

**TypeScript over plain JavaScript — decided.** TS compiles to JS with zero runtime cost, and its value lands exactly where this redesign's risk lives: the `DashboardAction` union and the Report Spec are type-checked end to end (rename a Pydantic field → the frontend fails to compile instead of sending `undefined`); discriminated unions give exhaustiveness checking so a new action type flags every handler that ignores it; and a ~9–13 week tab-by-tab rewrite is far safer with compiler-guided refactors. The costs (a compile step we already have via Vite, a modest learning curve) are dwarfed by the safety for a long-lived, agent-driven, multi-surface app.

**Honest trade-offs of the switch:**
- **Two languages, two build systems.** Mitigated by single-sourcing the contract: OpenAPI → generated TS client; the action + spec schemas mirrored in Pydantic and TS.
- **No SSR** with a Vite SPA — a non-issue for an internal authenticated tool (no SEO, no cold-first-paint concern), which is why we don't need Node/Next.js (§3).
- **ECharts parity cost.** Re-implementing chart configs (subplots/dual-axis/range tools) is real work; accepted as the no-compromise choice. Decoupling the backend to data-series means we can tune or swap chart types later without backend changes.
- **Upfront rewrite cost.** Phased (§8) so the live Dash app keeps serving until each tab reaches parity.

---

## 3. Backend: FastAPI, not Node (decision recap)

We keep the backend in **Python/FastAPI** and do **not** introduce a Node server:

- The data layer (`bituslabs_ds`: polars, S3 parquet, Redshift/Athena) and the agent layer (`ai_agent`: LangChain/LangGraph, tools, Slack bot) are Python and tightly coupled. A Node backend would mean rewriting all of it or running Python as a sidecar anyway.
- The agent-UI niceties people reach Node for (Vercel AI SDK et al.) are **client** libraries — they work against a FastAPI SSE endpoint with no Node server. LangGraph's `astream_events` streams tokens + tool events out of FastAPI cleanly.
- Deployment stays identical to today's uvicorn-on-Fargate; no new runtime, no third service.

Node appears **only as a build-time tool** (`vite build`); it never runs in production.

---

## 4. Target architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (React + Vite + TypeScript SPA, charts = ECharts)         │
│                                                                    │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│   │ Dashboard   │  │ Agent panel  │  │ Report editor          │   │
│   │ tabs/charts │  │ (streaming + │  │ (renders a Report Spec;│   │
│   │ (ECharts)   │  │  clarify     │  │  export to Confluence) │   │
│   │             │  │  chips)      │  │                        │   │
│   └─────┬───────┘  └──────┬───────┘  └─────────┬──────────────┘   │
│         └─────────┬───────┴────────────────────┘                  │
│             Zustand store (canonical UI state + pending intent +   │
│                            current Report Spec)                    │
│             action dispatcher ◄── validates + applies DashboardAction
│             typed API client (generated from OpenAPI)              │
└───────────────────────────┬────────────────────────────────────-─┘
                  REST (JSON) + SSE (token / tool / action / clarify)
                            │
                   ALB (one origin, path-routed)
              ┌─────────────┴──────────────┐
   /api/agent/*│                            │ everything else
┌──────────────┴────────────┐   ┌───────────┴───────────────────────────┐
│  ai_agent (heavy, ECS)     │   │  dashboard_api (light, ECS)            │
│  uvicorn + LangGraph       │   │  uvicorn, LLM-free, Plotly/kaleido-free│
│                            │   │                                        │
│  /api/agent/chat   (SSE)   │   │  /api/data/*    series+metadata (DATA) │
│  /api/agent/explain(SSE)   │   │  /api/report/*  spec CRUD + regenerate │
│  /api/report/desc & summary│◄──┤                 + export (client PNG)  │
│    (LLM one-shot, called   │   │  /api/skills    list skills (skills.md)│
│     by dashboard_api)      │   │  serves static SPA (frontend/dist)     │
│  Slack bot                 │   │                                        │
│  ├ agent core (LangGraph)  │   │  ├ bituslabs_ds (ETL/S3/polars)        │
│  └ bituslabs_ds (direct)   │   │  └ ReportSpec + skills.md (NEW)        │
└──────────────┬─────────────┘   └───────────┬────────────────────────────┘
               └───────────────┬─────────────┘
                    S3: parquet cache + versioned Report Specs
```

Both services import the shared `bituslabs_ds` library directly (no HTTP hop for data). The single library-vs-service distinction: `bituslabs_ds` and the report-generation helpers are *libraries* both services import; the *service* boundary exists only for deploy/scaling/dependency isolation. The lone cross-service call is `dashboard_api` → `ai_agent` for LLM-bound description/summary generation during report regeneration.

**The "agent drives the dashboard" loop:**
1. The store holds canonical state: active game config, active tab, selected metrics, date ranges, figures, and the current Report Spec.
2. A user message goes to `/api/agent/chat` with a **serialized snapshot** of relevant state.
3. The agent streams back tokens, tool events, and **structured actions** — e.g. `{type:"load_config", config:"fishhunter"}`, `{type:"navigate_tab", tab:"stats-by-date"}`, `{type:"set_metrics", group:"g1", metrics:["active_users"]}`.
4. The frontend **action dispatcher** validates each action against the typed schema and applies it to the store — re-rendering the dashboard. The agent never screen-scrapes; the action contract is the same one the UI uses.

**Clarifying questions:** when the agent needs disambiguation it emits a `clarify` event (`{question, options:["By date","By group"]}`) and ends the turn. The UI renders quick-reply chips and stores a **pending intent** (the parsed-but-incomplete request). The user's reply resolves the intent and the agent continues — fully supported over SSE (no WebSocket needed for v1).

---

## 5. Backend API surface (shared by UI and agent)

Single-sourced: FastAPI Pydantic models → OpenAPI → generated TS client. Both the React app and the agent's action/spec schemas derive from it.

| Endpoint | Method | Purpose | Streaming |
|---|---|---|---|
| `/api/data/configs` | GET | List game configs (ss01, ss02, fishhunter, …) | — |
| `/api/data/config/{name}` | GET | Resolved config: tabs, metrics, groups, date bounds | — |
| `/api/data/series` | POST | **Returns data series** (`{x, y, label, …}`) for a config/tab/group/metrics/date-range — the frontend encodes them into ECharts | — |
| `/api/agent/chat` | POST | Streaming agent turn: message + state snapshot + history → tokens, tool events, **actions**, **clarify** | **SSE** |
| `/api/agent/explain` | POST | Per-chart action ("explain"/"anomalies"/"compare") given a chart's data summary | **SSE** |
| `/api/report/spec` | GET/POST/PUT | CRUD a **Report Spec** (saved to S3) | — |
| `/api/report/regenerate` | POST | Given a spec (+ optional period override), re-fetch data and return rendered figures + (re)generated descriptions/summary | optional SSE |
| `/api/report/description` | POST | Generate/regenerate one figure description (existing logic, ported) | optional SSE |
| `/api/report/summary` | POST | Generate the overall summary (existing) | optional SSE |
| `/api/report/export` | POST | Export to Confluence — client posts the ECharts-rendered PNG(s) + text; server embeds (no server-side chart rendering) | — |
| `/api/agent/skills` | GET | The `skills.md` registry, raw markdown. *(Implemented on `ai_agent`, not dashboard_api as originally drawn: the consumer is the agent at prompt-build time, in-process, and `/api/agent/*` already routes there.)* | — |
| `/api/metadata/columns`, `/columns/{name}`, `/groups` | GET | Column/group metadata (existing) | — |
| `/api/health` | GET | Health check | — |

**DashboardAction schema** — a versioned discriminated union, validated on the client before dispatch and mirrored as Pydantic models so the agent emits validated actions:

```ts
type DashboardAction =
  | { type: "load_config";        config: ConfigId }            // switch game (fishhunter/ss01/…)
  | { type: "navigate_tab";       tab: TabId }
  | { type: "set_date_range";     group: GroupId; from: ISODate; to: ISODate }
  | { type: "set_metrics";        group: GroupId; metrics: string[] }
  | { type: "set_grouping";       group: GroupId; dimension: GroupDimension }
  | { type: "clip_data";          panel: PanelId; enable: boolean; min?: number; max?: number }
  | { type: "add_figure_to_report"; source: PanelId }
  | { type: "load_report_spec";   specId: string }
  | { type: "apply_report_spec";  spec: ReportSpec }            // agent-edited spec → render
  | { type: "open_chat" | "minimize_chat" }
```

**SSE event types:** `token`, `tool_start`, `tool_end`, `action`, `clarify`, `done`, `error`.

---

## 6. Report Spec & agent skills

### Report Spec — "the data that can recover the report"

A Report Spec is a declarative, serializable (JSON/YAML) definition stored in S3 and rendered by the Report tab. It is what use case #1 edits. A figure's `description`/`summary` can be either fixed prose **or** a generation `prompt` the agent fills in on regenerate.

```yaml
report:
  id: monthly-active-overview
  title: Monthly Active Users Overview
  language: en            # en | zh-Hans | zh-Hant
  period: { from: 2026-05-01, to: 2026-06-01 }   # ← the agent shifts this
  figures:
    - id: au-by-date
      config: ss01                # game config file
      tab: stats-by-date
      group: g1
      metrics: [active_users]
      date_range: inherit         # inherit report.period, or pin a window
      clip: { enable: false }
      description: { prompt: "Summarize the active-user trend and call out week-over-week changes." }
    - id: au-by-group
      config: ss01
      tab: stats-by-group
      group: g1
      metrics: [active_users]
      dimension: user_group
      description: { text: "Active users split by acquisition cohort." }   # fixed prose
  references: [ "https://confluence/.../baseline-defs" ]
  summary: { prompt: "Write a 3-bullet executive summary across the figures above." }
  export: { confluence_url: "https://confluence/.../monthly-report" }
```

**Regeneration flow** (prompt: *"update the report with new data from 2026-05-01 to 2026-06-01"*):
1. Load the current spec (from the store, or `/api/report/spec` by id).
2. The agent edits `report.period` (and any figures whose `date_range: inherit`).
3. `/api/report/regenerate` re-fetches each figure's data series, the frontend re-renders the ECharts, and any `description.prompt` / `summary.prompt` is (re)generated via the existing description/summary logic; fixed-prose entries are preserved.
4. The Report tab shows figures + descriptions; the user reviews/edits inline; **Export to Confluence** ships it.

Because the spec fully determines the report, reports are reproducible, diffable, version-controllable in S3, and **agent-editable**.

### Agent skills (`skills.md`)

A markdown registry of reusable agent procedures — names, descriptions, parameters, and the actions/endpoints each composes. *(As built: `src/ai_agent/skills.md`, merged into the agent's system prompt in-process and served raw at `GET /api/agent/skills`; mtime-cached, so edits apply on the next chat turn.)* Skills:

- **`update_report(period)`** — ✅ load a Report Spec, shift its period, regenerate (use case #1). Built on the report action tools: `set_report_period`, `regenerate_report`, `load/save_report_spec`, `add/patch/remove_report_figure` (targeted patches, not full-spec replacement — the state snapshot truncates prose to previews, so a round-tripped spec would lose text).
- **`show_metric(metric, configs[], breakdown?)`** — ✅ for each game config: `load_config` → choose tab (`stats-by-date` or `stats-by-group` based on `breakdown`, asking a `clarify` question if unspecified) → `set_metrics` → render (use case #2 + #3).
- **`edit_report_figure`** — ✅ patch one figure's title/description/inherit_period/source from chat.
- **`compare_periods(metric, periodA, periodB)`** — fetch both, render a comparison. *(Not yet built — needs a comparison surface.)*
- **`explain_chart(panel, mode)`** — back the per-chart contextual actions (use case #4). *(Not yet built.)*

Keeping skills in markdown (rather than hard-coded) means non-engineers can read/extend the agent's repertoire, and the AI agent can `grep` them the same way it greps the codebase today.

---

## 7. Repository layout

```
frontend/                         # NEW — React + Vite + TS SPA
├── package.json / vite.config.ts / tsconfig.json
├── src/
│   ├── main.tsx
│   ├── app/                      # routing, layout shell, theme
│   ├── store/                    # Zustand store, action dispatcher, pending-intent
│   ├── api/                      # generated OpenAPI client + SSE helpers
│   ├── charts/                   # ECharts wrappers (option builders from data series)
│   ├── features/
│   │   ├── stats-by-date/  stats-by-group/  stats-deepdive/
│   │   ├── weekly-report/  stats-by-bet/
│   │   ├── report/               # Report-Spec-driven editor + Confluence export
│   │   └── agent/                # chat panel, tool-call UI, clarify chips, inline actions
│   └── components/  theme/
└── tests/                        # Vitest + Playwright

src/dashboard_api/                # NEW — light service (LLM-free), serves the SPA
├── app.py                        # FastAPI app mounting the routers below
├── routers/  data.py  report.py  skills.py  metadata.py
├── schemas/
│   ├── actions.py                # DashboardAction union (mirrors frontend)
│   └── report_spec.py            # ReportSpec (mirrors frontend)
├── skills.md                     # agent skill registry
└── (imports bituslabs_ds, confluence_client; calls ai_agent for LLM text)

src/bituslabs_ds/                 # UNCHANGED — data layer (library, imported by both)
src/ai_agent/                     # heavy service: streaming agent + Slack
│   └── routers/ agent.py         # /api/agent/chat, /api/agent/explain (SSE)
│       + report description/summary generation (LLM one-shot, reused)
infra/dashboard/  infra/ai_agent/ # updated: dashboard builds vite + serves static
```

Static serving (Phase 1): `vite build` → `frontend/dist`, served by `dashboard_api` via FastAPI `StaticFiles` at `/`. Revisit S3+CloudFront if bundle/CDN concerns arise.

---

## 8. Migration plan (incremental cutover; the live app never goes dark)

The legacy Dash app keeps serving production until each tab reaches parity.

**Phase 0 — Foundations (1–2 wks)** — *complete*
- ✅ `frontend/` scaffolded (Vite + React + TS + Tailwind, Zustand, ECharts) with a HelloDashboard smoke page driving the full stack. Vitest wired (`npm test`) with a store unit test.
- ✅ `src/dashboard_api/` stood up with `/api/health`, `/api/data/configs`, `/api/data/config/{id}`, `/api/data/series`; `dashboard_api` poetry group + multi-stage Docker image (vite build → uvicorn).
- ✅ TS client generated from OpenAPI: `python -m dashboard_api.openapi` dumps the schema → `openapi-typescript` → `frontend/src/api/schema.d.ts`; typed `openapi-fetch` client in `client.ts`. One command: `make openapi`.
- ✅ CI: `.github/workflows/ci.yml` — backend (`make lint-new` + dashboard_api smoke tests), frontend (typecheck + vitest + build), and an OpenAPI-drift gate (`make openapi` must produce no diff). Backend lint is scoped to the new surfaces; the legacy tree has pre-existing flake8 debt and joins the gate per-module as the rewrite proceeds (a `.flake8` config at 120 chars was added to match black).
- ✅ ECS deploy wiring: `infra/dashboard_api/deploy_ecs.py` + `build.sh` (push to ECR). The service gets its **own ALB origin** (SPA at `/`, data API at `/api/data/*`) rather than a `/v2` path on the live dashboard ALB — see §12. Shared ALB/ECS plumbing was extracted into `infra/shared/ecs_helpers.py` and all four deploy scripts (dashboard, ai_agent, rag_service, dashboard_api) now compose it.
- Note: `/api/data/series` uses a placeholder per-date aggregation; user-row enrichment parity is Phase 1. Scale-to-zero for `dashboard_api` is deferred until the `UserRequestCount` middleware lands (the service runs one always-on task for now).

**Phase 1 — First read-only tab (1–2 wks)** — *complete*
- ✅ **Stats by Date** end-to-end: `/api/data/series` → ECharts, full parity validated side by side, plus saved views, sticky/responsive layout, per-panel incremental loading with request cancellation.

**Phase 2 — Remaining stats tabs (2–3 wks)** — *complete (Stats by Group + Deep Dive)*
- ✅ Stats by Group (box/bar, 3-range comparison) and Stats Deepdive (histogram/heatmap/scatter); shared chart/control components extracted. Stats by Bet and Weekly Report deferred to post-cutover.

**Phase 3 — Agent-native chat + actions (2 wks)** — *complete*
- ✅ `/api/agent/chat` SSE; docked resizable chat panel with token streaming + tool-call rendering (reusing the `ai_agent` LangGraph core).
- ✅ Action dispatcher with `load_config`, `navigate_tab`, `set_metrics`, `set_date_range`, `set_granularity` + **clarify** chips → use cases #2 and #3.

**Phase 4 — Report Spec + generative authoring (2–3 wks)** — *complete (4a manual UI + 4b agent skills)*
- ✅ `ReportSpec` (Pydantic + TS); `/api/report/spec/{name}` CRUD to S3 with a linked-view field (one view ↔ many reports); regeneration runs client-side from figure recipes (no `/api/report/regenerate` endpoint needed — the spec is the contract).
- ✅ Report tab renders from the spec (directly-editable JSON with Apply & render + staleness-aware prose refresh); description/summary generation proxied dashboard_api → ai_agent; Confluence export from client-rendered PNGs.
- ✅ `skills.md` + `GET /api/agent/skills` (on ai_agent — see §5/§6) and the `update_report`/`show_metric`/`edit_report_figure` skills via report action tools → use cases #1 and #4.

**Phase 5 — Cutover (1 wk)** — *complete*
- ✅ Flipped the ALB default to the React `dashboard_api` (atomic listener swap on the existing `game-stats-dashboard-alb`; `/api/agent/*` routed to ai_agent). No `/legacy` route — Dash assets aren't prefix-aware, so rollback was the listener flip during the bake instead.
- ✅ After a clean one-week `Dashboard/UserRequestCount` parity bake (steady traffic, zero errors), decommissioned Dash: deleted `src/dashboards/` (incl. `game_stats_monitor.py`) + `infra/dashboard/`, the `dashboards` package + `dashboard` poetry group, and the AWS resources (ECS service, task-def, target group, task SG, log group, ECR repo). See CHANGELOG → Removed.

Rough estimate: ~10–14 weeks; each phase independently shippable.

---

## 9. Deployment changes

- **Build**: `dashboard_api` uses a multi-stage Docker build — stage 1 `vite build` (Node *build-time only*), stage 2 the Python/uvicorn runtime serving the API + `frontend/dist`. Node never runs in prod. `ai_agent` keeps its existing build.
- **ECS**: two Fargate services — `dashboard_api` (light, replaces the old Dash service) and `ai_agent` (existing); reuse `infra/shared/ecs_helpers.py`. Net service count unchanged vs. today.
- **ALB routing**: `/api/agent/*` → `ai_agent`, everything else → `dashboard_api`, so the SPA sees one origin. *(As built: dashboard_api never got its own interim ALB — `infra/dashboard_api/deploy_ecs.py` deploys straight onto `game-stats-dashboard-alb`, adds a priority-10 `/api/agent/*` rule to a second agent target group [a TG belongs to one ALB, so the agent service carries two TGs via multi-TG `update_service` — recreating that service from scratch silently drops the extra TG], raises the idle timeout to 300s for SSE, then atomically flips the listener default from the Dash TG to `dashboard-api-tg`. No `/legacy` route — Dash assets aren't prefix-aware; rollback is scaling Dash back to 1 and flipping the default action back, one-liners printed by the deploy script. The agent's own `ai-chat-agent-alb` stays: the server-side `/api/report/*` proxy targets it via `AGENT_API_URL`.)*
- **Scale-to-zero**: preserve the `Dashboard/UserRequestCount` CloudWatch emission (move the after-request hook into FastAPI middleware).
- **Secrets/auth**: unchanged — `aws_secrets`, Confluence token, LLM keys via Secrets Manager; ALB behind SSO/VPN. New: S3 read/write for saved Report Specs (extend the task role).

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Two-language maintenance burden | Single-source the contract (OpenAPI → TS client); action + ReportSpec schemas mirrored in Pydantic + TS. |
| ECharts parity cost vs. dense Plotly charts | Backend returns data series (renderer-agnostic), so chart types are tuned/swapped without backend change; port tab-by-tab with side-by-side validation. |
| Agent actions misfire / mutate state unexpectedly | All actions validated against the typed schema before dispatch; actions are pure, inspectable, undoable store mutations; audit log of agent actions. |
| Agent edits a Report Spec incorrectly | Spec is validated (Pydantic) on regenerate; the user always reviews the rendered report before Confluence export; specs are versioned in S3 (revertible). |
| Ambiguous prompts → wrong dashboard state | Mandatory `clarify` step for under-specified requests (breakdown, config, period); pending-intent keeps the turn resumable. |
| Long-lived rewrite branch drifts from `dev` | Ship each phase to `dev` behind `/v2`; never one long branch. |
| SSE behind ALB idle timeouts | Tune ALB idle timeout + keep-alive (already 300s+/310s today); heartbeat events on the stream. |

---

## 11. Open questions

1. **Skills format** — plain `skills.md` parsed by the backend vs. adopting a structured skills framework. Start with markdown + a light parser.
2. **State store** — Zustand (leaning) vs. Redux Toolkit.
3. **Agent-UI library** — adopt `assistant-ui`/Vercel AI SDK components vs. a thin custom chat UI over our SSE format (evaluate in Phase 3).
4. **Auth** — does per-user report state / agent-action auditing warrant in-app auth, or stay ALB/SSO-only?

*Resolved (see §12): service topology (two services behind one ALB origin) and Report Spec storage (S3 with versioning).*

---

## 12. Decision log

- **2026-06-02** — Chose **Architecture A**: React + Vite + TS SPA + consolidated FastAPI backend (no Node server runtime). Keeps the Python data + agent stack intact; modern agent-UI is achievable client-side against FastAPI SSE; deployment unchanged. Rejected: Next.js full-stack (needless Node backend rewrite/sidecar) and hybrid Next.js BFF + FastAPI (highest ops complexity, two backend languages).
- **2026-06-03** — **TypeScript over JavaScript** (compiler-enforced action + ReportSpec contracts, discriminated-union exhaustiveness, safer large rewrite).
- **2026-06-03** — **ECharts** as the chart renderer, and the **backend returns data series, not figure specs** — decoupling the renderer so the agent can treat charts as data and chart types can be tuned/swapped without backend changes. Rejected `react-plotly.js` (unmaintained wrapper, full ~3 MB bundle, Plotly lock-in). Plotly-direct considered as a parity shortcut but dropped per the no-compromise directive.
- **2026-06-03** — Adopted the **Report Spec** primitive ("data that can recover the report") + a markdown **skills** registry to deliver schema-driven report regeneration and NL-driven UI control with clarifying questions.
- **2026-06-03** — **Two services behind one ALB origin**, not one consolidated service. A new lightweight `dashboard_api` (data + report orchestration + static SPA) and the existing `ai_agent` (streaming LangGraph agent + Slack). ALB path-routes `/api/agent/*` → `ai_agent`, everything else → `dashboard_api`, so the SPA sees one origin. Rationale: shared logic lives in libraries (`bituslabs_ds`, report generation) so the agent reaches data via direct imports, not HTTP hops; the only cross-service call is LLM-bound description/summary generation where a hop is negligible. Keeps the data service LLM-free (and Plotly/kaleido-free — see export note) for a small image + independent scaling/timeouts + smaller blast radius. Net service count is unchanged (the old Dash service is replaced by `dashboard_api`).
- **2026-06-03** — **Report Spec storage = S3 with bucket versioning** (`report_specs/{id}.json`), not DynamoDB. Zero new infra (reuses the bucket, task role, and `s3_utils`); S3 object versioning gives free revert history that maps directly to the agent-edit/undo flow; the access pattern (load-by-id + list ~tens) needs no database. Concurrency handled by last-writer-wins, optionally a conditional `If-Match` write. Revisit DynamoDB only for per-user spec libraries with attribute queries at scale.
- **2026-06-03** — **Confluence export sends a client-rendered PNG.** Because charts are now ECharts on the client, the browser exports the canvas (`getDataURL`) and POSTs the image to `/api/report/export`; the server embeds it. This removes server-side Plotly/kaleido entirely, keeping `dashboard_api` light.
- **2026-06-08** — **`dashboard_api` gets its own ALB origin**, not a `/v2` path on the live dashboard ALB (as Phase 0 originally sketched). Each existing service already has a dedicated ALB; a dedicated origin avoids building the SPA with a `/v2` base path and prefixing every API call, keeps blast radius off the production dashboard's ALB, and matches the Phase-5 end state (the SPA is served from `/`). Cost is one extra ALB during migration (~$16/mo), dropped at cutover when this origin becomes the default.
- **2026-06-08** — **Shared ECS deploy plumbing extracted to `infra/shared/ecs_helpers.py`.** The log-group / security-group / ALB / target-group / listener / task-def / service create-or-update steps were duplicated ~verbatim across the three existing `deploy_ecs.py` scripts. They now live as parameterized helpers; all four scripts (incl. the new `dashboard_api`) compose them. Per-service IAM policy, env vars, and autoscaling stay inline. ~750 duplicated lines removed.
- **2026-06-08** — **CI lint is scoped to new code, not the whole repo.** Repo-wide `make lint` surfaced 160 pre-existing flake8 violations (there had never been a flake8 config or CI). Rather than block the redesign on a large, risky legacy cleanup, CI gates `make lint-new` (currently `src/dashboard_api` + its tests) and grows that path list as each module is ported. A `.flake8` (max-line-length 120, black-compatible ignores) was added so flake8 finally agrees with black.
