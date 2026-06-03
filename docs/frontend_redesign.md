# Frontend Redesign: Agent-Native Dashboard (React + Vite + TypeScript)

**Status:** Proposed
**Author:** michael.niu@bituslabs.com
**Date:** 2026-06-02
**Scope:** Replace the monolithic Dash app (`src/dashboards/game_stats_monitor.py`, ~7,200 LOC) with a React + Vite + TypeScript frontend served as static assets, backed by a consolidated FastAPI backend. The goal is a dashboard where LLM/agentic capabilities are first-class, not bolted on.

---

## 1. Motivation

The current dashboard is a single 7,200-line `GameStatsDashboard` class with 34 Dash callbacks, inline CSS dicts, and HTML built in Python. It works, but it has hit structural limits:

- **Agentic features are bolted on.** The chat panel and Report-tab generation already POST to the `ai_agent` FastAPI service over `urllib` with a 120s blocking timeout — there is no streaming, no live tool-call visibility, and the agent cannot *act on* the dashboard (change a date range, add a figure, switch tabs). Every new agent capability means threading more state through the callback graph.
- **The callback graph is the bottleneck.** State lives in ~a dozen `dcc.Store`s wired through pattern-matching callbacks with `allow_duplicate=True`. Adding behavior means reasoning about the whole graph; there is no type safety on callback signatures (they rely on argument order).
- **The UI and the data/agent logic are fused.** Layout, styling, data fetching, and figure construction all happen in the same Python methods, so the UI can't be iterated independently and can't be tested in isolation.

We want to **rebuild the frontend so the agent is a first-class actor**: it can read dashboard state, drive the UI through a documented action API, stream its reasoning and tool calls live, and author content inline. That requires a clean client/server split and a real frontend framework.

### Goals

1. **Agent-native UX** — streaming chat with live tool-call rendering; the agent can read and *drive* dashboard state; per-chart contextual agent actions ("explain this", "find anomalies", "compare to last week").
2. **Clean client/server boundary** — the dashboard becomes a documented JSON/REST + SSE API instead of an in-process callback graph. The same API serves both the UI and the agent.
3. **Maintainable, testable frontend** — typed components, isolated rendering, previewable UI, a real state layer.
4. **Keep the Python data + agent stack intact** — polars/S3/Redshift data access and LangGraph agents stay in Python; we do not rewrite them.
5. **No regression in deployment story** — still ECS Fargate behind an ALB; no new server runtime.

### Non-goals

- Rewriting `bituslabs_ds` (ETL, S3, polars) or the LangGraph agent logic.
- Introducing a Node server runtime (see §3 — the backend stays FastAPI).
- Changing the data pipeline / parquet cache format.
- Adding end-user authentication (still assumed behind SSO/VPN at the ALB).

---

## 2. Why React + Vite + TypeScript over the current Dash/Plotly frontend

This is the question that prompted the redesign, so it's worth being explicit. The wins are not cosmetic — they are what make deep agentic integration tractable.

| Area | Today (Dash/Plotly in Python) | React + Vite + TypeScript | Why it matters for an agent-native app |
|---|---|---|---|
| **Type safety** | None — callbacks rely on positional `Input`/`Output`/`State` args; a renamed store ID fails silently at runtime | End-to-end types; with a generated client from FastAPI's OpenAPI schema, the API contract is compiler-checked | The agent's **action API** (the contract by which the agent drives the UI) is type-checked on both ends. Mismatches fail at build, not in production. |
| **State management** | ~12 `dcc.Store`s wired through 34 callbacks with `allow_duplicate=True`; data flow is implicit and hard to trace | Explicit store (Zustand/Redux) with a single source of truth; state transitions are inspectable and time-travel-debuggable | "Agent drives the dashboard" means the agent emits actions that mutate state. That only works if state is **centralized and serializable** — which is exactly what a React store gives you and a callback graph does not. |
| **Streaming / real-time** | `urllib` request/response with a 120s blocking timeout; no partial output | Native `fetch`/`EventSource`/WebSocket; React re-renders incrementally as tokens arrive | Agent UX *is* streaming: tokens, tool-call start/finish, intermediate steps. The current model can't show any of it. |
| **Tool-call & "generative UI"** | Not possible — the agent's output is a finished string dropped into a div | Mature ecosystem (Vercel AI SDK, `assistant-ui`, CopilotKit) renders tool calls, citations, and agent-generated UI components | The agent can return *structured* results the UI renders as interactive widgets (a chart, a table, a "apply this filter?" button), not just text. |
| **Component model** | HTML trees built in Python (`html.Div([...])`); no isolation, no preview | JSX components, hot-module reload, Storybook-able, unit-testable in isolation | Iterating on agent UI surfaces (chat, inline actions, report editor) is fast and safe; components are reused, not copy-pasted across tabs. |
| **Dev loop** | Restart the Python process; full page reload; slow | Vite HMR — sub-second updates preserving app state | Tightens the build-measure-learn loop for UX-heavy agent features. |
| **Styling** | 700+ lines of inline CSS dicts scattered through the class | Real CSS / Tailwind / CSS modules with a theme; consistent design system | Lower friction to a polished, modern UI; theming (incl. dark mode) becomes trivial. |
| **Charts** | Plotly figures built server-side, serialized, rendered by `dcc.Graph` | `react-plotly.js` renders the **same** figure JSON client-side | Charting ports ~1:1 — we keep Plotly, so no chart logic is lost — but the chart now lives next to its agent actions and client interactions. |
| **Performance / scale** | Every interaction round-trips to Python and re-renders server-side | Client-side rendering; only data fetches hit the server | Snappier UI; the FastAPI backend does data + agents, not view diffing. |
| **Separation of concerns** | UI + data + agent fused in one class | UI in React; data + agents in FastAPI behind a documented API | The same API serves the UI **and** the agent. One contract, two consumers — the core enabler of the whole redesign. |
| **Testing** | Callback testing is awkward; mostly manual | Vitest + React Testing Library for UI; Playwright for E2E; backend tested as plain API | Agent behaviors (action dispatch, streaming) become testable units instead of manual clicks. |
| **Ecosystem & hiring** | Dash-specific knowledge; small talent pool | Mainstream React/TS; huge ecosystem, libraries, and hiring pool | Easier to staff and to pull in best-in-class agent-UI components. |

**Honest trade-offs / costs of the switch:**

- **Two languages, two build systems.** The team now maintains a TS frontend and a Python backend. (Mitigated by generating the TS API client from FastAPI's OpenAPI schema, so the contract is single-sourced.)
- **No SSR out of the box** with a Vite SPA. For an internal authenticated tool this is a non-issue (no SEO, no cold first-paint concern), which is precisely why we don't need Next.js/Node — see §3.
- **Upfront rewrite cost.** This is real. §8 phases it so the live Dash app keeps running until the React app reaches parity tab-by-tab.
- **Plotly client bundle size.** `react-plotly.js` ships a large bundle; we'll use a partial/custom Plotly build or lazy-load the chart module to keep initial load reasonable.

---

## 3. Backend: FastAPI, not Node (decision recap)

We keep the backend in **Python/FastAPI** and do **not** introduce a Node server. Rationale, in brief (full discussion was in the design conversation):

- The entire data layer (`bituslabs_ds`: polars, S3 parquet, Redshift/Athena) and the entire agent layer (`ai_agent`: LangChain/LangGraph, tools, Slack bot) are Python and tightly coupled. Moving the backend to Node would mean rewriting all of it or running Python as a sidecar anyway.
- The modern *agent-UI* niceties people reach Node for (Vercel AI SDK et al.) are **client** libraries — they work against a FastAPI SSE endpoint without any Node server. LangGraph's `astream_events` streams tokens + tool events out of FastAPI cleanly.
- Deployment stays identical to today's uvicorn-on-Fargate story; no new runtime, no third service.

We **consolidate** the dashboard's data endpoints and the existing `ai_agent` endpoints behind one FastAPI surface (or two services sharing one OpenAPI schema — see §5/§7). The React SPA is served as static assets behind the same ALB.

---

## 4. Target architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Browser (React + Vite + TypeScript SPA)                       │
│                                                                │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│   │ Dashboard   │  │ Agent panel  │  │ Report editor      │   │
│   │ tabs/charts │  │ (streaming)  │  │ (generative)       │   │
│   │ react-plotly│  │ tool-call UI │  │                    │   │
│   └─────┬───────┘  └──────┬───────┘  └─────────┬──────────┘   │
│         │                 │                    │              │
│         └─────────┬───────┴────────────────────┘              │
│             Zustand store  ◄── agent actions mutate state     │
│             typed API client (generated from OpenAPI)         │
└───────────────────────────┬──────────────────────────────────┘
                  REST (JSON) + SSE (streaming)
                            │
┌───────────────────────────┴──────────────────────────────────┐
│  FastAPI backend (Python, uvicorn on ECS Fargate)             │
│                                                                │
│   /api/data/*      data + figure endpoints (polars/S3)         │
│   /api/agent/chat  streaming agent (SSE) + tool events         │
│   /api/agent/act   structured dashboard-control actions        │
│   /api/report/*    description / summary generation            │
│   /api/metadata/*  column/group metadata (existing)            │
│                                                                │
│   ├── bituslabs_ds  (ETL, S3, polars)  ── unchanged            │
│   └── agent core    (LangGraph, tools) ── from ai_agent        │
└───────────────────────────┬──────────────────────────────────┘
                            │
                    S3 parquet cache  ◄── existing ETL jobs
```

Key shift: **dashboard state is client-side and centralized**, and the agent reads/writes it through the same API the UI uses. The "agent drives the dashboard" loop is:

1. UI keeps canonical state in the Zustand store (active tab, selected metrics, date ranges, figures on the report).
2. When the user asks the agent something, the UI sends the message **plus a serialized snapshot of relevant state** to `/api/agent/chat`.
3. The agent streams back tokens and **structured actions** (e.g. `{type: "set_date_range", group: "g1", from, to}` or `{type: "add_figure_to_report", panel: "viz-p1"}`).
4. The frontend has an **action dispatcher** that validates each action against a typed schema and applies it to the store — re-rendering the dashboard. The agent thus manipulates the real UI without screen-scraping.

This is the same pattern as tool-calling, but the "tools" are dashboard mutations executed client-side.

---

## 5. Backend API surface (shared by UI and agent)

The contract is single-sourced: FastAPI Pydantic models → OpenAPI schema → generated TypeScript client (`openapi-typescript` / `orval`). Both the React app and the agent's action schema derive from it.

| Endpoint | Method | Purpose | Streaming |
|---|---|---|---|
| `/api/data/configs` | GET | List available dashboard configs (per game) | — |
| `/api/data/config/{name}` | GET | Resolved config: tabs, metrics, groups, date bounds | — |
| `/api/data/series` | POST | Fetch metric series for a tab/group/date-range → figure-ready JSON | — |
| `/api/data/figure` | POST | Build a Plotly figure spec server-side (optional; or build client-side from series) | — |
| `/api/agent/chat` | POST | Streaming agent turn: message + state snapshot + history → tokens, tool events, **actions** | **SSE** |
| `/api/agent/act` | POST | Validate/execute a single structured action (for agent-initiated or replayed actions) | — |
| `/api/agent/explain` | POST | Per-chart contextual action ("explain"/"anomalies"/"compare") given a data summary | **SSE** |
| `/api/report/description` | POST | Generate/regenerate a figure description (existing logic, ported) | optional SSE |
| `/api/report/summary` | POST | Generate the overall report summary (existing) | optional SSE |
| `/api/report/export` | POST | Export report to Confluence (existing exporter) | — |
| `/api/metadata/columns`, `/columns/{name}`, `/groups` | GET | Column/group metadata (existing) | — |
| `/api/health` | GET | Health check | — |

**Action schema (the agent↔UI contract).** A discriminated union, versioned, validated on the client before dispatch:

```ts
type DashboardAction =
  | { type: "navigate_tab";      tab: TabId }
  | { type: "set_date_range";    group: GroupId; from: ISODate; to: ISODate }
  | { type: "set_metrics";       group: GroupId; metrics: string[] }
  | { type: "set_grouping";      group: GroupId; dimension: GroupDimension }
  | { type: "add_figure_to_report"; source: PanelId }
  | { type: "clip_data";         panel: PanelId; enable: boolean; min?: number; max?: number }
  | { type: "open_chat" | "minimize_chat" }
  // ... extended as features land
```

The same union is mirrored as Pydantic models on the backend so the LangGraph agent emits validated actions via structured output / tool-calling.

---

## 6. Streaming: SSE over WebSocket (for now)

- **SSE** (`text/event-stream` via FastAPI `StreamingResponse`) is sufficient: agent turns are server→client streams (tokens, tool-call deltas, actions). It's simpler to deploy behind an ALB, survives proxies, and has first-class browser support (`EventSource` / `fetch` + `ReadableStream`).
- LangGraph's `astream_events` yields the token + tool-start/tool-end events we surface in the tool-call UI.
- **WebSocket** is reserved for a later phase if we need true bidirectional/interactive sessions (e.g. the agent asking a mid-turn clarifying question the user answers without a new request). Not needed for v1.

Event types over the SSE channel: `token`, `tool_start`, `tool_end`, `action`, `done`, `error`.

---

## 7. Repository layout

A new top-level `frontend/` for the SPA; the backend consolidates under a new `src/dashboard_api/` (or extends `ai_agent`), reusing existing modules.

```
frontend/                         # NEW — React + Vite + TS SPA
├── package.json
├── vite.config.ts
├── tsconfig.json
├── src/
│   ├── main.tsx
│   ├── app/                      # routing, layout shell
│   ├── store/                    # Zustand store + action dispatcher
│   ├── api/                      # generated OpenAPI client + SSE helpers
│   ├── features/
│   │   ├── stats-by-date/
│   │   ├── stats-by-group/
│   │   ├── stats-deepdive/
│   │   ├── weekly-report/
│   │   ├── stats-by-bet/
│   │   ├── report/               # report editor (generative)
│   │   └── agent/                # chat panel, tool-call UI, inline actions
│   ├── components/               # shared UI (charts via react-plotly.js)
│   └── theme/
└── tests/                        # Vitest + Playwright

src/dashboard_api/                # NEW or merged into ai_agent
├── app.py                        # FastAPI app: mounts routers below
├── routers/
│   ├── data.py                   # /api/data/*  (wraps bituslabs_ds, polars, S3)
│   ├── agent.py                  # /api/agent/* (LangGraph streaming + actions)
│   ├── report.py                 # /api/report/* (ported from report_agent)
│   └── metadata.py               # /api/metadata/* (existing)
├── schemas/                      # Pydantic models → OpenAPI → TS client
│   └── actions.py                # DashboardAction union (mirrors frontend)
└── ...

src/bituslabs_ds/                 # UNCHANGED — data layer
src/ai_agent/                     # agent core reused (chat_agent, tools, metadata)
infra/dashboard/                  # updated: build SPA + serve static + FastAPI
```

Static serving: build the SPA (`vite build` → `frontend/dist`) and serve it either (a) from FastAPI via `StaticFiles` mounted at `/`, or (b) from S3+CloudFront / an Nginx sidecar. Phase 1 uses (a) for simplicity (single container, single ALB target); revisit (b) if bundle/CDN concerns arise.

---

## 8. Migration plan (incremental cutover, live app never goes dark)

Even though we're rewriting comprehensively, we ship incrementally to de-risk. The legacy Dash app keeps serving production until each tab reaches parity.

**Phase 0 — Foundations (1–2 wks)**
- Scaffold `frontend/` (Vite + React + TS + Tailwind, Zustand, react-plotly.js).
- Stand up `src/dashboard_api/` FastAPI app with `/api/data/configs`, `/api/data/config`, `/api/health`; generate the TS client from OpenAPI.
- CI: lint/typecheck/test for both frontend and backend; `vite build` in the Docker image.
- Deploy a "hello dashboard" SPA behind a **new ALB path** (e.g. `/v2`) alongside the live Dash app.

**Phase 1 — First read-only tab (1–2 wks)**
- Port **Stats by Date** end-to-end: `/api/data/series` → `react-plotly.js`. Establishes the data contract and the chart component.
- Validate parity against the Dash version side by side.

**Phase 2 — Remaining stats tabs (2–3 wks)**
- Port Stats by Group, Stats Deepdive, Stats by Bet, Weekly Report. Extract shared chart/control components.

**Phase 3 — Agent-native chat (2 wks)**
- `/api/agent/chat` streaming via SSE; chat panel with token streaming + tool-call rendering (reuse `ai_agent` LangGraph core).
- Introduce the **action dispatcher** and the first `DashboardAction`s (navigate_tab, set_date_range, set_metrics) — the agent can now drive the dashboard.

**Phase 4 — Report editor + generative authoring (2 wks)**
- Port the Report tab: figure list, editable descriptions/summaries, Confluence export. Wire `/api/report/*` (port existing logic). Add inline per-chart agent actions (`/api/agent/explain`).

**Phase 5 — Cutover (1 wk)**
- Flip the ALB default to the React app; keep Dash on `/legacy` for one release as a fallback.
- Monitor `Dashboard/UserRequestCount` parity; then decommission the Dash app and delete `game_stats_monitor.py`.

Total rough estimate: ~9–13 weeks of focused work; each phase is independently shippable.

---

## 9. Deployment changes

- **Build**: multi-stage Docker — stage 1 `vite build` (Node build-time only, *not* a runtime), stage 2 the Python/uvicorn runtime that serves both the API and `frontend/dist` static assets. Node never runs in production; it's a build tool.
- **ECS**: same Fargate service pattern as today. The consolidated API replaces (or sits beside) the current dashboard + ai_agent services; reuse `infra/shared/ecs_helpers.py`.
- **ALB routing**: during migration, path-based routing (`/legacy` → Dash, `/` → React). After cutover, single target.
- **Scale-to-zero**: preserve the `Dashboard/UserRequestCount` CloudWatch metric emission (move the after-request hook into the FastAPI middleware).
- **Secrets/auth**: unchanged — `aws_secrets`, Confluence token, LLM keys via Secrets Manager; ALB behind SSO/VPN.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Two-language maintenance burden | Single-source the API contract (OpenAPI → generated TS client); shared action schema mirrored in Pydantic + TS. |
| Plotly bundle size hurts load time | Lazy-load the chart module; consider a custom partial Plotly build. |
| Parity gaps vs. the dense Dash UI | Incremental tab-by-tab port with side-by-side validation before cutover; keep `/legacy` for one release. |
| Agent-driven actions misfire / mutate state unexpectedly | All actions validated against the typed schema before dispatch; actions are pure store mutations (inspectable, undoable); audit log of agent actions. |
| Long-lived rewrite branch drifts from `dev` | Ship each phase to `dev` behind the `/v2` path; never a single long branch. |
| SSE behind ALB idle timeouts | Tune ALB idle timeout + keep-alive (already 300s+/310s today); heartbeat events on the stream. |

---

## 11. Open questions

1. **Consolidate vs. two services** — fold the data API into `ai_agent`, or keep a separate `dashboard_api` service that calls the agent service? (Leaning: one `dashboard_api` app that imports the agent core, to avoid an extra hop.)
2. **Static serving** — FastAPI `StaticFiles` (simplest) vs. S3+CloudFront (better caching/CDN). Start with the former.
3. **State store** — Zustand (lighter) vs. Redux Toolkit (more tooling). Leaning Zustand.
4. **Agent-UI library** — adopt `assistant-ui`/Vercel AI SDK components, or build a thin custom chat UI against our SSE format? (Evaluate in Phase 3.)
5. **Auth** — does the redesign warrant in-app auth (per-user report state, audit of agent actions), or stay ALB/SSO-only?

---

## 12. Decision log

- **2026-06-02** — Chose **Architecture A**: React + Vite + TS SPA + consolidated FastAPI backend (no Node server runtime). Rationale: keep the Python data + agent stack intact; modern agent-UI is achievable client-side against FastAPI SSE; deployment story unchanged. Alternatives considered: Next.js full-stack (rejected — needless Node backend rewrite/sidecar), hybrid Next.js BFF + FastAPI (rejected — highest ops complexity, two backend languages).
