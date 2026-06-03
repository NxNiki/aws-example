# frontend — Game Stats Dashboard (React + Vite + TypeScript + ECharts)

The new dashboard SPA. Replaces the legacy Dash app (`src/dashboards/game_stats_monitor.py`)
incrementally; see the plan in [`docs/frontend_redesign.md`](../docs/frontend_redesign.md).

**Phase 0 (this scaffold):** project setup + a smoke page that drives the full
stack — picks a game config and renders one metric as an ECharts line chart
from `dashboard_api`'s `/api/data/*` endpoints.

## Stack

- **React 18 + Vite 5 + TypeScript** (strict)
- **ECharts** for charts (thin wrapper in `src/charts/EChart.tsx`)
- **Zustand** for state (`src/store/dashboardStore.ts`)
- **Tailwind** for styling

## Develop

```bash
# 1) start the backend (from the repo root, in the poetry env)
poetry install --only main,dashboard_api
uvicorn dashboard_api.app:app --reload --port 8050   # needs AWS creds for /api/data/series

# 2) start the frontend dev server
cd frontend
npm install
npm run dev        # http://localhost:5173 ; proxies /api -> :8050
```

`npm run typecheck` type-checks without emitting. `npm run build` produces
`dist/`, which `dashboard_api` serves in production.

## Layout

```
src/
├── api/        # typed client + types mirroring dashboard_api schemas
│               #   (Phase 1+: generated from the OpenAPI schema)
├── charts/     # ECharts wrapper + option builders
├── store/      # Zustand store (+ agent action dispatcher in Phase 3)
├── features/   # one folder per tab (HelloDashboard is the Phase 0 placeholder)
└── App.tsx
```
