# dashboard_api

Lightweight FastAPI service backing the React dashboard. Serves the data/report
API the SPA and the agent share, and (in prod) the built SPA static assets.
LLM-free by design — agent turns go to the `ai_agent` service. Plan:
[`docs/frontend_redesign.md`](../../docs/frontend_redesign.md). Dataflow &
caching reference: [`docs/dashboard_dataflow.md`](../../docs/dashboard_dataflow.md).

## Endpoints (Phase 0)

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Health check |
| `GET /api/data/configs` | List game configs (`ss01`, `fishhunter`, …) |
| `GET /api/data/config/{id}` | Resolved config: tabs, metric groups, granularities |
| `POST /api/data/series` | Metric time-series (ECharts-ready) for a config/granularity/metrics |

OpenAPI docs at `/docs` when running.

## Run

```bash
poetry install --only main,dashboard_api
uvicorn dashboard_api.app:app --reload --port 8050
```

`/api/data/series` reads the S3 parquet cache via `bituslabs_ds.s3_utils`, so
it needs AWS credentials; `/configs` and `/config/{id}` only read local YAML.

## Layout

```
app.py                 # FastAPI factory; mounts routers + serves frontend/dist
settings.py            # env-driven settings (config dir, frontend dist, CORS, agent URL)
routers/  data.py health.py
services/ configs.py   # discover + parse dashboard_config-*.yaml
          series.py    # read parquet -> series (Phase 0: placeholder aggregation)
schemas/  data.py      # Pydantic models -> OpenAPI -> generated TS client
```

> **Phase 0 caveat:** `series.py` aggregates source rows per date with a mean
> placeholder. Parity with the legacy dashboard's user-row enrichment
> (`dashboards/user_stats_aggregates.py`) lands in Phase 1.
