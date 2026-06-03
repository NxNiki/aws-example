# dashboard_api (infra)

The new lightweight FastAPI service backing the React dashboard. Serves the
data/report API and the built SPA static assets from one uvicorn process on
port **8050**. LLM-free and Plotly/kaleido-free — agent turns go to the
`ai_agent` service. See [`docs/frontend_redesign.md`](../../docs/frontend_redesign.md).

## Build

```bash
bash infra/dashboard_api/build.sh        # builds dashboard-api:latest
```

The Dockerfile is multi-stage: stage 1 (`node:20`) runs `vite build`; the
runtime stage (`python:3.11-slim`) installs only `main,dashboard_api` and
serves both the API and `frontend/dist`. Node is build-time only.

## Run locally (Docker)

```bash
docker run --rm -p 8050:8050 \
  -e AWS_PROFILE -v ~/.aws:/root/.aws:ro \
  dashboard-api:latest
# open http://localhost:8050
```

## Deploy (follow-up)

ECS Fargate deploy wiring (`deploy_ecs.py`) reuses
`infra/shared/ecs_helpers.py` and is added in a later Phase-0 step. ALB routes
`/api/agent/*` to the `ai_agent` service and everything else here, so the SPA
sees one origin.
