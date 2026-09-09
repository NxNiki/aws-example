# dashboard_api (infra)

The new lightweight FastAPI service backing the React dashboard. Serves the
data/report API and the built SPA static assets from one uvicorn process on
port **8050**. LLM-free and Plotly/kaleido-free — agent turns go to the
`ai_agent` service. See [`docs/frontend_redesign.md`](../../docs/frontend_redesign.md).

## Build

```bash
bash infra/dashboard_api/build.sh        # build + push bituslabs-ds-dashboard-api:latest to ECR
```

The Dockerfile is multi-stage: stage 1 (`node:20`) runs `vite build`; the
runtime stage (`python:3.11-slim`) installs only `main,dashboard_api` and
serves both the API and `frontend/dist`. Node is build-time only.

## Run locally (Docker)

```bash
docker run --rm -p 8050:8050 \
  -e AWS_PROFILE -v ~/.aws:/root/.aws:ro \
  bituslabs-ds-dashboard-api
# open http://localhost:8050
```

## Deploy

```bash
bash infra/dashboard_api/build.sh           # build + push image to ECR
python infra/dashboard_api/deploy_ecs.py    # create/update the ECS service
# Or in one step:
python infra/dashboard_api/deploy_ecs.py --build-first
python infra/dashboard_api/deploy_ecs.py --dry-run   # preview without changes
```

`deploy_ecs.py` composes the shared ALB/ECS plumbing in
`infra/shared/ecs_helpers.py` and reuses the shared cluster (deploy the
dashboard first if it doesn't exist yet). For the Phase-0 migration the service
runs on **its own ALB origin** beside the live Dash app: the SPA is served at
`/` and the data API at `/api/data/*`. In Phase 3, `/api/agent/*` is path-routed
to the `ai_agent` service so the SPA still sees one origin; at cutover (Phase 5)
this origin becomes the default. It runs one always-on task — scale-to-zero is
added once the `UserRequestCount` middleware lands. See
[`docs/frontend_redesign.md`](../../docs/frontend_redesign.md) §8–§9.
