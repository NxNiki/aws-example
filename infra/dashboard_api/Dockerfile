# dashboard_api — serves the React SPA + the data/report API from one uvicorn
# process. Node appears only as a BUILD stage (vite build); it never runs in
# prod. See docs/frontend_redesign.md.

# --- Stage 1: build the SPA ---
FROM public.ecr.aws/docker/library/node:20-slim AS spa
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build   # -> /frontend/dist

# --- Stage 2: python deps ---
FROM public.ecr.aws/docker/library/python:3.11-slim AS builder
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
WORKDIR /app
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
COPY pyproject.toml poetry.lock ./
# LLM-free, Plotly/kaleido-free: FastAPI + uvicorn + polars (+ main), plus the
# confluence group for the Report tab's references + export endpoints.
RUN poetry install --only main,dashboard_api,confluence --no-root --no-cache && rm -rf $POETRY_CACHE_DIR

# --- Stage 3: runtime ---
FROM public.ecr.aws/docker/library/python:3.11-slim AS runtime
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    LOCAL_ROOT=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src/bituslabs_ds ./src/bituslabs_ds
COPY src/dashboard_api ./src/dashboard_api
# The shared modules (DataMetrics engine, Confluence client + secrets, report
# HTML builders) live in bituslabs_ds and ride the COPY above.
COPY configs/dashboard ./configs/dashboard
# TRANSITIONAL: also stage the YAMLs at the old path so this image works under
# a task definition that still sets DASHBOARD_CONFIG_DIR=/app/src/dashboards
# (:latest image / task-def env can skew during deploys). Drop once the new
# task def is live everywhere.
COPY configs/dashboard/ ./src/dashboards/
COPY --from=spa /frontend/dist ./frontend/dist

ENV DASHBOARD_CONFIG_DIR=/app/configs/dashboard \
    DASHBOARD_FRONTEND_DIST=/app/frontend/dist \
    AGENT_API_URL=http://localhost:8051
EXPOSE 8050

# --timeout-keep-alive 310 mirrors the ai_agent service (long SSE streams in
# later phases); --workers 2 for light data requests.
CMD ["uvicorn", "dashboard_api.app:app", \
     "--host", "0.0.0.0", "--port", "8050", \
     "--workers", "2", "--timeout-keep-alive", "310", "--access-log"]
