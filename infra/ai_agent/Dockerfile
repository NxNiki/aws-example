# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock ./

# Install main + 'llm' (langchain/langgraph) + 'confluence' (atlassian /
# requests, used by chat tools and the report-agent endpoint) + ai_agent
# (FastAPI/uvicorn/slack-bolt/aiohttp). No dash, plotly, polars, gunicorn.
RUN poetry install --only main,llm,confluence,ai_agent --no-root --no-cache && rm -rf $POETRY_CACHE_DIR

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    LOCAL_ROOT=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

# Application code — only the modules the chat agent needs.
#
# Layout:
#   src/ai_agent/  — chat_agent, chat_api, slack_handler, metadata_builder,
#                    report_agent/description (this service's own modules)
#   src/dashboards/ — only game_stats_monitor (source the agent reads via
#                    codebase_tools, never imported) and dashboard_config*.yaml
#                    (so the agent can answer data-location questions). The
#                    shared modules (confluence client, secrets, DataMetrics)
#                    moved into bituslabs_ds and ride that COPY.
#   src/rag_service/ — thin HTTP client only; no OpenSearch/Faiss/embeddings
#                    (search_confluence_rag would fail with "RAG service
#                    client is not installed" without these two files).
COPY src/bituslabs_ds ./src/bituslabs_ds
COPY src/ai_agent ./src/ai_agent
COPY src/dashboards/__init__.py               ./src/dashboards/__init__.py
COPY src/dashboards/game_stats_monitor.py     ./src/dashboards/game_stats_monitor.py
COPY src/dashboards/dashboard_config*.yaml    ./src/dashboards/
COPY src/rag_service/__init__.py              ./src/rag_service/__init__.py
COPY src/rag_service/client.py                ./src/rag_service/client.py
# ETL source files — explicit list of the scripts the agent should be able to
# inspect via codebase_tools (grep_codebase / read_source_file). Covers every
# ETL job scheduled by jobs/run_scheduled_etl_jobs.py PLUS the extra file
# scanned by metadata_builder (etl_game_stats_daily_by_group.py — required at
# startup; see src/ai_agent/column_metadata.yaml `etl_files`).
COPY jobs/run_scheduled_etl_jobs.py                                    ./jobs/run_scheduled_etl_jobs.py
COPY jobs/ss01_wucaishen/etl_game_stats_daily_by_user_group.py         ./jobs/ss01_wucaishen/etl_game_stats_daily_by_user_group.py
COPY jobs/ss02_deepdive/etl_game_stats_daily_by_user_group.py          ./jobs/ss02_deepdive/etl_game_stats_daily_by_user_group.py
COPY jobs/ss03_mahjiang_streak/etl_game_stats_daily_by_user_group.py   ./jobs/ss03_mahjiang_streak/etl_game_stats_daily_by_user_group.py
COPY jobs/fish_hunter/etl_game_stats_daily_by_user.py                  ./jobs/fish_hunter/etl_game_stats_daily_by_user.py
COPY jobs/operation_daily_report/run_daily_report.py                   ./jobs/operation_daily_report/run_daily_report.py
COPY jobs/operation_daily_report/etl_game_stats_daily_by_group.py      ./jobs/operation_daily_report/etl_game_stats_daily_by_group.py
COPY jobs/operation_daily_report/etl_weekly_report_all_games.py        ./jobs/operation_daily_report/etl_weekly_report_all_games.py

EXPOSE 8051

# Uvicorn ASGI server — async-friendly for LLM API calls.
#
# --workers 2          Two processes so the dashboard's per-figure description
#                      fan-out can run two LLM calls in parallel instead of
#                      serializing through a single worker (which caused 504s
#                      on 6+ figure reports). Each worker keeps its own
#                      metadata cache; cold-start happens twice but
#                      steady-state is unaffected. ~150 MB extra RAM.
# --timeout-keep-alive 310  Slightly above the ALB idle timeout (300s) so the
#                      server doesn't close keep-alive connections before
#                      the LB does — avoids spurious 502/504 races.
# --host 0.0.0.0       Reachable from outside the container.
# --port 8051          Separate from dashboard (8050).
CMD ["uvicorn", \
     "ai_agent.chat_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8051", \
     "--workers", "2", \
     "--timeout-keep-alive", "310", \
     "--access-log"]
