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
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

# Application code — only the modules the chat agent needs.
#
# Layout:
#   src/ai_agent/  — chat_agent, chat_api, slack_handler, metadata_builder,
#                    report_agent/description (this service's own modules)
#   src/dashboards/ — only the shared/auxiliary files: confluence_client +
#                    secrets (also used by rag_service), user_stats_aggregates
#                    (read at runtime by metadata_builder for Python aggregation
#                    docstrings), column_metadata.yaml + dashboard_config*.yaml.
#   src/rag_service/ — thin HTTP client only; no OpenSearch/Faiss/embeddings
#                    (search_confluence_rag would fail with "RAG service
#                    client is not installed" without these two files).
COPY src/bituslabs_ds ./src/bituslabs_ds
COPY src/ai_agent ./src/ai_agent
COPY src/dashboards/__init__.py               ./src/dashboards/__init__.py
COPY src/dashboards/confluence_client.py      ./src/dashboards/confluence_client.py
COPY src/dashboards/secrets.py                ./src/dashboards/secrets.py
COPY src/dashboards/user_stats_aggregates.py  ./src/dashboards/user_stats_aggregates.py
COPY src/dashboards/dashboard_config*.yaml    ./src/dashboards/
COPY src/rag_service/__init__.py              ./src/rag_service/__init__.py
COPY src/rag_service/client.py                ./src/rag_service/client.py
# ETL source files — only the .py scripts the metadata builder scans for SQL.
# Exclude output/, log/, __pycache__ (multi-GB data files not needed).
COPY jobs/ss01_wucaishen/*.py               ./jobs/ss01_wucaishen/
COPY jobs/ss02_deepdive/*.py                ./jobs/ss02_deepdive/
COPY jobs/ss03_mahjiang_streak/*.py         ./jobs/ss03_mahjiang_streak/
COPY jobs/fish_hunter/*.py                  ./jobs/fish_hunter/
COPY jobs/operation_daily_report/*.py       ./jobs/operation_daily_report/

EXPOSE 8051

# Uvicorn ASGI server — async-friendly for LLM API calls.
#
# --workers 1          Single process; metadata cache shared across requests.
# --timeout-keep-alive 65  Match ALB idle timeout (60s) + buffer.
# --host 0.0.0.0       Reachable from outside the container.
# --port 8051          Separate from dashboard (8050).
CMD ["uvicorn", \
     "ai_agent.chat_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8051", \
     "--workers", "1", \
     "--timeout-keep-alive", "65", \
     "--access-log"]
