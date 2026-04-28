# AI Chat Agent — ECS Deployment

Standalone FastAPI service that answers questions about dashboard columns, user groups, and game metrics using LangChain + LLM (OpenAI / Gemini).

## Architecture

```
 Dashboard (Gunicorn, port 8050)        AI Chat Agent (Uvicorn, port 8051)
 ┌────────────────────────────────┐     ┌──────────────────────────────────┐
 │ Dash / Plotly / Flask          │     │ FastAPI / LangChain / LLM API   │
 │ game_stats_monitor:server      │     │ chat_api:app                    │
 │ Dockerfile.dashboard           │     │ Dockerfile.ai_agent             │
 └──────────┬─────────────────────┘     └──────────┬───────────────────────┘
            │                                      │
       ALB :80                                ALB :80
   ai-team-dashboard-cluster            ai-team-dashboard-cluster
   (shared ECS cluster)                 (shared ECS cluster)
```

The two services share the same ECS cluster but run as **independent Fargate tasks**.
A crash or slow LLM response in the agent does not impact the dashboard.

## Quick Start

### 1. Build & push the Docker image

```bash
bash infra/docker_build_ai_agent.sh
```

### 2. Deploy to ECS

```bash
# Default: Gemini, scale-to-zero, reuse existing cluster
python infra/deploy_ai_agent_ecs.py

# With OpenAI instead:
python infra/deploy_ai_agent_ecs.py --chat-provider openai

# Build + deploy in one step:
python infra/deploy_ai_agent_ecs.py --build-first

# Dry run (print plan without executing):
python infra/deploy_ai_agent_ecs.py --dry-run
```

### 3. Run locally

```bash
docker run -p 8051:8051 \
  -e GOOGLE_API_KEY="your-key" \
  bituslabs-ds-ai-agent

# Or without Docker:
CHAT_PROVIDER=gemini GOOGLE_API_KEY=your-key \
  uvicorn dashboards.chat_api:app --host 0.0.0.0 --port 8051
```

## API Endpoints

| Method | Path                    | Description                      |
|--------|-------------------------|----------------------------------|
| GET    | `/health`               | Health check (used by ALB)       |
| POST   | `/api/chat`             | Send a question, get an answer   |
| GET    | `/api/metadata/columns` | List all column metadata         |
| GET    | `/api/metadata/groups`  | List all group definitions       |
| POST   | `/api/metadata/rebuild` | Force-rebuild metadata cache     |
| GET    | `/docs`                 | Swagger UI                       |

### Chat example

```bash
curl -X POST http://<ALB_DNS>/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does active_user_rtp_less_0_4_ratio mean?"}'
```

## Configuration

| Env var          | Default         | Description                           |
|------------------|-----------------|---------------------------------------|
| `CHAT_PROVIDER`  | `openai`        | `openai` or `gemini`                  |
| `CHAT_MODEL`     | provider default| Model name override                   |
| `OPENAI_API_KEY` | —               | Required when `CHAT_PROVIDER=openai`  |
| `GOOGLE_API_KEY` | —               | Required when `CHAT_PROVIDER=gemini`  |

For production, store API keys in **AWS Secrets Manager** and reference them in the task definition
instead of plain-text environment variables.

## Resource Sizing

| Setting     | Default | Notes                                               |
|-------------|---------|-----------------------------------------------------|
| CPU         | 512     | 0.5 vCPU — sufficient for API + LLM calls           |
| Memory      | 1024 MB | Metadata cache is small; LLM work is offloaded      |
| Workers     | 1       | Single Uvicorn worker; async handles concurrency     |
| Scale-to-0  | Yes     | Spins down after 60 min idle, wakes on first 503    |

Increase `--task-cpu` / `--task-memory` if you add embedding models or vector stores in the future.
