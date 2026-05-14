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
| `SLACK_BOT_TOKEN`      | —         | Optional — enables Slack bot endpoint |
| `SLACK_SIGNING_SECRET` | —         | Optional — required when bot enabled  |

For production, store API keys in **AWS Secrets Manager** and reference them in the task definition
instead of plain-text environment variables.

## Slack Bot (HTTPS Events)

The agent ships an optional Slack endpoint at `POST /slack/events`. It is
**only mounted when both `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` are
configured** (env var or AWS Secrets Manager entry `ai-dashboard_ai_agent`),
so local dev without Slack credentials still works.

Behavior: responds **only** to `app_mention` events (when the bot is
@-tagged). DMs and regular channel messages are ignored. Each reply
posts in the message's thread; if the @mention is already inside a
thread, the prior 20 messages are pulled into the LLM's history so
follow-ups have context.

### One-time Slack app setup

1. Create a Slack app at https://api.slack.com/apps → *From scratch*.
2. **OAuth & Permissions** → *Bot Token Scopes*. Add:
   - `app_mentions:read` — receive @mention events
   - `chat:write` — post replies
   - `channels:history` — read thread history for context (public channels)
   - `groups:history` — same, for private channels (optional)
3. **Event Subscriptions** → *Enable Events* → set Request URL to
   `https://<ai-agent-alb-dns>/slack/events`. Slack will send a one-time
   challenge; the agent must already be deployed and the secrets set.
4. Under *Subscribe to bot events*, add `app_mention` (and nothing else).
5. *Install to Workspace*. Copy the **Bot User OAuth Token** (`xoxb-…`).
6. Back on *Basic Information*, copy the **Signing Secret**.

### Wire up secrets

Add the two values to the existing Secrets Manager entry:

```bash
# Read current secret, edit, then re-put
aws secretsmanager get-secret-value \
  --secret-id ai-dashboard_ai_agent --region us-west-2 \
  --query SecretString --output text | jq '. + {
    SLACK_BOT_TOKEN: "xoxb-...",
    SLACK_SIGNING_SECRET: "32charhex..."
  }' > /tmp/secret.json
aws secretsmanager put-secret-value \
  --secret-id ai-dashboard_ai_agent --region us-west-2 \
  --secret-string file:///tmp/secret.json
rm /tmp/secret.json
```

Roll the ECS service to pick up the new secret on cold start:

```bash
python infra/deploy_ai_agent_ecs.py
```

### Cold-start caveat

The service runs with **scale-to-zero** by default. The first @mention
after idle waits ~30–60 s for ECS to launch a task; Slack will retry
the event up to 3 times during that window. The handler dedupes on
`event_id` so retries don't double-respond, but the human user sees a
slow first reply. If that's annoying, redeploy without scale-to-zero:

```bash
python infra/deploy_ai_agent_ecs.py --no-scale-to-zero
```

(always-on Fargate cost: ~$10–15/month for the default 0.5 vCPU task)

## Resource Sizing

| Setting     | Default | Notes                                               |
|-------------|---------|-----------------------------------------------------|
| CPU         | 512     | 0.5 vCPU — sufficient for API + LLM calls           |
| Memory      | 1024 MB | Metadata cache is small; LLM work is offloaded      |
| Workers     | 1       | Single Uvicorn worker; async handles concurrency     |
| Scale-to-0  | Yes     | Spins down after 60 min idle, wakes on first 503    |

Increase `--task-cpu` / `--task-memory` if you add embedding models or vector stores in the future.
