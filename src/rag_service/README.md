# rag_service

Isolated FastAPI microservice that powers the AI chat agent's documentation
lookups. It indexes Confluence pages declared in
[`config/rag_sources.yaml`](config/rag_sources.yaml) into OpenSearch and
exposes a hybrid (BM25 + vector) retrieval endpoint at
`POST /retrieve`.

The chat agent (`ai_agent.chat_agent`) calls the service over HTTP via
`rag_service.client.retrieve_passages` — no OpenSearch knowledge leaks into
the agent.

## Architecture

```
       rag_sources.yaml
              │
              ▼
   confluence_loader  ──►  chunker  ──►  embeddings (OpenAI)
                                                    │
                                                    ▼
                                       OpenSearch (Docker / AOSS)
                                                    ▲
                                                    │
   chat_agent ──HTTP──► /retrieve ──► retriever (BM25 + kNN, RRF merge)
```

| Concern | Module |
| --- | --- |
| What to index | `config/rag_sources.yaml` (YAML) |
| Connection (Docker / AOSS) | `opensearch_client.py` (env-driven) |
| Page fetching | `confluence_loader.py` (reuses `dashboards.confluence_client`) |
| Chunking | `chunker.py` |
| Embeddings (OpenAI / Gemini) | `embeddings.py` |
| Index build | `indexer.py` |
| Retrieval (hybrid + RRF) | `retriever.py` |
| HTTP API | `app.py` |
| Client used by agent | `client.py` |

## Local development

1. Start a local OpenSearch:

   ```bash
   docker compose -f infra/docker-compose.opensearch.yml up -d
   ```

   Confirm it's healthy: `curl -k -u admin:Admin@1234 https://localhost:9200`.
   The optional dashboards UI is at <http://localhost:5601>.

2. Copy `.env.example` → `.env` and fill in the `OPENSEARCH_*`,
   `RAG_SERVICE_*`, and `CONFLUENCE_*` blocks.

3. Edit [`config/rag_sources.yaml`](config/rag_sources.yaml) to point at the
   Confluence content you want indexed. You can use space keys, page IDs, or
   paste page URLs straight from the browser — the loader extracts IDs from
   `/pages/<id>` segments.

4. Install deps:

   ```bash
   poetry install --with rag_service,ai_agent
   ```

5. Build the index:

   ```bash
   poetry run python jobs/build_rag_index.py
   ```

   Add `--drop` if the schema changed and you want a clean rebuild.

6. Run the service:

   ```bash
   poetry run uvicorn rag_service.app:app --host 0.0.0.0 --port 8052
   ```

7. Smoke test:

   ```bash
   curl http://localhost:8052/health
   curl -X POST http://localhost:8052/retrieve \
        -H 'Content-Type: application/json' \
        -d '{"query": "how is RTP calculated", "top_k": 3}'
   ```

## Deploying to AWS

The service is environment-portable: the only changes between laptop and
production are env vars.

| Env var | Local Docker | AWS OpenSearch Serverless |
| --- | --- | --- |
| `OPENSEARCH_HOST` | `localhost` | `<collection-id>.<region>.aoss.amazonaws.com` |
| `OPENSEARCH_PORT` | `9200` | `443` |
| `OPENSEARCH_AUTH` | `basic` | `aws` |
| `OPENSEARCH_SERVICE` | _(unset)_ | `aoss` (use `es` for managed clusters) |
| `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD` | `admin` / `Admin@1234` | _(unset)_ |
| `AWS_REGION` | _(unset)_ | `us-west-2` |

Build and push the image:

```bash
./infra/rag_service/build.sh push
```

Then deploy as an ECS Fargate service (mirror the patterns in
`infra/ai_agent/deploy_ecs.py`). The IAM task role needs `aoss:APIAccessAll`
on the target collection plus read access to the Secrets Manager entries
for `OPENAI_API_KEY` and the `CONFLUENCE_*` triple.

## Embedding provider

The embedder supports both OpenAI and Gemini, selected by
`RAG_EMBEDDING_PROVIDER`:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use OpenAI if `OPENAI_API_KEY` is set; otherwise fall back to Gemini (`GOOGLE_API_KEY` / `GEMINI_API_KEY`). |
| `openai` | Force OpenAI. Default model `text-embedding-3-small` (1536 dim). |
| `gemini` | Force Gemini. Default model `text-embedding-004` (768 dim). |

Override the model with `RAG_EMBEDDING_MODEL`. If you also override the
dimension via `RAG_EMBEDDING_DIM`, make sure the OpenSearch mapping was
built with the same dimension.

> Switching providers (or using a model with a different dimension) requires
> a clean reindex because the `knn_vector` mapping is fixed at index
> creation time:
>
> ```bash
> poetry run python jobs/build_rag_index.py --drop
> ```

## Re-indexing

Two options:

- `POST /reindex` — runs the full pipeline in-process. Convenient but holds
  the request open until completion.
- Run `jobs/build_rag_index.py` as a scheduled EventBridge → Fargate task
  (recommended for production; mirror `jobs/run_scheduled_etl_jobs.py`).

The index is upserted by `_id = "{page_id}-{chunk_index}"`, so re-running
the job updates pages in place. Use `--drop` only when the schema changes.
