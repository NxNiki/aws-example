# RAG Service — Design Spec

The RAG service indexes Confluence documentation declared in
[`config/rag_sources.yaml`](../src/rag_service/config/rag_sources.yaml)
and exposes a retrieval HTTP endpoint at `POST /retrieve`. The chat
agent (`ai_agent.chat_agent`) calls it via `rag_service.client` so
no vector-store details leak into the agent.

The interaction is one-way: Confluence is the source of truth,
the service holds a derived snapshot, and refreshes are batch
rebuilds rather than live writes.


## End-to-end workflow

1. **Source declaration.** `rag_sources.yaml` lists named groups
   (e.g. `data_slot_games`, `model_fish_hunter`). Each group is a
   set of Confluence pages, folders, or whole spaces. Page URLs
   and folder URLs are auto-classified — folders use the Cloud REST
   v2 `direct-children` endpoint and recurse into subfolders.
2. **Page fetch + chunking.** `confluence_loader` resolves each
   source into page records reusing `bituslabs_ds.confluence.client`;
   `chunker` splits page text into overlapping windows (default
   1200 chars / 200 overlap).
3. **Embedding.** `embeddings.Embedder` calls OpenAI
   (`text-embedding-3-small`, 1536 dim) or Gemini
   (`text-embedding-004`, 768 dim) and emits one vector per chunk.
4. **Index persistence.** Vectors plus chunk metadata
   (`page_id`, `chunk_index`, `title`, `text`, `url`,
   `source_name`) are written to the configured backend.
5. **Retrieval.** `retriever` embeds the user's query, runs
   similarity search, and returns top-k chunks with full
   attribution.


## Storage backends

### Phase 1 — Faiss + S3 *(current target)*

At launch scale (≈100 source docs, ≈500 chunks, low query volume)
the index is a flat `faiss.IndexFlatIP` plus a JSON metadata
sidecar persisted to S3.

```
build_rag_index.py
       │
       ▼
   [chunks + embeddings]
       │
       ▼
   confluence_rag.faiss      ──► s3://bituslabs-team-ai/rag/
   confluence_rag.meta.json  ──► s3://bituslabs-team-ai/rag/
```

The FastAPI service downloads both files at startup, loads the
index into memory, and serves retrieval queries from there. A
daily EventBridge → Fargate task runs `build_rag_index.py` to
rebuild and re-upload the artifacts. The service either restarts
via ECS service update (triggered by the rebuild job) or polls the
S3 ETag on a short timer and hot-reloads.

**Why this for now**

- **Cost.** $0 of compute outside the rebuild job (which is
  minutes/day). AWS OpenSearch Serverless has a ~$175/month
  minimum even at idle.
- **Operational simplicity.** Nothing to patch, scale, or
  monitor — just an artifact in S3.
- **Headroom.** 500 chunks × 1536 dims ≈ 3 MB index file.
  Comfortably fits in a Fargate task's memory and an in-process
  load takes milliseconds.

**Tradeoffs**

- **No BM25 / keyword hybrid** — pure dense retrieval. Acceptable
  for our domain (mostly prose docs in Chinese + English). If
  recall becomes a problem we add a Python BM25 sidecar; we don't
  need OpenSearch for that.
- **Full rebuild per refresh.** Incremental skip-if-unchanged
  isn't built; daily batch is cheap enough at this scale.
- **Single-process index.** Multiple FastAPI replicas each load
  their own copy from S3. Fine; index is small.

### Phase 2 — Managed OpenSearch *(future, optional)*

Migrate when any of these signals appear:

- Corpus grows past ~10k chunks (Faiss in-process is still fine
  but rebuilds slow and the metadata file gets unwieldy).
- We need BM25 + dense hybrid search for relevance.
- We need live updates without a full rebuild (e.g. webhook
  from Confluence on doc edit).
- Filtering needs grow beyond `source_name` equality.

Target: a single-AZ `t3.small.search` managed domain at ≈$25/month
— *not* OpenSearch Serverless, whose minimum capacity floor is
overprovisioned for this workload.

**Migration path**

The OpenSearch client (`opensearch_client.py`) and matching
hybrid retriever stay in the repo behind a `RAG_BACKEND` env-var
switch. Migration is a config flip plus one full reindex:

```
RAG_BACKEND=opensearch
OPENSEARCH_HOST=<domain endpoint>
OPENSEARCH_AUTH=basic        # or 'aws' for SigV4
```

then `poetry run python jobs/build_rag_index.py --backend=opensearch --drop`
once. The data flow upstream of the index is unchanged.


## Modules

| Concern | Module |
| --- | --- |
| What to index | `config/rag_sources.yaml` |
| Page / folder fetch (auth, URL parsing, v2 folder API) | `bituslabs_ds.confluence.client` |
| Source resolution + dedup | `confluence_loader.py` |
| Chunking | `chunker.py` |
| Embedding (OpenAI / Gemini) | `embeddings.py` |
| Vector store — Faiss + S3 | `faiss_store.py` *(new in Phase 1)* |
| Vector store — OpenSearch | `opensearch_client.py` *(kept for Phase 2)* |
| Index build | `indexer.py` |
| Retrieval | `retriever.py` |
| HTTP API | `app.py` |
| Agent-side client | `client.py` |

`bituslabs_ds.confluence.client` is the single home for Confluence
auth and HTTP — both `rag_service` and `report_agent` consume it,
so adding a new Confluence capability (folders, attachments,
labels, …) only happens once.


## Build cadence

A daily EventBridge → Fargate task runs
`jobs/build_rag_index.py`. Each run is a full rebuild: walk all
sources, re-fetch every page, re-chunk, re-embed, swap the S3
artifact atomically.

At current scale a full rebuild takes minutes and the embedding
cost is negligible. We'll move to incremental refresh only when
daily rebuild becomes the bottleneck — at that point the natural
checkpoint is `version.when` from Confluence vs.
`updated_at` stored per chunk.


## Configuration

Two layers, kept separate so content edits don't churn infra:

- **Content** lives in `rag_sources.yaml` (source list; optional
  `build:` block for chunk size / embedding provider).
- **Infrastructure** lives in env vars.

| Env var | Purpose |
| --- | --- |
| `RAG_BACKEND` | `faiss` (default) \| `opensearch` |
| `RAG_CHUNK_CHARS` / `RAG_CHUNK_OVERLAP` | Override YAML build params |
| `RAG_EMBEDDING_PROVIDER` | `auto` \| `openai` \| `gemini` |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Provider auth |
| `CONFLUENCE_URL` / `CONFLUENCE_EMAIL` / `CONFLUENCE_TOKEN` | Source auth |
| `OPENSEARCH_*` | Phase 2 only |

The Faiss artifact location is fixed in code at
`bituslabs_ds.config.DEFAULT_RAG_INDEX_URI` — it isn't a secret and isn't
expected to vary between environments, so it lives in config.py rather
than `.env`.


## Smoke-test queries

Use these to confirm the service is wired up after a deploy or index
rebuild. Each query targets internal-only terminology that a generic
LLM can't fabricate, so a sensible-looking response definitely came
from the indexed corpus.

```bash
# Set this once
ALB=http://rag-service-alb-1946335648.us-west-2.elb.amazonaws.com

# 1. Health — should report doc_count > 0
curl -s "$ALB/health" | jq

# 2. Sources — should list data_slot_games, data_fish_hunter, model_slot_games, model_fish_hunter
curl -s "$ALB/sources" | jq

# 3. Slot-games retrieval — expect SS01 AI 组数学表对比 or 老虎机数学表优化方案
curl -sX POST "$ALB/retrieve" -H 'Content-Type: application/json' \
    -d '{"query": "How is RTP calculated for slot games?", "top_k": 3}' | jq
```

Suggested test questions, grouped by which `source_name` should
dominate the top hits. The third column shows the kind of evidence
that proves retrieval is real (vs. the model bluffing from prior
knowledge).

| # | Query | Expected `source_name` | What to look for in the top passage |
| - | ----- | ---------------------- | ----------------------------------- |
| 1 | `"rollerCoaster math table RTP"` | `data_slot_games` | mentions specific RTP percentages for `rollerCoaster` / `carousels` / `risky2` etc. |
| 2 | `"giftShop free game trigger probability"` | `data_slot_games` | `giftShop` / `波动性` / `free game RTP` numbers |
| 3 | `"定向奖池分配 RTP 组别"` | `data_fish_hunter` | RTP=105/115/120/125 cohort rules, Round-Robin filling |
| 4 | `"fish hunter killing intervals streak"` | `data_fish_hunter` | mentions kill-streak thresholds, fish_type categories |
| 5 | `"BOOST_POOL DYNAMIC_RTP strategy"` | `data_fish_hunter` | fish_hunter strategy names from the ETL `daily_group` |
| 6 | `"动态规划求解老虎机 RTP"` | `model_slot_games` | DP[r][s1][s2] transition, three-reel combinations |

Quick way to sanity-check that the chat agent (not just the raw
service) is actually calling `/retrieve`: tail the rag-service log
while issuing a chat-side question. If the chat reached the RAG, you'll
see a `POST /retrieve HTTP/1.1 200 OK` entry within a second or two.

```bash
aws logs tail /ecs/rag-service --since 1m --region us-west-2 --follow \
  | grep -E 'POST /retrieve'
```


## Open questions

- **Service restart vs. hot reload.** Whether `app.py` polls S3
  ETag for index updates or relies on ECS deploys after each
  rebuild. Hot reload is mildly more complex; ECS deploys are
  zero-extra-code.
- **Index baked into image vs. fetched at startup.** Bake-in
  removes the S3 dependency at runtime but couples redeploys to
  reindexes. Default plan: fetch at startup, since reindexes
  shouldn't require redeploys.
- **Per-turn retrieval vs. tool-use pattern.** Currently the
  agent calls `/retrieve` on every user turn. A tool-use pattern
  where the LLM decides when to retrieve is cheaper and more
  precise but needs prompt engineering work.
