# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS-based data science and ETL platform for game analytics (Bituslabs). Fetches game statistics from Redshift via SSH bastion tunnel, processes data, caches to S3, and serves a Dash dashboard and AI agent on ECS Fargate.

## Common Commands

```bash
# Setup
conda env create -f environment.yml --prune && conda activate aws-example
poetry config virtualenvs.create false --local
poetry install                          # main deps only
poetry install --with dev,test          # development
pre-commit install                      # git hooks

# Testing
poetry run pytest                       # all tests
poetry run pytest tests/unit/ -m "not slow"   # fast unit tests
poetry run pytest tests/integration/    # integration tests
poetry run pytest -m "not slow"         # skip slow tests
poetry run pytest -n auto               # parallel execution
poetry run pytest --cov=bituslabs_ds --cov-report=html --cov-report=term-missing  # coverage

# Run a single test
poetry run pytest tests/unit/test_utils.py::test_function_name -v

# Linting & formatting
make lint       # flake8, mypy, black --check, isort --check
make format     # auto-format with black + isort

# Run jobs
poetry run python jobs/<script>.py
```

## Architecture

**Four source packages** (all under `src/`, configured in pyproject.toml):
- `bituslabs_ds` — Core library: ETL, S3 utilities, ML, EDA, Athena, PySpark helpers
- `dashboards` — LEGACY Dash web app (replaced in production by dashboard_api + the React SPA; deleted after the post-cutover bake). Shared utilities formerly here now live in `bituslabs_ds` (`metrics/user_stats_aggregates.py`, `confluence/{client,export_html,references}.py`, `aws_secrets.py`); one-line shims remain at the old import paths until deletion
- `ai_agent` — FastAPI chat service: LangChain ReAct agent, Slack bot, metadata cache, Report-tab LLM endpoints
- `rag_service` — FastAPI retrieval microservice: Confluence loader, embeddings, Faiss/OpenSearch backed retriever

**Data flow:**
```
Redshift (prod) → SSH bastion tunnel → DataLoader (etl.py) → S3 parquet cache → Dashboard / Analysis
```

**Key modules in `bituslabs_ds`:**
- `config.py` — AWS regions, S3 buckets, Redshift credentials, SageMaker roles. Auto-loads `.env` locally.
- `etl.py` — `DataLoader` class: Redshift/Athena queries, SSH tunneling, S3 caching
- `s3_utils.py` — S3 read/write/list/cache operations
- `ml.py` — ML training utilities, SageMaker integration
- `eda.py` — Exploratory data analysis tools
- `models/gail/` — GAIL (Generative Adversarial Imitation Learning) implementation

**Entry points:**
- `jobs/run_scheduled_etl_jobs.py` — Master ETL orchestrator (EventBridge/Fargate)
- `entry_points/etl_dispatcher.py` — Dynamic job runner
- `infra/dashboard/deploy_ecs.py` — Dashboard deployment to ECS
- `infra/ai_agent/deploy_ecs.py` — AI agent deployment to ECS
- `infra/emr/deploy.py` — EMR cluster management

**Infrastructure:** `infra/` is organized one subfolder per service — `dashboard/`, `ai_agent/`, `rag_service/`, `etl/`, `sagemaker/`, `operation_report/`, `emr/` — each containing its own `Dockerfile`, `build.sh`, deploy script, and README where relevant. Shared deploy plumbing (base Docker image, ECS helpers) lives in `infra/shared/`. See `infra/README.md` for the index.

## Code Style

- **Black** formatter: line length 120, target Python 3.11+
- **isort**: black profile, line length 120
- **mypy**: `--ignore-missing-imports`
- **Pre-commit hooks** run black, isort, mypy on every commit
- Pyright `extraPaths = ["src"]` for import resolution

## Comments

Default to writing no comments. Only add one when the WHY is non-obvious: a
hidden constraint, a subtle invariant, a workaround for a specific bug,
behavior that would surprise a reader. Don't restate what well-named code
already conveys. Don't reference the current task, PR, or callers — those
belong in commit messages and rot fast.

**Narrow exception — user-facing feature surfaces.** Functions that
implement a user-facing feature *do* get a purpose docstring, because they
sit on a vocabulary boundary: a user asks about "clip data", the AI agent
greps the repo, and without a comment naming the feature, neither the
agent nor a new engineer can bridge the user's words to the code. This
applies to:

- Dashboard callbacks / panel builders / chart controls (`src/dashboards/`)
- AI-agent API endpoints + tool functions (`src/ai_agent/`)
- ETL jobs producing named dashboard metrics (`jobs/*/etl_*.py`)
- Scheduled jobs invoked from `jobs/run_scheduled_etl_jobs.py`

It does NOT apply to: private helpers, internal data-plumbing utilities,
boilerplate callback wiring, or anything purely internal whose name is
already self-explanatory.

### Docstring template for user-facing features

```python
def _apply_outlier_clipping(df: pl.DataFrame, lo: float, hi: float) -> pl.DataFrame:
    """Clip per-row metric values to a user-configured [min, max] for display.

    Dashboard feature: "Clip data" toggle in each chart's controls
    (element IDs ``group-{group_id}-clip-{enable|min|max}`` in
    Group/Range mode, ``viz-{panel_id}-clip-{enable|min|max}`` in
    Viz mode).

    Behavior: when the toggle is on, values outside [lo, hi] are pinned
    to the bound — NOT removed. This stops a single outlier from
    squashing the visible y-axis range without dropping data points.
    The original values stay intact upstream; clipping is purely a
    display transformation applied right before the figure is rendered.
    """
```

Structure to follow (skip a section if it genuinely doesn't apply):

1. **One-line summary** in user/UX vocabulary, not code vocabulary.
2. **`Dashboard feature:`** (or `API endpoint:` / `ETL job:`) line naming
   the feature as users / dashboards / API callers refer to it, plus
   *where it surfaces* — the panel/tab/control IDs, the URL path, the
   output table/column, whichever is the searchable handle. This is what
   lets the AI agent's `grep_codebase` connect a question like
   *"how does the clip data toggle work?"* to this function.
3. **`Behavior:`** 2-4 lines on *what it does and why*, including the
   trigger (button click, config toggle, scheduled run) and any
   non-obvious invariant. Skip what well-named code already conveys.
4. **`Inputs:`** only if a parameter's meaning isn't obvious from its
   name + type (e.g. units, ranges, what it represents in dashboard
   terms).

Style: keep it tight. A user-facing-feature docstring should usually be
6–15 lines. If it grows past 20, the function is probably doing two
features and should be split.

## Testing

- pytest config in `tests/pytest.ini`
- Markers: `slow`, `integration`, `aws`, `spark`, `unit`, `smoke`
- AWS mocking via `moto`; shared fixtures in `tests/conftest.py`
- 300-second timeout per test

## Credentials

Never commit secrets. Copy `.env.example` → `.env` and set `REDSHIFT_USER`, `REDSHIFT_PASSWORD`, `BASTION_KEY_PATH`. On ECS, secrets come from environment variables / Secrets Manager.

A populated `.env` is present locally with the full set of credentials the services and jobs use:
- **Redshift + bastion:** `REDSHIFT_USER`, `REDSHIFT_PASSWORD`, `BASTION_KEY_PATH` — run ground-truth Redshift queries via `bituslabs_ds.etl.DataLoader` (e.g. to validate dashboard/ETL numbers against the source).
- **Confluence:** `CONFLUENCE_URL`, `CONFLUENCE_EMAIL`, `CONFLUENCE_TOKEN`.
- **LLM:** `GOOGLE_API_KEY` (Gemini).
- **Slack:** `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `SLACK_USER_TOKEN`, `SLACK_CHANNEL_ID`.
- **Services:** `RAG_SERVICE_URL`.

The bastion **IP is not stored** (it changes) — pass the current one to `DataLoader(bastion_ip=...)` or a job's `--bastion-ip`. Never print `.env` values or commit them (`.env` is gitignored).

## Branch & Merge Workflow

Full workflow rules live in `CONTRIBUTING.md`. Highlights:

- **Never commit to `main` directly.** Releases flow `feature/* → dev → main`.
- **`dev` accepts direct commits** for small fixes; non-trivial work goes through a feature branch + PR.
- **Feature → dev**: rebase onto `dev` during development, **squash-merge** the PR. Feature branches are kept (not deleted) and reused after rebasing onto fresh `dev`.
- **dev → main**: **`--no-ff` merge commit** (preserves history), then create an **annotated semver tag** (`MAJOR.MINOR.PATCH`, no `v` prefix) on the merge commit. Update `CHANGELOG.md` (Keep a Changelog format) before the merge.
- **Pause for confirmation** before any push to `main`, any tag push, or any force-push.
- When opening a PR, draft a structured description (Summary / Changes / Test plan / Breaking changes / Related). Template in `CONTRIBUTING.md`.

## Commit Guidelines

### Message format
```
[type] short imperative description

Optional longer explanation if the why is non-obvious.
```

**Types:** `feat` · `fix` · `chore` · `refactor` · `docs` · `test`

### How to split commits
- One commit per logical concern. Ask: "would reverting this commit make sense on its own?"
- All changes to a single file go in one commit — never split one file across commits.
- Related changes across multiple files that serve the same purpose belong together (e.g., renaming a metric in three ETL jobs + the dashboard config that references it).
- Unrelated changes that happen to land at the same time should be separate commits (e.g., a Dockerfile tweak is separate from an ETL query change).

### Examples of good groupings
| Commit | Files |
|--------|-------|
| `[feat] add delta bet metrics to fish_hunter ETL` | one ETL file |
| `[feat] rename metric to _bg and restrict to BASE game in ss01/ss02/ss03` | three parallel ETL files |
| `[chore] use ECR mirror for Python base image` | one Dockerfile |
| `[feat] update dashboard configs for renamed/new metrics` | dashboard config + metadata files |

### What not to do
- Don't commit all modified files in one giant commit.
- Don't add unrelated cleanup to a feature commit.
- Don't use vague messages like `update` or `fix stuff`.

## Poetry Dependency Groups

All optional groups: `ds` (scipy, pymc), `ml` (scikit-learn, sagemaker), `dl` (pytorch), `dashboard` (dash, plotly, polars, kaleido), `llm` (langchain, langgraph — ai_agent-only now), `confluence` (atlassian-python-api, requests — shared by dashboard + ai_agent + rag_service), `ai_agent` (fastapi, uvicorn, slack-bolt, aiohttp — install with `llm,confluence`), `rag_service` (faiss-cpu, opensearch-py, openai, google-generativeai), `etl` (redshift, paramiko, slack), `spark` (pyspark — do NOT bundle when deploying to EMR).
