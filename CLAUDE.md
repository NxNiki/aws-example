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

**Two source packages** (both under `src/`, configured in pyproject.toml):
- `bituslabs_ds` — Core library: ETL, S3 utilities, ML, EDA, Athena, PySpark helpers
- `dashboards` — Dash web apps for game analytics with AI chat integration

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
- `infra/deploy_dashboard_ecs.py` — Dashboard deployment to ECS
- `infra/deploy_ai_agent_ecs.py` — AI agent deployment to ECS
- `infra/deploy_emr.py` — EMR cluster management

**Infrastructure:** Five Dockerfiles in `infra/` (base, dashboard, etl, ai_agent, sagemaker) with corresponding build scripts.

## Code Style

- **Black** formatter: line length 120, target Python 3.11+
- **isort**: black profile, line length 120
- **mypy**: `--ignore-missing-imports`
- **Pre-commit hooks** run black, isort, mypy on every commit
- Pyright `extraPaths = ["src"]` for import resolution

## Testing

- pytest config in `tests/pytest.ini`
- Markers: `slow`, `integration`, `aws`, `spark`, `unit`, `smoke`
- AWS mocking via `moto`; shared fixtures in `tests/conftest.py`
- 300-second timeout per test

## Credentials

Never commit secrets. Copy `.env.example` → `.env` and set `REDSHIFT_USER`, `REDSHIFT_PASSWORD`, `BASTION_KEY_PATH`. On ECS, secrets come from environment variables / Secrets Manager.

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

All optional groups: `ds` (scipy, pymc), `ml` (scikit-learn, sagemaker), `dl` (pytorch), `dashboard` (dash, plotly, polars), `llm` (langchain, langgraph, atlassian-python-api — shared by dashboard + ai_agent images), `ai_agent` (fastapi, uvicorn — install together with `llm`), `etl` (redshift, paramiko, slack), `spark` (pyspark — do NOT bundle when deploying to EMR).
