# Infra

One folder per service. Each service folder contains its own `Dockerfile`,
`build.sh`, deploy script, and (where useful) a README explaining
operational details. Shared deploy plumbing lives in `shared/`.

| Folder | Purpose | Entry points |
| --- | --- | --- |
| [`shared/`](shared/) | Common deploy helpers + base Docker image | `ecs_helpers.py`, `Dockerfile.base`, `build_base.sh` |
| [`dashboard_api/`](dashboard_api/README.md) | React SPA + data/report API (FastAPI) on ECS Fargate behind an ALB | `deploy_ecs.py`, `build.sh`, `Dockerfile` |
| [`ai_agent/`](ai_agent/README.md) | FastAPI chat service + Slack bot on ECS Fargate | `deploy_ecs.py`, `build.sh`, `Dockerfile` |
| [`rag_service/`](rag_service/) | RAG retrieval microservice on ECS Fargate | `deploy_ecs.py`, `build.sh`, `Dockerfile`, `docker-compose.opensearch.yml` |
| [`etl/`](etl/README.md) | Fargate-scheduled ETL jobs (EventBridge → ECS) | `build.sh`, `setup_schedule.sh`, `ecs_task_def.json`, `Dockerfile` |
| [`sagemaker/`](sagemaker/) | Image used for SageMaker training jobs | `build.sh`, `Dockerfile` |
| [`operation_report/`](operation_report/) | Daily operation report — macOS launchd + ECS schedule variants | `setup_schedule_mac.sh`, `setup_ecs_schedule.sh`, `daily-report.plist`, `ecs_task_def.json` |
| [`emr/`](emr/) | EMR cluster spin-up helper | `deploy.py` |
| [`bootstrap/`](bootstrap/) | EMR/SageMaker bootstrap scripts | `install-package.sh` |

## Conventions

- **Run from the project root.** Every `build.sh` and `deploy_ecs.py` expects
  the project root as the working directory so Docker's build context picks
  up `pyproject.toml`, `poetry.lock`, `src/`, `jobs/`, etc.
  ```bash
  bash infra/dashboard_api/build.sh
  python infra/dashboard_api/deploy_ecs.py --build-first
  ```
- **Shared helpers** in `shared/ecs_helpers.py` (account-id lookup, default
  VPC discovery, ECR repo + ECS execution role idempotent create,
  `ECS_CLUSTER_NAME`). Deploy scripts import them as
  `from infra.shared.ecs_helpers import …` after prepending the project
  root to `sys.path`.
- **`--build-first` flag** on each deploy script invokes the matching
  `build.sh` from the project root before provisioning, so a single command
  rebuilds the image and rolls the ECS service.
