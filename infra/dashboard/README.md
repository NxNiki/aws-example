# Game Stats Dashboard — AWS ECS deployment

This folder contains everything needed to build the dashboard container and deploy it to **ECS Fargate** behind an Application Load Balancer.

## Quick start

From the **repository root** (build context must include `pyproject.toml`, `src/`, etc.):

```bash
python infra/dashboard/deploy_ecs.py --build-first
```

Use **`--build-first`** whenever you have changed application code, dependencies, or dashboard config under `src/` so ECS runs a **new image**. The flag runs `infra/dashboard/build.sh` before provisioning or updating the service; without it, deployment reuses whatever image is already in ECR.

If you only need to tweak ECS/ALB settings and the image is already current:

```bash
python infra/dashboard/deploy_ecs.py
```

## What each piece does

| Artifact | Role |
|----------|------|
| [`deploy_ecs.py`](deploy_ecs.py) | Creates or updates the ECS cluster, task definition, service, ALB, target group, and related AWS resources. Points the task at the ECR image `bituslabs-ds-dashboard:latest`. |
| [`Dockerfile.dashboard`](Dockerfile.dashboard) | Multi-stage build: installs Poetry deps (`main` + `dashboard` groups), copies `src/bituslabs_ds` and `src/dashboards`, exposes port **8050**, runs `game_stats_monitor.py`. |
| [`infra/dashboard/build.sh`](infra/dashboard/build.sh) | Ensures the ECR repo exists, logs in to ECR, runs `docker build -f infra/dashboard/Dockerfile ./`, tags and pushes **`bituslabs-ds-dashboard:latest`** to ECR. |

`--build-first` is implemented by invoking `infra/dashboard/build.sh` from the project root (same as running it manually).

Equivalent manual flow:

```bash
bash infra/dashboard/build.sh
python infra/dashboard/deploy_ecs.py
```

## Common options

Run `python infra/dashboard/deploy_ecs.py --help` for the full list. Useful flags include:

- **`--region`** — AWS region (default `us-west-2`).
- **`--cluster-name`**, **`--service-name`** — ECS cluster and service names.
- **`--dry-run`** — Print planned actions without changing AWS.
- **`--wait`** — Wait for the service to stabilize after deploy.

## Prerequisites (short)

- AWS CLI configured; permissions for ECS, ELB, EC2/VPC, IAM, Logs, and ECR as needed for create/update.
- **Docker** available locally when using `--build-first` or `infra/dashboard/build.sh`.
- Task execution role / S3 access for dashboard data (see the docstring at the top of `deploy_ecs.py`).

For scheduled ETL and operational context, see [`ETL_SCHEDULED_RUNS.md`](ETL_SCHEDULED_RUNS.md).
