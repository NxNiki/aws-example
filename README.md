# aws-example

Data science and ETL utilities for AWS: Redshift-backed ETL, SageMaker training, EMR Spark, and a Dash game-stats dashboard on ECS.

---

## Setup (macOS and Python environment)

### Conda + Poetry

From the **repository root**:

```bash
conda env create -f environment.yml --prune
conda activate aws-example
```

Configure Poetry to use the conda environment (avoid a separate Poetry venv):

```bash
poetry config virtualenvs.create false --local
poetry install
```

If Poetry creates its own virtualenv, it will diverge from conda—keep `virtualenvs.create false` for this project.

### Pre-commit

```bash
pre-commit install
```

Hooks run on `git commit` using the definitions in `.pre-commit-config.yaml`.

### AWS CLI and credentials

Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Configure a profile (or default credentials):

```bash
aws configure
```

You typically use an **IAM user** access key for local development, or SSO / assumed roles per your org. The CLI is used for S3, SageMaker, EMR, ECR, etc.

---

## IAM permissions (summary)

Attach policies that match what you run. Common needs:

| Task | Example IAM access |
|------|---------------------|
| S3 read/write for project bucket | e.g. `AmazonS3FullAccess` or a scoped policy on your bucket/prefix |
| Athena SQL | e.g. `AmazonAthenaFullAccess` (or tighter: query + results bucket) |
| EMR: submit steps, manage cluster | e.g. `AmazonEMRFullAccessPolicy_v2` |
| SageMaker: training, tuning | Execution role (see `SAGEMAKER_ROLE` in `src/bituslabs_ds/config.py`) plus `sagemaker:*` / PassRole as required |
| ECR push/pull (Docker deploys) | ECR login + push policies for your repos |
| ECS / EventBridge (scheduled ETL, dashboard) | See [infra/etl/README.md](infra/etl/README.md) and [infra/dashboard_api/README.md](infra/dashboard_api/README.md) |

Narrow permissions in production; the table above matches the older README’s intent.

---

## Run jobs locally

### Clustering (K-means and pipeline)

The **`ClusterAnalysis`** class and YAML configs live under **`jobs/cluster_analysis/`**. See the full guide:

- **[jobs/cluster_analysis/README.md](jobs/cluster_analysis/README.md)** — feature selection, elbow method, K-means, pipeline, configs.

Example:

```bash
poetry run python jobs/cluster_analysis/cluster_analysis_pipeline.py --config_file jobs/cluster_analysis/cluster_config-wucaishen.yaml
```

### EDA and exploratory analysis

Examples at the repo root of `jobs/` and in subfolders:

- `jobs/analyses/analysis_eda_*` — one-off EDA (wucaishen, dragon/tiger, fish hunter, deepdive slot type)
- `jobs/ss01_wucaishen/analysis_eda_correlation.py` — correlation EDA
- `jobs/deep_dive/eda_bet_pattern.py` — bet-pattern EDA

Run with `poetry run python jobs/<script>.py` (add any script-specific flags).

### Smoke tests: S3, EMR, SageMaker

**[jobs/examples/README.md](jobs/examples/README.md)** — minimal scripts to verify AWS wiring without heavy `src` setup.

---

## Submit jobs to SageMaker

Training and tuning submit scripts live under **`jobs/examples/`**:

| Script | Purpose |
|--------|---------|
| [sagemaker_training_job_submit.py](jobs/examples/sagemaker_training_job_submit.py) | Single training job |
| [sagemaker_training_job_dist_submit.py](jobs/examples/sagemaker_training_job_dist_submit.py) | **Distributed** PyTorch training |
| [sagemaker_training_job_dist.py](jobs/examples/sagemaker_training_job_dist.py) | Training entrypoint (used by the dist job) |
| [sagemaker_tuning_job_submit.py](jobs/examples/sagemaker_tuning_job_submit.py) | Hyperparameter tuning |

Shared settings (region, bucket, role, image URI) are in **`src/bituslabs_ds/config.py`** (`SAGEMAKER_ROLE`, `IMAGE_URI`, etc.). Adjust for your account if needed.

---

## Submit jobs to EMR (Spark)

1. Create or use an EMR cluster in the AWS console (see [EMR getting started](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html)).
2. **IAM:** policy such as `AmazonEMRFullAccessPolicy_v2` for submitting steps and managing flows.
3. **Example:** upload your driver script to S3 and add a Spark step:
   - [jobs/examples/data_loader_pyspark_example_submit.py](jobs/examples/data_loader_pyspark_example_submit.py) — uses `submit_spark_step` with `cluster_id`, script URI, data and output S3 paths.
4. Optional automation: **`infra/emr/deploy.py`** can start a cluster and upload bootstrap/package artifacts (see script docstring and code).

Local PySpark driver for development: [jobs/examples/data_loader_pyspark_example.py](jobs/examples/data_loader_pyspark_example.py).

---

## ETL: fetch data from Redshift (`DataLoader`)

The **`DataLoader`** class in **`src/bituslabs_ds/etl.py`** connects to Redshift (often via an SSH bastion tunnel), runs SQL, and can cache results locally or on S3.

### Credentials (never commit secrets)

1. Copy **`.env.example`** to **`.env`** in the repo root.
2. Set **`REDSHIFT_USER`** and **`REDSHIFT_PASSWORD`**.
3. For bastion SSH: **`BASTION_KEY_PATH`** (path to `.pem`) or **`BASTION_KEY_CONTENT`** (e.g. from Secrets Manager on ECS).

Host and port are defined in **`src/bituslabs_ds/config.py`** (`REDSHIFT_HOST`, `REDSHIFT_PORT`). Default bastion IP is documented there (`DEFAULT_BASTION_IP`).

ETL jobs auto-load `.env` when importing `bituslabs_ds.config` locally (see `_maybe_load_dotenv` in that module).

### Example jobs

- `jobs/fish_hunter/etl_*.py` — game stats, bets, retention, etc.
- `jobs/ss01_wucaishen/etl_game_stats_daily_by_user_group.py` — builds `DataLoader` and runs queries
- `jobs/operation_daily_report/run_daily_report.py` — daily report orchestration ([README](jobs/operation_daily_report/README.md))
- `jobs/run_scheduled_etl_jobs.py` — sequential ETL/report orchestrator entrypoint for scheduled runs

### Scheduled ETL orchestrator script

Use `jobs/run_scheduled_etl_jobs.py` as a single entrypoint for cron/EventBridge/ECS scheduled execution.
It combines the four jobs listed below into one fixed sequence and returns a non-zero exit code if any step fails.

Current sequence:

1. `jobs/operation_daily_report/run_daily_report.py --lookback-days 1 --send-slack`
2. `jobs/ss01_wucaishen/etl_game_stats_daily_by_user_group.py`
3. `jobs/fish_hunter/etl_game_stats_daily_by_user.py`
4. `jobs/operation_daily_report/etl_weekly_report_all_games.py`

Run locally:

```bash
poetry run python jobs/run_scheduled_etl_jobs.py
```

### Scheduled ETL on AWS (Fargate / EventBridge)

Long-running ETL is better suited to **ECS Fargate** than Lambda. See **[infra/etl/README.md](infra/etl/README.md)** for Docker build, schedules, and IAM.

---

## Athena (optional)

For SQL against Athena instead of Redshift, add **`AmazonAthenaFullAccess`** (or equivalent) and use utilities under **`src/bituslabs_ds/`** (e.g. Athena helpers referenced in historical scripts).

---

## Update and deploy the dashboard

The Game Stats Dashboard runs as a container on **ECS Fargate** with an ALB.

- **Full instructions:** **[infra/dashboard_api/README.md](infra/dashboard_api/README.md)**
- **`infra/dashboard_api/deploy_ecs.py`** — deploy or update the service.
- **`infra/dashboard_api/Dockerfile`** + **`infra/dashboard_api/build.sh`** — build and push the image to ECR.

After code or dependency changes, **rebuild the image** before deploy:

```bash
python infra/dashboard_api/deploy_ecs.py --build-first
```

---

## Documentation index

| Topic | Location |
|------------|----------|
| Dashboard (ECS, Docker, `--build-first`) | [infra/dashboard_api/README.md](infra/dashboard_api/README.md) |
| Scheduled ETL on Fargate | [infra/etl/README.md](infra/etl/README.md) |
| Clustering / K-means | [jobs/cluster_analysis/README.md](jobs/cluster_analysis/README.md) |
| Operation daily report | [jobs/operation_daily_report/README.md](jobs/operation_daily_report/README.md) |
| Examples (S3, EMR, SageMaker) | [jobs/examples/README.md](jobs/examples/README.md) |
