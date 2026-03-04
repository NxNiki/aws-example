# Running ETL Jobs Periodically on AWS

ETL jobs can run **~30+ minutes** (Redshift queries, multiple runs). **Lambda is not suitable** (15 min max timeout). Use **Fargate** (or ECS with Fargate) so a container runs on a schedule with no timeout limit and no servers to manage.

## Why Fargate (and not Lambda)

| | Lambda | Fargate (ECS) |
|---|--------|----------------|
| Max duration | 15 min | No practical limit (hours) |
| Memory / CPU | Fixed tiers | Choose task size (e.g. 1 vCPU, 2 GB) |
| VPC / Redshift | Supported | Full control (same VPC as Redshift if needed) |
| Cost | Per invocation + GB‑sec | Per vCPU/memory per second while task runs |

**Recommendation:** Use **EventBridge (schedule) → ECS Fargate task** that runs your existing ETL Docker image.

## High-level flow

1. **EventBridge rule** runs on a schedule (e.g. daily at 2:00 UTC).
2. Rule triggers an **ECS RunTask** (Fargate) with your ETL image and task definition.
3. Container starts, runs the ETL script (e.g. `fish_hunter/etl_game_stats_daily_by_user.py`), then exits.
4. Logs go to **CloudWatch Logs**; you can add alerts on failure.

## Prerequisites

- ETL image built and pushed to **ECR** (you already have `infra/docker_build_etl.sh`).
- **VPC + subnets** where the task runs. If Redshift is in a VPC, run the task in the same VPC (or a peered one) so it can reach Redshift; if you use a bastion, ensure the task can reach it or pass a bastion IP.
- **IAM**: task execution role (pull image, write logs), task role (S3, Redshift if needed, Secrets Manager if you store DB credentials there).

## Option A: One-time setup with AWS Console

1. **ECS cluster** (Fargate)
   - ECS → Clusters → Create cluster → Name e.g. `etl-cluster`, no EC2, Create.

2. **Task definition**
   - ECS → Task definitions → Create new task definition.
   - Launch type: **Fargate**.
   - Task size: e.g. **1 vCPU, 2 GB** (increase if the job is heavy).
   - Container: image = your ECR URI, e.g. `338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-etl:latest`.
   - Command override (optional): e.g. `jobs/fish_hunter/etl_game_stats_daily_by_user.py,--bastion-ip,` (comma-separated; leave bastion-ip empty if not used).
   - Log configuration: **awslogs** group e.g. `/ecs/etl-fish-hunter`, region = your region.
   - Environment (optional): `BASTION_IP` if you use it.
   - Create.

3. **EventBridge rule**
   - EventBridge → Rules → Create rule.
   - Name: `etl-fish-hunter-daily`, Schedule: e.g. `cron(0 2 * * ? *)` (2:00 UTC daily).
   - Target: **ECS Task**.
   - Cluster: `etl-cluster`, Task definition: the one above, Subnets: at least one private (or public if no VPC requirements), Security group: allow outbound (and to Redshift/bastion if needed).
   - Create.

After this, the rule will start one Fargate task at the scheduled time; the task runs the ETL and then stops.

## Option B: Scripted setup (recommended)

Use the provided script to create the ECS cluster, log group, task definition, and EventBridge rule + target in one go:

1. **Build and push the ETL image** (from project root):
   ```bash
   bash infra/docker_build_etl.sh
   ```

2. **Create IAM role for EventBridge** (one-time; if you already use ECS scheduled tasks you may have this):
   - IAM → Roles → Create role → Trusted entity: **EventBridge** (Events).
   - Attach policy **AmazonEC2ContainerServiceEventsRole** (allows EventBridge to run ECS tasks).
   - Name it e.g. `ecsEventsRole`.

3. **Run the setup script** with your VPC subnet(s) and security group:
   ```bash
   export SUBNETS="subnet-xxx,subnet-yyy"   # at least one; use private if Redshift is in VPC
   export SECURITY_GROUP="sg-xxx"          # allow outbound; allow Redshift/bastion if needed
   bash infra/setup_etl_schedule.sh
   ```

The script creates:

- ECS cluster `etl-cluster`
- Log group `/ecs/etl-fish-hunter`
- Fargate task definition `etl-fish-hunter` (from `infra/ecs_etl_task_def.json` with your ECR image)
- EventBridge rule `etl-fish-hunter-daily` (cron: daily at 02:00 UTC) and target to run the task

Override names with env vars: `ETL_CLUSTER_NAME`, `ETL_TASK_FAMILY`, `ETL_LOG_GROUP`, `ETL_RULE_NAME`, `AWS_REGION`.

## Running multiple ETL jobs

One task definition, one image. Override the **command** to run different jobs.

### Option A: One task definition, override command per run

When you run a task (RunTask API or EventBridge), pass a different `command`:

| Job | Command |
|-----|---------|
| fish_hunter daily | `["jobs/fish_hunter/etl_game_stats_daily_by_user.py", "--bastion-ip", "13.215.212.244"]` |
| ss01 by user group | `["jobs/ss01_wucaishen/etl_game_stats_daily_by_user_group.py", "--bastion-ip", "13.215.212.244"]` |
| ss01 by group | `["jobs/operation_daily_report/etl_game_stats_daily_by_group.py", "--bastion-ip", "13.215.212.244"]` |

**EventBridge**: Create one rule per job. Each rule targets the same task definition but overrides the container command in the target.

**RunTask (CLI)**:
```bash
aws ecs run-task --cluster etl-cluster --task-definition etl-fishhunter \
  --overrides '{"containerOverrides":[{"name":"etl","command":["jobs/ss01_wucaishen/etl_game_stats_daily_by_user_group.py","--bastion-ip","13.215.212.244"]}]}'
```

### Option B: Separate task definitions per job

Useful if jobs need different CPU/memory or different secrets. Create `etl-fishhunter`, `etl-ss01-user-group`, etc., each with a different default `command`.

## Secrets (Redshift credentials, bastion key)

ETL jobs read credentials from environment variables. **Never commit credentials to the repo.**

- **Local**: Copy `.env.example` to `.env`, fill in `REDSHIFT_USER` and `REDSHIFT_PASSWORD`, then `source .env` or use `python-dotenv` before running jobs.
- **ECS**: Add to task definition `secrets` (from Secrets Manager):
  - `REDSHIFT_USER`
  - `REDSHIFT_PASSWORD`
  - `BASTION_KEY_CONTENT` (if using bastion tunnel)

Grant the task **execution role** `secretsmanager:GetSecretValue` on those secrets.

## Bastion tunnel (cross-region ECS ↔ Redshift)

When ECS and Redshift are in different regions, use an SSH bastion host. **Do not copy the `.pem` key into the Docker image** (security risk).

### ECS Fargate (Secrets Manager)

1. **Store secrets in Secrets Manager**
   ```bash
   aws secretsmanager create-secret --name ai_etl/bastion-key \
     --secret-string "$(cat ~/.ssh/your-bastion.pem)" --region us-west-2
   aws secretsmanager create-secret --name ai_etl/redshift-user \
     --secret-string "anaylsis_user" --region us-west-2
   aws secretsmanager create-secret --name ai_etl/redshift-password \
     --secret-string "your-password" --region us-west-2
   ```

2. **Grant the ECS task execution role** (ecsTaskExecutionRole) `secretsmanager:GetSecretValue` on that secret. Attach a policy or add an inline policy with that permission.

3. **Update the task definition**
   - Replace `ACCOUNT_ID` in all secret ARNs with your AWS account ID.
   - Add `REDSHIFT_USER` and `REDSHIFT_PASSWORD` to `secrets` (pointing to your Secrets Manager ARNs).

4. `etl.py` reads `BASTION_KEY_CONTENT` at connect time and uses it for the SSH tunnel.

### Local Docker (volume mount)

```bash
docker run --rm \
  -v ~/.ssh/your-bastion.pem:/tmp/bastion_key.pem \
  -e BASTION_KEY_PATH=/tmp/bastion_key.pem \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
  338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-etl:latest \
  fish_hunter/etl_game_stats_daily_by_user.py --bastion-ip 13.215.212.244
```

## Summary

- **Use Fargate** (via ECS) for long-running ETL; avoid Lambda for 30‑minute jobs.
- **EventBridge (cron) → ECS RunTask (Fargate)** with your existing ETL image is the standard pattern.
- Build/push image with `infra/docker_build_etl.sh`, then create an ECS cluster, Fargate task definition, and EventBridge rule as above.
