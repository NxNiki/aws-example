#!/usr/bin/env python3
"""
Deploy (update) the dashboard_api service on ECS Fargate.

Steady-state deployer: the dashboard_api service already lives behind the
production ALB (``game-stats-dashboard-alb``) — its listener default forwards
to ``dashboard-api-tg`` and ``/api/agent/*`` routes to ai-chat-agent. Those
were established once by the Phase 5 cutover and the legacy Dash app is
decommissioned, so a routine deploy just rolls a new task definition:

  1. Ensure the ECR repo + cluster, target group, task SG, and exec role.
  2. Register a fresh task definition (new ``:latest`` image) and force a
     rolling deployment (old task serves until the new one is healthy).
  3. Wait for the service to stabilize and the target group to go healthy.

The one-time cutover/decommission logic (listener flip, agent-TG wiring,
legacy teardown) lived here through 1bd7d6c; it was removed once the legacy
app was deleted. Recover it from git history if a similar cutover is ever
needed again.

Prerequisites: ``bash infra/dashboard_api/build.sh`` (or pass --build-first).

Usage:
  python infra/dashboard_api/deploy_ecs.py
  python infra/dashboard_api/deploy_ecs.py --build-first
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import boto3
except ImportError:
    print("boto3 required. Run: poetry add boto3  # or pip install boto3", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bituslabs_ds.config import DASHBOARD_CONFIG_S3_PATH, S3_BUCKET  # noqa: E402
from bituslabs_ds.s3_utils import parse_s3_path, upload_file_to_s3  # noqa: E402
from infra.shared.ecs_helpers import (  # noqa: E402
    ECS_CLUSTER_NAME,
    ECS_TASK_EXECUTION_ROLE_NAME,
    create_or_update_service,
    ensure_ecr_repo,
    ensure_ecs_task_execution_role,
    ensure_log_group,
    ensure_target_group,
    ensure_task_security_group,
    get_account_id,
    get_default_vpc_id,
    get_default_vpc_subnets,
    register_task_definition,
    require_active_cluster,
    wait_for_service_stable,
    wait_for_targets_healthy,
)

# ---- Configuration (edit these directly) ------------------------------------
REGION = "us-west-2"
SERVICE_NAME = "dashboard-api"
IMAGE_NAME = "bituslabs-ds-dashboard-api"
DASHBOARD_API_PORT = 8050
# Memory sized from production OOMs: 1 GB died on the first /api/data/series;
# 2 GB died on parallel per-panel collects (fixed by the single-flight window
# cache in services/common.py); 4 GB still died on Stats-by-Group spans — the
# user-row frames are simply large. The window cache is additionally
# size-budgeted (see _WINDOW_CACHE_MAX_BYTES). One worker — more workers
# multiply it all.
# 8 GB is the 1-vCPU Fargate ceiling. 1 vCPU pinned at 100% under concurrent
# users when collects loaded every column; the column-projected window collect
# (services/common.py projection_columns) shrinks that work 5-40x, so we try
# the current size first. If CPUUtilization still pins during busy windows,
# bump to 2048/12288 (12 GB needs >=2 vCPU) and raise _WINDOW_CACHE_MAX_BYTES.
TASK_CPU = 1024
TASK_MEMORY = 8192
DESIRED_COUNT = 1  # scale-to-zero: see ensure_autoscaling (idle >1h -> 0, ALB 5xx wakes)

# Scale-to-zero policy knobs. Idle = no Dashboard/UserRequestCount datapoint
# (the app.py middleware emits one per non-health request) for IDLE_MINUTES.
# Wake = any ELB 5xx on the ALB: with zero tasks the listener returns 503,
# which is exactly a user knocking while the service sleeps. First visit after
# a sleep therefore shows an error page for the ~2-3 min the task needs to
# start; the SPA/user retry lands normally.
IDLE_MINUTES = 60
IDLE_PERIOD_SECONDS = 300  # alarm granularity; IDLE_MINUTES must divide by this
MAX_TASKS = 1

# The production ALB dashboard-api lives behind (its listener default already
# forwards to dashboard-api-tg). We read its DNS for the CORS env and keep the
# idle timeout at 300s so SSE chat streams (/api/agent/chat) don't 504.
ALB_NAME = "game-stats-dashboard-alb"
ALB_IDLE_TIMEOUT = 300
# Where the server-side /api/report/* proxy reaches the agent (its own ALB).
AGENT_ALB_NAME = "ai-chat-agent-alb"

HEALTH_CHECK_PATH = "/api/health"  # registered before the SPA catch-all mount
TG_HEALTH_CHECK_INTERVAL = 30
TG_HEALTHY_THRESHOLD = 2
TG_DEREGISTRATION_DELAY = 30

# S3_BUCKET / DASHBOARD_CONFIG_S3_PATH come from bituslabs_ds.config (single source of
# truth). Dashboard config YAMLs are read from that S3 path at runtime (not baked into
# the image), so a new config is just an S3 upload — no rebuild/redeploy. Deploy syncs
# the repo's configs/dashboard/*.yaml there (without deleting S3-only configs).
S3_POLICY_NAME = "ecsTaskExecutionRole-s3-dashboard-api-rw"
S3_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [f"arn:aws:s3:::{S3_BUCKET}", f"arn:aws:s3:::{S3_BUCKET}/*"],
        },
        {
            "Effect": "Allow",
            "Action": ["cloudwatch:PutMetricData"],
            "Resource": "*",
            "Condition": {"StringEquals": {"cloudwatch:namespace": ["Dashboard"]}},
        },
    ],
}


def sync_configs_to_s3() -> None:
    """Upload the repo's dashboard_config-*.yaml to S3 (no delete).

    Keeps the repo configs as the source of truth in S3 while preserving any configs
    added directly to the bucket (the whole point: add a config without a redeploy).
    """
    config_dir = PROJECT_ROOT / "configs" / "dashboard"
    files = sorted(config_dir.glob("dashboard_config-*.yaml"))
    if not files:
        print(f"  No configs found under {config_dir}; skipping S3 sync.")
        return
    _, prefix = parse_s3_path(DASHBOARD_CONFIG_S3_PATH)
    prefix = prefix.rstrip("/")
    for path in files:
        upload_file_to_s3(path, S3_BUCKET, f"{prefix}/{path.name}")
    print(f"  Synced {len(files)} config(s) to {DASHBOARD_CONFIG_S3_PATH} (no delete).")


def ensure_autoscaling(session, alb_arn: str) -> None:
    """Scale the service to zero after IDLE_MINUTES without user requests and
    wake it on the next visit.

    Two step-scaling policies on the ECS service (min 0, max MAX_TASKS):

    - scale-in: alarm on ``Dashboard/UserRequestCount`` (Sum < 1 per
      IDLE_PERIOD_SECONDS window, IDLE_MINUTES/IDLE_PERIOD consecutive
      windows, missing data = breaching since the middleware only emits on
      traffic) -> ExactCapacity 0.
    - wake: alarm on the ALB's ``HTTPCode_ELB_5XX_Count`` (a request hitting
      the empty target group 503s at the ALB) -> ExactCapacity 1. Any real
      5xx burst also wakes the service, which is harmless.

    After a wake, the idle alarm stays in ALARM until the first 5-minute
    window with traffic lands; the scale-in policy's cooldown (2x the wake
    window) blocks it from re-zeroing the service in that gap.
    """
    aas = session.client("application-autoscaling")
    cw = session.client("cloudwatch")
    resource_id = f"service/{ECS_CLUSTER_NAME}/{SERVICE_NAME}"
    dimension = "ecs:service:DesiredCount"

    aas.register_scalable_target(
        ServiceNamespace="ecs",
        ResourceId=resource_id,
        ScalableDimension=dimension,
        MinCapacity=0,
        MaxCapacity=MAX_TASKS,
    )

    scale_in_arn = aas.put_scaling_policy(
        PolicyName=f"{SERVICE_NAME}-scale-to-zero",
        ServiceNamespace="ecs",
        ResourceId=resource_id,
        ScalableDimension=dimension,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity",
            "StepAdjustments": [{"MetricIntervalUpperBound": 0.0, "ScalingAdjustment": 0}],
            "Cooldown": 2 * IDLE_PERIOD_SECONDS,
        },
    )["PolicyARN"]
    scale_out_arn = aas.put_scaling_policy(
        PolicyName=f"{SERVICE_NAME}-wake-on-request",
        ServiceNamespace="ecs",
        ResourceId=resource_id,
        ScalableDimension=dimension,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity",
            "StepAdjustments": [{"MetricIntervalLowerBound": 0.0, "ScalingAdjustment": MAX_TASKS}],
            "Cooldown": 60,
        },
    )["PolicyARN"]

    idle_windows = IDLE_MINUTES * 60 // IDLE_PERIOD_SECONDS
    cw.put_metric_alarm(
        AlarmName=f"{SERVICE_NAME}-idle-scale-in",
        AlarmDescription=f"No dashboard user requests for {IDLE_MINUTES} min -> scale service to 0",
        Namespace="Dashboard",
        MetricName="UserRequestCount",
        Dimensions=[{"Name": "Service", "Value": SERVICE_NAME}],
        Statistic="Sum",
        Period=IDLE_PERIOD_SECONDS,
        EvaluationPeriods=idle_windows,
        DatapointsToAlarm=idle_windows,
        Threshold=1.0,
        ComparisonOperator="LessThanThreshold",
        TreatMissingData="breaching",
        AlarmActions=[scale_in_arn],
    )
    # ALB dimension value is the ARN suffix: app/<name>/<hash>.
    alb_dimension = alb_arn.split(":loadbalancer/", 1)[1]
    cw.put_metric_alarm(
        AlarmName=f"{SERVICE_NAME}-wake-on-request",
        AlarmDescription="Request hit the ALB while the service is scaled to 0 -> scale to 1",
        Namespace="AWS/ApplicationELB",
        MetricName="HTTPCode_ELB_5XX_Count",
        Dimensions=[{"Name": "LoadBalancer", "Value": alb_dimension}],
        Statistic="Sum",
        Period=60,
        EvaluationPeriods=1,
        Threshold=0.0,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[scale_out_arn],
    )
    print(f"  Scalable target: {resource_id} (min 0, max {MAX_TASKS})")
    print(f"  Scale-in:  {SERVICE_NAME}-idle-scale-in ({IDLE_MINUTES} min without UserRequestCount)")
    print(f"  Wake:      {SERVICE_NAME}-wake-on-request (ELB 5xx on {alb_dimension})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy (roll a new task definition for) the dashboard_api service")
    parser.add_argument("--region", default=REGION, help="AWS region")
    parser.add_argument("--alb-name", default=ALB_NAME, help="Production ALB dashboard-api lives behind")
    parser.add_argument("--desired-count", type=int, default=DESIRED_COUNT, help="Number of tasks")
    parser.add_argument("--agent-api-url", default=None, help="AGENT_API_URL override (default: ai-chat-agent-alb)")
    parser.add_argument("--cors-origins", default=None, help="DASHBOARD_API_CORS override")
    parser.add_argument("--build-first", action="store_true", help="Run infra/dashboard_api/build.sh first")
    args = parser.parse_args()

    if args.build_first:
        print("Building and pushing dashboard_api Docker image...")
        subprocess.run(["bash", str(SCRIPT_DIR / "build.sh")], check=True, cwd=PROJECT_ROOT)

    region = args.region
    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    ec2 = session.client("ec2")
    ecs = session.client("ecs")
    elbv2 = session.client("elbv2")
    logs = session.client("logs")
    iam = session.client("iam")

    account_id = get_account_id(sts)
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{IMAGE_NAME}:latest"
    print(f"Region: {region}, Account: {account_id}")
    print(f"Image: {ecr_uri}")

    vpc_id = get_default_vpc_id(ec2)
    subnet_ids = get_default_vpc_subnets(ec2, vpc_id)
    print(f"VPC: {vpc_id}, Subnets: {subnet_ids}")

    print("\n1. ECR repository + cluster...")
    ensure_ecr_repo(session.client("ecr"), IMAGE_NAME)
    require_active_cluster(ecs)

    print(f"\n1b. Syncing dashboard configs to {DASHBOARD_CONFIG_S3_PATH}...")
    sync_configs_to_s3()

    print(f"\n2. Production ALB: {args.alb_name}")
    alb = elbv2.describe_load_balancers(Names=[args.alb_name])["LoadBalancers"][0]
    alb_arn, alb_dns = alb["LoadBalancerArn"], alb["DNSName"]
    alb_sg_id = alb["SecurityGroups"][0]
    elbv2.modify_load_balancer_attributes(
        LoadBalancerArn=alb_arn,
        Attributes=[{"Key": "idle_timeout.timeout_seconds", "Value": str(ALB_IDLE_TIMEOUT)}],
    )
    print(f"  ALB: {alb_dns} (idle timeout → {ALB_IDLE_TIMEOUT}s, SG {alb_sg_id})")

    print("\n3. dashboard-api target group + task security group...")
    tg_arn = ensure_target_group(
        elbv2,
        f"{SERVICE_NAME}-tg",
        vpc_id,
        DASHBOARD_API_PORT,
        health_check_path=HEALTH_CHECK_PATH,
        health_check_interval=TG_HEALTH_CHECK_INTERVAL,
        healthy_threshold=TG_HEALTHY_THRESHOLD,
        deregistration_delay=TG_DEREGISTRATION_DELAY,
    )
    sg_task_id = ensure_task_security_group(ec2, vpc_id, f"{SERVICE_NAME}-task-sg", DASHBOARD_API_PORT, alb_sg_id)

    print("\n4. Task execution role + inline policy...")
    exec_role_arn = ensure_ecs_task_execution_role(iam, account_id)
    iam.put_role_policy(
        RoleName=ECS_TASK_EXECUTION_ROLE_NAME, PolicyName=S3_POLICY_NAME, PolicyDocument=json.dumps(S3_POLICY)
    )

    print("\n5. Task definition + service...")
    agent_api_url = args.agent_api_url
    if not agent_api_url:
        agent_albs = elbv2.describe_load_balancers(Names=[AGENT_ALB_NAME])["LoadBalancers"]
        agent_api_url = f"http://{agent_albs[0]['DNSName']}"
        print(f"  Auto-detected AI agent ALB: {agent_api_url}")
    env_vars = [
        {"name": "DASHBOARD_CONFIG_DIR", "value": DASHBOARD_CONFIG_S3_PATH},
        {"name": "DASHBOARD_FRONTEND_DIST", "value": "/app/frontend/dist"},
        # Enables the UserRequestCount middleware (scale-to-zero idle signal).
        {"name": "DASHBOARD_SERVICE_NAME", "value": SERVICE_NAME},
        {"name": "DASHBOARD_API_CORS", "value": args.cors_origins or f"http://{alb_dns}"},
        {"name": "AGENT_API_URL", "value": agent_api_url},
    ]
    log_group = ensure_log_group(logs, f"/ecs/{SERVICE_NAME}")
    task_def = {
        "family": SERVICE_NAME,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": str(TASK_CPU),
        "memory": str(TASK_MEMORY),
        "executionRoleArn": exec_role_arn,
        "taskRoleArn": exec_role_arn,
        "containerDefinitions": [
            {
                "name": SERVICE_NAME,
                "image": ecr_uri,
                # Override the image's --workers 2: a second worker doubles the
                # in-process parquet cache memory (see TASK_MEMORY note). Sync
                # endpoints run in Starlette's threadpool, so one worker still
                # serves concurrent requests.
                "command": [
                    "uvicorn",
                    "dashboard_api.app:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(DASHBOARD_API_PORT),
                    "--workers",
                    "1",
                    "--timeout-keep-alive",
                    "310",
                    "--access-log",
                ],
                "portMappings": [{"containerPort": DASHBOARD_API_PORT, "protocol": "tcp"}],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": SERVICE_NAME,
                    },
                },
                "environment": env_vars,
                "healthCheck": {
                    "command": [
                        "CMD-SHELL",
                        f"curl -sf http://localhost:{DASHBOARD_API_PORT}{HEALTH_CHECK_PATH} || exit 1",
                    ],
                    "interval": 10,
                    "timeout": 5,
                    "retries": 3,
                    "startPeriod": 45,
                },
            }
        ],
    }
    register_task_definition(ecs, task_def)
    create_or_update_service(
        ecs,
        cluster=ECS_CLUSTER_NAME,
        service_name=SERVICE_NAME,
        container_name=SERVICE_NAME,
        container_port=DASHBOARD_API_PORT,
        tg_arn=tg_arn,
        subnet_ids=subnet_ids,
        sg_task_id=sg_task_id,
        desired_count=args.desired_count,
    )
    wait_for_service_stable(ecs, ECS_CLUSTER_NAME, SERVICE_NAME)
    wait_for_targets_healthy(elbv2, tg_arn)

    print("\n6. Autoscaling (scale-to-zero + wake-on-request)...")
    ensure_autoscaling(session, alb_arn)

    print("\n" + "=" * 60)
    print("dashboard-api deployed.")
    print(f"  SPA:     http://{alb_dns}/")
    print(f"  Health:  http://{alb_dns}{HEALTH_CHECK_PATH}")
    print(f"  Configs: http://{alb_dns}/api/data/configs")
    print(f"  Agent:   http://{alb_dns}/api/agent/skills")
    print("\nRollback: redeploy a prior image — rebuild from an earlier git SHA, or")
    print("  re-tag a previous ECR image digest to :latest and re-run this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
