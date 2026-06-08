#!/usr/bin/env python3
"""
Deploy the new dashboard_api service (React SPA + data/report API) to AWS ECS
Fargate behind its own public Application Load Balancer.

Phase-0 migration model (see docs/frontend_redesign.md §8/§9): this service
runs *beside* the live Dash app on its own ALB origin, so the React rewrite can
be previewed end to end without touching the production dashboard's ALB. The
SPA is served at ``/``, the data API under ``/api/data/*``; ``/api/agent/*``
will be path-routed to the existing ``ai_agent`` service in Phase 3. At cutover
(Phase 5) the team flips DNS / the default origin to this service.

The service does NOT yet emit the ``Dashboard/UserRequestCount`` metric the live
dashboard uses for scale-to-zero (that middleware is a later step), so this runs
a single always-on task. Scale-to-zero is wired in once the metric lands.

Prerequisites:
  1. Run `bash infra/dashboard_api/build.sh` to build and push the image to ECR.
  2. The shared ECS cluster must exist (deploy the dashboard first if it doesn't).

Usage:
  python infra/dashboard_api/deploy_ecs.py
  python infra/dashboard_api/deploy_ecs.py --build-first
  python infra/dashboard_api/deploy_ecs.py --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 required. Run: poetry add boto3  # or pip install boto3", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infra.shared.ecs_helpers import (  # noqa: E402
    ECS_CLUSTER_NAME,
    ECS_TASK_EXECUTION_ROLE_NAME,
    create_or_update_service,
    ensure_alb,
    ensure_ecr_repo,
    ensure_ecs_task_execution_role,
    ensure_http_listener,
    ensure_log_group,
    ensure_service_security_groups,
    ensure_target_group,
    get_account_id,
    get_default_vpc_id,
    get_default_vpc_subnets,
    register_task_definition,
    require_active_cluster,
    wait_for_service_stable,
)

# ---- Configuration (edit these directly) ------------------------------------
REGION = "us-west-2"
SERVICE_NAME = "dashboard-api"
IMAGE_NAME = "bituslabs-ds-dashboard-api"
DASHBOARD_API_PORT = 8050
TASK_CPU = 512  # 0.5 vCPU — light: data fetches + static assets, LLM-free
TASK_MEMORY = 1024  # 1 GB
DESIRED_COUNT = 1  # always-on for now (no scale-to-zero metric yet)

# Health check hits the API, not the SPA root: the data router registers
# /api/health *before* the SPA StaticFiles mount, so it returns 200 JSON even
# when the catch-all SPA mount would otherwise serve index.html.
HEALTH_CHECK_PATH = "/api/health"
TG_HEALTH_CHECK_INTERVAL = 30
TG_HEALTHY_THRESHOLD = 2
TG_DEREGISTRATION_DELAY = 30
# SSE streams in Phase 3 (/api/agent proxy + report regeneration) hold the
# connection open; match the ai_agent ALB's 300s idle timeout so long turns
# don't 504. Harmless for the data API.
ALB_IDLE_TIMEOUT = 300

# The data API reads the parquet cache from S3 (bituslabs_ds.s3_utils); the
# Report Spec store (Phase 4) read/writes report_specs/ in the same bucket.
# CloudWatch PutMetricData is pre-granted so the UserRequestCount middleware
# (scale-to-zero, later) works without a policy change.
S3_BUCKET = "bituslabs-team-ai"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy dashboard_api (SPA + data API) to ECS Fargate")
    parser.add_argument("--region", default=REGION, help="AWS region")
    parser.add_argument("--service-name", default=SERVICE_NAME, help="ECS service name")
    parser.add_argument("--vpc-id", default=None, help="VPC ID (default: default VPC)")
    parser.add_argument("--subnet-ids", default=None, help="Comma-separated subnet IDs (default: public subnets)")
    parser.add_argument("--task-cpu", type=int, default=TASK_CPU, help="Task CPU units (1 vCPU = 1024)")
    parser.add_argument("--task-memory", type=int, default=TASK_MEMORY, help="Task memory MB")
    parser.add_argument("--desired-count", type=int, default=DESIRED_COUNT, help="Number of tasks")
    parser.add_argument(
        "--agent-api-url",
        default=None,
        help="AGENT_API_URL the backend uses for /api/agent/* (LLM text). "
        "Auto-detected from the ai-chat-agent ALB if not provided.",
    )
    parser.add_argument(
        "--cors-origins",
        default=None,
        help="DASHBOARD_API_CORS value (comma-separated). Defaults to the ALB origin.",
    )
    parser.add_argument("--build-first", action="store_true", help="Run infra/dashboard_api/build.sh before deploying")
    parser.add_argument("--wait", action="store_true", help="Wait for service to stabilize")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    args = parser.parse_args()

    if args.build_first and not args.dry_run:
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

    vpc_id = args.vpc_id or get_default_vpc_id(ec2)
    if args.subnet_ids:
        subnet_ids = [s.strip() for s in args.subnet_ids.split(",")]
    else:
        subnet_ids = get_default_vpc_subnets(ec2, vpc_id)
    print(f"VPC: {vpc_id}, Subnets: {subnet_ids}")

    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        print("  - Log group, security groups")
        print("  - ALB, target group, listener")
        print(f"  - Task definition ({args.service_name}), ECS service")
        return

    print("\n1. ECR repository...")
    ensure_ecr_repo(session.client("ecr"), IMAGE_NAME)

    print("\n2. ECS cluster...")
    require_active_cluster(ecs)

    print(f"\n3. Log group: /ecs/{args.service_name}")
    log_group = ensure_log_group(logs, f"/ecs/{args.service_name}")

    print("\n4. Security groups...")
    sg_alb_id, sg_task_id = ensure_service_security_groups(
        ec2,
        vpc_id,
        args.service_name,
        DASHBOARD_API_PORT,
        alb_description="ALB for dashboard_api",
        task_description="ECS tasks for dashboard_api",
        egress_description="All outbound (S3, agent)",
    )

    print("\n5. Application Load Balancer...")
    alb_arn, alb_dns = ensure_alb(
        elbv2, f"{args.service_name}-alb", subnet_ids, sg_alb_id, idle_timeout_seconds=ALB_IDLE_TIMEOUT
    )

    print("\n6. Target group...")
    tg_arn = ensure_target_group(
        elbv2,
        f"{args.service_name}-tg",
        vpc_id,
        DASHBOARD_API_PORT,
        health_check_path=HEALTH_CHECK_PATH,
        health_check_interval=TG_HEALTH_CHECK_INTERVAL,
        healthy_threshold=TG_HEALTHY_THRESHOLD,
        deregistration_delay=TG_DEREGISTRATION_DELAY,
    )

    print("\n7. Listener...")
    ensure_http_listener(elbv2, alb_arn, tg_arn)

    print("\n7b. ECS task execution role + inline policy...")
    exec_role_arn = ensure_ecs_task_execution_role(iam, account_id)
    iam.put_role_policy(
        RoleName=ECS_TASK_EXECUTION_ROLE_NAME,
        PolicyName=S3_POLICY_NAME,
        PolicyDocument=json.dumps(S3_POLICY),
    )
    print(f"  Attached S3 RW + CloudWatch policy: s3://{S3_BUCKET}/")

    print("\n8. Task definition...")
    env_vars = [
        {"name": "DASHBOARD_CONFIG_DIR", "value": "/app/src/dashboards"},
        {"name": "DASHBOARD_FRONTEND_DIST", "value": "/app/frontend/dist"},
        {"name": "DASHBOARD_SERVICE_NAME", "value": args.service_name},
    ]
    # The SPA is served from this same ALB origin, so it needs no CORS entry;
    # default CORS to the ALB origin for out-of-band callers (e.g. local dev
    # pointed at the deployed API). Override with --cors-origins.
    env_vars.append({"name": "DASHBOARD_API_CORS", "value": args.cors_origins or f"http://{alb_dns}"})

    agent_api_url = args.agent_api_url
    if not agent_api_url:
        try:
            agent_albs = elbv2.describe_load_balancers(Names=["ai-chat-agent-alb"])["LoadBalancers"]
            if agent_albs:
                agent_api_url = f"http://{agent_albs[0]['DNSName']}"
                print(f"  Auto-detected AI agent ALB: {agent_api_url}")
        except ClientError:
            pass
    if agent_api_url:
        env_vars.append({"name": "AGENT_API_URL", "value": agent_api_url})
    else:
        print(
            "  Warning: AGENT_API_URL not set. Agent-backed endpoints (Phase 3+) "
            "fall back to localhost:8051. Deploy ai-chat-agent first or pass --agent-api-url."
        )

    task_def = {
        "family": args.service_name,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": str(args.task_cpu),
        "memory": str(args.task_memory),
        "executionRoleArn": exec_role_arn,
        "taskRoleArn": exec_role_arn,
        "containerDefinitions": [
            {
                "name": "dashboard-api",
                "image": ecr_uri,
                "portMappings": [{"containerPort": DASHBOARD_API_PORT, "protocol": "tcp"}],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": "dashboard-api",
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

    print("\n9. ECS service...")
    create_or_update_service(
        ecs,
        cluster=ECS_CLUSTER_NAME,
        service_name=args.service_name,
        container_name="dashboard-api",
        container_port=DASHBOARD_API_PORT,
        tg_arn=tg_arn,
        subnet_ids=subnet_ids,
        sg_task_id=sg_task_id,
        desired_count=args.desired_count,
    )

    if args.wait:
        wait_for_service_stable(ecs, ECS_CLUSTER_NAME, args.service_name)
    else:
        print("\nSkipping wait. Check ECS console and CloudWatch logs if tasks fail.")

    print("\n" + "=" * 60)
    print("dashboard_api deployed successfully!")
    print(f"  SPA:     http://{alb_dns}/")
    print(f"  Health:  http://{alb_dns}{HEALTH_CHECK_PATH}")
    print(f"  Configs: http://{alb_dns}/api/data/configs")
    print(f"  Docs:    http://{alb_dns}/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
