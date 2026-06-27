#!/usr/bin/env python3
"""
Deploy the AI Chat Agent to AWS ECS Fargate with a public Application Load Balancer.

The agent runs as a standalone service, separate from the dashboard.
This gives fault-isolation: a chat agent crash or slow LLM call never
blocks the dashboard.

Prerequisites:
  1. Run `bash infra/ai_agent/build.sh` to build and push the image to ECR.
  2. Ensure ecsTaskExecutionRole exists (shared with the dashboard).
  3. Store your LLM API key(s) in AWS Secrets Manager (or pass via env vars).

Usage:
  python infra/ai_agent/deploy_ecs.py
  python infra/ai_agent/deploy_ecs.py --build-first
  python infra/ai_agent/deploy_ecs.py --dry-run
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
SERVICE_NAME = "ai-chat-agent"
IMAGE_NAME = "bituslabs-ds-ai-agent"
AGENT_PORT = 8051
TASK_CPU = 512  # 0.5 vCPU
TASK_MEMORY = 1024  # 1 GB
DESIRED_COUNT = 1  # always keep 1 task running
CHAT_PROVIDER = "gemini"

# API keys are fetched at runtime by the app from AWS Secrets Manager
# (secret name: "ai-dashboard_ai_agent"). No need to inject them here.

HEALTH_CHECK_PATH = "/health"
TG_HEALTH_CHECK_INTERVAL = 300
TG_HEALTHY_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy AI Chat Agent to ECS Fargate")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    parser.add_argument("--build-first", action="store_true", help="Run infra/ai_agent/build.sh before deploying")
    parser.add_argument("--wait", action="store_true", help="Wait for service to stabilize")
    parser.add_argument(
        "--rag-service-url",
        default=None,
        help="RAG_SERVICE_URL the agent uses to call /retrieve. "
        "Auto-detected from the rag-service ALB if not provided.",
    )
    args = parser.parse_args()

    if args.build_first and not args.dry_run:
        print("Building and pushing AI Agent Docker image...")
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "build.sh")],
            check=True,
            cwd=PROJECT_ROOT,
        )

    session = boto3.Session(region_name=REGION)
    sts = session.client("sts")
    ec2 = session.client("ec2")
    ecs = session.client("ecs")
    elbv2 = session.client("elbv2")
    logs = session.client("logs")
    iam = session.client("iam")

    account_id = get_account_id(sts)
    ecr_uri = f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/{IMAGE_NAME}:latest"

    print(f"Region: {REGION}, Account: {account_id}")
    print(f"Image: {ecr_uri}")

    vpc_id = get_default_vpc_id(ec2)
    subnet_ids = get_default_vpc_subnets(ec2, vpc_id)
    print(f"VPC: {vpc_id}, Subnets: {subnet_ids}")

    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        print("  - Log group, security groups")
        print("  - ALB, target group, listener")
        print(f"  - Task definition ({SERVICE_NAME}), ECS service")
        return

    # 1. ECR
    print("\n1. ECR repository...")
    ensure_ecr_repo(session.client("ecr"), IMAGE_NAME)

    # 2. ECS cluster (reuse the dashboard cluster)
    print("\n2. ECS cluster...")
    require_active_cluster(ecs)

    # 3. Log group
    print(f"\n3. Log group: /ecs/{SERVICE_NAME}")
    log_group = ensure_log_group(logs, f"/ecs/{SERVICE_NAME}")

    # 4. Security groups
    print("\n4. Security groups...")
    sg_alb_id, sg_task_id = ensure_service_security_groups(
        ec2,
        vpc_id,
        SERVICE_NAME,
        AGENT_PORT,
        alb_description="ALB for AI chat agent",
        task_description="ECS tasks for AI chat agent",
        egress_description="All outbound (LLM APIs)",
    )

    # 5. ALB
    # Idle timeout: the chat/report endpoints can take 30s+ per LLM call, and the
    # dashboard fan-outs N description requests for an N-figure report. With the
    # default 60s, queued requests return 504 even though the agent eventually
    # finishes. 300s buffers ~10 stacked calls.
    print("\n5. Application Load Balancer...")
    alb_arn, alb_dns = ensure_alb(elbv2, f"{SERVICE_NAME}-alb", subnet_ids, sg_alb_id, idle_timeout_seconds=300)

    # 6. Target group
    print("\n6. Target group...")
    tg_arn = ensure_target_group(
        elbv2,
        f"{SERVICE_NAME}-tg",
        vpc_id,
        AGENT_PORT,
        health_check_path=HEALTH_CHECK_PATH,
        health_check_interval=TG_HEALTH_CHECK_INTERVAL,
        healthy_threshold=TG_HEALTHY_THRESHOLD,
    )

    # 7. Listener
    print("\n7. Listener...")
    ensure_http_listener(elbv2, alb_arn, tg_arn)

    # 7b. Execution role + Secrets Manager access for the task
    print("\n7b. ECS task execution role...")
    exec_role_arn = ensure_ecs_task_execution_role(iam, account_id)
    iam.put_role_policy(
        RoleName=ECS_TASK_EXECUTION_ROLE_NAME,
        PolicyName="ecsTaskRole-secrets-ai-agent",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["secretsmanager:GetSecretValue"],
                        "Resource": [f"arn:aws:secretsmanager:{REGION}:{account_id}:secret:ai-dashboard_ai_agent*"],
                    }
                ],
            }
        ),
    )
    print("  Attached Secrets Manager read policy for ai-dashboard_ai_agent")

    # 8. Task definition
    print("\n8. Task definition...")
    env_vars = [
        {"name": "CHAT_PROVIDER", "value": CHAT_PROVIDER},
    ]

    # Resolve the RAG service URL (auto-detect from its ALB if not provided)
    # so the agent's rag_service.client can call /retrieve.
    rag_service_url = args.rag_service_url
    if not rag_service_url:
        try:
            rag_albs = elbv2.describe_load_balancers(Names=["rag-service-alb"])["LoadBalancers"]
            if rag_albs:
                rag_service_url = f"http://{rag_albs[0]['DNSName']}"
                print(f"  Auto-detected RAG service ALB: {rag_service_url}")
        except ClientError:
            pass
    if rag_service_url:
        env_vars.append({"name": "RAG_SERVICE_URL", "value": rag_service_url})
    else:
        print(
            "  Warning: RAG_SERVICE_URL not set. /retrieve will fall back to "
            "localhost:8052 (which won't exist on Fargate). Deploy rag-service "
            "first or pass --rag-service-url."
        )

    container_def = {
        "name": "ai-agent",
        "image": ecr_uri,
        "portMappings": [{"containerPort": AGENT_PORT, "protocol": "tcp"}],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group,
                "awslogs-region": REGION,
                "awslogs-stream-prefix": "ai-agent",
            },
        },
        "environment": env_vars,
        "healthCheck": {
            "command": [
                "CMD-SHELL",
                f"curl -sf http://localhost:{AGENT_PORT}{HEALTH_CHECK_PATH} || exit 1",
            ],
            "interval": 10,
            "timeout": 5,
            "retries": 3,
            "startPeriod": 30,
        },
    }
    task_def = {
        "family": SERVICE_NAME,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": str(TASK_CPU),
        "memory": str(TASK_MEMORY),
        "executionRoleArn": exec_role_arn,
        "taskRoleArn": exec_role_arn,
        "containerDefinitions": [container_def],
    }

    register_task_definition(ecs, task_def)

    # 9. ECS service
    print("\n9. ECS service...")
    create_or_update_service(
        ecs,
        cluster=ECS_CLUSTER_NAME,
        service_name=SERVICE_NAME,
        container_name="ai-agent",
        container_port=AGENT_PORT,
        tg_arn=tg_arn,
        subnet_ids=subnet_ids,
        sg_task_id=sg_task_id,
        desired_count=DESIRED_COUNT,
    )

    if args.wait:
        wait_for_service_stable(ecs, ECS_CLUSTER_NAME, SERVICE_NAME)
    else:
        print("\nSkipping wait. Check ECS console and CloudWatch logs if tasks fail.")

    print("\n" + "=" * 60)
    print("AI Chat Agent deployed successfully!")
    print(f"  API URL:  http://{alb_dns}")
    print(f"  Health:   http://{alb_dns}/health")
    print(f"  Chat:     POST http://{alb_dns}/api/chat")
    print(f"  Docs:     http://{alb_dns}/docs")
    print("=" * 60)
    print("\nTo point the dashboard at this agent, set CHAT_API_URL:")
    print(f"  export CHAT_API_URL=http://{alb_dns}")


if __name__ == "__main__":
    main()
