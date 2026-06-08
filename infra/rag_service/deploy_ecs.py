#!/usr/bin/env python3
"""
Deploy the RAG service to AWS ECS Fargate with a public Application Load Balancer.

Mirrors infra/ai_agent/deploy_ecs.py — same VPC / cluster / execution
role patterns. The RAG service is its own ECS service so its lifecycle
(daily index reload, FastAPI serving) is independent of the chat agent
or dashboard.

The Faiss artifact lives in S3 at bituslabs_ds.config.DEFAULT_RAG_INDEX_URI;
the task role gets read access to that prefix.

Prerequisites:
  1. Run `bash infra/rag_service/build.sh` to build/push the image.
  2. ECS cluster (infra/shared/ecs_helpers.ECS_CLUSTER_NAME) must already exist —
     it's created by the dashboard deploy script.
  3. AWS Secrets Manager entry "ai-dashboard_ai_agent" must contain
     GOOGLE_API_KEY (the embedder reads it via ai_agent.chat_agent._get_secret).

Usage:
  python infra/rag_service/deploy_ecs.py
  python infra/rag_service/deploy_ecs.py --build-first
  python infra/rag_service/deploy_ecs.py --dry-run
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
SERVICE_NAME = "rag-service"
IMAGE_NAME = "bituslabs-ds-rag-service"
SERVICE_PORT = 8052
TASK_CPU = 512  # 0.5 vCPU
TASK_MEMORY = 1024  # 1 GB — Faiss index is ~5 MB; headroom for embed-query tensors.
DESIRED_COUNT = 1

# S3 prefix where build_rag_index.py writes the Faiss artifact.
# Must match bituslabs_ds.config.DEFAULT_RAG_INDEX_URI.
S3_BUCKET = "bituslabs-team-ai"
RAG_INDEX_PREFIX = "rag"

# Same Secrets Manager entry the AI agent uses (contains GOOGLE_API_KEY).
SECRETS_MANAGER_NAME = "ai-dashboard_ai_agent"

HEALTH_CHECK_PATH = "/health"
TG_HEALTH_CHECK_INTERVAL = 30
TG_HEALTHY_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy RAG service to ECS Fargate")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    parser.add_argument("--build-first", action="store_true", help="Run infra/rag_service/build.sh before deploying")
    parser.add_argument("--wait", action="store_true", help="Wait for service to stabilize")
    args = parser.parse_args()

    if args.build_first and not args.dry_run:
        print("Building and pushing RAG service Docker image...")
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
        print(f"  - S3 read policy on s3://{S3_BUCKET}/{RAG_INDEX_PREFIX}/*")
        print(f"  - Secrets Manager read policy on {SECRETS_MANAGER_NAME}")
        return

    # 1. ECR
    print("\n1. ECR repository...")
    ensure_ecr_repo(session.client("ecr"), IMAGE_NAME)

    # 2. ECS cluster (must already exist — created by the dashboard deploy)
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
        SERVICE_PORT,
        alb_description="ALB for RAG service",
        task_description="ECS tasks for RAG service",
        egress_description="All outbound (Gemini, S3)",
    )

    # 5. ALB
    print("\n5. Application Load Balancer...")
    alb_arn, alb_dns = ensure_alb(elbv2, f"{SERVICE_NAME}-alb", subnet_ids, sg_alb_id)

    # 6. Target group
    print("\n6. Target group...")
    tg_arn = ensure_target_group(
        elbv2,
        f"{SERVICE_NAME}-tg",
        vpc_id,
        SERVICE_PORT,
        health_check_path=HEALTH_CHECK_PATH,
        health_check_interval=TG_HEALTH_CHECK_INTERVAL,
        healthy_threshold=TG_HEALTHY_THRESHOLD,
    )

    # 7. Listener
    print("\n7. Listener...")
    ensure_http_listener(elbv2, alb_arn, tg_arn)

    # 7b. Execution role + S3 read for the Faiss artifact + Secrets Manager read.
    print("\n7b. ECS task execution role + inline policies...")
    exec_role_arn = ensure_ecs_task_execution_role(iam, account_id)
    iam.put_role_policy(
        RoleName=ECS_TASK_EXECUTION_ROLE_NAME,
        PolicyName="ecsTaskRole-rag-service",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "ReadFaissArtifact",
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": [
                            f"arn:aws:s3:::{S3_BUCKET}",
                            f"arn:aws:s3:::{S3_BUCKET}/{RAG_INDEX_PREFIX}/*",
                        ],
                    },
                    {
                        "Sid": "ReadEmbeddingApiKey",
                        "Effect": "Allow",
                        "Action": ["secretsmanager:GetSecretValue"],
                        "Resource": [f"arn:aws:secretsmanager:{REGION}:{account_id}:secret:{SECRETS_MANAGER_NAME}*"],
                    },
                ],
            }
        ),
    )
    print(f"  Attached S3 read policy on s3://{S3_BUCKET}/{RAG_INDEX_PREFIX}/*")
    print(f"  Attached Secrets Manager read policy on {SECRETS_MANAGER_NAME}")

    # 8. Task definition
    print("\n8. Task definition...")
    env_vars = [
        {"name": "RAG_BACKEND", "value": "faiss"},
        {"name": "RAG_EMBEDDING_PROVIDER", "value": "gemini"},
        {"name": "AWS_REGION", "value": REGION},
    ]

    container_def = {
        "name": "rag-service",
        "image": ecr_uri,
        "portMappings": [{"containerPort": SERVICE_PORT, "protocol": "tcp"}],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group,
                "awslogs-region": REGION,
                "awslogs-stream-prefix": "rag-service",
            },
        },
        "environment": env_vars,
        "healthCheck": {
            "command": [
                "CMD-SHELL",
                f"curl -sf http://localhost:{SERVICE_PORT}{HEALTH_CHECK_PATH} || exit 1",
            ],
            "interval": 30,
            "timeout": 5,
            "retries": 3,
            "startPeriod": 60,
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
        container_name="rag-service",
        container_port=SERVICE_PORT,
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
    print("RAG service deployed successfully!")
    print(f"  Health:    http://{alb_dns}/health")
    print(f"  Retrieve:  POST http://{alb_dns}/retrieve")
    print(f"  Sources:   GET  http://{alb_dns}/sources")
    print("=" * 60)
    print("\nTo point the chat agent at this RAG service, set RAG_SERVICE_URL:")
    print(f"  export RAG_SERVICE_URL=http://{alb_dns}")


if __name__ == "__main__":
    main()
