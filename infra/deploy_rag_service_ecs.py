#!/usr/bin/env python3
"""
Deploy the RAG service to AWS ECS Fargate with a public Application Load Balancer.

Mirrors infra/deploy_ai_agent_ecs.py — same VPC / cluster / execution
role patterns. The RAG service is its own ECS service so its lifecycle
(daily index reload, FastAPI serving) is independent of the chat agent
or dashboard.

The Faiss artifact lives in S3 at bituslabs_ds.config.DEFAULT_RAG_INDEX_URI;
the task role gets read access to that prefix.

Prerequisites:
  1. Run `bash infra/docker_build_rag_service.sh` to build/push the image.
  2. ECS cluster (infra/ecs_helpers.ECS_CLUSTER_NAME) must already exist —
     it's created by the dashboard deploy script.
  3. AWS Secrets Manager entry "ai-dashboard_ai_agent" must contain
     GOOGLE_API_KEY (the embedder reads it via dashboards.chat_agent._get_secret).

Usage:
  python infra/deploy_rag_service_ecs.py
  python infra/deploy_rag_service_ecs.py --build-first
  python infra/deploy_rag_service_ecs.py --dry-run
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
PROJECT_ROOT = SCRIPT_DIR.parent

from ecs_helpers import (  # noqa: E402  (sibling module on sys.path[0])
    ECS_CLUSTER_NAME,
    ECS_TASK_EXECUTION_ROLE_NAME,
    ensure_ecr_repo,
    ensure_ecs_task_execution_role,
    get_account_id,
    get_default_vpc_id,
    get_default_vpc_subnets,
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
    parser.add_argument("--build-first", action="store_true", help="Run docker_build_rag_service.sh before deploying")
    parser.add_argument("--wait", action="store_true", help="Wait for service to stabilize")
    args = parser.parse_args()

    if args.build_first and not args.dry_run:
        print("Building and pushing RAG service Docker image...")
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "docker_build_rag_service.sh")],
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
    ecr = session.client("ecr")
    ensure_ecr_repo(ecr, IMAGE_NAME)

    # 2. ECS cluster (must already exist — created by the dashboard deploy)
    print("\n2. ECS cluster...")
    resp = ecs.describe_clusters(clusters=[ECS_CLUSTER_NAME])
    if resp.get("failures") or not resp.get("clusters") or resp["clusters"][0].get("status") != "ACTIVE":
        raise RuntimeError(
            f"Cluster {ECS_CLUSTER_NAME} not found or not ACTIVE. Deploy the dashboard first to create it."
        )
    print(f"  Using existing cluster: {ECS_CLUSTER_NAME}")

    # 3. Log group
    log_group = f"/ecs/{SERVICE_NAME}"
    print(f"\n3. Log group: {log_group}")
    try:
        logs.create_log_group(logGroupName=log_group)
        print(f"  Created log group: {log_group}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
        print(f"  Log group exists: {log_group}")

    # 4. Security groups
    print("\n4. Security groups...")
    sg_name_alb = f"{SERVICE_NAME}-alb-sg"
    sg_name_task = f"{SERVICE_NAME}-task-sg"

    try:
        sg_alb = ec2.create_security_group(GroupName=sg_name_alb, Description="ALB for RAG service", VpcId=vpc_id)
        sg_alb_id = sg_alb["GroupId"]
        print(f"  Created ALB security group: {sg_alb_id}")
    except ClientError as e:
        if "InvalidGroup.Duplicate" not in str(e):
            raise
        sg_alb_id = ec2.describe_security_groups(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [sg_name_alb]}]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  ALB security group exists: {sg_alb_id}")

    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_alb_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTP public"}],
                }
            ],
        )
        print("  Added ALB inbound rule (HTTP 80)")
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise

    try:
        sg_task = ec2.create_security_group(
            GroupName=sg_name_task, Description="ECS tasks for RAG service", VpcId=vpc_id
        )
        sg_task_id = sg_task["GroupId"]
        print(f"  Created task security group: {sg_task_id}")
    except ClientError as e:
        if "InvalidGroup.Duplicate" not in str(e):
            raise
        sg_task_id = ec2.describe_security_groups(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [sg_name_task]}]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  Task security group exists: {sg_task_id}")

    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_task_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": SERVICE_PORT,
                    "ToPort": SERVICE_PORT,
                    "UserIdGroupPairs": [{"GroupId": sg_alb_id, "Description": "From ALB"}],
                }
            ],
        )
        print(f"  Added task inbound rule ({SERVICE_PORT} from ALB)")
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise

    try:
        ec2.authorize_security_group_egress(
            GroupId=sg_task_id,
            IpPermissions=[
                {
                    "IpProtocol": "-1",
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "All outbound (Gemini, S3)"}],
                }
            ],
        )
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise
        print("  Task outbound rule already exists, skipping")

    # 5. ALB
    print("\n5. Application Load Balancer...")
    lb_name = f"{SERVICE_NAME}-alb"
    try:
        alb = elbv2.create_load_balancer(
            Name=lb_name,
            Subnets=subnet_ids,
            SecurityGroups=[sg_alb_id],
            Scheme="internet-facing",
            Type="application",
            Tags=[{"Key": "Name", "Value": lb_name}],
        )
        alb_arn = alb["LoadBalancers"][0]["LoadBalancerArn"]
        alb_dns = alb["LoadBalancers"][0]["DNSName"]
        print(f"  Created ALB: {alb_dns}")
    except ClientError as e:
        if "DuplicateLoadBalancerName" not in str(e):
            raise
        albs = elbv2.describe_load_balancers(Names=[lb_name])["LoadBalancers"]
        alb_arn = albs[0]["LoadBalancerArn"]
        alb_dns = albs[0]["DNSName"]
        print(f"  ALB exists: {alb_dns}")

    waiter = elbv2.get_waiter("load_balancer_available")
    waiter.wait(LoadBalancerArns=[alb_arn])
    print("  ALB is active")

    # 6. Target group
    print("\n6. Target group...")
    tg_name = f"{SERVICE_NAME}-tg"
    try:
        tg = elbv2.create_target_group(
            Name=tg_name,
            Protocol="HTTP",
            Port=SERVICE_PORT,
            VpcId=vpc_id,
            TargetType="ip",
            HealthCheckProtocol="HTTP",
            HealthCheckPath=HEALTH_CHECK_PATH,
            HealthCheckIntervalSeconds=TG_HEALTH_CHECK_INTERVAL,
            HealthyThresholdCount=TG_HEALTHY_THRESHOLD,
            UnhealthyThresholdCount=3,
        )
        tg_arn = tg["TargetGroups"][0]["TargetGroupArn"]
        print(f"  Created target group: {tg_arn}")
    except ClientError as e:
        if "DuplicateTargetGroupName" not in str(e):
            raise
        tgs = elbv2.describe_target_groups(Names=[tg_name])["TargetGroups"]
        tg_arn = tgs[0]["TargetGroupArn"]
        print(f"  Target group exists: {tg_arn}")

    elbv2.modify_target_group(
        TargetGroupArn=tg_arn,
        HealthCheckPath=HEALTH_CHECK_PATH,
        HealthCheckIntervalSeconds=TG_HEALTH_CHECK_INTERVAL,
        HealthyThresholdCount=TG_HEALTHY_THRESHOLD,
    )

    # 7. Listener
    print("\n7. Listener...")
    try:
        elbv2.create_listener(
            LoadBalancerArn=alb_arn,
            Protocol="HTTP",
            Port=80,
            DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}],
        )
        print("  Created HTTP listener on port 80")
    except ClientError as e:
        if "Duplicate" not in str(e) and "ResourceInUse" not in str(e):
            raise
        print("  Listener exists")

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

    resp_td = ecs.register_task_definition(**task_def)
    new_revision = resp_td["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered task definition: {SERVICE_NAME} (revision {resp_td['taskDefinition']['revision']})")

    old_revisions = ecs.list_task_definitions(familyPrefix=SERVICE_NAME, status="ACTIVE")["taskDefinitionArns"]
    for old_arn in old_revisions:
        if old_arn != new_revision:
            ecs.deregister_task_definition(taskDefinition=old_arn)
            rev = old_arn.split(":")[-1]
            print(f"  Deregistered old revision: {rev}")

    # 9. ECS service
    print("\n9. ECS service...")
    try:
        ecs.create_service(
            cluster=ECS_CLUSTER_NAME,
            serviceName=SERVICE_NAME,
            taskDefinition=SERVICE_NAME,
            desiredCount=DESIRED_COUNT,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": subnet_ids,
                    "securityGroups": [sg_task_id],
                    "assignPublicIp": "ENABLED",
                }
            },
            loadBalancers=[
                {
                    "targetGroupArn": tg_arn,
                    "containerName": "rag-service",
                    "containerPort": SERVICE_PORT,
                }
            ],
        )
        print(f"  Created service: {SERVICE_NAME}")
    except ClientError as e:
        err = e.response.get("Error", {})
        code, msg = err.get("Code", ""), err.get("Message", "")
        if "ServiceAlreadyExists" in str(e) or (
            code == "InvalidParameterException" and "not idempotent" in (msg or "").lower()
        ):
            ecs.update_service(
                cluster=ECS_CLUSTER_NAME,
                service=SERVICE_NAME,
                taskDefinition=SERVICE_NAME,
                desiredCount=DESIRED_COUNT,
                deploymentConfiguration={
                    "minimumHealthyPercent": 100,
                    "maximumPercent": 200,
                },
                forceNewDeployment=True,
            )
            print(f"  Updated existing service: {SERVICE_NAME}")
        else:
            raise

    if not args.wait:
        print("\nSkipping wait. Check ECS console and CloudWatch logs if tasks fail.")
    else:
        print("\nWaiting for service to stabilize (up to 5 min)...")
        try:
            waiter = ecs.get_waiter("services_stable")
            waiter.wait(
                cluster=ECS_CLUSTER_NAME,
                services=[SERVICE_NAME],
                WaiterConfig={"Delay": 10, "MaxAttempts": 30},
            )
            print("  Service is stable")
        except Exception as e:
            print(f"\n  Warning: Service did not stabilize: {e}")

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
