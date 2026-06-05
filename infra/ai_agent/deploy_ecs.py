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
    ensure_ecr_repo,
    ensure_ecs_task_execution_role,
    get_account_id,
    get_default_vpc_id,
    get_default_vpc_subnets,
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
    ecr = session.client("ecr")
    ensure_ecr_repo(ecr, IMAGE_NAME)

    # 2. ECS cluster (reuse the dashboard cluster)
    print("\n2. ECS cluster...")
    resp = ecs.describe_clusters(clusters=[ECS_CLUSTER_NAME])
    if resp.get("failures") or not resp.get("clusters") or resp["clusters"][0].get("status") != "ACTIVE":
        raise RuntimeError(
            f"Cluster {ECS_CLUSTER_NAME} not found or not ACTIVE. " "Deploy the dashboard first to create the cluster."
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
        sg_alb = ec2.create_security_group(GroupName=sg_name_alb, Description="ALB for AI chat agent", VpcId=vpc_id)
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
            GroupName=sg_name_task, Description="ECS tasks for AI chat agent", VpcId=vpc_id
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
                    "FromPort": AGENT_PORT,
                    "ToPort": AGENT_PORT,
                    "UserIdGroupPairs": [{"GroupId": sg_alb_id, "Description": "From ALB"}],
                }
            ],
        )
        print(f"  Added task inbound rule ({AGENT_PORT} from ALB)")
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise

    try:
        ec2.authorize_security_group_egress(
            GroupId=sg_task_id,
            IpPermissions=[
                {
                    "IpProtocol": "-1",
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "All outbound (LLM APIs)"}],
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

    # Idle timeout: the chat/report endpoints can take 30s+ per LLM call,
    # and the dashboard fan-outs N description requests for an N-figure
    # report. With the default 60s, queued requests return 504 even though
    # the agent eventually finishes. 300s buffers ~10 stacked calls.
    elbv2.modify_load_balancer_attributes(
        LoadBalancerArn=alb_arn,
        Attributes=[{"Key": "idle_timeout.timeout_seconds", "Value": "300"}],
    )
    print("  ALB idle_timeout set to 300s")

    # 6. Target group
    print("\n6. Target group...")
    tg_name = f"{SERVICE_NAME}-tg"
    try:
        tg = elbv2.create_target_group(
            Name=tg_name,
            Protocol="HTTP",
            Port=AGENT_PORT,
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

    resp_td = ecs.register_task_definition(**task_def)
    new_revision = resp_td["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered task definition: {SERVICE_NAME} (revision {resp_td['taskDefinition']['revision']})")

    # Deregister old revisions (keep only the new one)
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
                    "containerName": "ai-agent",
                    "containerPort": AGENT_PORT,
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

    # Wait
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
