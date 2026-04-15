#!/usr/bin/env python3
"""
Deploy the AI Chat Agent to AWS ECS Fargate with a public Application Load Balancer.

The agent runs as a standalone service, separate from the dashboard.
This gives fault-isolation: a chat agent crash or slow LLM call never
blocks the dashboard.

Prerequisites:
  1. Run `bash infra/docker_build_ai_agent.sh` to build and push the image to ECR.
  2. Ensure ecsTaskExecutionRole exists (shared with the dashboard).
  3. Store your LLM API key(s) in AWS Secrets Manager (or pass via env vars).

Usage:
  python infra/deploy_ai_agent_ecs.py
  python infra/deploy_ai_agent_ecs.py --build-first
  python infra/deploy_ai_agent_ecs.py --dry-run
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

# ---- Configuration (edit these directly) ------------------------------------
REGION = "us-west-2"
CLUSTER_NAME = "ai-team-dashboard-cluster"
SERVICE_NAME = "ai-chat-agent"
IMAGE_NAME = "bituslabs-ds-ai-agent"
AGENT_PORT = 8051
TASK_CPU = 512  # 0.5 vCPU
TASK_MEMORY = 1024  # 1 GB
DESIRED_COUNT = 1  # always keep 1 task running
CHAT_PROVIDER = "gemini"

HEALTH_CHECK_PATH = "/health"
TG_HEALTH_CHECK_INTERVAL = 300
TG_HEALTHY_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_account_id(sts_client) -> str:
    return sts_client.get_caller_identity()["Account"]


def get_default_vpc_id(ec2_client) -> str:
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError("No default VPC found.")
    return vpcs[0]["VpcId"]


def get_default_vpc_subnets(ec2_client, vpc_id: str) -> list[str]:
    subnets = ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    if not subnets:
        raise RuntimeError(f"No subnets found in VPC {vpc_id}")
    public = [s for s in subnets if s.get("MapPublicIpOnLaunch")]
    return [s["SubnetId"] for s in (public if public else subnets)]


def ensure_ecr_repo(ecr_client, repo_name: str) -> None:
    try:
        ecr_client.create_repository(repositoryName=repo_name)
        print(f"  Created ECR repository: {repo_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "RepositoryAlreadyExistsException":
            raise
        print(f"  ECR repository exists: {repo_name}")


ECS_TASK_EXECUTION_ROLE_NAME = "ecsTaskExecutionRole"
ECS_TASK_EXECUTION_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}
ECS_TASK_EXECUTION_MANAGED_POLICY = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"


def ensure_ecs_task_execution_role(iam_client, account_id: str) -> str:
    role_name = ECS_TASK_EXECUTION_ROLE_NAME
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    expected_trust = json.dumps(ECS_TASK_EXECUTION_TRUST_POLICY, sort_keys=True)

    try:
        resp = iam_client.get_role(RoleName=role_name)
        current_doc = resp["Role"]["AssumeRolePolicyDocument"]
        current_str = json.dumps(current_doc, sort_keys=True) if isinstance(current_doc, dict) else current_doc
        if current_str != expected_trust:
            print(f"  Updating trust policy for {role_name}...")
            iam_client.update_assume_role_policy(
                RoleName=role_name,
                PolicyDocument=json.dumps(ECS_TASK_EXECUTION_TRUST_POLICY),
            )
        else:
            print(f"  Role exists with correct trust: {role_arn}")

        attached = iam_client.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]
        if not any(p["PolicyArn"] == ECS_TASK_EXECUTION_MANAGED_POLICY for p in attached):
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=ECS_TASK_EXECUTION_MANAGED_POLICY)

        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    print(f"  Creating role {role_name}...")
    iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(ECS_TASK_EXECUTION_TRUST_POLICY),
        Description="Allows ECS tasks to pull images and write logs",
    )
    iam_client.attach_role_policy(RoleName=role_name, PolicyArn=ECS_TASK_EXECUTION_MANAGED_POLICY)
    print("  Created role, attached execution policy")
    return role_arn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy AI Chat Agent to ECS Fargate")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    parser.add_argument("--build-first", action="store_true", help="Run docker_build_ai_agent.sh before deploying")
    parser.add_argument("--wait", action="store_true", help="Wait for service to stabilize")
    args = parser.parse_args()

    if args.build_first and not args.dry_run:
        print("Building and pushing AI Agent Docker image...")
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "docker_build_ai_agent.sh")],
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
    resp = ecs.describe_clusters(clusters=[CLUSTER_NAME])
    if resp.get("failures") or not resp.get("clusters") or resp["clusters"][0].get("status") != "ACTIVE":
        raise RuntimeError(
            f"Cluster {CLUSTER_NAME} not found or not ACTIVE. " "Deploy the dashboard first to create the cluster."
        )
    print(f"  Using existing cluster: {CLUSTER_NAME}")

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

    # 7b. Execution role
    print("\n7b. ECS task execution role...")
    exec_role_arn = ensure_ecs_task_execution_role(iam, account_id)

    # 8. Task definition
    print("\n8. Task definition...")
    env_vars = [
        {"name": "CHAT_PROVIDER", "value": CHAT_PROVIDER},
    ]

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
        ],
    }

    ecs.register_task_definition(**task_def)
    print(f"  Registered task definition: {SERVICE_NAME}")

    # 9. ECS service
    print("\n9. ECS service...")
    try:
        ecs.create_service(
            cluster=CLUSTER_NAME,
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
                cluster=CLUSTER_NAME,
                service=SERVICE_NAME,
                taskDefinition=SERVICE_NAME,
                desiredCount=DESIRED_COUNT,
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
                cluster=CLUSTER_NAME,
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
