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
  python infra/deploy_ai_agent_ecs.py --region us-west-2 --cluster-name ai-team-dashboard-cluster

Options:
  --region              AWS region (default: us-west-2)
  --cluster-name        ECS cluster name (default: ai-team-dashboard-cluster)
  --service-name        ECS service name (default: ai-chat-agent)
  --vpc-id              VPC ID (default: use default VPC)
  --subnet-ids          Comma-separated subnet IDs (default: public subnets of default VPC)
  --task-cpu            Task CPU units (default: 512)
  --task-memory         Task memory MB (default: 1024)
  --desired-count       Number of tasks (default: 0 with scale-to-zero, else 1)
  --no-scale-to-zero    Disable scale-to-zero; keep 1 task always running
  --chat-provider       CHAT_PROVIDER env (openai | gemini, default: gemini)
  --chat-model          CHAT_MODEL env (default: provider default)
  --build-first         Run docker_build_ai_agent.sh before deploying
  --dry-run             Print planned actions without executing
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

DEFAULT_REGION = "us-west-2"
IMAGE_NAME = "bituslabs-ds-ai-agent"
AGENT_PORT = 8051

HEALTH_CHECK_PATH = "/health"
TG_HEALTH_CHECK_INTERVAL = 300
TG_HEALTHY_THRESHOLD = 2
SCALE_IN_IDLE_MINUTES = 60  # scale to 0 after 1 hour idle (lighter than dashboard)


# ---------------------------------------------------------------------------
# Helpers (shared with deploy_dashboard_ecs.py — kept inline so this script
# is self-contained and runnable without importing the dashboard deploy)
# ---------------------------------------------------------------------------


def get_account_id(sts_client) -> str:
    return sts_client.get_caller_identity()["Account"]


def get_default_vpc_id(ec2_client) -> str:
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError("No default VPC found. Set --vpc-id and --subnet-ids explicitly.")
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
    print(f"  Created role, attached execution policy")
    return role_arn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy AI Chat Agent to ECS Fargate with public ALB")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--cluster-name",
        default="ai-team-dashboard-cluster",
        help="ECS cluster name (reuse the dashboard cluster for cost savings)",
    )
    parser.add_argument("--service-name", default="ai-chat-agent")
    parser.add_argument("--vpc-id", default=None)
    parser.add_argument("--subnet-ids", default=None, help="Comma-separated subnet IDs")
    parser.add_argument("--task-cpu", type=int, default=512, help="Task CPU units (0.5 vCPU)")
    parser.add_argument("--task-memory", type=int, default=1024, help="Task memory MB")
    parser.add_argument("--desired-count", type=int, default=None)
    parser.add_argument("--no-scale-to-zero", action="store_true")
    parser.add_argument(
        "--chat-provider", default="gemini", choices=["openai", "gemini"], help="LLM provider (default: gemini)"
    )
    parser.add_argument("--chat-model", default=None, help="Model name override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build-first", action="store_true")
    parser.add_argument("--create-cluster", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    args.use_existing_cluster = not args.create_cluster
    args.skip_wait = not args.wait
    args.scale_to_zero = not args.no_scale_to_zero
    if args.desired_count is None:
        args.desired_count = 0 if args.scale_to_zero else 1

    if args.build_first and not args.dry_run:
        print("Building and pushing AI Agent Docker image...")
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "docker_build_ai_agent.sh")],
            check=True,
            cwd=PROJECT_ROOT,
        )

    region = args.region
    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    ec2 = session.client("ec2")
    ecs = session.client("ecs")
    elbv2 = session.client("elbv2")
    logs = session.client("logs")
    iam = session.client("iam")
    as_client = session.client("application-autoscaling")

    account_id = get_account_id(sts)
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{IMAGE_NAME}:latest"

    print(f"Region: {region}, Account: {account_id}")
    print(f"Image: {ecr_uri}")

    vpc_id = args.vpc_id or get_default_vpc_id(ec2)
    subnet_ids = (
        [s.strip() for s in args.subnet_ids.split(",")] if args.subnet_ids else get_default_vpc_subnets(ec2, vpc_id)
    )
    print(f"VPC: {vpc_id}, Subnets: {subnet_ids}")

    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        print("  - ECS cluster (reuse), log group, security groups")
        print("  - ALB, target group, listener")
        print(f"  - Task definition ({args.service_name}), ECS service")
        return

    # 1. ECR
    print("\n1. ECR repository...")
    ecr = session.client("ecr")
    ensure_ecr_repo(ecr, IMAGE_NAME)

    # 2. ECS cluster (reuse the dashboard cluster by default)
    print("\n2. ECS cluster...")
    if args.use_existing_cluster:
        resp = ecs.describe_clusters(clusters=[args.cluster_name])
        if resp.get("failures") or not resp.get("clusters") or resp["clusters"][0].get("status") != "ACTIVE":
            raise RuntimeError(
                f"Cluster {args.cluster_name} not found or not ACTIVE. "
                "Deploy the dashboard first or use --create-cluster."
            )
        print(f"  Using existing cluster: {args.cluster_name}")
    else:
        try:
            ecs.create_cluster(clusterName=args.cluster_name)
            print(f"  Created cluster: {args.cluster_name}")
        except ClientError as e:
            if "AlreadyExists" in str(e):
                print(f"  Cluster exists: {args.cluster_name}")
            else:
                raise

    # 3. Log group
    log_group = f"/ecs/{args.service_name}"
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
    sg_name_alb = f"{args.service_name}-alb-sg"
    sg_name_task = f"{args.service_name}-task-sg"

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
                {"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "All outbound (LLM APIs)"}]}
            ],
        )
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise
        print("  Task outbound rule already exists, skipping")

    # 5. ALB
    print("\n5. Application Load Balancer...")
    lb_name = f"{args.service_name}-alb"
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
    tg_name = f"{args.service_name}-tg"
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
        {"name": "CHAT_PROVIDER", "value": args.chat_provider},
    ]
    if args.chat_model:
        env_vars.append({"name": "CHAT_MODEL", "value": args.chat_model})

    # API keys — In production, use AWS Secrets Manager or SSM Parameter Store.
    # For initial deployment you can pass them as env vars; the container
    # also reads .env if present in the image (not recommended for secrets).
    # Example: add {"name": "GOOGLE_API_KEY", "value": "<key>"} here.

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
                "name": "ai-agent",
                "image": ecr_uri,
                "portMappings": [{"containerPort": AGENT_PORT, "protocol": "tcp"}],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": "ai-agent",
                    },
                },
                "environment": env_vars,
                "healthCheck": {
                    "command": ["CMD-SHELL", f"curl -sf http://localhost:{AGENT_PORT}{HEALTH_CHECK_PATH} || exit 1"],
                    "interval": 10,
                    "timeout": 5,
                    "retries": 3,
                    "startPeriod": 30,
                },
            }
        ],
    }

    ecs.register_task_definition(**task_def)
    print(f"  Registered task definition: {args.service_name}")

    # 9. ECS service
    print("\n9. ECS service...")
    try:
        ecs.create_service(
            cluster=args.cluster_name,
            serviceName=args.service_name,
            taskDefinition=args.service_name,
            desiredCount=args.desired_count,
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
        print(f"  Created service: {args.service_name}")
    except ClientError as e:
        err = e.response.get("Error", {})
        code, msg = err.get("Code", ""), err.get("Message", "")
        if "ServiceAlreadyExists" in str(e) or (
            code == "InvalidParameterException" and "not idempotent" in (msg or "").lower()
        ):
            ecs.update_service(
                cluster=args.cluster_name,
                service=args.service_name,
                taskDefinition=args.service_name,
                desiredCount=args.desired_count,
                forceNewDeployment=True,
            )
            print(f"  Updated existing service: {args.service_name}")
        else:
            raise

    # 9b. Auto scaling
    if args.scale_to_zero:
        print("\n9b. Application Auto Scaling (scale to 0 when idle)...")
        resource_id = f"service/{args.cluster_name}/{args.service_name}"
        alb_dim = alb_arn.split(":")[-1].replace("loadbalancer/", "")

        try:
            as_client.register_scalable_target(
                ServiceNamespace="ecs",
                ResourceId=resource_id,
                ScalableDimension="ecs:service:DesiredCount",
                MinCapacity=0,
                MaxCapacity=1,
            )
            print("  Registered scalable target (min=0, max=1)")

            cw = session.client("cloudwatch")

            # Scale out on ALB 503 (no running tasks)
            resp_out = as_client.put_scaling_policy(
                ServiceNamespace="ecs",
                ResourceId=resource_id,
                ScalableDimension="ecs:service:DesiredCount",
                PolicyName=f"{args.service_name}-scale-out",
                PolicyType="StepScaling",
                StepScalingPolicyConfiguration={
                    "AdjustmentType": "ChangeInCapacity",
                    "Cooldown": 60,
                    "MetricAggregationType": "Average",
                    "StepAdjustments": [{"MetricIntervalLowerBound": 0.0, "ScalingAdjustment": 1}],
                },
            )
            policy_arn_out = resp_out["PolicyARN"]
            cw.put_metric_alarm(
                AlarmName=f"{args.service_name}-scale-out-on-503",
                MetricName="HTTPCode_ELB_503_Count",
                Namespace="AWS/ApplicationELB",
                Dimensions=[{"Name": "LoadBalancer", "Value": alb_dim}],
                Statistic="Sum",
                Period=60,
                EvaluationPeriods=1,
                Threshold=1.0,
                ComparisonOperator="GreaterThanOrEqualToThreshold",
                AlarmActions=[policy_arn_out],
            )
            print("  Scale-out: 503 -> add 1 task")

            # Scale in after idle period
            resp_in = as_client.put_scaling_policy(
                ServiceNamespace="ecs",
                ResourceId=resource_id,
                ScalableDimension="ecs:service:DesiredCount",
                PolicyName=f"{args.service_name}-scale-in",
                PolicyType="StepScaling",
                StepScalingPolicyConfiguration={
                    "AdjustmentType": "ChangeInCapacity",
                    "Cooldown": 300,
                    "MetricAggregationType": "Average",
                    "StepAdjustments": [{"MetricIntervalUpperBound": 0.0, "ScalingAdjustment": -1}],
                },
            )
            policy_arn_in = resp_in["PolicyARN"]
            scale_in_period = 60
            scale_in_evals = (SCALE_IN_IDLE_MINUTES * 60) // scale_in_period
            tg_dim = tg_arn.split(":")[-1]
            cw.put_metric_alarm(
                AlarmName=f"{args.service_name}-scale-in-on-idle",
                MetricName="RequestCount",
                Namespace="AWS/ApplicationELB",
                Dimensions=[
                    {"Name": "TargetGroup", "Value": tg_dim},
                    {"Name": "LoadBalancer", "Value": alb_dim},
                ],
                Statistic="Sum",
                Period=scale_in_period,
                EvaluationPeriods=scale_in_evals,
                Threshold=1.0,
                ComparisonOperator="LessThanThreshold",
                TreatMissingData="breaching",
                AlarmActions=[policy_arn_in],
            )
            print(f"  Scale-in: no requests for {SCALE_IN_IDLE_MINUTES} min -> remove 1 task (min 0)")
        except ClientError as e:
            print(f"  Warning: Auto Scaling setup failed: {e}")
            print("  Configure manually in ECS Console -> Service -> Auto Scaling")

    # Wait
    if args.skip_wait:
        print("\nSkipping wait. Check ECS console and CloudWatch logs if tasks fail.")
    else:
        print("\nWaiting for service to stabilize (up to 5 min)...")
        try:
            waiter = ecs.get_waiter("services_stable")
            waiter.wait(
                cluster=args.cluster_name,
                services=[args.service_name],
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
