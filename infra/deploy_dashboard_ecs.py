#!/usr/bin/env python3
"""
Deploy the Game Stats Dashboard to AWS ECS Fargate with a public Application Load Balancer.

Prerequisites:
  1. Run `bash infra/docker_build_dashboard.sh` to build and push the image to ECR.
  2. Ensure ecsTaskExecutionRole exists (ECS console creates it, or use the default).
  3. ecsTaskExecutionRole (or your task role) needs S3 read/write access for dashboard data:
     Add policy: s3:GetObject, s3:PutObject, s3:ListBucket on s3://bituslabs-team-ai/*
  4. Default: use existing cluster. Pass --create-cluster to create new (requires ecs:CreateCluster).
     IAM needs: ecs:*, elasticloadbalancing:*, ec2:*, logs:*, iam:PassRole/CreateRole/AttachRolePolicy/PutRolePolicy.

Usage:
  python infra/deploy_dashboard_ecs.py
  python infra/deploy_dashboard_ecs.py --region us-west-2

The cluster name is fixed in ``infra/ecs_helpers.py`` so all three deploy
scripts (dashboard, ai-agent, rag-service) target the same ECS cluster.

Options:
  --region              AWS region (default: us-west-2)
  --service-name        ECS service name (default: game-stats-dashboard)
  --vpc-id              VPC ID (default: use default VPC)
  --subnet-ids          Comma-separated subnet IDs (default: public subnets of default VPC)
  --task-cpu            Task CPU units (default: 512)
  --task-memory         Task memory MB (default: 1024)
  --desired-count       Number of tasks (default: 0 with scale-to-zero, else 1)
  --no-scale-to-zero    Disable scale-to-zero; keep 1 task always running
  --public-url          DASHBOARD_PUBLIC_URL env (default: ALB URL)
  --chat-api-url        CHAT_API_URL for the AI agent (auto-detected from ai-chat-agent ALB)
  --build-first         Run docker_build_dashboard.sh before deploying
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

# Project root (script lives in infra/)
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

# Defaults from config
DEFAULT_REGION = "us-west-2"
S3_BUCKET = "bituslabs-team-ai"
IMAGE_NAME = "bituslabs-ds-dashboard"
DASHBOARD_PORT = 8050

HEALTH_CHECK_PATH = "/health"
TG_HEALTH_CHECK_INTERVAL = 30  # fast health checks so new tasks become healthy quickly
TG_HEALTHY_THRESHOLD = 2  # 2 consecutive checks = ~60s to become healthy
TG_DEREGISTRATION_DELAY = 30  # seconds to drain old task before removing from ALB
SCALE_IN_IDLE_MINUTES = 180  # Scale to 0 after 3 hours with no user requests


S3_DASHBOARD_BUCKET = "bituslabs-team-ai"
S3_DASHBOARD_POLICY_NAME = "ecsTaskExecutionRole-s3-dashboard-rw"
S3_DASHBOARD_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::{S3_DASHBOARD_BUCKET}",
                f"arn:aws:s3:::{S3_DASHBOARD_BUCKET}/*",
            ],
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
    parser = argparse.ArgumentParser(description="Deploy Game Stats Dashboard to ECS Fargate with public ALB")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--service-name", default="game-stats-dashboard", help="ECS service name")
    parser.add_argument("--vpc-id", default=None, help="VPC ID (default: default VPC)")
    parser.add_argument(
        "--subnet-ids",
        default=None,
        help="Comma-separated subnet IDs (default: public subnets)",
    )
    parser.add_argument("--task-cpu", type=int, default=1024, help="Task CPU units (1 vCPU = 1024)")
    parser.add_argument("--task-memory", type=int, default=2048, help="Task memory MB")
    parser.add_argument(
        "--desired-count",
        type=int,
        default=None,
        help="Number of tasks (default: 0 with scale-to-zero, else 1)",
    )
    parser.add_argument(
        "--no-scale-to-zero",
        action="store_true",
        help="Disable scale-to-zero; keep 1 task always running",
    )
    parser.add_argument(
        "--public-url",
        default=None,
        help="DASHBOARD_PUBLIC_URL env (default: ALB URL)",
    )
    parser.add_argument(
        "--chat-api-url",
        default=None,
        help="CHAT_API_URL for the AI agent service (e.g. http://ai-chat-agent-alb-123.us-west-2.elb.amazonaws.com). "
        "Auto-detected from the ai-chat-agent ALB if not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing",
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        default=False,
        help="Run docker_build_dashboard.sh before deploying",
    )
    parser.add_argument(
        "--create-cluster",
        action="store_true",
        help=f"Create new cluster (default: use existing {ECS_CLUSTER_NAME})",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for service to stabilize (default: skip)",
    )
    args = parser.parse_args()
    args.use_existing_cluster = not args.create_cluster
    args.skip_wait = not args.wait
    args.scale_to_zero = not args.no_scale_to_zero
    if args.desired_count is None:
        args.desired_count = 0 if args.scale_to_zero else 1

    if args.build_first and not args.dry_run:
        print("Building and pushing Docker image...")
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "docker_build_dashboard.sh")],
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

    # Resolve VPC and subnets
    vpc_id = args.vpc_id or get_default_vpc_id(ec2)
    if args.subnet_ids:
        subnet_ids = [s.strip() for s in args.subnet_ids.split(",")]
    else:
        subnet_ids = get_default_vpc_subnets(ec2, vpc_id)
    print(f"VPC: {vpc_id}, Subnets: {subnet_ids}")

    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        print("  - ECS cluster, log group, security groups")
        print("  - ALB, target group, listener")
        print("  - Task definition, ECS service")
        return

    # 1. Ensure ECR repo
    print("\n1. ECR repository...")
    ecr = session.client("ecr")
    ensure_ecr_repo(ecr, IMAGE_NAME)

    # 2. ECS cluster
    print("\n2. ECS cluster...")
    if args.use_existing_cluster:
        resp = ecs.describe_clusters(clusters=[ECS_CLUSTER_NAME])
        if resp.get("failures"):
            raise RuntimeError(
                f"Cluster {ECS_CLUSTER_NAME} not found. "
                "Pass --create-cluster to create it, or update infra/ecs_helpers.py."
            )
        if not resp.get("clusters") or resp["clusters"][0].get("status") != "ACTIVE":
            raise RuntimeError(f"Cluster {ECS_CLUSTER_NAME} not found or not ACTIVE.")
        print(f"  Using existing cluster: {ECS_CLUSTER_NAME}")
    else:
        try:
            ecs.create_cluster(clusterName=ECS_CLUSTER_NAME)
            print(f"  Created cluster: {ECS_CLUSTER_NAME}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ClusterAlreadyExistsException":
                print(f"  Cluster exists: {ECS_CLUSTER_NAME}")
            elif e.response["Error"]["Code"] == "AccessDeniedException":
                print("\n  ! AccessDenied: You lack ecs:CreateCluster permission.")
                print(f"  Drop --create-cluster if {ECS_CLUSTER_NAME} already exists,")
                print("  or ask your admin to add ECS permissions (see script docstring for IAM policy).")
                sys.exit(1)
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
    vpc = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]
    sg_name_alb = f"{args.service_name}-alb-sg"
    sg_name_task = f"{args.service_name}-task-sg"

    try:
        sg_alb = ec2.create_security_group(
            GroupName=sg_name_alb,
            Description="ALB for dashboard",
            VpcId=vpc_id,
        )
        sg_alb_id = sg_alb["GroupId"]
        print(f"  Created ALB security group: {sg_alb_id}")
    except ClientError as e:
        if "InvalidGroup.Duplicate" not in str(e):
            raise
        sg_alb_id = ec2.describe_security_groups(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "group-name", "Values": [sg_name_alb]},
            ]
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
            GroupName=sg_name_task,
            Description="ECS tasks for dashboard",
            VpcId=vpc_id,
        )
        sg_task_id = sg_task["GroupId"]
        print(f"  Created task security group: {sg_task_id}")
    except ClientError as e:
        if "InvalidGroup.Duplicate" not in str(e):
            raise
        sg_task_id = ec2.describe_security_groups(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "group-name", "Values": [sg_name_task]},
            ]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  Task security group exists: {sg_task_id}")

    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_task_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": DASHBOARD_PORT,
                    "ToPort": DASHBOARD_PORT,
                    "UserIdGroupPairs": [{"GroupId": sg_alb_id, "Description": "From ALB"}],
                }
            ],
        )
        print("  Added task inbound rule (8050 from ALB)")
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise

    try:
        ec2.authorize_security_group_egress(
            GroupId=sg_task_id,
            IpPermissions=[
                {
                    "IpProtocol": "-1",
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "All outbound"}],
                }
            ],
        )
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise
        print("  Task outbound rule already exists, skipping")

    # 5. Application Load Balancer
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
        alb_hosted_zone_id = alb["LoadBalancers"][0]["CanonicalHostedZoneId"]
        print(f"  Created ALB: {alb_dns}")
    except ClientError as e:
        if "DuplicateLoadBalancerName" not in str(e):
            raise
        albs = elbv2.describe_load_balancers(Names=[lb_name])["LoadBalancers"]
        alb_arn = albs[0]["LoadBalancerArn"]
        alb_dns = albs[0]["DNSName"]
        alb_hosted_zone_id = albs[0]["CanonicalHostedZoneId"]
        print(f"  ALB exists: {alb_dns}")

    # Wait for ALB to be active
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
            Port=DASHBOARD_PORT,
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
    elbv2.modify_target_group_attributes(
        TargetGroupArn=tg_arn,
        Attributes=[
            {"Key": "deregistration_delay.timeout_seconds", "Value": str(TG_DEREGISTRATION_DELAY)},
        ],
    )

    # 7. Listener
    print("\n7. Listener...")
    try:
        elbv2.create_listener(
            LoadBalancerArn=alb_arn,
            Protocol="HTTP",
            Port=80,
            DefaultActions=[
                {
                    "Type": "forward",
                    "TargetGroupArn": tg_arn,
                }
            ],
        )
        print("  Created HTTP listener on port 80")
    except ClientError as e:
        if "Duplicate" not in str(e) and "ResourceInUse" not in str(e):
            raise
        print("  Listener exists")

    # 7b. Ensure ECS task execution role exists with correct trust policy,
    # then layer on dashboard-specific S3 read/write + CloudWatch metric access.
    print("\n7b. ECS task execution role + inline policies...")
    exec_role_arn = ensure_ecs_task_execution_role(iam, account_id)
    iam.put_role_policy(
        RoleName=ECS_TASK_EXECUTION_ROLE_NAME,
        PolicyName=S3_DASHBOARD_POLICY_NAME,
        PolicyDocument=json.dumps(S3_DASHBOARD_POLICY),
    )
    print(f"  Attached S3 RW + CloudWatch policy: s3://{S3_DASHBOARD_BUCKET}/")

    # 8. Task definition
    print("\n8. Task definition...")
    task_role_arn = exec_role_arn  # Same role; ensure it has S3 read for dashboard data

    env_vars = [
        {"name": "DASHBOARD_CONFIG_DIR", "value": "/app/src/dashboards"},
        {"name": "DASHBOARD_SERVICE_NAME", "value": args.service_name},
    ]
    public_url = args.public_url or f"http://{alb_dns}"
    if public_url:
        env_vars.append({"name": "DASHBOARD_PUBLIC_URL", "value": public_url})

    # Resolve AI chat agent URL so the dashboard can call it over HTTP
    chat_api_url = args.chat_api_url
    if not chat_api_url:
        try:
            agent_albs = elbv2.describe_load_balancers(Names=["ai-chat-agent-alb"])["LoadBalancers"]
            if agent_albs:
                chat_api_url = f"http://{agent_albs[0]['DNSName']}"
                print(f"  Auto-detected AI agent ALB: {chat_api_url}")
        except ClientError:
            pass
    if chat_api_url:
        env_vars.append({"name": "CHAT_API_URL", "value": chat_api_url})
    else:
        print(
            "  Warning: CHAT_API_URL not set. AI chat will not work until you "
            "deploy the ai-chat-agent service and redeploy the dashboard with --chat-api-url."
        )

    task_def = {
        "family": args.service_name,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": str(args.task_cpu),
        "memory": str(args.task_memory),
        "executionRoleArn": exec_role_arn,
        "taskRoleArn": task_role_arn,
        "containerDefinitions": [
            {
                "name": "dashboard",
                "image": ecr_uri,
                "portMappings": [{"containerPort": DASHBOARD_PORT, "protocol": "tcp"}],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": "dashboard",
                    },
                },
                "environment": env_vars,
                "healthCheck": {
                    "command": [
                        "CMD-SHELL",
                        f"curl -sf http://localhost:{DASHBOARD_PORT}{HEALTH_CHECK_PATH} || exit 1",
                    ],
                    "interval": 10,
                    "timeout": 5,
                    "retries": 3,
                    "startPeriod": 45,
                },
            }
        ],
    }

    resp_td = ecs.register_task_definition(**task_def)
    new_revision = resp_td["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered task definition: {args.service_name} (revision {resp_td['taskDefinition']['revision']})")

    # Deregister old revisions (keep only the new one)
    old_revisions = ecs.list_task_definitions(familyPrefix=args.service_name, status="ACTIVE")["taskDefinitionArns"]
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
            serviceName=args.service_name,
            taskDefinition=args.service_name,
            desiredCount=args.desired_count,
            launchType="FARGATE",
            deploymentConfiguration={
                "minimumHealthyPercent": 100,
                "maximumPercent": 200,
            },
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
                    "containerName": "dashboard",
                    "containerPort": DASHBOARD_PORT,
                }
            ],
        )
        print(f"  Created service: {args.service_name}")
    except ClientError as e:
        err = e.response.get("Error", {})
        code, msg = err.get("Code", ""), err.get("Message", "")
        if "ClusterNotFoundException" in str(e):
            raise RuntimeError(f"Cluster {ECS_CLUSTER_NAME} not found") from e
        if "ServiceAlreadyExists" in str(e) or (
            code == "InvalidParameterException" and "not idempotent" in (msg or "").lower()
        ):
            ecs.update_service(
                cluster=ECS_CLUSTER_NAME,
                service=args.service_name,
                taskDefinition=args.service_name,
                desiredCount=args.desired_count,
                deploymentConfiguration={
                    "minimumHealthyPercent": 100,
                    "maximumPercent": 200,
                },
                forceNewDeployment=True,
            )
            print(f"  Updated existing service: {args.service_name}")
        else:
            raise

    # 9b. Application Auto Scaling (scale to 0 when idle)
    if args.scale_to_zero:
        print("\n9b. Application Auto Scaling (scale to 0 when idle)...")
        resource_id = f"service/{ECS_CLUSTER_NAME}/{args.service_name}"
        # Extract dimension values for ALB RequestCount metric
        tg_dim = tg_arn.split(":")[-1]  # targetgroup/name/id
        alb_dim = alb_arn.split(":")[-1].replace("loadbalancer/", "")  # app/name/id

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

            # Delete any old scale-out alarms (avoids "multiple alarms" error)
            old_scale_out_alarms = [
                f"{args.service_name}-scale-out-on-request",  # legacy name
                f"{args.service_name}-scale-out-on-503",
            ]
            for name in old_scale_out_alarms:
                try:
                    cw.delete_alarms(AlarmNames=[name])
                    print(f"  Removed old alarm: {name}")
                except ClientError:
                    pass

            # Scale out: create policy first, then alarm with AlarmActions=[policy ARN]
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
            alarm_name_out = f"{args.service_name}-scale-out-on-503"
            # Use HTTPCode_ELB_503_Count: RequestCount is NOT incremented when there
            # are 0 targets (ALB returns 503), so we must scale out on 503 instead.
            cw.put_metric_alarm(
                AlarmName=alarm_name_out,
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
            print("  Scale-out: 503 (no targets) -> add 1 task")

            # Scale in: when no user requests for 3 hours (excludes /health), scale to 0
            # Uses custom metric UserRequestCount from dashboard app (excludes health checks)
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
            alarm_name_in = f"{args.service_name}-scale-in-on-idle"
            scale_in_period_sec = 60
            scale_in_eval_periods = (SCALE_IN_IDLE_MINUTES * 60) // scale_in_period_sec  # 180
            cw.put_metric_alarm(
                AlarmName=alarm_name_in,
                MetricName="UserRequestCount",
                Namespace="Dashboard",
                Dimensions=[{"Name": "Service", "Value": args.service_name}],
                Statistic="Sum",
                Period=scale_in_period_sec,
                EvaluationPeriods=scale_in_eval_periods,
                Threshold=1.0,
                ComparisonOperator="LessThanThreshold",
                TreatMissingData="breaching",
                AlarmActions=[policy_arn_in],
            )
            print(f"  Scale-in: no user requests for {SCALE_IN_IDLE_MINUTES} min -> remove 1 task (min 0)")
        except ClientError as e:
            print(f"  Warning: Auto Scaling setup failed: {e}")
            print("  Configure manually in ECS Console -> Service -> Auto Scaling")

    # Wait for service to stabilize
    if args.skip_wait:
        print("\nSkipping wait (--skip-wait). Check ECS console and CloudWatch logs if tasks fail.")
    else:
        print("\nWaiting for service to stabilize (up to 5 min)...")
        try:
            waiter = ecs.get_waiter("services_stable")
            waiter.wait(
                cluster=ECS_CLUSTER_NAME,
                services=[args.service_name],
                WaiterConfig={"Delay": 10, "MaxAttempts": 30},
            )
            print("  Service is stable")
        except Exception as e:
            print(f"\n  Warning: Service did not stabilize: {e}")
            print("  Run with --skip-wait next time. Debug: ECS Console → Service → Events + Logs.")

    print("\n" + "=" * 60)
    print("Dashboard deployed successfully!")
    print(f"  URL: http://{alb_dns}")
    print("=" * 60)


if __name__ == "__main__":
    main()
