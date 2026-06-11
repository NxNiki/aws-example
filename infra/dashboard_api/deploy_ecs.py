#!/usr/bin/env python3
"""
Deploy dashboard_api (React SPA + data/report API) to ECS Fargate behind the
PRODUCTION dashboard ALB, with path routing to ai_agent — the Phase 5 cutover
(docs/frontend_redesign.md §8/§9).

What one run does (idempotent):
  1. Reuses the production ALB (``game-stats-dashboard-alb``) and raises its
     idle timeout to 300s for SSE chat streams.
  2. Stands up dashboard-api: target group (health ``/api/health``), task
     security group (8050 from the ALB's SG), task definition, service —
     production traffic is untouched up to here.
  3. Wires ``/api/agent/*`` to the existing ai-chat-agent service: a second
     target group on this ALB (a TG can belong to only one ALB), attached to
     the agent service via multi-TG update_service, SG ingress for 8051 from
     the ALB's SG, then a priority-10 listener rule.
  4. THE FLIP: repoints the listener's default action from the legacy Dash
     target group to dashboard-api. Atomic; rollback is one call (printed at
     the end).
  5. Decommissions the legacy Dash service (``--keep-legacy`` to skip):
     deletes its CloudWatch alarms FIRST (the scale-out alarm watches this
     ALB's 503 count and would otherwise resurrect Dash), deregisters its
     scalable target, scales it to 0. The service/task-def/TG are kept for a
     rollback bake; tear them down once the React dashboard has soaked.

Prerequisites:
  1. ``bash infra/ai_agent/build.sh`` + ``python infra/ai_agent/deploy_ecs.py``
     (the agent must run the /api/agent/* + skills code).
  2. ``bash infra/dashboard_api/build.sh`` (or pass --build-first).

Usage:
  python infra/dashboard_api/deploy_ecs.py --build-first
  python infra/dashboard_api/deploy_ecs.py --no-flip      # stage everything, don't cut over
  python infra/dashboard_api/deploy_ecs.py --keep-legacy  # cut over but leave Dash running
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
    add_service_load_balancer,
    create_or_update_service,
    ensure_ecr_repo,
    ensure_ecs_task_execution_role,
    ensure_listener_rule,
    ensure_log_group,
    ensure_sg_ingress_from_sg,
    ensure_target_group,
    ensure_task_security_group,
    get_account_id,
    get_default_vpc_id,
    get_default_vpc_subnets,
    get_http_listener_arn,
    register_task_definition,
    require_active_cluster,
    set_listener_default_tg,
    wait_for_service_stable,
    wait_for_targets_healthy,
)

# ---- Configuration (edit these directly) ------------------------------------
REGION = "us-west-2"
SERVICE_NAME = "dashboard-api"
IMAGE_NAME = "bituslabs-ds-dashboard-api"
DASHBOARD_API_PORT = 8050
# Sized from production OOMs: 1 GB died on the first /api/data/series, 2 GB
# died when one tab render fired its per-panel requests in parallel (each
# collected its own copy of the user-row window before the single-flight
# window cache landed in services/common.py). 4 GB = window cache (≤3 frames)
# + per-request compute headroom. One worker — more workers multiply it all.
TASK_CPU = 1024
TASK_MEMORY = 4096
DESIRED_COUNT = 1  # always-on; scale-to-zero policies wait for metric parity

# The production ALB the legacy Dash app currently owns; at cutover its
# listener default flips to dashboard-api and /api/agent/* routes to the agent.
ALB_NAME = "game-stats-dashboard-alb"
# SSE chat streams (/api/agent/chat) hold the connection open between events;
# the legacy 60s idle timeout would 504 long agent turns.
ALB_IDLE_TIMEOUT = 300

HEALTH_CHECK_PATH = "/api/health"  # registered before the SPA catch-all mount
TG_HEALTH_CHECK_INTERVAL = 30
TG_HEALTHY_THRESHOLD = 2
TG_DEREGISTRATION_DELAY = 30

# ai-chat-agent wiring: its own ALB (ai-chat-agent-alb) stays — the dashboard's
# server-side /api/report/* proxy targets it via AGENT_API_URL — but browser
# traffic to /api/agent/* needs the agent in a TG on THIS ALB too.
AGENT_SERVICE_NAME = "ai-chat-agent"
AGENT_CONTAINER_NAME = "ai-agent"
AGENT_PORT = 8051
AGENT_DASH_TG_NAME = "ai-chat-agent-dash-tg"
AGENT_RULE_PRIORITY = 10
AGENT_HEALTH_CHECK_PATH = "/health"
STAGING_RULE_PRIORITY = 9

# Legacy Dash service decommission targets.
LEGACY_SERVICE_NAME = "game-stats-dashboard"
LEGACY_TG_NAME = "game-stats-dashboard-tg"
LEGACY_ALARMS = ["game-stats-dashboard-scale-in-on-idle", "game-stats-dashboard-scale-out-on-503"]

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
    parser = argparse.ArgumentParser(description="Deploy dashboard_api to ECS + cut the production ALB over to it")
    parser.add_argument("--region", default=REGION, help="AWS region")
    parser.add_argument("--alb-name", default=ALB_NAME, help="Existing production ALB to deploy behind")
    parser.add_argument("--desired-count", type=int, default=DESIRED_COUNT, help="Number of tasks")
    parser.add_argument("--agent-api-url", default=None, help="AGENT_API_URL override (default: ai-chat-agent-alb)")
    parser.add_argument("--cors-origins", default=None, help="DASHBOARD_API_CORS override")
    parser.add_argument("--build-first", action="store_true", help="Run infra/dashboard_api/build.sh first")
    parser.add_argument("--no-flip", action="store_true", help="Stage service + routing but keep Dash as default")
    parser.add_argument("--keep-legacy", action="store_true", help="Flip, but leave the Dash service running")
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
    cloudwatch = session.client("cloudwatch")
    autoscaling = session.client("application-autoscaling")

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

    print(f"\n2. Production ALB: {args.alb_name}")
    alb = elbv2.describe_load_balancers(Names=[args.alb_name])["LoadBalancers"][0]
    alb_arn, alb_dns = alb["LoadBalancerArn"], alb["DNSName"]
    alb_sg_id = alb["SecurityGroups"][0]
    listener_arn = get_http_listener_arn(elbv2, alb_arn)
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
    # ECS create_service rejects a TG with no load-balancer association, but the
    # default action must stay on Dash until the new service is healthy — so
    # associate via a staging rule on a path no real traffic matches; it's
    # removed after the flip.
    staging_rule_arn = ensure_listener_rule(
        elbv2, listener_arn, STAGING_RULE_PRIORITY, ["/_dashboard-api-staging*"], tg_arn
    )

    print("\n4. Task execution role + inline policy...")
    exec_role_arn = ensure_ecs_task_execution_role(iam, account_id)
    iam.put_role_policy(
        RoleName=ECS_TASK_EXECUTION_ROLE_NAME, PolicyName=S3_POLICY_NAME, PolicyDocument=json.dumps(S3_POLICY)
    )

    print("\n5. Task definition + service...")
    agent_api_url = args.agent_api_url
    if not agent_api_url:
        agent_albs = elbv2.describe_load_balancers(Names=["ai-chat-agent-alb"])["LoadBalancers"]
        agent_api_url = f"http://{agent_albs[0]['DNSName']}"
        print(f"  Auto-detected AI agent ALB: {agent_api_url}")
    env_vars = [
        {"name": "DASHBOARD_CONFIG_DIR", "value": "/app/src/dashboards"},
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

    print("\n6. /api/agent/* routing to ai-chat-agent (SG ingress BEFORE TG attach)...")
    agent_task_sg = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": [f"{AGENT_SERVICE_NAME}-task-sg"]},
        ]
    )["SecurityGroups"][0]["GroupId"]
    ensure_sg_ingress_from_sg(ec2, agent_task_sg, AGENT_PORT, alb_sg_id, "From dashboard ALB (/api/agent/*)")
    agent_tg_arn = ensure_target_group(
        elbv2,
        AGENT_DASH_TG_NAME,
        vpc_id,
        AGENT_PORT,
        health_check_path=AGENT_HEALTH_CHECK_PATH,
        health_check_interval=TG_HEALTH_CHECK_INTERVAL,  # NOT the agent's own 300s TG interval
        healthy_threshold=TG_HEALTHY_THRESHOLD,
        deregistration_delay=TG_DEREGISTRATION_DELAY,
    )
    # Rule BEFORE attach: ECS rejects a TG with no load-balancer association.
    # Inert until the flip — neither the legacy Dash UI nor anything else sends
    # /api/agent/* to this ALB yet.
    ensure_listener_rule(elbv2, listener_arn, AGENT_RULE_PRIORITY, ["/api/agent/*"], agent_tg_arn)
    add_service_load_balancer(
        ecs,
        cluster=ECS_CLUSTER_NAME,
        service_name=AGENT_SERVICE_NAME,
        container_name=AGENT_CONTAINER_NAME,
        container_port=AGENT_PORT,
        tg_arn=agent_tg_arn,
    )
    # The agent's OWN TG checks every 300s, so this rollout takes ~10 min.
    wait_for_service_stable(ecs, ECS_CLUSTER_NAME, AGENT_SERVICE_NAME, max_attempts=90)
    wait_for_targets_healthy(elbv2, agent_tg_arn, timeout_seconds=900)

    if args.no_flip:
        print("\n--no-flip: staged. Legacy Dash still serves the default; flip later by re-running without it.")
        return

    print("\n7. CUTOVER: listener default → dashboard-api")
    set_listener_default_tg(elbv2, listener_arn, tg_arn)
    elbv2.delete_rule(RuleArn=staging_rule_arn)
    print("  Removed staging rule (TG now associated via the default action)")

    if not args.keep_legacy:
        print("\n8. Decommission legacy Dash (alarms first — the 503 alarm would resurrect it)...")
        for alarm in LEGACY_ALARMS:
            cloudwatch.delete_alarms(AlarmNames=[alarm])
            print(f"  Deleted alarm: {alarm}")
        try:
            autoscaling.deregister_scalable_target(
                ServiceNamespace="ecs",
                ResourceId=f"service/{ECS_CLUSTER_NAME}/{LEGACY_SERVICE_NAME}",
                ScalableDimension="ecs:service:DesiredCount",
            )
            print("  Deregistered legacy scalable target (policies removed)")
        except ClientError as e:
            if "ObjectNotFoundException" not in str(e):
                raise
        ecs.update_service(cluster=ECS_CLUSTER_NAME, service=LEGACY_SERVICE_NAME, desiredCount=0)
        print(f"  {LEGACY_SERVICE_NAME} scaled to 0 (service/task-def/TG kept for the rollback bake)")

    print("\n" + "=" * 60)
    print("Cutover complete!")
    print(f"  SPA:     http://{alb_dns}/")
    print(f"  Health:  http://{alb_dns}{HEALTH_CHECK_PATH}")
    print(f"  Configs: http://{alb_dns}/api/data/configs")
    print(f"  Agent:   http://{alb_dns}/api/agent/skills")
    print("\nRollback (legacy Dash):")
    print(f"  aws ecs update-service --cluster {ECS_CLUSTER_NAME} --service {LEGACY_SERVICE_NAME} \\")
    print(f"      --desired-count 1 --region {region}")
    print(f"  # wait for {LEGACY_TG_NAME} healthy, then:")
    legacy_tg = elbv2.describe_target_groups(Names=[LEGACY_TG_NAME])["TargetGroups"][0]["TargetGroupArn"]
    print(f"  aws elbv2 modify-listener --listener-arn {listener_arn} \\")
    print(f"      --default-actions Type=forward,TargetGroupArn={legacy_tg} --region {region}")
    print("=" * 60)


if __name__ == "__main__":
    main()
