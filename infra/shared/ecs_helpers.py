"""
Shared boto3 helpers for the ECS Fargate deploy scripts.

Lives under ``infra/`` (not under ``bituslabs_ds``) because these helpers
are only consumed by deploy scripts. The deploy scripts import this as a
sibling module — Python adds the script's directory to ``sys.path[0]``
when invoked as ``python infra/deploy_*.py``, so a bare
``from infra.shared.ecs_helpers import ...`` works without sys.path tinkering.

What's here vs what stays in the deploy scripts
-----------------------------------------------
- Generic AWS lookups (account ID, default VPC, default subnets) and
  generic ECS resource ensures (ECR repo, base task-execution role) live
  here.
- Service-specific concerns (inline policies for S3 read/write, Secrets
  Manager access, env vars, ALB sizing) stay in each deploy script.

The base ``ensure_ecs_task_execution_role`` only attaches the AWS-managed
``AmazonECSTaskExecutionRolePolicy``. Each script then layers its own
inline policy on top via ``iam.put_role_policy(PolicyName=...)`` so the
service-specific permissions stay co-located with the service that needs
them.
"""

from __future__ import annotations

import json

from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Shared ECS Fargate cluster used by the dashboard, AI chat agent, and
# RAG service. All three are co-located so they share the cluster's
# capacity providers and IAM trust relationships.
ECS_CLUSTER_NAME = "ai-team-dashboard-cluster"

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


# ---------------------------------------------------------------------------
# AWS lookups
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


# ---------------------------------------------------------------------------
# ECR / IAM ensures
# ---------------------------------------------------------------------------


def ensure_ecr_repo(ecr_client, repo_name: str) -> None:
    try:
        ecr_client.create_repository(repositoryName=repo_name)
        print(f"  Created ECR repository: {repo_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "RepositoryAlreadyExistsException":
            raise
        print(f"  ECR repository exists: {repo_name}")


def ensure_ecs_task_execution_role(iam_client, account_id: str) -> str:
    """Ensure ``ecsTaskExecutionRole`` exists with the ECS trust policy and
    the AWS-managed execution policy. Returns the role ARN.

    Service-specific inline policies (S3 reads, Secrets Manager access,
    etc.) are NOT attached here — each deploy script does that itself
    via ``iam.put_role_policy(PolicyName=...)`` so the permissions live
    next to the service that needs them.
    """
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
# ALB + ECS service plumbing (shared by every deploy_ecs.py)
# ---------------------------------------------------------------------------
#
# Each service is a public FastAPI/Dash app behind its own internet-facing ALB:
# one always-on listener on :80 forwarding to an IP target group, an ALB SG that
# allows HTTP in, and a task SG that allows the container port only from the ALB.
# These helpers ensure (create-or-reuse) each piece idempotently and print the
# same "  Created ..."/"  ... exists" detail lines the scripts used to print
# inline. The per-service variation — IAM inline policy, env vars, task sizing,
# autoscaling — stays in each deploy script; only the identical plumbing lives
# here. ``require_active_cluster`` covers services that reuse the shared cluster
# (the dashboard script keeps its own create-or-reuse path).


def require_active_cluster(ecs_client, cluster_name: str = ECS_CLUSTER_NAME) -> None:
    """Raise unless ``cluster_name`` exists and is ACTIVE. For services that
    reuse the shared cluster rather than create it (the dashboard deploy is
    what creates the cluster)."""
    resp = ecs_client.describe_clusters(clusters=[cluster_name])
    if resp.get("failures") or not resp.get("clusters") or resp["clusters"][0].get("status") != "ACTIVE":
        raise RuntimeError(
            f"Cluster {cluster_name} not found or not ACTIVE. Deploy the dashboard first to create the cluster."
        )
    print(f"  Using existing cluster: {cluster_name}")


def ensure_log_group(logs_client, name: str) -> str:
    try:
        logs_client.create_log_group(logGroupName=name)
        print(f"  Created log group: {name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
        print(f"  Log group exists: {name}")
    return name


def ensure_service_security_groups(
    ec2_client,
    vpc_id: str,
    service_name: str,
    container_port: int,
    *,
    alb_description: str | None = None,
    task_description: str | None = None,
    egress_description: str = "All outbound",
) -> tuple[str, str]:
    """Ensure the ALB + task security groups for a public service. Returns
    ``(sg_alb_id, sg_task_id)``.

    ALB SG: inbound HTTP 80 from anywhere. Task SG: inbound ``container_port``
    only from the ALB SG, plus all-outbound egress. Idempotent — reuses groups
    by name on ``InvalidGroup.Duplicate``.
    """
    sg_name_alb = f"{service_name}-alb-sg"
    sg_name_task = f"{service_name}-task-sg"
    alb_description = alb_description or f"ALB for {service_name}"
    task_description = task_description or f"ECS tasks for {service_name}"

    def _lookup(group_name: str) -> str:
        return ec2_client.describe_security_groups(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [group_name]}]
        )["SecurityGroups"][0]["GroupId"]

    try:
        sg_alb_id = ec2_client.create_security_group(GroupName=sg_name_alb, Description=alb_description, VpcId=vpc_id)[
            "GroupId"
        ]
        print(f"  Created ALB security group: {sg_alb_id}")
    except ClientError as e:
        if "InvalidGroup.Duplicate" not in str(e):
            raise
        sg_alb_id = _lookup(sg_name_alb)
        print(f"  ALB security group exists: {sg_alb_id}")

    try:
        ec2_client.authorize_security_group_ingress(
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
        sg_task_id = ec2_client.create_security_group(
            GroupName=sg_name_task, Description=task_description, VpcId=vpc_id
        )["GroupId"]
        print(f"  Created task security group: {sg_task_id}")
    except ClientError as e:
        if "InvalidGroup.Duplicate" not in str(e):
            raise
        sg_task_id = _lookup(sg_name_task)
        print(f"  Task security group exists: {sg_task_id}")

    try:
        ec2_client.authorize_security_group_ingress(
            GroupId=sg_task_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": container_port,
                    "ToPort": container_port,
                    "UserIdGroupPairs": [{"GroupId": sg_alb_id, "Description": "From ALB"}],
                }
            ],
        )
        print(f"  Added task inbound rule ({container_port} from ALB)")
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise

    try:
        ec2_client.authorize_security_group_egress(
            GroupId=sg_task_id,
            IpPermissions=[
                {"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": egress_description}]}
            ],
        )
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise
        print("  Task outbound rule already exists, skipping")

    return sg_alb_id, sg_task_id


def ensure_alb(
    elbv2_client,
    name: str,
    subnet_ids: list[str],
    sg_alb_id: str,
    *,
    idle_timeout_seconds: int | None = None,
) -> tuple[str, str]:
    """Ensure an internet-facing ALB. Returns ``(alb_arn, alb_dns)``. Waits for
    the ALB to become available; sets the idle timeout when given (needed for
    long SSE streams)."""
    try:
        alb = elbv2_client.create_load_balancer(
            Name=name,
            Subnets=subnet_ids,
            SecurityGroups=[sg_alb_id],
            Scheme="internet-facing",
            Type="application",
            Tags=[{"Key": "Name", "Value": name}],
        )["LoadBalancers"][0]
        alb_arn, alb_dns = alb["LoadBalancerArn"], alb["DNSName"]
        print(f"  Created ALB: {alb_dns}")
    except ClientError as e:
        if "DuplicateLoadBalancerName" not in str(e):
            raise
        alb = elbv2_client.describe_load_balancers(Names=[name])["LoadBalancers"][0]
        alb_arn, alb_dns = alb["LoadBalancerArn"], alb["DNSName"]
        print(f"  ALB exists: {alb_dns}")

    elbv2_client.get_waiter("load_balancer_available").wait(LoadBalancerArns=[alb_arn])
    print("  ALB is active")

    if idle_timeout_seconds is not None:
        elbv2_client.modify_load_balancer_attributes(
            LoadBalancerArn=alb_arn,
            Attributes=[{"Key": "idle_timeout.timeout_seconds", "Value": str(idle_timeout_seconds)}],
        )
        print(f"  ALB idle_timeout set to {idle_timeout_seconds}s")

    return alb_arn, alb_dns


def ensure_target_group(
    elbv2_client,
    name: str,
    vpc_id: str,
    port: int,
    *,
    health_check_path: str,
    health_check_interval: int,
    healthy_threshold: int,
    deregistration_delay: int | None = None,
) -> str:
    """Ensure an IP target group with an HTTP health check. Returns its ARN.
    Reapplies health-check settings on reuse; sets the deregistration delay when
    given (faster task drain on redeploy)."""
    try:
        tg_arn = elbv2_client.create_target_group(
            Name=name,
            Protocol="HTTP",
            Port=port,
            VpcId=vpc_id,
            TargetType="ip",
            HealthCheckProtocol="HTTP",
            HealthCheckPath=health_check_path,
            HealthCheckIntervalSeconds=health_check_interval,
            HealthyThresholdCount=healthy_threshold,
            UnhealthyThresholdCount=3,
        )["TargetGroups"][0]["TargetGroupArn"]
        print(f"  Created target group: {tg_arn}")
    except ClientError as e:
        if "DuplicateTargetGroupName" not in str(e):
            raise
        tg_arn = elbv2_client.describe_target_groups(Names=[name])["TargetGroups"][0]["TargetGroupArn"]
        print(f"  Target group exists: {tg_arn}")

    elbv2_client.modify_target_group(
        TargetGroupArn=tg_arn,
        HealthCheckPath=health_check_path,
        HealthCheckIntervalSeconds=health_check_interval,
        HealthyThresholdCount=healthy_threshold,
    )
    if deregistration_delay is not None:
        elbv2_client.modify_target_group_attributes(
            TargetGroupArn=tg_arn,
            Attributes=[{"Key": "deregistration_delay.timeout_seconds", "Value": str(deregistration_delay)}],
        )
    return tg_arn


def ensure_http_listener(elbv2_client, alb_arn: str, tg_arn: str) -> None:
    """Ensure a :80 HTTP listener forwarding to ``tg_arn``."""
    try:
        elbv2_client.create_listener(
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


def register_task_definition(ecs_client, task_def: dict) -> str:
    """Register a task definition and deregister the family's older ACTIVE
    revisions, keeping only the new one. Returns the new revision ARN."""
    resp = ecs_client.register_task_definition(**task_def)
    new_arn = resp["taskDefinition"]["taskDefinitionArn"]
    family = task_def["family"]
    print(f"  Registered task definition: {family} (revision {resp['taskDefinition']['revision']})")

    for old_arn in ecs_client.list_task_definitions(familyPrefix=family, status="ACTIVE")["taskDefinitionArns"]:
        if old_arn != new_arn:
            ecs_client.deregister_task_definition(taskDefinition=old_arn)
            print(f"  Deregistered old revision: {old_arn.split(':')[-1]}")
    return new_arn


DEFAULT_DEPLOYMENT_CONFIG = {"minimumHealthyPercent": 100, "maximumPercent": 200}


def create_or_update_service(
    ecs_client,
    *,
    cluster: str,
    service_name: str,
    container_name: str,
    container_port: int,
    tg_arn: str,
    subnet_ids: list[str],
    sg_task_id: str,
    desired_count: int,
    deployment_config: dict | None = None,
) -> None:
    """Create the Fargate service, or force a new deployment if it already
    exists (idempotent re-runs roll the task definition). ``deployment_config``
    defaults to 100%/200% rolling, applied on both create and update."""
    deployment_config = deployment_config or DEFAULT_DEPLOYMENT_CONFIG
    network_config = {
        "awsvpcConfiguration": {
            "subnets": subnet_ids,
            "securityGroups": [sg_task_id],
            "assignPublicIp": "ENABLED",
        }
    }
    load_balancers = [{"targetGroupArn": tg_arn, "containerName": container_name, "containerPort": container_port}]
    try:
        ecs_client.create_service(
            cluster=cluster,
            serviceName=service_name,
            taskDefinition=service_name,
            desiredCount=desired_count,
            launchType="FARGATE",
            deploymentConfiguration=deployment_config,
            networkConfiguration=network_config,
            loadBalancers=load_balancers,
        )
        print(f"  Created service: {service_name}")
    except ClientError as e:
        err = e.response.get("Error", {})
        code, msg = err.get("Code", ""), err.get("Message", "")
        if "ServiceAlreadyExists" in str(e) or (
            code == "InvalidParameterException" and "not idempotent" in (msg or "").lower()
        ):
            ecs_client.update_service(
                cluster=cluster,
                service=service_name,
                taskDefinition=service_name,
                desiredCount=desired_count,
                deploymentConfiguration=deployment_config,
                forceNewDeployment=True,
            )
            print(f"  Updated existing service: {service_name}")
        else:
            raise


def wait_for_service_stable(ecs_client, cluster: str, service_name: str, *, max_attempts: int = 30) -> None:
    """Block until the service reaches a steady state (or warn after
    ``max_attempts`` × 10s; a target group with a slow health-check interval
    can gate a rollout for 10+ minutes)."""
    print(f"\nWaiting for service to stabilize (up to {max_attempts * 10 // 60} min)...")
    try:
        ecs_client.get_waiter("services_stable").wait(
            cluster=cluster,
            services=[service_name],
            WaiterConfig={"Delay": 10, "MaxAttempts": max_attempts},
        )
        print("  Service is stable")
    except Exception as e:
        print(f"\n  Warning: Service did not stabilize: {e}")


# --- Cutover helpers (Phase 5): path routing + default flip on a shared ALB ---


def get_http_listener_arn(elbv2_client, alb_arn: str) -> str:
    """ARN of the ALB's single :80 HTTP listener (asserts exactly one)."""
    listeners = [ls for ls in elbv2_client.describe_listeners(LoadBalancerArn=alb_arn)["Listeners"] if ls["Port"] == 80]
    if len(listeners) != 1:
        raise RuntimeError(f"Expected exactly one :80 listener on {alb_arn}, found {len(listeners)}")
    return listeners[0]["ListenerArn"]


def ensure_sg_ingress_from_sg(ec2_client, group_id: str, port: int, source_sg_id: str, description: str) -> None:
    """Idempotently allow TCP ``port`` into ``group_id`` from ``source_sg_id``."""
    try:
        ec2_client.authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": port,
                    "ToPort": port,
                    "UserIdGroupPairs": [{"GroupId": source_sg_id, "Description": description}],
                }
            ],
        )
        print(f"  Added ingress: {port} from {source_sg_id} into {group_id}")
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise
        print(f"  Ingress {port} from {source_sg_id} already present on {group_id}")


def ensure_task_security_group(
    ec2_client, vpc_id: str, name: str, port: int, source_sg_id: str, *, description: str | None = None
) -> str:
    """Ensure a task security group admitting ``port`` from an EXISTING ALB
    security group (unlike ``ensure_service_security_groups``, which always
    creates a fresh ALB SG pair). Returns the group id."""
    try:
        sg_id = ec2_client.create_security_group(
            GroupName=name, Description=description or f"ECS tasks for {name}", VpcId=vpc_id
        )["GroupId"]
        print(f"  Created task security group: {sg_id}")
    except ClientError as e:
        if "InvalidGroup.Duplicate" not in str(e):
            raise
        sg_id = ec2_client.describe_security_groups(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [name]}]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  Task security group exists: {sg_id}")

    ensure_sg_ingress_from_sg(ec2_client, sg_id, port, source_sg_id, "From shared ALB")
    try:
        ec2_client.authorize_security_group_egress(
            GroupId=sg_id,
            IpPermissions=[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "All outbound"}]}],
        )
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise
    return sg_id


def ensure_listener_rule(elbv2_client, listener_arn: str, priority: int, path_patterns: list[str], tg_arn: str) -> str:
    """Ensure a path-pattern forward rule on the listener. Matched by path
    patterns: an existing rule with the same patterns gets its action/priority
    left alone (idempotent re-runs). Returns the rule ARN."""
    for rule in elbv2_client.describe_rules(ListenerArn=listener_arn)["Rules"]:
        for cond in rule.get("Conditions", []):
            if cond.get("Field") == "path-pattern" and sorted(cond.get("Values", [])) == sorted(path_patterns):
                print(f"  Listener rule for {path_patterns} exists: {rule['RuleArn']}")
                return rule["RuleArn"]
    rule_arn = elbv2_client.create_rule(
        ListenerArn=listener_arn,
        Priority=priority,
        Conditions=[{"Field": "path-pattern", "Values": path_patterns}],
        Actions=[{"Type": "forward", "TargetGroupArn": tg_arn}],
    )["Rules"][0]["RuleArn"]
    print(f"  Created listener rule p{priority}: {path_patterns} → {tg_arn.split('/')[-2]}")
    return rule_arn


def set_listener_default_tg(elbv2_client, listener_arn: str, tg_arn: str) -> None:
    """Atomically repoint the listener's default action — the cutover flip."""
    elbv2_client.modify_listener(
        ListenerArn=listener_arn, DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}]
    )
    print(f"  Listener default action → {tg_arn.split('/')[-2]}")


def add_service_load_balancer(
    ecs_client, *, cluster: str, service_name: str, container_name: str, container_port: int, tg_arn: str
) -> None:
    """Attach an additional target group to a running service (multi-TG).

    ``update_service`` REPLACES the whole loadBalancers list, so the existing
    attachments are re-sent alongside the new one; the call starts a rolling
    deployment. NOTE: ``create_or_update_service``'s create path attaches only
    its own TG — recreating the service from scratch drops extra TGs added here.
    """
    svc = ecs_client.describe_services(cluster=cluster, services=[service_name])["services"][0]
    current = svc.get("loadBalancers", [])
    if any(lb.get("targetGroupArn") == tg_arn for lb in current):
        print(f"  Service {service_name} already attached to {tg_arn.split('/')[-2]}")
        return
    ecs_client.update_service(
        cluster=cluster,
        service=service_name,
        loadBalancers=current
        + [{"targetGroupArn": tg_arn, "containerName": container_name, "containerPort": container_port}],
    )
    print(f"  Attached {tg_arn.split('/')[-2]} to {service_name} (rolling deployment started)")


def wait_for_targets_healthy(elbv2_client, tg_arn: str, *, timeout_seconds: int = 600) -> None:
    """Block until the target group reports at least one healthy target."""
    import time

    print(f"  Waiting for healthy targets in {tg_arn.split('/')[-2]} (up to {timeout_seconds // 60} min)...")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        states = [
            t["TargetHealth"]["State"]
            for t in elbv2_client.describe_target_health(TargetGroupArn=tg_arn)["TargetHealthDescriptions"]
        ]
        if "healthy" in states:
            print(f"  Target group healthy ({states})")
            return
        time.sleep(15)
    raise RuntimeError(f"No healthy targets in {tg_arn} after {timeout_seconds}s")
