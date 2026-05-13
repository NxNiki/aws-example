"""
Shared boto3 helpers for the ECS Fargate deploy scripts.

Lives under ``infra/`` (not under ``bituslabs_ds``) because these helpers
are only consumed by deploy scripts. The deploy scripts import this as a
sibling module — Python adds the script's directory to ``sys.path[0]``
when invoked as ``python infra/deploy_*.py``, so a bare
``from ecs_helpers import ...`` works without sys.path tinkering.

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
