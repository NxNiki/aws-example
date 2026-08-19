"""Shared plumbing for SageMaker PySpark ETL pipelines.

Every cold-data ETL uses the same shape: a ``PySparkProcessor`` under the one
IAM role the game warehouses trust, optionally chained into a SageMaker
Pipeline, started daily/monthly by an EventBridge Scheduler rule. The
per-pipeline variation (which job scripts, arguments, cron) stays in each
deploy script under ``infra/etl/``; the identical plumbing lives here.

Notes baked into the helpers:

- ``TRUSTED_SPARK_ROLE`` is NOT the repo-wide ``SAGEMAKER_ROLE``: the
  slotmachine/oceanhunter warehouse bucket policies (owned by another
  account) only trust this specific role.
- EventBridge's universal target for ``StartPipelineExecution`` must send a
  ``ClientRequestToken``; the ``<aws.scheduler.execution-id>`` context
  attribute is unique per firing, so retries of one firing dedupe while
  scheduled runs don't.
- ``iam:*``/``scheduler:*`` are admin-only for regular users in this
  account, so the ensure_* helpers degrade to printing exactly what an
  admin must create (role trust/policy JSON, or the CloudShell CLI).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

import boto3

from bituslabs_ds.config import LOCAL_ROOT, REGION

if TYPE_CHECKING:  # sagemaker is an optional (ml) dependency group
    from sagemaker.spark.processing import PySparkProcessor

TRUSTED_SPARK_ROLE = "arn:aws:iam::338568447110:role/service-role/AmazonSageMaker-ExecutionRole-20250102T151291"

# Container-side helpers shared by every Spark job; ship with each submit via
# ``processor.run(..., submit_py_files=SPARK_COMMON_PY_FILES)``.
SPARK_COMMON_PY_FILES = [
    f"{LOCAL_ROOT}/jobs/etl/sagemaker/spark_etl_common.py",
    f"{LOCAL_ROOT}/jobs/etl/sagemaker/group_policy.py",
    f"{LOCAL_ROOT}/jobs/etl/sagemaker/group_policy_sql.py",
]

SPARK_FRAMEWORK_VERSION = "3.3"  # container runs Python 3.9 — job scripts must stay 3.9-compatible
DEFAULT_INSTANCE_TYPE = "ml.m5.4xlarge"  # 16 vCPU, 64 GB per node
DEFAULT_INSTANCE_COUNT = 3  # account quota is 4 concurrent — schedule pipelines apart
DEFAULT_TIMEZONE = "America/Los_Angeles"  # DST handled by EventBridge Scheduler


def spark_processor(
    base_job_name: str,
    sagemaker_session,
    *,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    volume_size_gb: int = 100,
    max_runtime_hours: int = 2,
    role: str = TRUSTED_SPARK_ROLE,
) -> "PySparkProcessor":
    """The standard cold-data PySpark processor. ``sagemaker_session`` is a
    plain ``Session`` for direct submits or a ``PipelineSession`` inside a
    pipeline build. Raise ``volume_size_gb`` for jobs with heavy shuffle
    spill (per-user window functions); the 30 GB SageMaker default is never
    enough."""
    from sagemaker.spark.processing import PySparkProcessor

    return PySparkProcessor(
        base_job_name=base_job_name,
        framework_version=SPARK_FRAMEWORK_VERSION,
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size_in_gb=volume_size_gb,
        max_runtime_in_seconds=max_runtime_hours * 60 * 60,
        sagemaker_session=sagemaker_session,
    )


def scheduler_role_policy(pipeline_name: str, account_id: str, region: str = REGION) -> dict:
    """Least-privilege inline policy for an EventBridge scheduler role: start
    exactly one pipeline."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sagemaker:StartPipelineExecution",
                "Resource": f"arn:aws:sagemaker:{region}:{account_id}:pipeline/{pipeline_name}",
            }
        ],
    }


_SCHEDULER_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "scheduler.amazonaws.com"}, "Action": "sts:AssumeRole"}],
}


def ensure_scheduler_role(role_name: str, pipeline_name: str, account_id: Optional[str] = None) -> str:
    """Create-or-reuse the IAM role EventBridge assumes to start the pipeline.
    Degrades to printing the trust/policy JSON when IAM access is denied (an
    admin creates it out of band); always returns the expected role ARN."""
    iam = boto3.client("iam")
    if account_id is None:
        account_id = boto3.client("sts").get_caller_identity()["Account"]
    policy = scheduler_role_policy(pipeline_name, account_id)
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    try:
        iam.get_role(RoleName=role_name)
        print("role exists", role_name)
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        pass
    except Exception as e:
        print(f"cannot read role ({e}); assuming it exists: {role_arn}")
        print("An admin can grant this pipeline to the role (roles hold one inline policy per pipeline) with:")
        print(
            f"aws iam put-role-policy --role-name {role_name}"
            f" --policy-name start-{pipeline_name}"
            f" --policy-document '{json.dumps(policy)}'"
        )
        print("If the role itself does not exist yet, create it first with trust policy:")
        print(json.dumps(_SCHEDULER_TRUST, indent=2))
        return role_arn
    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(_SCHEDULER_TRUST),
            Description=f"EventBridge Scheduler role: start the {pipeline_name} SageMaker pipeline",
        )
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"start-{pipeline_name}",
            PolicyDocument=json.dumps(policy),
        )
        print("created role", role_name)
    except Exception as e:
        print(f"cannot create role ({e}); an admin must create it with trust=scheduler.amazonaws.com and policy:")
        print(json.dumps(policy, indent=2))
    return role_arn


def ensure_pipeline_schedule(
    schedule_name: str,
    cron: str,
    pipeline_name: str,
    role_arn: str,
    *,
    description: str = "",
    timezone: str = DEFAULT_TIMEZONE,
    region: str = REGION,
) -> None:
    """Create-or-update the EventBridge Scheduler rule that starts the
    pipeline. Degrades to printing the exact CloudShell CLI when scheduler
    access is denied — the pipeline upsert alone is still complete, because
    the schedule targets the pipeline by NAME."""
    target = {
        "Arn": "arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution",
        "RoleArn": role_arn,
        "Input": json.dumps({"PipelineName": pipeline_name, "ClientRequestToken": "<aws.scheduler.execution-id>"}),
        "RetryPolicy": {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 3600},
    }
    schedule = dict(
        Name=schedule_name,
        ScheduleExpression=cron,
        ScheduleExpressionTimezone=timezone,
        FlexibleTimeWindow={"Mode": "OFF"},
        Target=target,
        Description=description,
        State="ENABLED",
    )
    scheduler = boto3.client("scheduler", region_name=region)
    try:
        scheduler.create_schedule(**schedule)
        print("created schedule", schedule_name, cron, timezone)
    except scheduler.exceptions.ConflictException:
        scheduler.update_schedule(**schedule)
        print("updated schedule", schedule_name, cron, timezone)
    except Exception as e:
        print(f"schedule not created (no permission here): {e}")
        print("An admin can create it in CloudShell with:")
        print(
            f"aws scheduler create-schedule --name {schedule_name}"
            f" --schedule-expression '{cron}'"
            f" --schedule-expression-timezone {timezone}"
            " --flexible-time-window Mode=OFF"
            f" --target '{json.dumps(target)}'"
        )


def upsert_pipeline_with_schedule(
    pipeline,
    *,
    schedule_name: str,
    cron: str,
    scheduler_role_name: str,
    description: str = "",
    timezone: str = DEFAULT_TIMEZONE,
    run_now: bool = False,
) -> None:
    """The whole deploy tail every pipeline script shares: upsert the
    pipeline under the trusted role, ensure the scheduler role + schedule,
    optionally start one execution immediately."""
    upserted = pipeline.upsert(role_arn=TRUSTED_SPARK_ROLE)
    print("pipeline upserted:", upserted["PipelineArn"])
    role_arn = ensure_scheduler_role(scheduler_role_name, pipeline.name)
    ensure_pipeline_schedule(schedule_name, cron, pipeline.name, role_arn, description=description, timezone=timezone)
    if run_now:
        execution = pipeline.start()
        print("started execution:", execution.arn)
