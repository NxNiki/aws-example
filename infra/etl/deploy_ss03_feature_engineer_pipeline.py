"""Create/update the monthly SS03 feature-engineering ETL schedule.

Builds a SageMaker Pipeline (``ss03-feature-engineer-monthly``) with a single
PySpark processing step that computes all four ai_group slices from the
cold-data warehouse (see ``jobs/etl/sagemaker/slot_machine/
etl_feature_engineer_cold_data.py``), and an EventBridge Scheduler rule that
starts it on the 1st of every month. The step passes no date window, so each
firing recomputes the just-finished month (plus current month-to-date) via
dynamic ``year=/month=`` partition overwrite.

Usage:
    poetry run python infra/etl/deploy_ss03_feature_engineer_pipeline.py            # upsert pipeline + schedule
    poetry run python infra/etl/deploy_ss03_feature_engineer_pipeline.py --run-now  # also start one execution
"""

import argparse
import json

import boto3
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET

PIPELINE_NAME = "ss03-feature-engineer-monthly"
SCHEDULE_NAME = "ss03-feature-engineer-monthly"
SCHEDULE_CRON = "cron(0 20 1 * ? *)"  # 1st of each month, 20:00 LA — clear of the 9am daily
# stats pipeline (both need 3x ml.m5.4xlarge from the account's 4-instance quota,
# so overlapping starts would fail on ResourceLimitExceeded)
SCHEDULE_TIMEZONE = "America/Los_Angeles"
SCHEDULER_ROLE_NAME = "ss03-feature-engineer-scheduler-role"

# NOT the repo-wide SAGEMAKER_ROLE: the slotmachine bucket policy (owned by
# another account) only trusts this role.
ROLE = "arn:aws:iam::338568447110:role/service-role/AmazonSageMaker-ExecutionRole-20250102T151291"

INPUT_ROOT = "s3://slotmachine-production-data-warehouse/transformed_data/partition_cold_data/bet_order"
OUTPUT_ROOT_BASE = f"s3://{S3_BUCKET}/etl-results/jobs"


def build_pipeline(session: PipelineSession) -> Pipeline:
    # 300 GB / 6 h (vs the daily stats steps' 100 GB / 2 h): the per-user
    # window functions shuffle-spill heavily, and a self-healing window can
    # span several months after missed firings.
    processor = PySparkProcessor(
        base_job_name="ss03-feature-engineer-cold-data",
        framework_version="3.3",
        role=ROLE,
        instance_type="ml.m5.4xlarge",
        instance_count=3,
        volume_size_in_gb=300,
        max_runtime_in_seconds=6 * 60 * 60,
        sagemaker_session=session,
    )
    step_args = processor.run(
        submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_feature_engineer_cold_data.py",
        arguments=["--input-root", INPUT_ROOT, "--output-root-base", OUTPUT_ROOT_BASE],
    )
    step = ProcessingStep(name="feature-engineer-ss03", step_args=step_args)
    return Pipeline(name=PIPELINE_NAME, steps=[step], sagemaker_session=session)


def ensure_scheduler_role(account_id: str) -> str:
    iam = boto3.client("iam")
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "scheduler.amazonaws.com"}, "Action": "sts:AssumeRole"}
        ],
    }
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sagemaker:StartPipelineExecution",
                "Resource": f"arn:aws:sagemaker:{REGION}:{account_id}:pipeline/{PIPELINE_NAME}",
            }
        ],
    }
    role_arn = f"arn:aws:iam::{account_id}:role/{SCHEDULER_ROLE_NAME}"
    try:
        iam.get_role(RoleName=SCHEDULER_ROLE_NAME)
        print("role exists", SCHEDULER_ROLE_NAME)
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        pass
    except Exception as e:
        print(f"cannot read role ({e}); assuming it exists: {role_arn}")
        print("If it does not, an admin must create it with trust policy:")
        print(json.dumps(trust, indent=2))
        print("and inline policy:")
        print(json.dumps(policy, indent=2))
        return role_arn
    try:
        iam.create_role(
            RoleName=SCHEDULER_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="EventBridge Scheduler role: start the ss03-feature-engineer-monthly SageMaker pipeline",
        )
        iam.put_role_policy(
            RoleName=SCHEDULER_ROLE_NAME,
            PolicyName="start-ss03-feature-engineer-pipeline",
            PolicyDocument=json.dumps(policy),
        )
        print("created role", SCHEDULER_ROLE_NAME)
    except Exception as e:
        print(f"cannot create role ({e}); an admin must create it with trust=scheduler.amazonaws.com and policy:")
        print(json.dumps(policy, indent=2))
    return role_arn


def ensure_schedule(role_arn: str) -> None:
    scheduler = boto3.client("scheduler", region_name=REGION)
    schedule = dict(
        Name=SCHEDULE_NAME,
        ScheduleExpression=SCHEDULE_CRON,
        ScheduleExpressionTimezone=SCHEDULE_TIMEZONE,
        FlexibleTimeWindow={"Mode": "OFF"},
        Target={
            "Arn": "arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution",
            "RoleArn": role_arn,
            # StartPipelineExecution requires ClientRequestToken; the scheduler
            # context attribute is unique per firing, so retries of one firing
            # dedupe while monthly runs don't.
            "Input": json.dumps({"PipelineName": PIPELINE_NAME, "ClientRequestToken": "<aws.scheduler.execution-id>"}),
            "RetryPolicy": {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 3600},
        },
        Description="Monthly SS03 feature-engineering ETL (recomputes previous month, cold-data source)",
        State="ENABLED",
    )
    try:
        scheduler.create_schedule(**schedule)
        print("created schedule", SCHEDULE_NAME, SCHEDULE_CRON, SCHEDULE_TIMEZONE)
    except scheduler.exceptions.ConflictException:
        scheduler.update_schedule(**schedule)
        print("updated schedule", SCHEDULE_NAME, SCHEDULE_CRON, SCHEDULE_TIMEZONE)
    except Exception as e:
        # scheduler:* is admin-only in this account; if the schedule is managed
        # out of band (CloudShell), the pipeline upsert alone is still complete —
        # the schedule starts the pipeline by name.
        print(f"schedule not created (no permission here): {e}")
        print("An admin can create it in CloudShell with:")
        print(
            f"aws scheduler create-schedule --name {SCHEDULE_NAME}"
            f" --schedule-expression '{SCHEDULE_CRON}'"
            f" --schedule-expression-timezone {SCHEDULE_TIMEZONE}"
            " --flexible-time-window Mode=OFF"
            f" --target '{json.dumps({'Arn': 'arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution', 'RoleArn': role_arn, 'Input': json.dumps({'PipelineName': PIPELINE_NAME, 'ClientRequestToken': '<aws.scheduler.execution-id>'}), 'RetryPolicy': {'MaximumRetryAttempts': 2, 'MaximumEventAgeInSeconds': 3600}})}'"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-now", action="store_true", help="start one pipeline execution after upsert")
    args = parser.parse_args()

    session = PipelineSession(boto_session=boto3.Session(region_name=REGION))
    pipeline = build_pipeline(session)
    upserted = pipeline.upsert(role_arn=ROLE)
    print("pipeline upserted:", upserted["PipelineArn"])

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = ensure_scheduler_role(account_id)
    ensure_schedule(role_arn)

    if args.run_now:
        execution = pipeline.start()
        print("started execution:", execution.arn)


if __name__ == "__main__":
    main()
