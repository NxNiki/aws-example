"""Create/update the daily slot-machine cold-data ETL schedule.

Builds a SageMaker Pipeline (``slot-cold-data-daily``) with one PySpark
processing step per game, chained sequentially to stay inside the account's
4x ml.m5.4xlarge processing quota, and an EventBridge Scheduler rule that
starts it once a day. Steps pass no date window, so each run recomputes the
job's rolling incremental window (last 3 Beijing days; dynamic period=
overwrite leaves history untouched).

Usage:
    poetry run python infra/etl/deploy_slot_cold_data_pipeline.py            # upsert pipeline + schedule
    poetry run python infra/etl/deploy_slot_cold_data_pipeline.py --run-now  # also start one execution
"""

import argparse
import json

import boto3
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET

PIPELINE_NAME = "slot-cold-data-daily"
SCHEDULE_NAME = "slot-cold-data-daily"
SCHEDULE_CRON = "cron(0 9 * * ? *)"
SCHEDULE_TIMEZONE = "America/Los_Angeles"  # DST handled by EventBridge Scheduler
SCHEDULER_ROLE_NAME = "slot-cold-data-scheduler-role"

# NOT the repo-wide SAGEMAKER_ROLE: the slotmachine bucket policy (owned by
# another account) only trusts this role.
ROLE = "arn:aws:iam::338568447110:role/service-role/AmazonSageMaker-ExecutionRole-20250102T151291"

INPUT_ROOT = "s3://slotmachine-production-data-warehouse/transformed_data/partition_cold_data/bet_order"
GAME_OUTPUT_ROOTS = {
    "SS01": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss01_wucaishen_v2_cold_data",
    "SS01A": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss01a_golden_goal_v2_cold_data",
    "SS02": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss02_deepdive_v2_cold_data",
    "SS03": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_mahjiang_streak_v2_cold_data",
    "SS06": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss06_pocket_soccer_v2_cold_data",
}


def build_pipeline(session: PipelineSession) -> Pipeline:
    steps = []
    prev = None
    for game_id, output_root in GAME_OUTPUT_ROOTS.items():
        processor = PySparkProcessor(
            base_job_name=f"{game_id.lower()}-cold-data-daily",
            framework_version="3.3",
            role=ROLE,
            instance_type="ml.m5.4xlarge",
            instance_count=3,
            volume_size_in_gb=100,
            max_runtime_in_seconds=2 * 60 * 60,
            sagemaker_session=session,
        )
        step_args = processor.run(
            submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_game_stats_daily_by_user_group_cold_data.py",
            arguments=[
                "--game-id",
                game_id,
                "--input-root",
                INPUT_ROOT,
                "--output-root",
                output_root,
            ],
        )
        step = ProcessingStep(
            name=f"etl-{game_id.lower()}",
            step_args=step_args,
            depends_on=[prev.name] if prev else None,
        )
        steps.append(step)
        prev = step
    return Pipeline(name=PIPELINE_NAME, steps=steps, sagemaker_session=session)


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
        # IAM reads may be denied for this principal; if an admin already
        # created the role out of band, the schedule call validates the ARN.
        print(f"cannot read role ({e}); assuming it exists: {role_arn}")
        return role_arn
    iam.create_role(
        RoleName=SCHEDULER_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="EventBridge Scheduler role: start the slot-cold-data-daily SageMaker pipeline",
    )
    iam.put_role_policy(
        RoleName=SCHEDULER_ROLE_NAME,
        PolicyName="start-slot-cold-data-pipeline",
        PolicyDocument=json.dumps(policy),
    )
    print("created role", SCHEDULER_ROLE_NAME)
    return role_arn


def ensure_schedule(pipeline_arn: str, role_arn: str) -> None:
    scheduler = boto3.client("scheduler", region_name=REGION)
    schedule = dict(
        Name=SCHEDULE_NAME,
        ScheduleExpression=SCHEDULE_CRON,
        ScheduleExpressionTimezone=SCHEDULE_TIMEZONE,
        FlexibleTimeWindow={"Mode": "OFF"},
        Target={
            "Arn": "arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution",
            "RoleArn": role_arn,
            # StartPipelineExecution requires ClientRequestToken (SDKs autofill
            # it; raw API targets must send it). The scheduler context attribute
            # resolves to a unique id per firing, so retries of one firing
            # dedupe while daily runs don't.
            "Input": json.dumps({"PipelineName": PIPELINE_NAME, "ClientRequestToken": "<aws.scheduler.execution-id>"}),
            "RetryPolicy": {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 3600},
        },
        Description="Daily slot-machine cold-data ETL (rolling 3-day incremental window)",
        State="ENABLED",
    )
    try:
        scheduler.create_schedule(**schedule)
        print("created schedule", SCHEDULE_NAME, SCHEDULE_CRON, SCHEDULE_TIMEZONE)
    except scheduler.exceptions.ConflictException:
        scheduler.update_schedule(**schedule)
        print("updated schedule", SCHEDULE_NAME, SCHEDULE_CRON, SCHEDULE_TIMEZONE)
    except Exception as e:
        # scheduler:* is admin-only in this account; the schedule is managed
        # out of band (CloudShell) and keeps starting the pipeline by name,
        # so a pipeline-only upsert is complete without touching it.
        print(f"schedule unchanged (no permission to manage it here): {e}")


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
    ensure_schedule(upserted["PipelineArn"], role_arn)

    if args.run_now:
        execution = pipeline.start()
        print("started execution:", execution.arn)


if __name__ == "__main__":
    main()
