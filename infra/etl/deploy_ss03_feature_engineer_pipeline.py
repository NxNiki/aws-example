"""Create/update the monthly SS03 feature-engineering ETL schedule.

Builds a SageMaker Pipeline (``ss03-feature-engineer-monthly``) with a single
PySpark processing step that computes all four ai_group slices from the
cold-data warehouse (see ``jobs/etl/sagemaker/slot_machine/
etl_feature_engineer_cold_data.py``), and an EventBridge Scheduler rule that
starts it on the 1st of every month. The step passes no date window, so each
firing recomputes the just-finished month (plus current month-to-date, and
any months a missed firing left behind) via dynamic ``year=/month=``
partition overwrite.

Shared pipeline/schedule plumbing lives in ``bituslabs_ds.sagemaker_etl``.

Usage:
    poetry run python infra/etl/deploy_ss03_feature_engineer_pipeline.py            # upsert pipeline + schedule
    poetry run python infra/etl/deploy_ss03_feature_engineer_pipeline.py --run-now  # also start one execution
"""

import argparse

import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET
from bituslabs_ds.sagemaker_etl import SPARK_COMMON_PY_FILES, spark_processor, upsert_pipeline_with_schedule

PIPELINE_NAME = "ss03-feature-engineer-monthly"
SCHEDULE_NAME = "ss03-feature-engineer-monthly"
# 1st of each month, 20:00 LA — clear of the 9am daily stats pipeline (both
# need 3x ml.m5.4xlarge from the account's 4-instance quota, so overlapping
# starts would fail on ResourceLimitExceeded).
SCHEDULE_CRON = "cron(0 20 1 * ? *)"
# Shared with the daily stats schedule: the role's trust policy is the same
# (scheduler.amazonaws.com) and each pipeline attaches its own least-privilege
# start-<pipeline> inline policy, so no second role is needed.
SCHEDULER_ROLE_NAME = "slot-cold-data-scheduler-role"

INPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_slot_orders_ab_group/orders"
OUTPUT_ROOT_BASE = f"s3://{S3_BUCKET}/etl-results/jobs"


def build_pipeline(session: PipelineSession) -> Pipeline:
    # 300 GB / 6 h (vs the daily stats steps' 100 GB / 2 h): the per-user
    # window functions shuffle-spill heavily, and a self-healing window can
    # span several months after missed firings.
    processor = spark_processor("ss03-feature-engineer-cold-data", session, volume_size_gb=300, max_runtime_hours=6)
    step_args = processor.run(
        submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_feature_engineer_cold_data.py",
        submit_py_files=SPARK_COMMON_PY_FILES,
        arguments=["--input-root", INPUT_ROOT, "--output-root-base", OUTPUT_ROOT_BASE],
    )
    step = ProcessingStep(name="feature-engineer-ss03", step_args=step_args)
    return Pipeline(name=PIPELINE_NAME, steps=[step], sagemaker_session=session)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-now", action="store_true", help="start one pipeline execution after upsert")
    args = parser.parse_args()

    session = PipelineSession(boto_session=boto3.Session(region_name=REGION))
    upsert_pipeline_with_schedule(
        build_pipeline(session),
        schedule_name=SCHEDULE_NAME,
        cron=SCHEDULE_CRON,
        scheduler_role_name=SCHEDULER_ROLE_NAME,
        description="Monthly SS03 feature-engineering ETL (recomputes previous month, cold-data source)",
        run_now=args.run_now,
    )


if __name__ == "__main__":
    main()
