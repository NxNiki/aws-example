"""Submit feature_engineer_life_cycle.py as a SageMaker PySpark processing job.

Runs from a local machine. The job executes under SageMakerExecutionRole,
which (unlike local IAM users) has read access to the oceanhunter source
bucket; all outputs are written to S3 under OUTPUT_ROOT.
"""

import boto3
from sagemaker.session import Session
from sagemaker.spark.processing import PySparkProcessor

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET, SAGEMAKER_ROLE

INPUT_ROOT = "s3://oceanhunter-production-data-warehouse/transformed_data/cold_data/bullet"
OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/lifecycle_feature_engineering/fm01_cny"

# The job derives the raw-data scan range from this window (one extra day on
# each side to keep sessions crossing midnight whole).
OUTPUT_START = "2025-05-01"  # keep rows with bet_date >= this
OUTPUT_END = "2026-06-01"  # keep rows with bet_date < this

INSTANCE_TYPE = "ml.m5.2xlarge"  # 8 vCPU, 32 GB memory per node
INSTANCE_COUNT = 3

# True blocks and streams the job logs (per-month summaries) to this terminal;
# False submits and returns -- watch the job in SageMaker console -> Processing jobs.
WAIT = True

processor = PySparkProcessor(
    base_job_name="fm01-lifecycle-features",
    framework_version="3.3",
    role=SAGEMAKER_ROLE,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    max_runtime_in_seconds=6 * 60 * 60,
    sagemaker_session=Session(boto3.Session(region_name=REGION)),
)

processor.run(
    submit_app=f"{LOCAL_ROOT}/jobs/fish_hunter/feature_engineer_life_cycle.py",
    arguments=[
        "--input-root",
        INPUT_ROOT,
        "--output-root",
        OUTPUT_ROOT,
        "--output-start",
        OUTPUT_START,
        "--output-end",
        OUTPUT_END,
    ],
    spark_event_logs_s3_uri=f"{OUTPUT_ROOT}/spark-event-logs",
    logs=WAIT,
    wait=WAIT,
)
