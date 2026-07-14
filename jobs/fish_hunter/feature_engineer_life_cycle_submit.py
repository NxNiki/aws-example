"""Submit feature_engineer_life_cycle.py as a SageMaker PySpark processing job.

Runs from a local machine. The job executes under SageMakerExecutionRole,
which (unlike local IAM users) has read access to the oceanhunter source
bucket; all outputs are written to S3 under OUTPUT_ROOT.
"""

import boto3
from sagemaker.session import Session
from sagemaker.spark.processing import PySparkProcessor

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET

# NOT the repo-wide SAGEMAKER_ROLE: the oceanhunter bucket policy (owned by
# another account) only trusts this role -- the one the DataScience notebook
# instance runs under. SageMakerExecutionRole gets 403 on the source bucket.
ROLE = "arn:aws:iam::338568447110:role/service-role/AmazonSageMaker-ExecutionRole-20250102T151291"

INPUT_ROOT = "s3://oceanhunter-production-data-warehouse/transformed_data/cold_data/bullet"
OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/lifecycle_feature_engineering/fm01_cny"

# The job derives the raw-data scan range from this window (one extra day on
# each side to keep sessions crossing midnight whole).
OUTPUT_START = "2025-05-01"  # keep rows with bet_date >= this
OUTPUT_END = "2026-07-01"  # keep rows with bet_date < this

INSTANCE_TYPE = "ml.m5.4xlarge"  # 16 vCPU, 64 GB memory per node
INSTANCE_COUNT = 3
# Shuffle spill from the sessionization windows lands on local disk; the
# SageMaker default of 30 GB per node is far too small for the larger months.
VOLUME_SIZE_GB = 300

# True blocks and streams the job logs (per-month summaries) to this terminal;
# False submits and returns -- watch the job in SageMaker console -> Processing jobs.
WAIT = True

processor = PySparkProcessor(
    base_job_name="fm01-lifecycle-features",
    framework_version="3.3",
    role=ROLE,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    volume_size_in_gb=VOLUME_SIZE_GB,
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
