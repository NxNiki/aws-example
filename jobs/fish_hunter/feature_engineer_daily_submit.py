"""Submit feature_engineer_daily.py as a SageMaker PySpark processing job.

Runs from a local machine. Reads the fish_bullets_group_tag bullets dataset
(build/refresh it first with
jobs/etl/sagemaker/fish_hunter/etl_fish_bullets_group_tag_submit.py); the
daily grouped features are written to S3 under OUTPUT_ROOT.
"""

import boto3
from sagemaker.session import Session

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET
from bituslabs_ds.sagemaker_etl import SPARK_COMMON_PY_FILES, spark_processor

INPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_fish_bullets_group_tag/bullets"
OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_fishunter_feature_engineer"

OUTPUT_START = "2025-05-01"  # keep rows with bet_date >= this
OUTPUT_END = "2026-07-01"  # keep rows with bet_date < this

# Shuffle spill from the per-user window functions lands on local disk; the
# SageMaker default of 30 GB per node is far too small for the larger months.
VOLUME_SIZE_GB = 300

# True blocks and streams the job logs (per-month summaries) to this terminal;
# False submits and returns -- watch the job in SageMaker console -> Processing jobs.
WAIT = True

processor = spark_processor(
    "fm01-feature-daily",
    Session(boto3.Session(region_name=REGION)),
    volume_size_gb=VOLUME_SIZE_GB,
    max_runtime_hours=6,
)

processor.run(
    submit_app=f"{LOCAL_ROOT}/jobs/fish_hunter/feature_engineer_daily.py",
    submit_py_files=SPARK_COMMON_PY_FILES,
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
