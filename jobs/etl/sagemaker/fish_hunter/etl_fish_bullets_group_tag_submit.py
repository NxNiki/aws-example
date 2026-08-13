"""Submit etl_fish_bullets_group_tag.py as a SageMaker PySpark processing job.

Runs from a local machine or a scheduler under the DataScience notebook role
(the only principal the oceanhunter source bucket trusts). Without
``--start``/``--end`` this is the daily incremental run over the rolling
last-3-Beijing-days window; pass an explicit window for backfills:

    # daily incremental
    poetry run python etl_fish_bullets_group_tag_submit.py
    # full-history backfill
    poetry run python etl_fish_bullets_group_tag_submit.py --start 2025-01-01
"""

import argparse

import boto3
from sagemaker.session import Session

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET
from bituslabs_ds.sagemaker_etl import SPARK_COMMON_PY_FILES, spark_processor

INPUT_ROOT = "s3://oceanhunter-production-data-warehouse/transformed_data/cold_data/bullet"
OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_fish_bullets_group_tag"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start", help="Beijing activity date >= this (default: rolling lookback window)")
parser.add_argument("--end", help="Beijing activity date < this (default: tomorrow in Beijing time)")
args = parser.parse_args()

job_args = ["--input-root", INPUT_ROOT, "--output-root", OUTPUT_ROOT]
if args.start:
    job_args += ["--output-start", args.start]
if args.end:
    job_args += ["--output-end", args.end]

processor = spark_processor(
    "fish-bullets-group-tag",
    Session(boto3.Session(region_name=REGION)),
    volume_size_gb=300,
    max_runtime_hours=6,
)
processor.run(
    submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/fish_hunter/etl_fish_bullets_group_tag.py",
    submit_py_files=SPARK_COMMON_PY_FILES,
    arguments=job_args,
    spark_event_logs_s3_uri=f"{OUTPUT_ROOT}/spark-event-logs",
    logs=True,
    wait=True,
)
