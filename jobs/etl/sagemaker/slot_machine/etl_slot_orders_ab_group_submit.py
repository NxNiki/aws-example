"""Submit etl_slot_orders_ab_group.py as a SageMaker PySpark processing job.

Runs from a local machine or a scheduler under the DataScience notebook role
(the only principal the slotmachine source bucket trusts). Without
``--start``/``--end`` this is the daily incremental run over the rolling
last-3-Beijing-days window; pass an explicit window for backfills:

    # daily incremental
    poetry run python etl_slot_orders_ab_group_submit.py
    # full-history backfill
    poetry run python etl_slot_orders_ab_group_submit.py --start 2025-11-24
"""

import argparse

import boto3
from sagemaker.session import Session

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET
from bituslabs_ds.sagemaker_etl import SPARK_COMMON_PY_FILES, spark_processor

INPUT_ROOT = "s3://slotmachine-production-data-warehouse/transformed_data/partition_cold_data/bet_order"
OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_slot_orders_ab_group"

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
    "slot-orders-ab-group",
    Session(boto3.Session(region_name=REGION)),
    volume_size_gb=300,
    max_runtime_hours=6,
)
processor.run(
    submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_slot_orders_ab_group.py",
    submit_py_files=SPARK_COMMON_PY_FILES,
    arguments=job_args,
    spark_event_logs_s3_uri=f"{OUTPUT_ROOT}/spark-event-logs",
    logs=True,
    wait=True,
)
