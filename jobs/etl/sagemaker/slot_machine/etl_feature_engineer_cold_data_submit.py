"""Submit etl_feature_engineer_cold_data.py as one SageMaker PySpark processing job.

Runs from a local machine. The job executes under the DataScience notebook
role (the only principal the slotmachine source bucket trusts) and computes
all four SS03 ai_group slices in a single Spark app (one source scan).

Without ``--output-start``/``--output-end`` this is the monthly incremental
run: it recomputes whole months from the previous calendar month through
today (dynamic ``year=/month=`` partition overwrite; older history untouched).

    # monthly incremental, all groups
    poetry run python etl_feature_engineer_cold_data_submit.py
    # explicit backfill window / subset of groups
    poetry run python ..._submit.py --groups ai --output-start 2026-02-01 --output-end 2026-08-01
"""

import argparse

import boto3
from sagemaker.session import Session
from sagemaker.spark.processing import PySparkProcessor

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET

# NOT the repo-wide SAGEMAKER_ROLE: the slotmachine bucket policy (owned by
# another account) only trusts this role.
ROLE = "arn:aws:iam::338568447110:role/service-role/AmazonSageMaker-ExecutionRole-20250102T151291"

INPUT_ROOT = "s3://slotmachine-production-data-warehouse/transformed_data/partition_cold_data/bet_order"
OUTPUT_ROOT_BASE = f"s3://{S3_BUCKET}/etl-results/jobs"

INSTANCE_TYPE = "ml.m5.4xlarge"  # 16 vCPU, 64 GB memory per node
INSTANCE_COUNT = 3
# Per-user window functions shuffle-spill to local disk; the 30 GB SageMaker
# default is too small for multi-month windows.
VOLUME_SIZE_GB = 300

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--groups", default="default,ai,ab_test_a,ab_test_b")
parser.add_argument("--output-start", help="recompute months from this date's month (default: previous month)")
parser.add_argument("--output-end", help="exclusive end date (default: tomorrow UTC)")
parser.add_argument("--no-wait", action="store_true", help="submit and return without streaming logs")
parser.add_argument(
    "--output-root-base",
    default=OUTPUT_ROOT_BASE,
    help="override the output prefix (e.g. a scratch prefix for a validation run)",
)
args = parser.parse_args()

job_args = ["--input-root", INPUT_ROOT, "--output-root-base", args.output_root_base, "--groups", args.groups]
if args.output_start:
    job_args += ["--output-start", args.output_start]
if args.output_end:
    job_args += ["--output-end", args.output_end]

processor = PySparkProcessor(
    base_job_name="ss03-feature-engineer-cold-data",
    framework_version="3.3",
    role=ROLE,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    volume_size_in_gb=VOLUME_SIZE_GB,
    max_runtime_in_seconds=6 * 60 * 60,
    sagemaker_session=Session(boto3.Session(region_name=REGION)),
)
processor.run(
    submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_feature_engineer_cold_data.py",
    arguments=job_args,
    spark_event_logs_s3_uri=f"{OUTPUT_ROOT_BASE}/output_ss03_feature_engineer/spark-event-logs",
    logs=not args.no_wait,
    wait=not args.no_wait,
)
