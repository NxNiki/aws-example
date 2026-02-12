"""
Submit predict_transition_mcmc.py to SageMaker Processing (ScriptProcessor).
Uploads merged_with_transitions.parquet from local output dir to S3, then starts the job.

Usage:
  1. Run analysis_cluster_transition_stats.py locally so merged_with_transitions.parquet exists.
  2. python -m jobs.ss01_wucaishen.predict_transition_mcmc_submit

  Use --no-upload to skip upload and use whatever is already at the S3 input path.
  Use --no-wait to submit and exit without waiting (no console log streaming).
"""

import argparse
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET

# Local path: same layout as analysis_cluster_transition_stats.py output
LOCAL_MERGED_PARQUET = Path(LOCAL_ROOT) / "output_ss01_cluster_transition" / "merged_with_transitions.parquet"

# SageMaker processing paths (inside the container)
INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

# S3: prefix under bucket where merged_with_transitions.parquet is (or will be uploaded)
S3_INPUT_PREFIX = "ds-data-ss01_cluster_transition/input"
S3_OUTPUT_PREFIX = "ds-data-ss01_cluster_transition/mcmc_output"


def upload_merged_to_s3() -> None:
    if not LOCAL_MERGED_PARQUET.exists():
        raise FileNotFoundError(
            f"Local merged parquet not found: {LOCAL_MERGED_PARQUET}\n"
            "Run analysis_cluster_transition_stats.py first."
        )
    s3_key = f"{S3_INPUT_PREFIX}/merged_with_transitions.parquet"
    print(f"Uploading {LOCAL_MERGED_PARQUET} -> s3://{S3_BUCKET}/{s3_key}")
    boto3.client("s3").upload_file(str(LOCAL_MERGED_PARQUET), S3_BUCKET, s3_key)
    print("Upload done.")


# SageMaker job config
ROLE = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
IMAGE_URI = "338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-sagemaker:latest"
INSTANCE_TYPE = "ml.m5.4xlarge"  # 16 vCPU, 64 GB — MCMC is CPU-heavy
INSTANCE_COUNT = 1
MAX_RUNTIME_SECONDS = 24 * 60 * 60  # 24 hours

session = sagemaker.Session()

processor = ScriptProcessor(
    image_uri=IMAGE_URI,
    command=["python3"],
    role=ROLE,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    base_job_name="ss01-predict-transition-mcmc",
    sagemaker_session=session,
    max_runtime_in_seconds=MAX_RUNTIME_SECONDS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit MCMC transition job to SageMaker.")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading merged parquet; use existing data at S3 input path.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit job and exit; do not wait or stream logs to console.",
    )
    args = parser.parse_args()

    if not args.no_upload:
        upload_merged_to_s3()
    else:
        print("Skipping upload (--no-upload). Using existing S3 input.")

    time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    inputs = [
        ProcessingInput(
            source=f"s3://{S3_BUCKET}/{S3_INPUT_PREFIX}",
            destination=INPUT_DIR,
        )
    ]
    outputs = [
        ProcessingOutput(
            source=OUTPUT_DIR,
            destination=f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/result_{time_tag}/",
        )
    ]

    processor.run(
        code=f"{LOCAL_ROOT}/jobs/ss01_wucaishen/predict_transition_mcmc.py",
        inputs=inputs,
        outputs=outputs,
        arguments=[
            "--input",
            INPUT_DIR,
            "--output",
            OUTPUT_DIR,
        ],
        wait=not args.no_wait,
        logs=not args.no_wait,
    )


if __name__ == "__main__":
    main()
