"""
get some statistics from mahjiang streak user data to support math tabel inferernce.
"""

from datetime import datetime

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET

# directories on sagemaker to hold data:
input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"

role = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
image_uri = "338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-sagemaker:latest"
session = sagemaker.Session()

print(f"image_uri: {image_uri}")
processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.12xlarge",
    instance_count=1,
    base_job_name="mahjiang-streak-stats",
    sagemaker_session=session,
)


time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outputs = [
    ProcessingOutput(
        source=f"{output_dir}",
        destination=f"s3://{S3_BUCKET}/ds-data-mahjiang_streak_stats/2025_{time_tag}/",
    )
]

processor.run(
    code=f"{LOCAL_ROOT}/jobs/mahjiang_streak/analysis_mahjiang_streak_stats.py",
    outputs=outputs,
    arguments=[
        "--output",
        output_dir,
    ],
    wait=False,
)
