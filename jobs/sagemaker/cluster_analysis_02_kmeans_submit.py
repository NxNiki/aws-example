"""
WIP...
"""

from datetime import datetime

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import IMAGE_URI, S3_BUCKET, SAGEMAKER_ROLE

# directory to save output data locally on sagemaker instance. it will be uploaded to s3.
# the input data is read and saved to s3 in the previous step, so we can read it directly.
input_path_data = "/opt/ml/processing/input/output"
input_path_features = "/opt/ml/processing/input/features"
output_dir = "/opt/ml/processing/output"

session = sagemaker.Session()
processor = ScriptProcessor(
    image_uri=IMAGE_URI,
    command=["python3"],
    role=SAGEMAKER_ROLE,
    instance_type="ml.m5.4xlarge",
    instance_count=1,
    base_job_name="elbow-method",
    sagemaker_session=session,
)

input_time_tag = "_2025-06-18_17-17-08"
inputs = [
    ProcessingInput(
        source=f"s3://{S3_BUCKET}/ds-data-kmeans/elbow_method{input_time_tag}/output/wucaishen_grouped_stat_output_24.csv",
        destination="/opt/ml/processing/input/output",
    ),
    ProcessingInput(
        source=f"s3://{S3_BUCKET}/ds-data-kmeans/elbow_method{input_time_tag}/features/",
        destination="/opt/ml/processing/input/features",
    ),
]

output_time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outputs = [
    ProcessingOutput(
        source=f"{output_dir}",
        destination=f"s3://{S3_BUCKET}/ds-data-kmeans/kmeans_{output_time_tag}",
    )
]

processor.run(
    code="cluster_analysis_02_kmeans.py",
    inputs=inputs,
    outputs=outputs,
    arguments=[
        "--input_path_data",
        input_path_data,
        "--input_path_features",
        input_path_features,
        "--output",
        output_dir,
    ],
)
