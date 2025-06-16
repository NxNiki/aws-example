from pathlib import Path

import sagemaker
from sagemaker import image_uris
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import S3_BUCKET

input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"

role = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
session = sagemaker.Session()

# image_uri = image_uris.retrieve(
#     framework="sklearn",           # or "pytorch", "xgboost", "tensorflow", etc.
#     region="us-west-2",            # your region
#     version="1.0-1",               # framework version
#     instance_type="ml.m5.xlarge",  # optional; helps select CPU vs GPU image
# )

image_uri = "338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-sagemaker:latest"

print(f"image_uri: {image_uri}")

processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.12xlarge",
    instance_count=1,
    base_job_name="attach-cluster-index",
    sagemaker_session=session,
)

inputs = [
    ProcessingInput(
        source=f"s3://bituslabs-team-ai/wucaishen_processed_data/",
        destination=f"{input_dir}/wucaishen_processed_data/",
    ),
    ProcessingInput(
        source="s3://bituslabs-team-ai/wucaishen_analysis_kmeans/output/", destination=f"{input_dir}/kmeans_output/"
    ),
]

outputs = [
    ProcessingOutput(
        source=f"{output_dir}",
        destination=f"s3://{S3_BUCKET}/ds-data-kmeans/test",
    )
]

processor.run(
    code="processing_attach_cluster_index.py",
    inputs=inputs,
    outputs=outputs,
    arguments=[
        "--input",
        input_dir,
        "--output",
        output_dir,
    ],
)
