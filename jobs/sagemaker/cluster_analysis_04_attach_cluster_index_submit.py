"""
the final step of cluster analysis on wucaishen data:
the cluster index is identified with grouped data (stats on 40 consecutive trials) in previous steps.
this script merges cluster index to enriched data (original metrics for single trials).
"""

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import S3_BUCKET

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
        destination=f"s3://{S3_BUCKET}/ds-data-kmeans/2025",
    )
]

processor.run(
    code="cluster_analysis_04_attach_cluster_index.py",
    inputs=inputs,
    outputs=outputs,
    arguments=[
        "--input",
        input_dir,
        "--output",
        output_dir,
    ],
)
