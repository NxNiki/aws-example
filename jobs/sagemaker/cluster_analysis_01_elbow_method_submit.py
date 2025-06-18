from datetime import datetime

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import S3_BUCKET

# directory to save output data locally on sagemaker instance. it will be uploaded to s3.
# input data is directly read from s3 bucket, so we do not define it here.
output_dir = "/opt/ml/processing/output"

role = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
session = sagemaker.Session()

image_uri = "338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-sagemaker:latest"
print(f"image_uri: {image_uri}")

processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.12xlarge",
    instance_count=1,
    base_job_name="elbow-method",
    sagemaker_session=session,
)

time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outputs = [
    ProcessingOutput(
        source=f"{output_dir}",
        destination=f"s3://{S3_BUCKET}/ds-data-kmeans/elbow_method_{time_tag}",
    )
]

processor.run(
    code="processing_cluster_analysis_01_elbow_method.py",
    outputs=outputs,
    arguments=[
        "--output",
        output_dir,
    ],
)
