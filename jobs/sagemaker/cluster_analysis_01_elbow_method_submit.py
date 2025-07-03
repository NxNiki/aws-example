from datetime import datetime

import sagemaker
from sagemaker.processing import ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import IMAGE_URI, S3_BUCKET, SAGEMAKER_ROLE

# directory to save output data locally on sagemaker instance. it will be uploaded to s3.
# input data is directly read from s3 bucket, we do not define it here as that will make all data in s3 prefix
# downloaded to sagemaker
output_dir = "/opt/ml/processing/output"

session = sagemaker.Session()
processor = ScriptProcessor(
    image_uri=IMAGE_URI,
    command=["python3"],
    role=SAGEMAKER_ROLE,
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
    code="cluster_analysis_01_elbow_method.py",
    outputs=outputs,
    arguments=[
        "--output",
        output_dir,
    ],
)
