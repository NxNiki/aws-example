import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from bituslabs_ds.config import S3_BUCKET
from infra.deploy_emr import build_package

input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"

role = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
session = sagemaker.Session()

# Use built-in scikit-learn image (includes pandas)
image_uri = sagemaker.image_uris.retrieve(framework="sklearn", region=session.boto_region_name, version="1.2-1")

processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="etl-processing-job",
    sagemaker_session=session,
)

package_file_uri, package_name = build_package()
inputs = [
    ProcessingInput(
        source=package_file_uri,
        destination=f"{input_dir}/dependencies",
    ),
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
    arguments=["--input", input_dir, "--output", output_dir, "--package", f"{input_dir}/dependencies/{package_name}"],
)
