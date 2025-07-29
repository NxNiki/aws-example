import logging
import os

import boto3

from bituslabs_ds.s3_utils import upload_file_to_s3

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


def submit_spark_step(
    cluster_id: str, script_s3_path: str, data_source: str, output_uri: str, region: str = "us-west-2"
):
    """
    Submits a Spark step to an EMR cluster.
    """
    emr_client = boto3.client("emr", region_name=region)

    step = {
        "Name": "RedViolationJob",
        "ActionOnFailure": "CONTINUE",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                script_s3_path,
                "--data_source",
                data_source,
                "--output_uri",
                output_uri,
            ],
        },
    }

    response = emr_client.add_job_flow_steps(JobFlowId=cluster_id, Steps=[step])
    step_id = response["StepIds"][0]
    logger.info(f"Step submitted successfully. Step ID: {step_id}")
    return step_id


if __name__ == "__main__":

    # Setup logging
    os.makedirs("../.log", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(".log/emr_job_submit.log"), logging.StreamHandler()],
    )

    # === Configuration ===
    LOCAL_SCRIPT = "data_loader_pyspark_example.py"
    BUCKET = "bituslabs-team-ai"
    S3_KEY = "data_loader_pyspark_example.py"
    DATA_SOURCE = f"s3://{BUCKET}/test/food_establishment_data.csv"
    OUTPUT_URI = f"s3://{BUCKET}/test/restaurant_violation_results"
    CLUSTER_ID = "j-WD67QQS5JW2Z"  # Your EMR Cluster ID
    REGION = "us-west-2"

    # === Upload and Submit ===
    s3_script_uri = upload_file_to_s3(LOCAL_SCRIPT, BUCKET, S3_KEY)
    if s3_script_uri:
        submit_spark_step(CLUSTER_ID, s3_script_uri, DATA_SOURCE, OUTPUT_URI, region=REGION)
