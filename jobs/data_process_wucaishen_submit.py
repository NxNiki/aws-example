import logging
import os

import boto3

from bituslabs_ds.config import REGION, S3_BUCKET
from bituslabs_ds.s3_utils import upload_file_to_s3
from infra.deploy_emr import build_package, start_emr_cluster, upload_bootstrap_script, wait_for_cluster_ready

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def submit_spark_step(cluster_id: str, script_s3_path: str, region: str = "us-west-2"):
    """
    Submits a Spark step to an EMR cluster.
    """
    emr_client = boto3.client("emr", region_name=region)

    step = {
        "Name": "RedViolationJob",
        "ActionOnFailure": "CONTINUE",  # "CONTINUE", "CANCEL_AND_WAIT", "TERMINATE_CLUSTER"
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                script_s3_path,
            ],
        },
    }

    response = emr_client.add_job_flow_steps(JobFlowId=cluster_id, Steps=[step])
    step_id = response["StepIds"][0]
    logger.info(f"Step submitted successfully. Step ID: {step_id}")
    return step_id


if __name__ == "__main__":

    # Setup logging
    os.makedirs(".log", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(".log/data_process_wucaishen_submit.log"), logging.StreamHandler()],
    )

    package_file_uri = build_package()
    bootstrap_script_uri = upload_bootstrap_script()
    cluster_id, status = start_emr_cluster(
        cluster_name="xin-spark-cluster",
        log_uri=f"s3://{S3_BUCKET}/emr-logs/data_process_wucaishen/",
        instance_type="m5.4xlarge",  # 64GB memory
        instance_count=3,
        region=REGION,
        bootstrap_script_uri=bootstrap_script_uri,
        package_uri=package_file_uri,
    )
    print(f"Cluster ID: {cluster_id}, Status Code: {status}")

    # === Configuration ===
    LOCAL_SCRIPT = "data_process_wucaishen.py"
    S3_KEY = "emr_jobs/data_process_wucaishen.py"

    # === Upload and Submit ===
    s3_script_uri = upload_file_to_s3(LOCAL_SCRIPT, S3_BUCKET, S3_KEY)

    if s3_script_uri:
        wait_for_cluster_ready(cluster_id, region=REGION)
        submit_spark_step(cluster_id, s3_script_uri, region=REGION)
