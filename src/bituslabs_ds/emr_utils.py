import logging
import time
from typing import Tuple

import boto3

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


def start_emr_cluster(
    cluster_name: str,
    log_uri: str,
    release_label: str = "emr-6.15.0",
    instance_type: str = "m5.xlarge",
    instance_count: int = 3,
    region: str = "us-west-2",
) -> Tuple[str, str]:
    """
    Starts a new EMR cluster and returns the cluster ID and state.

    Args:
        cluster_name: Name of the EMR cluster.
        log_uri: S3 path for storing logs (e.g., s3://your-bucket/logs/).
        release_label: EMR release label.
        instance_type: Type of EC2 instances.
        instance_count: Number of core nodes (1 master, rest core).
        region: AWS region.

    Returns:
        Tuple of (cluster_id, initial_cluster_state).
    """
    emr_client = boto3.client("emr", region_name=region)

    response = emr_client.run_job_flow(
        Name=cluster_name,
        LogUri=log_uri,
        ReleaseLabel=release_label,
        Applications=[{"Name": "Spark"}],
        Instances={
            "InstanceGroups": [
                {
                    "Name": "Master nodes",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                    "InstanceType": instance_type,
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core nodes",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                    "InstanceType": instance_type,
                    "InstanceCount": instance_count - 1,
                },
            ],
            "Ec2KeyName": "your-ec2-key-name",  # Optional, if you want SSH access
            "KeepJobFlowAliveWhenNoSteps": True,
            "TerminationProtected": False,
        },
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
        VisibleToAllUsers=True,
    )

    cluster_id = response["JobFlowId"]
    print(f"Started EMR cluster with ID: {cluster_id}")
    return cluster_id, response["ResponseMetadata"]["HTTPStatusCode"]


def wait_for_cluster_ready(cluster_id: str, region: str = "us-west-2"):
    emr = boto3.client("emr", region_name=region)
    logger.info(f"Waiting for EMR cluster {cluster_id} to be ready...")

    while True:
        response = emr.describe_cluster(ClusterId=cluster_id)
        state = response["Cluster"]["Status"]["State"]
        logger.info(f"Cluster state: {state}")
        if state in ["WAITING", "RUNNING"]:
            logger.info("Cluster is ready!")
            return
        elif state in ["TERMINATING", "TERMINATED", "TERMINATED_WITH_ERRORS"]:
            logger.error(f"Cluster is not usable (state: {state})")
            raise Exception(f"Cluster is not usable (state: {state})")
        time.sleep(15)


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
