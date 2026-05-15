import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import boto3

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import upload_file_to_s3

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

rootdir = Path(__file__).parent.parent


def upload_bootstrap_script() -> Optional[str]:
    bootstrap_script = rootdir / "infra/bootstrap/install-package.sh"
    bootstrap_script_uri = upload_file_to_s3(bootstrap_script, S3_BUCKET, "package/install-package.sh")

    return bootstrap_script_uri


def build_package() -> Tuple[str, str]:
    print("Building package with poetry...")
    subprocess.run(["poetry", "build"], check=True)
    dist_files = os.listdir(rootdir / "dist")
    wheel = next(f for f in dist_files if f.endswith(".whl"))
    package_file = rootdir / os.path.join("dist", wheel)
    package_file_uri = upload_file_to_s3(package_file, S3_BUCKET, f"package/{package_file.name}")
    if package_file_uri is None:
        raise RuntimeError(f"Failed to upload package {package_file.name} to s3://{S3_BUCKET}")
    return package_file_uri, package_file.name


def start_emr_cluster(
    cluster_name: str,
    log_uri: str,
    release_label: str = "emr-7.9.0",
    instance_type: str = "m5.xlarge",
    instance_count: int = 3,
    region: str = "us-west-2",
    bootstrap_script_uri: Optional[str] = None,
    package_uri: Optional[str] = None,
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
        bootstrap_script_uri: S3 path to the bootstrap script (optional).
        package_uri: S3 path to the Python package or requirements file (optional).

    Returns:
        Tuple of (cluster_id, initial_cluster_state).
    """
    emr_client = boto3.client("emr", region_name=region)

    instances_config = {
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
        "Ec2KeyName": "xin-key-us-west2",  # Optional, if you want SSH access
        "KeepJobFlowAliveWhenNoSteps": True,  # Set to True to make sure log files are uploaded to s3.
        "TerminationProtected": False,
        "Ec2SubnetId": "subnet-02571e70cb058d27a",  # your public subnet here
        "EmrManagedMasterSecurityGroup": "sg-069ce0aa8db40f042",
        "EmrManagedSlaveSecurityGroup": "sg-0df70a2677f24ff31",
    }

    job_flow_args = {
        "Name": cluster_name,
        "LogUri": log_uri,
        "ReleaseLabel": release_label,
        "Applications": [{"Name": "Spark"}],
        "Instances": instances_config,
        "JobFlowRole": "EMR_EC2_DefaultRole",
        "ServiceRole": "EMR_DefaultRole",
        "VisibleToAllUsers": True,
        "AutoTerminationPolicy": {"IdleTimeout": 600},  # terminate clusters after it being idle for 600 seconds
    }

    if bootstrap_script_uri and package_uri:
        job_flow_args["BootstrapActions"] = [
            {
                "Name": "Install package",
                "ScriptBootstrapAction": {
                    "Path": bootstrap_script_uri,
                    "Args": [package_uri],
                },
            }
        ]

    response = emr_client.run_job_flow(**job_flow_args)

    cluster_id = response["JobFlowId"]
    print(f"Started EMR cluster with ID: {cluster_id}")
    return cluster_id, response["ResponseMetadata"]["HTTPStatusCode"]


def wait_for_cluster_ready(cluster_id: str, region: str = "us-west-2"):
    emr = boto3.client("emr", region_name=region)
    logger.info(f"Waiting for EMR cluster {cluster_id} to be ready...")

    last_state = ""
    while True:
        response = emr.describe_cluster(ClusterId=cluster_id)
        state = response["Cluster"]["Status"]["State"]
        if state != last_state:
            logger.info(f"Cluster state: {state}")
            last_state = state
        if state in ["WAITING", "RUNNING"]:
            logger.info("Cluster is ready!")
            return
        elif state in ["TERMINATING", "TERMINATED", "TERMINATED_WITH_ERRORS"]:
            logger.error(f"Cluster is not usable (state: {state})")
            raise Exception(f"Cluster is not usable (state: {state})")
        time.sleep(30)
