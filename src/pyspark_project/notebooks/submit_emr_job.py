import boto3
import os
import time
import logging
from src.pyspark_project.s3_utils import upload_file_to_s3

# Setup logging
os.makedirs('.log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(".log/emr_job_submit.log"),
        logging.StreamHandler()
    ]
)



def wait_for_cluster_ready(cluster_id, region='us-west-2'):
    emr = boto3.client('emr', region_name=region)
    logging.info(f"Waiting for EMR cluster {cluster_id} to be ready...")

    while True:
        response = emr.describe_cluster(ClusterId=cluster_id)
        state = response['Cluster']['Status']['State']
        logging.info(f"Cluster state: {state}")
        if state in ['WAITING', 'RUNNING']:
            logging.info("Cluster is ready!")
            return
        elif state in ['TERMINATING', 'TERMINATED', 'TERMINATED_WITH_ERRORS']:
            logging.error(f"Cluster is not usable (state: {state})")
            raise Exception(f"Cluster is not usable (state: {state})")
        time.sleep(15)


def submit_spark_step(cluster_id, script_s3_path, data_source, output_uri, region='us-west-2'):
    """
    Submits a Spark step to an EMR cluster.
    """
    emr_client = boto3.client('emr', region_name=region)

    step = {
        'Name': 'RedViolationJob',
        'ActionOnFailure': 'CONTINUE',
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                'spark-submit',
                script_s3_path,
                '--data_source', data_source,
                '--output_uri', output_uri
            ]
        }
    }

    response = emr_client.add_job_flow_steps(JobFlowId=cluster_id, Steps=[step])
    step_id = response['StepIds'][0]
    print(f"Step submitted successfully. Step ID: {step_id}")
    return step_id


if __name__ == "__main__":
    # === Configuration ===
    LOCAL_SCRIPT = "example_emr.py"  # Your local Python script
    BUCKET = "xin-config"  # Your S3 bucket
    S3_KEY = "scripts/example_emr.py"
    DATA_SOURCE = "s3://xin-config/food_establishment_data.csv"
    OUTPUT_URI = "s3://xin-config/restaurant_violation_results"
    CLUSTER_ID = "j-WD67QQS5JW2Z"  # Your EMR Cluster ID
    REGION = "us-west-2"

    # === Upload and Submit ===
    s3_script_uri = upload_file_to_s3(LOCAL_SCRIPT, BUCKET, S3_KEY)
    if s3_script_uri:
        submit_spark_step(CLUSTER_ID, s3_script_uri, DATA_SOURCE, OUTPUT_URI, region=REGION)
