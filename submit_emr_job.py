import boto3
import os
from botocore.exceptions import NoCredentialsError


def upload_script_to_s3(local_path, s3_bucket, s3_key):
    """
    Uploads a local Python script to an S3 bucket.

    :param local_path: Path to the local Python file.
    :param s3_bucket: Name of the S3 bucket.
    :param s3_key: S3 object key (e.g., 'scripts/red_violations.py').
    :return: Full S3 URI of the uploaded script.
    """
    s3 = boto3.client('s3')

    try:
        s3.upload_file(local_path, s3_bucket, s3_key)
        print(f"Uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
        return f"s3://{s3_bucket}/{s3_key}"
    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except NoCredentialsError:
        print("Error: AWS credentials not available.")
    except Exception as e:
        print(f"Unexpected error: {e}")


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
    DATA_SOURCE = "s3://amzn-s3-demo-bucket/food-establishment-data.csv"
    OUTPUT_URI = "s3://xin-config/restaurant_violation_results"
    CLUSTER_ID = "j-2YEHPSL08JSF4"  # Your EMR Cluster ID
    REGION = "us-west-2"

    # === Upload and Submit ===
    s3_script_uri = upload_script_to_s3(LOCAL_SCRIPT, BUCKET, S3_KEY)
    if s3_script_uri:
        submit_spark_step(CLUSTER_ID, s3_script_uri, DATA_SOURCE, OUTPUT_URI, region=REGION)
