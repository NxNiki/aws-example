import io
import logging
import os
from typing import List, Optional

import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError

os.makedirs("../.log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("../.log/s3_utils.log"), logging.StreamHandler()],
)

# Initialize S3 client globally
s3_client = boto3.client("s3")


def read_to_pandas_df(bucket: str, key: str) -> pd.DataFrame:
    """Read a CSV file from S3 and return it as a Pandas DataFrame."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(response["Body"])


def write_pandas_df(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write a Pandas DataFrame to a CSV file in S3."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())


def upload_file_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> Optional[str]:
    """
    Uploads a local file to an S3 bucket.

    :param local_path: Path to the local Python file.
    :param s3_bucket: Name of the S3 bucket.
    :param s3_key: S3 object key (e.g., 'scripts/red_violations.py').
    :return: Full S3 URI of the uploaded script.
    """

    try:
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        logging.info(f"Uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
        return f"s3://{s3_bucket}/{s3_key}"
    except FileNotFoundError:
        logging.info("Error: The specified file was not found.")
    except NoCredentialsError:
        logging.info("Error: AWS credentials not available.")
    except Exception as e:
        logging.info(f"Unexpected error: {e}")

    return None


def list_s3_files(bucket: str, prefix: str, suffix: Optional[str] = None) -> List[str]:
    """
    list all S3 files in bucket with prefix and suffix
    :param bucket:
    :param prefix:
    :param suffix:
    :return:
    """
    logging.info(f"list s3 files: {bucket}/{prefix}*{suffix}")
    matching_keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                matching_keys.append(f"s3a://{bucket}/{key}")
                logging.info(f"Found {key}")

    return matching_keys


if __name__ == "__main__":
    list_s3_files("hyber-slot", "wucaishen_oringaldata/", ".csv.gz")
    # list_s3_files("xin-config", "", ".csv")
