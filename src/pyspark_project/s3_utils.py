import logging
import os
from typing import List, Optional

import boto3
from botocore.exceptions import NoCredentialsError

os.makedirs("../.log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("../.log/s3_utils.log"), logging.StreamHandler()],
)


def upload_file_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> Optional[str]:
    """
    Uploads a local file to an S3 bucket.

    :param local_path: Path to the local Python file.
    :param s3_bucket: Name of the S3 bucket.
    :param s3_key: S3 object key (e.g., 'scripts/red_violations.py').
    :return: Full S3 URI of the uploaded script.
    """
    s3 = boto3.client("s3")

    try:
        s3.upload_file(local_path, s3_bucket, s3_key)
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
    s3 = boto3.client("s3")
    logging.info(f"list s3 files: {bucket}/{prefix}*{suffix}")
    matching_keys = []
    paginator = s3.get_paginator("list_objects_v2")
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
