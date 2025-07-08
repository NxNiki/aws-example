from __future__ import annotations

import io
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError
from pyarrow import fs
from pyarrow.dataset import Dataset, dataset
from pyspark.sql import DataFrame as SparkDataFrame

from bituslabs_ds.config import DEFAULT_MAX_JOBS, REGION

s3_client = boto3.client("s3")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


def parse_bucket_name(bucket: str) -> str:
    if bucket.endswith("/"):
        logger.info(f"remove '/' from {bucket}")
        bucket = bucket[:-1]
    if bucket.startswith("s3a://"):
        logger.info(f"remove 's3a://' from {bucket}")
        bucket = bucket[len("s3a://") :]
    if bucket.startswith("s3://"):
        logger.info(f"remove 's3://' from {bucket}")
        bucket = bucket[len("s3://") :]

    return bucket


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parse an S3 URI (s3://, s3a://, s3n://) into (bucket, key).

    :param s3_path: S3 URI like s3://bucket/key
    :return: Tuple of (bucket, key)
    """
    if not s3_path.startswith(("s3://", "s3a://", "s3n://")):
        raise ValueError(f"Unsupported S3 URI scheme: {s3_path}")

    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket:
        raise ValueError(f"Missing bucket in S3 URI: {s3_path}")

    return bucket, key


def read_to_pandas_df(bucket: str, key: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a CSV file from S3 and return it as a Pandas DataFrame."""
    bucket = parse_bucket_name(bucket)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(response["Body"], usecols=columns)


def write_df_to_s3(data: Union[pd.DataFrame, SparkDataFrame], bucket: str, key: str) -> None:
    if isinstance(data, pd.DataFrame):
        return write_pandas_to_s3(data, bucket, key)
    elif isinstance(data, SparkDataFrame):
        return write_spark_to_s3(data, bucket, key)
    else:
        raise TypeError("Unsupported DataFrame type.")


def write_pandas_to_s3(data: pd.DataFrame, bucket: str, key: str) -> None:
    """Write a Pandas DataFrame to a CSV file in S3."""
    bucket = parse_bucket_name(bucket)
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

    logger.info(f"Writing {key} to {bucket}")


def write_spark_to_s3(data: SparkDataFrame, bucket: str, key: str, file_format: str = "csv") -> None:
    """Write a PySpark DataFrame to S3 in CSV or Parquet format."""
    bucket = parse_bucket_name(bucket)
    s3_path = f"s3://{bucket}/{key}"

    if file_format not in ("csv", "parquet"):
        raise ValueError("PySpark supports only 'csv' or 'parquet' formats.")

    try:
        write_options = {"header": "true", "compression": "gzip"} if file_format == "csv" else {}
        data.write.mode("overwrite").options(**write_options).format(file_format).save(s3_path)
        logger.info(f"Successfully wrote PySpark DataFrame to {s3_path}")
    except Exception as e:
        logger.error(f"Failed to write PySpark DataFrame to S3: {e}")
        raise


def upload_file_to_s3(local_path: Union[str, Path], s3_bucket: str, s3_key: str) -> Optional[str]:
    """
    Uploads a local file to an S3 bucket. Add content type so we can open uploaded files directly on aws.

    :param local_path: Path to the local Python file.
    :param s3_bucket: Name of the S3 bucket.
    :param s3_key: S3 object key (e.g., 'scripts/red_violations.py').
    :return: Full S3 URI of the uploaded script.
    """

    filename = os.path.basename(local_path).lower()
    content_type_map = {
        # ".csv": "text/csv",
        ".csv": "text/plain",  # make it plain so that we can view small csv file online
        ".json": "application/json",
        ".log": "text/plain",
        ".txt": "text/plain",
        ".parquet": "application/x-parquet",
        ".html": "text/html",
    }
    extra_args = {"ContentType": "application/octet-stream"}
    for ext, content_type in content_type_map.items():
        if filename.endswith(ext + ".gz"):
            extra_args = {"ContentType": content_type, "ContentEncoding": "gzip"}
        if filename.endswith(ext):
            extra_args = {"ContentType": content_type}

    s3_bucket = parse_bucket_name(s3_bucket)

    try:
        s3_client.upload_file(local_path, s3_bucket, s3_key, ExtraArgs=extra_args)
        logger.info(f"Uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
        return f"s3://{s3_bucket}/{s3_key}"
    except FileNotFoundError:
        logger.info("Error: The specified file was not found.")
    except NoCredentialsError:
        logger.info("Error: AWS credentials not available.")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")

    return None


def list_s3_files(bucket: str, prefix: str, pattern: Optional[str] = None) -> List[str]:
    """
    List all S3 files in bucket starting from the prefix, filtering with an optional regex pattern.

    :param bucket: S3 bucket name
    :param prefix: S3 key prefix (acts like a root folder)
    :param pattern: Optional regex pattern to match the key
    :return: List of matching s3 keys (not including bucket)
    """
    logger.info(f"Listing S3 files in: {bucket}/{prefix}, with pattern: {pattern}")
    matching_keys = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if pattern is None or re.search(pattern, key):  # Changed from match to search
                s3_uri = f"s3://{bucket}/{key}"
                matching_keys.append(s3_uri)
                logging.info(f"Found {s3_uri}")

    logger.info(f"Found {len(matching_keys)} S3 keys")
    return matching_keys


def _read_file(file: str, columns: Optional[List[str]]) -> pd.DataFrame:
    logger.info(f"Reading {file}")
    bucket, file = parse_s3_path(file)
    return read_to_pandas_df(bucket, file, columns)


def read_files(
    files: Union[List[str], str],
    local_cache_path: Optional[str] = None,
    columns: Optional[List[str]] = None,
    max_workers: int = DEFAULT_MAX_JOBS,
    parallel_mode: Literal["thread", "process", "none"] = "thread",
    reload: bool = False,
) -> pd.DataFrame:
    """
    Read CSV files from S3 using optional parallelization.

    :param files: List of S3 paths to CSV files.
    :param local_cache_path: Local cache path.
    :param columns: List of column names to read from each file.
    :param max_workers: Number of workers to use in parallel execution.
    :param parallel_mode: Parallel execution strategy: 'thread', 'process', or 'none'.
    :param reload: Whether to reload files from S3 or not.
    :return: Concatenated DataFrame of all read files.
    """

    if local_cache_path is not None and os.path.exists(local_cache_path) and not reload:
        logger.info(f"Found local cache at {local_cache_path}")
        data = pd.read_csv(local_cache_path)
        return data

    read_func = partial(_read_file, columns=columns)

    if isinstance(files, str):
        files = [files]

    if parallel_mode == "none" or max_workers <= 1:
        dfs = [read_func(file) for file in files]
    else:
        logger.info(f"read files using {max_workers} workers")
        executor_cls: Callable = ThreadPoolExecutor if parallel_mode == "thread" else ProcessPoolExecutor
        dfs = []

        with executor_cls(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(read_func, file): file for file in files}
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    dfs.append(future.result())
                except Exception as e:
                    logger.error(f"Failed to read {file}: {e}")

    data = pd.concat(dfs, ignore_index=True)
    if local_cache_path is not None:
        data.to_csv(local_cache_path, index=False)
    return data


def read_dataset(file_path: str, region: str = REGION, data_format: str = "parquet") -> Dataset:
    file_path = parse_bucket_name(file_path)
    logger.info(f"Read data from: {file_path}")
    s3 = fs.S3FileSystem(region=region)
    ds = dataset(file_path, format=data_format, partitioning="hive", filesystem=s3)  # recognizes year=, month=, etc.
    logger.info("Finished reading dataset")

    return ds
