"""
This is obsolete code, refer the etl.py for query execution on athena and redshift
"""

import logging
import os
import time
from typing import Optional

import boto3
import pandas as pd

from bituslabs_ds.config import DEFAULT_ATHENA_OUTPUT, S3_BUCKET
from bituslabs_ds.s3_utils import parse_s3_path, read_to_pandas_df, write_df_to_s3

athena = boto3.client("athena")
s3 = boto3.client("s3")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


def execute_query(
    query: str,
    database: str,
    s3_file_path: Optional[str] = None,
    local_cache: Optional[str] = None,
    overwrite_cache: Optional[bool] = False,
    return_result: Optional[bool] = False,
) -> Optional[pd.DataFrame]:
    """
    execute a query and return results to a pandas DataFrame.
    :param query:
    :param database:
    :param s3_file_path: save results to s3_file_path; otherwise, results will be saved to the default s3 path for
        athena outputs
    :param local_cache: specify a local cache path to avoid repeated running queries.
    :param overwrite_cache: whether to overwrite local cache if it exists.
    :param return_result:
    :return:
    """

    if local_cache is not None and os.path.exists(local_cache) and not overwrite_cache:
        logger.info(f"Found local cache at {local_cache}")
        df = pd.read_csv(local_cache)
    else:
        query_execution_id = submit_query(query, database)

        if not return_result:
            return None

        df = get_query_result(query_execution_id, DEFAULT_ATHENA_OUTPUT)
        if local_cache is not None:
            os.makedirs(os.path.dirname(local_cache), exist_ok=True)
            logger.info(f"Writing to local cache at {local_cache}")
            df.to_csv(local_cache, index=False)

        if s3_file_path is not None:
            logger.info(f"copy result from {DEFAULT_ATHENA_OUTPUT}/{query_execution_id}.csv to {s3_file_path}")
            bucket, key = parse_s3_path(s3_file_path)
            bucket_source, key_source = parse_s3_path(f"{DEFAULT_ATHENA_OUTPUT}/{query_execution_id}.csv")

            if bucket != bucket_source:
                raise Exception(f"cannot copy output across buckets: dest: {bucket}, source: {bucket_source}")

            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key_source}, Key=key)
            s3.delete_object(Bucket=bucket, Key=key_source)

    return df


def submit_query(query: str, database: str) -> str:
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": DEFAULT_ATHENA_OUTPUT},
    )
    query_execution_id = response["QueryExecutionId"]
    logger.info(f"Started query: {query_execution_id}")

    return query_execution_id


def check_query_status(query_execution_id: str) -> str:
    result = athena.get_query_execution(QueryExecutionId=query_execution_id)
    status = result["QueryExecution"]["Status"]["State"]
    logger.info(f"check query {query_execution_id} status: {status}")

    return status


def wait_query_finish(query_execution_id: str, interval: int = 10):
    while True:
        status = check_query_status(query_execution_id)
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(interval)

    logger.info(f"Query finished with status: {status}")
    if status != "SUCCEEDED":
        raise Exception(f"Query failed with status: {status}")


def get_query_result(query_execution_id: str, s3_path: Optional[str] = None) -> pd.DataFrame:
    """
    :param query_execution_id:
    :param s3_path: read results from s3_path which is faster.
    :return:
    """
    wait_query_finish(query_execution_id)
    logger.info(f"get query result of job: {query_execution_id}")

    if s3_path is None:
        # This works for relative small dataset. For large data directly download data from S3 bucket.
        paginator = athena.get_paginator("get_query_results")
        response_iterator = paginator.paginate(QueryExecutionId=query_execution_id)

        all_rows = []
        columns = []

        for i, page in enumerate(response_iterator):
            rows = page["ResultSet"]["Rows"]

            # The first row is the header
            if i == 0:
                columns = [col.get("VarCharValue", "") for col in rows[0]["Data"]]
                data_rows = rows[1:]
            else:
                data_rows = rows

            for row in data_rows:
                values = [col.get("VarCharValue", "") for col in row["Data"]]
                all_rows.append(values)

        df = pd.DataFrame(all_rows, columns=columns)
    else:
        # read directly from s3 path:
        s3_path = s3_path.rstrip("/")
        full_s3_path = f"{s3_path}/{query_execution_id}.csv"
        logger.info(f"get data: {full_s3_path}")
        bucket, key = parse_s3_path(full_s3_path)
        df = read_to_pandas_df(bucket, key)

    return df


if __name__ == "__main__":

    athena_database = "ag_share_data"
    athena_query = """
        SELECT type, column1, column2, column3, column4, column5, column6, column7, column8
        FROM sampled_slotorders_timeblocks
        WHERE bet_amount <= 500
        ORDER BY bet_time
        LIMIT 100
    """

    execution_id = submit_query(athena_query, athena_database)
    dataframe = get_query_result(execution_id)
    print(dataframe.head())
