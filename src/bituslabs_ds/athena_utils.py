import time

import boto3
import pandas as pd

from bituslabs_ds.config import ATHENA_OUTPUT, S3_BUCKET
from bituslabs_ds.s3_utils import write_df_to_s3

athena = boto3.client("athena")
s3 = boto3.client("s3")


def execute_query(query: str, database: str, data_output: str) -> pd.DataFrame:
    query_execution_id = submit_query(query, database)
    df = get_query_result(query_execution_id)
    write_df_to_s3(df, S3_BUCKET, data_output)
    return df


def submit_query(query: str, database: str) -> str:
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
    )
    query_execution_id = response["QueryExecutionId"]
    print(f"Started query: {query_execution_id}")
    print(f"result will be saved to {ATHENA_OUTPUT}")

    return query_execution_id


def check_query_status(query_execution_id: str) -> str:
    result = athena.get_query_execution(QueryExecutionId=query_execution_id)
    status = result["QueryExecution"]["Status"]["State"]
    print(f"check query {query_execution_id} status: {status}")

    return status


def wait_query_finish(query_execution_id: str, interval: int = 10):
    while True:
        status = check_query_status(query_execution_id)
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(interval)

    print(f"Query finished with status: {status}")
    if status != "SUCCEEDED":
        raise Exception(f"Query failed with status: {status}")


def get_query_result(query_execution_id: str) -> pd.DataFrame:
    """
    This works for relative small dataset. For large data directly download data from S3 bucket.
    :param query_execution_id:
    :return:
    """
    wait_query_finish(query_execution_id)

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
