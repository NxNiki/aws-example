import io

import boto3
import pandas as pd

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


if __name__ == "__main__":
    # Define your bucket and object key
    bucket_name = "xin-config"
    object_key = "example.csv"

    # Read the body of the response into a pandas DataFrame
    df = read_to_pandas_df(bucket_name, object_key)

    df["read_success"] = True

    # Display the first few rows
    print(df.head())

    object_key = "example_read.csv"
    write_pandas_df(df, bucket_name, object_key)
