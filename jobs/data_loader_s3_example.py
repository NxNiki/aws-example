from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import read_to_pandas_df, write_df_to_s3

if __name__ == "__main__":
    # Define your bucket and object key
    object_key = "xin-config/example.csv"

    # Read the body of the response into a pandas DataFrame
    df = read_to_pandas_df(S3_BUCKET, object_key)

    df["read_success"] = True

    # Display the first few rows
    print(df.head())

    object_key = "example_read.csv"
    write_df_to_s3(df, S3_BUCKET, object_key)
