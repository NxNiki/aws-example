from pyspark_project.s3_utils import read_to_pandas_df, write_pandas_df

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
