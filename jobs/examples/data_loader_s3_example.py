import pandas as pd

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import read_to_pandas_df, write_df_to_s3

if __name__ == "__main__":
    # Define your bucket and object key
    object_key = "test/example.csv"

    df_write = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
    write_df_to_s3(df_write, S3_BUCKET, object_key)

    # Read the body of the response into a pandas DataFrame
    df_read = read_to_pandas_df(S3_BUCKET, object_key)

    # Display the first few rows
    print(df_read.head())
