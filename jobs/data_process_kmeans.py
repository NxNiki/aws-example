from typing import List

import pandas as pd

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_to_pandas_df


def read_features(bucket: str, files: List[str]) -> pd.DataFrame:
    """
    Read a CSV file from S3 and return it as a Pandas DataFrame.
    :param bucket: S3 bucket name
    :param files: s3 Path to the CSV file.
    """

    dfs = []
    for file in files:
        print(f"Reading {file}")
        dfs.append(read_to_pandas_df(bucket, file))

    df = pd.concat(dfs)
    return df


if __name__ == "__main__":

    wucaishen_files = list_s3_files(
        S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_2401.*\.csv$"
    )
    wucaishen_data = read_features(S3_BUCKET, wucaishen_files)
