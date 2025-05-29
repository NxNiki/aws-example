from typing import List

import pandas as pd

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_files

if __name__ == "__main__":

    wucaishen_files = list_s3_files(
        S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_2401.*\.csv$"
    )
    wucaishen_data = read_files(S3_BUCKET, wucaishen_files)
