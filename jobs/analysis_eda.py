import logging
import os

import pandas as pd

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_files, upload_file_to_s3

os.makedirs("./.log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("./.log/analysis_eda.log"), logging.StreamHandler()],
)

for month in range(3, 13):
    wucaishen_files = list_s3_files(
        S3_BUCKET, "wucaishen_processed_data", rf"wucaishen_enriched_output_24{month:02d}.*\.csv$"
    )
    wucaishen_data = read_files(S3_BUCKET, wucaishen_files, f"./output/wucaishen_enriched_output_24{month:02d}.csv")

    print(wucaishen_data.head())
