"""
apply kmeans model to 2025 data and make predictions
"""

import json

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_files

output_path = "."

wucaishen_files = list_s3_files(S3_BUCKET, "wucaishen_processed_data", r"wucaishen_grouped_stat_output_25.*\.csv$")
wucaishen_data = read_files(
    wucaishen_files, local_cache_path=f"{output_path}/output/wucaishen_grouped_stat_output_25.csv", reload=False
)


important_features = json.load(open(f"{output_path}/features/important_features.json", "r"))
print(important_features[:25])
