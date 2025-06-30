"""
bet statistics for basepoint, all account (bet amount) values with its frequency.
"""

import json
from collections import Counter

from bituslabs_ds.config import S3_BUCKET, setup_logging
from bituslabs_ds.s3_utils import read_files, upload_file_to_s3

setup_logging(".log/data_process_wucaishen_cluster_stats.log")

wucaishen_enriched_with_clusters = [
    "s3://bituslabs-team-ai/ds-data-kmeans/wucaishen_with_cluster_2024_0.csv",
    "s3://bituslabs-team-ai/ds-data-kmeans/wucaishen_with_cluster_2024_1.csv",
    "s3://bituslabs-team-ai/ds-data-kmeans/wucaishen_with_cluster_2024_2.csv",
]

cluster_stats = {}
for cluster_index in range(3):
    data = read_files(
        [wucaishen_enriched_with_clusters[cluster_index]],
        local_cache_path=f"./output/wucaishen_enriched_with_clusters_{cluster_index}.csv",
        columns=["basepoint", "account"],
    )
    stats = {
        "basepoint_mean": data["basepoint"].mean(),
        "basepoint_median": data["basepoint"].median(),
        "basepoint_std": data["basepoint"].std(),
        "account_counter": Counter(data["account"]),
    }
    cluster_stats[f"cluster_{cluster_index}"] = stats

json.dump(cluster_stats, open("./output/cluster_stats.json", "w"), indent=4)
upload_file_to_s3("./output/cluster_stats.json", S3_BUCKET, "ds-data-kmeans/wucaishen_cluster_stats_2024.json")
