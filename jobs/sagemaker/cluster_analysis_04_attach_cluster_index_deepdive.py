import argparse
import glob
import logging
import os
import time
from datetime import datetime

import pandas as pd
from cluster_config import ENVIRONMENT, OUTPUT_PATH, USE_CNY, WORK_DIR

from bituslabs_ds.config import S3_BUCKET, get_cpu_cores, setup_logging
from bituslabs_ds.eda import read_csv_cols
from bituslabs_ds.s3_utils import list_s3_files, read_files, upload_folder_to_s3

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def main(input_dir: str, output_dir: str, upload_output: bool):
    start_time = time.time()
    columns_to_read = [
        "loginname",
        "billno",
        "billtime",
        "account",
        "cus_account",
        "currency",
        "slottype",
        "basepoint",
        "delta_t",
        "delta_bet",
        "delta_profit",
        "delta_payout",
        "streak",
        "win_streak",
        "lose_streak",
        "result_clean",
        "group_id",
    ]

    files = list_s3_files(
        bucket="hyber-slot", prefix="deepdive_groupdata", pattern=r"25\d{2}/deepdive_enriched_output/part-.*\.csv"
    )
    data = read_files(files, local_cache_path=f"{input_dir}/deepdive_enriched_data.csv", columns=columns_to_read)

    if USE_CNY:
        logger.info("user only CNY gamers.")
        data = data[data["currency"] == "CNY"]

    print(data.head(10))
    print(data.shape)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(3):
        cluster_file = f"{output_dir}/output/original_data_cluster_{i}.csv"
        cluster_data = pd.read_csv(cluster_file, usecols=["group_id"])
        cluster_data["cluster"] = i
        cluster_data = pd.merge(data, cluster_data, how="inner", on="group_id")
        # cluster_data.drop("group_id", axis=1, inplace=True)

        f_name = f"deepdive_enrich_with_cluster_2025_{i}.csv"
        cluster_data.to_csv(f"{output_dir}/output/{f_name}", index=False)

    if upload_output:
        # Add a time tag to the S3 output path
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_output_prefix = f"deepdive_analysis_kmeans/{ENVIRONMENT}_{time_tag}/"
        upload_folder_to_s3(args.output, S3_BUCKET, s3_output_prefix)

    logger.info(f"job took {time.time() - start_time} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=WORK_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--upload_to_s3", default=True)
    args = parser.parse_args()

    setup_logging(f"{args.output}/log", "analysis_cluster_04_attach_cluster_index.log")

    main(args.input, args.output, args.upload_to_s3)
