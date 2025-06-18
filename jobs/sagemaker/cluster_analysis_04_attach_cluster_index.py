import argparse
import glob
import logging
import os
import sys
import time

import pandas as pd

from bituslabs_ds.eda import read_csv_cols

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def main(input_dir: str, output_dir: str):
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
        "group_id",
    ]

    files = glob.glob(f"{input_dir}/wucaishen_processed_data/wucaishen_enriched_output_25??/*.csv")
    data = read_csv_cols(files, columns=columns_to_read, max_workers=12)

    print(data.head(10))
    print(data.shape)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(3):
        cluster_file = f"{input_dir}/kmeans_output/grouped_data_2025_cluster_{i}.csv"
        cluster_data = pd.read_csv(cluster_file, usecols=["group_id"])
        cluster_data["cluster"] = i
        cluster_data = pd.merge(data, cluster_data, how="inner", on="group_id")
        # cluster_data.drop("group_id", axis=1, inplace=True)

        f_name = f"wucaishen_with_cluster_2025_{i}.csv"
        cluster_data.to_csv(f"{output_dir}/{f_name}", index=False)

    logger.info(f"job took {time.time() - start_time} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, default="./output")
    args = parser.parse_args()

    os.makedirs(f"{args.output}/.log/")
    log_file_path = f"{args.output}/.log/processing_attach_cluster_index.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    main(args.input, args.output)
