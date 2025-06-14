import argparse
import glob

import pandas as pd

from bituslabs_ds.eda import read_csv_cols


def main(input_dir, output_dir):
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

    files = glob.glob(f"{input_dir}/wucaishen_processed_data/wucaishen_enriched_output_24??/*.csv")
    data = read_csv_cols(files, columns=columns_to_read)

    print(data.head(10))
    print(data.shape)

    for i in range(3):
        cluster_file = f"{input_dir}/kmeans_output/original_data_cluster_{i}.csv"
        cluster_data = pd.read_csv(cluster_file, usecols=["group_id"])
        cluster_data["cluster"] = i
        cluster_data = pd.merge(data, cluster_data, how="inner", on="group_id")
        # cluster_data.drop("group_id", axis=1, inplace=True)

        f_name = f"wucaishen_with_cluster_2024_{i}.csv"
        cluster_data.to_csv(f"{output_dir}/{f_name}", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    # parser.add_argument("--package", required=True)
    args = parser.parse_args()

    # subprocess.check_call(["pip", "install", args.package])
    main(args.input, args.output)
