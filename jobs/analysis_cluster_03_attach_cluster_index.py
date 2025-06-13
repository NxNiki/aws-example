import glob

import pandas as pd

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.eda import read_csv_cols
from bituslabs_ds.s3_utils import list_s3_files, read_files, upload_file_to_s3

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
    # "result_clean",
    # "result_pos1",
    # "result_pos2",
    # "result_pos3",
    # "result_pos4",
    # "result_pos5",
    # "result_pos6",
    # "result_pos7",
    # "result_pos8",
    # "result_pos9",
    # "result_pos10",
    # "result_pos11",
    # "result_pos12",
    # "result_pos13",
    # "result_pos14",
    # "result_pos15",
    "streak",
    "win_streak",
    "lose_streak",
    "group_id",
]

files = glob.glob(f"./output/wucaishen_enriched_output_24??.csv")
data = read_csv_cols(
    files,
    columns=columns_to_read,
)

# files_s3_path = list_s3_files(S3_BUCKET, "wucaishen_enriched_output_24*.csv")
# data = read_files()

print(data.head(10))
print(data.shape)


for i in range(3):
    cluster_file = f"./output/original_data_cluster_{i}.csv"
    cluster_data = pd.read_csv(cluster_file, usecols=["group_id"])
    cluster_data["cluster"] = i
    data = pd.merge(data, cluster_data, how="inner", on="group_id")

    f_name = f"./output/wucaishen_with_cluster_2024_{i}.csv"
    data.to_csv(f_name)
    upload_file_to_s3(f_name, S3_BUCKET, f"wucaishen_analysis_kmeans/output/{f_name}")
