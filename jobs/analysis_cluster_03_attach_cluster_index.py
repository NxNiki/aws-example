import glob

import pandas as pd

from bituslabs_ds.eda import read_csv_cols

cluster_data = []
cluster_file_name = [f"./output/original_data_cluster_{i}.csv" for i in range(3)]
for file_name in cluster_file_name:
    cluster_data.append(pd.read_csv(file_name, usecols=["group_id"]))

cluster_df = pd.concat(cluster_data, keys=[0, 1, 2]).reset_index(level=0).rename(columns={"level_0": "cluster"})
print(cluster_df.head(10))


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
    "result_clean",
    "result_pos1",
    "result_pos2",
    "result_pos3",
    "result_pos4",
    "result_pos5",
    "result_pos6",
    "result_pos7",
    "result_pos8",
    "result_pos9",
    "result_pos10",
    "result_pos11",
    "result_pos12",
    "result_pos13",
    "result_pos14",
    "result_pos15",
    "streak",
    "win_streak",
    "lose_streak",
    "group_id",
]

files = glob.glob(f"./output/wucaishen_enriched_output_2401.csv")

data = read_csv_cols(
    files,
    columns=columns_to_read,
)

data = pd.merge(cluster_data, data, how="left", on="group_id")
data.to_csv("./output/wucaishen_with_cluster_2024.csv")
