"""
check the bet pattern over time to verify simulation result.
"""

import random

import matplotlib.pyplot as plt
import pandas as pd

# data file is fetch by user segmentation module, reuse here:
data_file = "/Users/niuxin/Documents/aws-example/jobs/output_deepdive/deepdive_enriched_data_2025.parquet"


data = pd.read_parquet(data_file)
data = data[(data["currency"] == "CNY") & (data["slottype"] == 1)]


# Re-identify sessions just in case previous state needed
data["session_start"] = data.groupby("loginname")["delta_t"].transform(lambda x: x > 200)
data["session_id"] = data.groupby("loginname")["session_start"].cumsum()

# Ensure sorted by time
data = data.sort_values("billtime")

plt.figure(figsize=(14, 7))
count = 0
for session_id, session_df in data.groupby(["loginname", "session_id"]):
    if random.random() > 0.01:
        continue
    sorted_session = session_df.sort_values("billtime")
    plt.plot(sorted_session["account"].values[:50], linewidth=1, alpha=0.8, marker="*")
    plt.yscale("log")
    if count > 500:
        break
    count += 1

plt.xlabel("billtime")
plt.ylabel("account")
plt.title("Account over Time by Session")
plt.tight_layout()
plt.show()
