from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

bet_file = "/Users/niuxin/Downloads/slot_orders_distinct_account.csv"
df = pd.read_csv(bet_file)

df["account"] = pd.to_numeric(df["account"], errors="coerce")
print("number of samples:", len(df))

print("Account Value Statistics:")
print(df["account"].describe())

print("number of distinct product Id and account:")
print(df["productid"].value_counts())
print(df["account"].value_counts())

account_range = df.groupby("productid")["account"].agg(["min", "max"]).reset_index()
account_range.rename(columns={"min": "min_by_productid", "max": "max_by_productid"}, inplace=True)
print(account_range)


def plot_account_distribution(df, col_name: str, value_thresh: Optional[float] = None):

    if value_thresh is not None:
        df = df[df[col_name] <= value_thresh]

    # Plot histogram of account values
    plt.figure(figsize=(10, 6))
    plt.hist(df[col_name], bins=50, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of {col_name}")
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot histogram with log-scaled y-axis
    plt.figure(figsize=(10, 6))
    plt.hist(df[col_name], bins=50, color="skyblue", edgecolor="black")
    plt.yscale("log")  # Logarithmic scale on frequency
    plt.title(f"Distribution of {col_name} (Log Frequency)")
    plt.xlabel(col_name)
    plt.ylabel("Frequency (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Remove zero or negative values for log scale
    df = df[df[col_name] > 0]

    # Plot histogram with log-scaled x-axis
    plt.figure(figsize=(10, 6))
    plt.hist(
        df[col_name],
        bins=np.logspace(np.log10(df[col_name].min()), np.log10(df[col_name].max()), 50),
        color="skyblue",
        edgecolor="black",
    )
    plt.xscale("log")
    plt.title(f"Distribution of {col_name} (Log Scale)")
    plt.xlabel(f"{col_name} (log scale)")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# plot_account_distribution(dataframe, 'account', 5000)
plot_account_distribution(account_range, "max_by_productid")
plot_account_distribution(account_range, "min_by_productid")
