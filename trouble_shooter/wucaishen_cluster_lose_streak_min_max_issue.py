"""
In the radar plot of raw features. One cluster has highest lose_streak_min, but lowest lose_streak_max. This is logically possible but very strange.
"""

import pandas as pd

cluster_label = pd.read_parquet(
    "/Users/niuxin/Documents/aws-example/jobs/output_wucaishen/wucaishen_2024/output/cluster_label_top_features_30_k_4.parquet"
)
grouped_data = pd.read_parquet(
    "/Users/niuxin/Documents/aws-example/jobs/output_wucaishen/wucaishen_grouped_data_2024.parquet"
)

grouped_data = grouped_data.merge(cluster_label, on="group_id")
grouped_data["lose_streak_range"] = grouped_data["lose_streak_max"] - grouped_data["lose_streak_min"]
print(grouped_data.groupby("cluster_label")[["lose_streak_min", "lose_streak_max", "lose_streak_range"]].mean())

# this should be empty but we see some strange cases that lost_streak_min > 0 but the range is larger than 40.
print(
    grouped_data.loc[
        (grouped_data["lose_streak_range"] > 40) & (grouped_data["lose_streak_min"] > 1),
        ["lose_streak_max", "lose_streak_min", "lose_streak_range"],
    ]
)

# we also have data with streak range > 40, but streak_min > 0:
grouped_data["streak_range"] = grouped_data["streak_max"] - grouped_data["streak_min"]
print(
    grouped_data.loc[
        (grouped_data["streak_range"] > 45) & (grouped_data["streak_min"] > 1),
        ["group_id", "streak_max", "streak_min", "streak_range"],
    ]
)

# Issue: cluster_label is not correctly merged to enriched data which yields duplicated samples with (potentially) incorrect cluster_label. We may need to re-run gail model training and simulation.
# display group_id with cluster_label to check potentially wrong cluster labels:

print(grouped_data[["group_id", "cluster_label"]].drop_duplicates().to_csv("group_id_cluster_label.csv", index=False))

import matplotlib.pyplot as plt


def plot_hist_by_cluster(data, column, bins=30):
    """
    Plots stacked histogram of the given column for each level of cluster_label,
    uses log y-scale, and saves the figure to the current directory.
    Each cluster is visible as a separate color in the stack.
    """
    import numpy as np

    cluster_labels = sorted(data["cluster_label"].unique())
    # Prepare a list of arrays: each cluster's data for the column
    data_by_cluster = [data[data["cluster_label"] == cl][column].values for cl in cluster_labels]

    plt.figure(figsize=(12, 6))
    # Use stacked histogram; each cluster is a stack
    # To truly "stack" the histograms, let's plot the total counts (not density)
    # and set stacked=True. Additionally, let's check for the underlying cause of overlapping.
    # We'll also assign a color for each cluster for clear visibility.
    colors = plt.get_cmap("tab10").colors
    counts, bins_, patches = plt.hist(
        data_by_cluster,
        bins=bins,
        stacked=False,
        alpha=0.85,
        density=False,  # Change to False for true stacking
        label=[f"Cluster {cl}" for cl in cluster_labels],
        color=colors[: len(cluster_labels)],
        edgecolor="black",
    )
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title(f"Stacked Histogram of {column} by Cluster Label (log scaled y)")
    plt.legend()
    fname = f"stacked_hist_{column}_by_cluster_label.png"
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved figure: {fname}")
    plt.close()


# Show histograms for lose_streak_min and lose_streak_max
plot_hist_by_cluster(grouped_data, "lose_streak_min")
plot_hist_by_cluster(grouped_data, "lose_streak_max")
plot_hist_by_cluster(grouped_data, "lose_streak_range")
