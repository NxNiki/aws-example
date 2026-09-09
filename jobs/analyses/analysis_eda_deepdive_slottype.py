import os

import pandas as pd

from bituslabs_ds.config import setup_logging
from bituslabs_ds.eda import DataProfiler, DataVisualizer
from bituslabs_ds.s3_utils import read_files

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_logging(f"{script_dir}/.log", log_filename="analysis_eda_deepdive.log")

    s3_file_path = "s3://bituslabs-tsplayerai/slotmachine/deep_dive_MA51_0827.csv"
    numeric_columns = ["basepoint", "currpoint", "account", "valid_account", "bet_multiple"]
    read_cols = ["username", "slottype", "currency", "billtime"] + numeric_columns

    data = read_files(s3_file_path, local_cache_path=f"{script_dir}/output/deep_dive_ma51_0827.csv", columns=read_cols)

    currency_counts = data["currency"].value_counts()
    print("Number of rows for each value in column 'currency':")
    print(currency_counts.to_markdown())

    slottype_counts = data["slottype"].value_counts()
    print("Number of rows for each value in column 'slottype':")
    print(slottype_counts.to_markdown())

    data = data[data["currency"] == "CNY"]
    slottype_summary = (
        data.drop(columns=["username", "currency", "billtime"])
        .groupby("slottype")
        .agg({"mean", "median", "max", "min"})
    )
    print(slottype_summary.transpose().to_markdown())

    # select a piece of consecutive bet for slottype 16 for a single user:
    user_summary = data.drop(columns=["currency", "billtime"]).groupby(["username", "slottype"]).agg({"count"})
    print(user_summary.sort_values(by=user_summary.columns[0], ascending=False).head(10).to_markdown())
    print("\nTop users for slottype == 16:")
    print(
        user_summary.loc[(slice(None), 16), :]
        .sort_values(by=user_summary.columns[0], ascending=False)
        .head(10)
        .to_markdown()
    )

    print("\nTop users for slottype == 17:")
    print(
        user_summary.loc[(slice(None), 17), :]
        .sort_values(by=user_summary.columns[0], ascending=False)
        .head(10)
        .to_markdown()
    )

    data_slottype16 = data[(data["username"] == "ydw_438461") & (data["slottype"] == 16)]
    print(data_slottype16.head(10).to_markdown())
    data_slottype16.to_csv(f"{script_dir}/output/deep_dive_slottype16.csv")

    data_slottype17 = data[(data["username"] == "ydw_438461") & (data["slottype"] == 17)]
    print(data_slottype17.head(10).to_markdown())
    data_slottype17.to_csv(f"{script_dir}/output/deep_dive_slottype17.csv")

    # get the mean and median of daily number of bets across all players, and number of unique players in each months:
    data["billtime"] = pd.to_datetime(data["billtime"])
    data["bill_year"] = data["billtime"].dt.year
    print(data["bill_year"].value_counts())
    # data = data[data["bill_year"] == 2025]
    data["bill_day"] = data["billtime"].dt.date
    data["bill_month"] = data["billtime"].dt.month
    data_monthly_bet_stats = (
        data[["bill_year", "bill_month", "bill_day", "username", "slottype"]]
        .groupby(["bill_year", "bill_month", "bill_day", "username"])
        .agg({"count"})
        .reset_index()
    )
    data_monthly_bet_stats.columns = ["bill_year", "bill_month", "bill_day", "username", "num_bets"]
    data_monthly_bet_stats = (
        data_monthly_bet_stats.groupby(["bill_year", "bill_month"])
        .agg({"num_bets": ["mean", "median"], "username": "count"})
        .reset_index()
    )
    print(data_monthly_bet_stats.to_markdown())

    # data_profiler = DataProfiler(data, skewness_threshold=1.5)

    # viz = DataVisualizer(data_profiler)
    # # Plot distributions with distribution statistics
    # viz.create_figure(
    #     layout_cols=["basepoint", "currpoint", "account", "valid_account"], group_col="slottype", n_cols=5, fig_size=(6, 4.5)
    # )
    # viz.add_histogram(show_distribution_stats=False, kde=False, bins=50, log_scale=True)
    # viz.figure.subplots_adjust(left=0.05, bottom=0.1, top=0.9, right=0.95)
    # viz.display()
    # viz.save(f"{script_dir}/figures/deep_dive_distribution.png")

    # viz.create_figure(
    #     layout_cols=["basepoint", "currpoint", "account", "valid_account"], n_cols=4, fig_size=(6, 3.5)
    # )
    # viz.add_boxplot(x_col="slottype", log_scale=True)
    # viz.figure.subplots_adjust(left=0.05, bottom=0.1, top=0.9, right=0.95)
    # viz.display()
    # viz.save(f"{script_dir}/figures/deep_dive_boxplot.png")
