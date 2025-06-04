import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series


def read_csv_cols(
    files: List[str],
    columns: List[str],
    filters: Optional[Dict[str, Any]] = None,
    sampling: Optional[int | float] = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Reads specific columns from multiple CSV files, filters rows based on criteria,
    and concatenates the result into a single DataFrame using parallel processing.

    Parameters:
    - files (List[str]): List of CSV file paths.
    - columns (List[str]): Columns to include in final output.
    - filters (Optional[Dict[str, Any]]): Optional {column: value} filter conditions.
    - sampling (Optional[int | float]): Optional row sampling (count or fraction).
    - max_workers (int): Number of threads to use for parallelism.

    Returns:
    - pd.DataFrame: Concatenated DataFrame with selected columns and filtered rows.
    """

    def process_file(file: str) -> pd.DataFrame:
        try:
            needed_cols = set(columns)
            if filters:
                needed_cols.update(filters.keys())

            df = pd.read_csv(file, usecols=list(needed_cols))

            if filters:
                for col, val in filters.items():
                    if col not in df.columns:
                        warnings.warn(
                            f"Filter column '{col}' not found in '{file}'; skipping this filter.", UserWarning
                        )
                        continue
                    df = df[df[col] == val]

            df = df[columns]  # Ensure final column order

            if sampling:
                n = sampling
                if isinstance(sampling, float):
                    n = int(df.shape[0] * sampling)
                if df.shape[0] > n:
                    df = df.sample(n=n)

            return df

        except Exception as e:
            warnings.warn(f"Error processing '{file}': {e}", UserWarning)
            return pd.DataFrame(columns=columns)

    df_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                df_list.append(df)

    if not df_list:
        return pd.DataFrame(columns=columns)

    return pd.concat(df_list, ignore_index=True)


def plot_df_distribution(
    data: Union[pd.DataFrame, pd.Series], bins: int = 30, alpha: float = 0.5, log: bool = False
) -> None:
    """
    Plots the distribution (histogram) of each numeric column in a DataFrame in separate subplots.
    Maximum of 5 columns per row.

    Parameters:
    - data (Union[pd.DataFrame, pd.Series]): Data to plot.
    - bins (int): Number of histogram bins.
    - alpha (float): Transparency level for histograms.
    - log (bool): Whether to use logarithmic scale on y-axis.

    Returns:
    - None: Displays a matplotlib plot.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Filter only numeric columns
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

    if not numeric_cols:
        print("No numeric columns to plot.")
        return

    n_cols = 5
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        values = data[col].dropna()
        n_bins = min(bins, values.nunique())
        counts, bin_edges, _ = ax.hist(values, bins=n_bins, alpha=alpha, edgecolor="black")

        # Annotate with column name at max bin
        if len(counts) > 0:
            max_bin_index = counts.argmax()
            x_pos = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
            y_pos = counts[max_bin_index]
            ax.text(x_pos, y_pos, col, fontsize=9, ha="center", va="bottom")

        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        if log:
            ax.set_yscale("log")
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_scatter_pairs(
    data: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str, bool, bool]]] = None,
    max_per_row: int = 5,
    figsize_per_plot: Tuple[int, int] = (4, 4),
    alpha: float = 0.7,
) -> None:
    """
    Plots scatter plots for every unique pair of numeric columns in the DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing numeric columns.
    - pairs (List[Tuple[str, str]]): List of pairs of numeric column names.
    - max_per_row (int): Maximum number of plots per row.
    - figsize_per_plot (Tuple[int, int]): Size of each subplot (width, height).
    - alpha (float): Marker transparency.

    Returns:
    - None: Shows matplotlib scatter plots.
    """
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include="number").columns.tolist()

    if pairs is None or len(pairs) == 0:
        # Generate all unique pairs (combinations)
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                pairs.append((numeric_cols[i], numeric_cols[j], False, False))

    num_plots = len(pairs)
    if num_plots == 0:
        print("No numeric column pairs to plot.")
        return

    nrows = (num_plots - 1) // max_per_row + 1
    ncols = min(num_plots, max_per_row)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    axes = axes.flatten() if num_plots > 1 else [axes]

    for ax_idx, (col_x, col_y, logx, logy) in enumerate(pairs):
        ax = axes[ax_idx]
        ax.scatter(data[col_x], data[col_y], alpha=alpha, s=2)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"{col_x} vs {col_y}")

        if logx:
            ax.set_xscale("symlog")
        if logy:
            ax.set_yscale("symlog")

    # Hide unused subplots if any
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def _prepare_dataframe(time: Series, data: Union[Series, DataFrame], freq: str) -> DataFrame:
    """Convert data to DataFrame, align with time, and extract seasonal period."""
    if isinstance(data, Series):
        data = data.to_frame()

    df = data.copy()
    df.insert(0, "time", time)

    if freq == "monthly":
        df["period"] = df["time"].dt.month
    elif freq == "weekly":
        df["period"] = df["time"].dt.dayofweek
    elif freq == "daily":
        df["period"] = df["time"].dt.day
    elif freq == "hourly":
        df["period"] = df["time"].dt.hour
    else:
        raise ValueError("`freq` must be one of 'monthly', 'weekly', 'daily', 'hourly'")

    return df


def _plot_grouped_data(ax, df: DataFrame, marker: str, shift_scale: float, colors, alpha: float = 0.7):
    """Group by period and plot mean with error bars."""
    n_series = len(df.columns) - 2  # Exclude 'time' and 'period'
    for i, (col, color) in enumerate(zip(df.columns[1:-1], colors)):
        grouped = df.groupby("period")[col]
        seasonal_mean = grouped.mean()
        seasonal_sem = grouped.sem()
        x = seasonal_mean.index + (i - n_series / 2) * shift_scale
        ax.errorbar(x, seasonal_mean, yerr=seasonal_sem, fmt=marker, capsize=4, label=col, alpha=alpha, color=color)


def _get_x_labels(freq: str):
    """Return x-axis tick labels based on frequency."""
    if freq == "monthly":
        return list(range(1, 13)), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    elif freq == "weekly":
        return list(range(7)), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    elif freq == "daily":
        return list(range(1, 32)), [str(i) for i in range(1, 32)]
    elif freq == "hourly":
        return list(range(24)), [str(i) for i in range(24)]
    else:
        raise ValueError("Invalid frequency")


def plot_seasonality(
    time: Series,
    data_left: Union[Series, DataFrame],
    data_right: Optional[Union[Series, DataFrame]] = None,
    freq: str = "monthly",
) -> None:
    """
    Plots the seasonality of one or two datasets with optional dual y-axes.

    Parameters:
    - time (pd.Series): Series with datetime-like values.
    - data_left (Union[pd.Series, pd.DataFrame]): Left axis data.
    - data_right (Optional[Union[pd.Series, pd.DataFrame]]): Right axis data (optional).
    - freq (str): Frequency for seasonal grouping.

    Returns:
    - None: Displays a matplotlib plot.
    """
    if not pd.api.types.is_datetime64_any_dtype(time):
        try:
            time = pd.to_datetime(time)
        except Exception as e:
            raise ValueError("Failed to convert `time` to datetime.") from e

    df_left = _prepare_dataframe(time, data_left, freq)
    df_right = _prepare_dataframe(time, data_right, freq) if data_right is not None else None

    n_left = len(df_left.columns) - 2
    n_right = len(df_right.columns) - 2 if df_right is not None else 0
    total = n_left + n_right

    # Use a single colormap and split the colors
    color_map = cm.get_cmap("tab10", total)
    colors = [color_map(i) for i in range(total)]
    left_colors = colors[:n_left]
    right_colors = colors[n_left:]

    ticks, labels = _get_x_labels(freq)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    _plot_grouped_data(ax1, df_left, marker="-o", shift_scale=0.1, colors=left_colors)
    ax1.set_xlabel(freq.capitalize())
    ax1.set_ylabel("Left Axis")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels[: len(ticks)])
    ax1.grid(True)

    if df_right is not None:
        ax2 = ax1.twinx()
        _plot_grouped_data(ax2, df_right, marker="-s", shift_scale=0.1, colors=right_colors)
        ax2.set_ylabel("Right Axis")

        left_handles, left_labels = ax1.get_legend_handles_labels()
        right_handles, right_labels = ax2.get_legend_handles_labels()
        ax1.legend(left_handles, left_labels, loc="upper left")
        ax2.legend(right_handles, right_labels, loc="upper right")
    else:
        ax1.legend(loc="upper left")

    plt.title(f"Seasonality Plot ({freq.capitalize()})")
    plt.tight_layout()
    plt.show()
