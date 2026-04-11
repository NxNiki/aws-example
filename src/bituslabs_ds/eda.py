import itertools
import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import matplotlib.cm as cm
import matplotlib.legend_handler as lh
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from diptest import diptest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.typing import ColorType
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import energy_distance, kurtosis as scipy_kurtosis, pearsonr, skew, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, power_transform
from statannotations.Annotator import Annotator

from bituslabs_ds.config import DEFAULT_MAX_JOBS
from bituslabs_ds.descriptors import ListProperty
from bituslabs_ds.utils import batch_iterator, convert_to_list, group_iterator, keep_numeric_columns

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NoSymbolHandler(lh.HandlerBase):
    """
    Custom legend handler to suppress the symbol for a legend entry.
    Useful for adding text-only entries or spacing in matplotlib legends.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Return an invisible Line2D artist so that no symbol appears in the legend.
        return [Line2D([], [], visible=False)]


def read_csv_cols(
    files: List[str],
    columns: List[str],
    filters: Optional[Dict[str, Any]] = None,
    sampling: Optional[Union[int, float]] = None,
    max_workers: int = DEFAULT_MAX_JOBS,
) -> pd.DataFrame:
    """
    Efficiently reads specified columns from multiple CSV files, applies optional row filters,
    and concatenates the results into a single DataFrame using parallel threads.

    Parameters:
        files (List[str]): List of local CSV file paths.
        columns (List[str]): Columns to include in the final output.
        filters (Optional[Dict[str, Any]]): Optional {column: value} filter conditions.
        sampling (Optional[int | float]): Optional row sampling (count or fraction).
        max_workers (int): Number of threads to use for parallelism.

    Returns:
        pd.DataFrame: Concatenated DataFrame with selected columns and filtered rows.
    """

    def process_file(file: str) -> pd.DataFrame:
        try:
            needed_cols = set(columns)
            if filters:
                needed_cols.update(filters.keys())

            # Read only the necessary columns
            df = pd.read_csv(file, usecols=list(needed_cols))

            if filters:
                for col, val in filters.items():
                    if col not in df.columns:
                        logger.warning(
                            f"Filter column '{col}' not found in '{file}'; skipping this filter.",
                            UserWarning,
                        )
                        continue
                    df = df[df[col] == val]

            # Ensure final column order and presence
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                for col in missing_cols:
                    df[col] = pd.NA
            df = df[columns]

            if sampling:
                n = sampling
                if isinstance(sampling, float):
                    n = int(df.shape[0] * sampling)
                if df.shape[0] > n:
                    df = df.sample(n=n)

            return df

        except Exception as e:
            logger.warning(f"Error processing '{file}': {e}", UserWarning)
            return pd.DataFrame(columns=columns)

    df_list = []
    logger.info(f"run jobs on {max_workers} threads")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                df_list.append(df)

    if not df_list:
        return pd.DataFrame(columns=columns)

    return pd.concat(df_list, ignore_index=True)


def read_excel(file_path: str, sheet_name_col: str = "sheet_name") -> pd.DataFrame:
    """
    Reads all sheets from an Excel file and combines them into a single DataFrame.

    If the Excel file contains only one sheet, returns that sheet as a DataFrame.
    If there are multiple sheets, concatenates them into a single DataFrame, adding a column
    to indicate the sheet name for each row.

    Drops columns that contain only NaN values.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name_col (str): Name of the column to store sheet names when combining multiple sheets.
                              Defaults to "sheet_name".

    Returns:
        pd.DataFrame: Combined DataFrame containing data from all sheets, with a sheet name column if applicable.
    """
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    sheet_names = list(all_sheets.keys())

    if len(sheet_names) == 1:
        df = all_sheets[sheet_names[0]]
        logger.info(f"Only one sheet '{sheet_names[0]}' shape: {df.shape}")
    else:
        df = pd.concat(all_sheets.values(), keys=sheet_names)
        df = df.reset_index(level=0).rename(columns={"level_0": sheet_name_col})

        for sheet_name, sheet_df in all_sheets.items():
            logger.info(f"Sheet '{sheet_name}' shape: {sheet_df.shape}")

        logger.info(f"Combined data shape: {df.shape}")

    cols_to_drop = df.columns[df.isna().all()].tolist()
    if len(cols_to_drop) > 0:
        logger.info(f"Drop Columns with all NaNs: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def split_column_by_threshold(
    data: pd.DataFrame, columns: Union[str, List[str]], threshold: Union[float, List[float]] = 0.0
) -> pd.DataFrame:

    if isinstance(columns, str):
        columns = [columns]

    if not isinstance(threshold, list):
        threshold = [threshold]

    for col, thresh in zip_longest(columns, threshold, fillvalue=threshold[-1]):
        data[f"{col}_below_{thresh}"] = data[col]
        data.loc[data[col] > thresh, f"{col}_below_{thresh}"] = pd.NA

        data[f"{col}_above_{thresh}"] = data[col]
        data.loc[data[col] <= thresh, f"{col}_above_{thresh}"] = pd.NA

    return data


def remove_rows_with_zeros(data: pd.DataFrame, columns: List[str], eps: float = 1e-10) -> pd.DataFrame:
    """
    Drop rows where any of the specified columns is zero (or below eps).
    Useful before power transform for zero-inflated data.

    Args:
        data: Input DataFrame.
        columns: Column names to check for zeros.
        eps: Values below this are treated as zero. Default 1e-10.

    Returns:
        Filtered DataFrame.
    """
    mask = (data[columns] > eps).all(axis=1)
    return data.loc[mask].copy()


def correlation_matrix_with_pvalues(
    df: pd.DataFrame,
    method: str = "pearson",
    remove_pairwise_zeros: bool = False,
    zeros_eps: float = 1e-10,
    zeros_removal_threshold: float = 0.3,
    df_for_zero_check: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix and p-values.

    Args:
        df: DataFrame with numeric columns (used for correlation).
        method: 'pearson' or 'spearman'.
        remove_pairwise_zeros: If True, for each pair independently: if col A has >threshold zeros,
            drop those rows (and corresponding col B values); same for col B. Applied per pair.
        zeros_eps: Values below this are treated as zero.
        zeros_removal_threshold: Remove pairwise zeros only when zero proportion > this (0–1). Default 0.3.
        df_for_zero_check: If provided and remove_pairwise_zeros is True, use this (e.g. raw pre-transform
            data) to detect zeros. Required when df is power-transformed, since transform maps 0 to non-zero.

    Returns:
        Tuple of (correlation_matrix, pvalue_matrix).
    """
    cols = df.columns.tolist()
    corr = np.zeros((len(cols), len(cols)))
    pval = np.zeros((len(cols), len(cols)))
    corr_func = spearmanr if method == "spearman" else pearsonr
    df_zero = df_for_zero_check if df_for_zero_check is not None else df
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                corr[i, j] = 1.0
                pval[i, j] = 0.0
            else:
                valid = df[[c1, c2]].dropna()
                valid_zero = df_zero[[c1, c2]].reindex(valid.index).dropna()
                if remove_pairwise_zeros and len(valid) >= 3 and len(valid_zero) >= 3:
                    zero_frac_c1 = (valid_zero[c1] <= zeros_eps).mean()
                    zero_frac_c2 = (valid_zero[c2] <= zeros_eps).mean()
                    keep = ((valid_zero[c1] > zeros_eps) | (zero_frac_c1 <= zeros_removal_threshold)) & (
                        (valid_zero[c2] > zeros_eps) | (zero_frac_c2 <= zeros_removal_threshold)
                    )
                    valid = valid[keep]
                if len(valid) >= 3:
                    r, p = corr_func(valid[c1], valid[c2])
                    corr[i, j] = r
                    pval[i, j] = p
                else:
                    corr[i, j] = np.nan
                    pval[i, j] = np.nan
    return pd.DataFrame(corr, index=cols, columns=cols), pd.DataFrame(pval, index=cols, columns=cols)


def get_significant_correlation_pairs(
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    significance_level: float = 0.05,
    min_abs_corr: float = 0.2,
) -> List[Tuple[str, str, float, float]]:
    """
    Return pairs with significant correlation.

    Returns:
        List of (var1, var2, r, p).
    """
    pairs: List[Tuple[str, str, float, float]] = []
    for i, c1 in enumerate(corr_df.index):
        for j, c2 in enumerate(corr_df.columns):
            if i >= j:
                continue
            r = corr_df.loc[c1, c2]
            p = pval_df.loc[c1, c2]
            if np.isfinite(r) and np.isfinite(p) and p < significance_level and abs(r) >= min_abs_corr:
                pairs.append((c1, c2, float(r), float(p)))
    return pairs


def split_column_by_multiple_separators(data: pd.DataFrame, column: str, sep: str = ";") -> pd.DataFrame:
    """
    Splits a DataFrame column into multiple new columns based on a separator.

    This function handles entries with a variable number of separators. It creates
    as many new columns as needed based on the maximum number of parts found in
    any single entry. The new columns are named by appending '_1', '_2', '_3',
    etc., to the original column name. The original column is kept.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to split.
        sep (str): The separator string to split on. Defaults to ";".

    Returns:
        pd.DataFrame: The DataFrame with the new split columns added.
    """
    data = data.copy()

    # Split the column into a new temporary DataFrame.
    # `expand=True` automatically creates the necessary number of columns
    # and fills missing parts with None.
    split_df = data[column].str.split(sep, expand=True)

    new_col_names = [f"{column}_{i+1}" for i in range(split_df.shape[1])]
    split_df.columns = new_col_names
    result_df = pd.concat([data, split_df], axis=1)

    return result_df


def barplot(data: pd.DataFrame, output_path: str, title: str):

    plt.figure(figsize=(18, 10))
    colors = sns.color_palette("viridis", len(data))
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=data, palette=colors, legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.savefig(f"{output_path}/{title}.png")
    plt.show()


def plot_by_time(
    data: pd.DataFrame,
    time_col: str,
    var_columns: List[str],
    time_window: timedelta,
    group_col: Optional[str] = None,
    title: str = "Time Series Plot (Gaps Removed)",
    vline_time: Union[str, datetime, pd.Timestamp] = None,
    vline_label: str = "Event",
    vars_in_logscale: Set[str] = set(),
):
    """
    Plots time series data, visually removing time gaps where no data exists,
    while still allowing an accurately placed vertical line.

    Parameters:
    - data (pd.DataFrame): The input dataframe.
    - time_col (str): The name of the time column.
    - var_columns (List[str]): List of variable columns to plot.
    - time_window (timedelta): The time window for resampling.
    - group_col (Optional[str]): Plot separate curve for each group.
    - title (str): Title of the entire plot.
    - vline_time (Union[str, datetime, pd.Timestamp], optional): A timestamp for the vertical line.
    - vline_label (str, optional): The label for the vertical line.
    - vars_in_logscale (Set[str]): Set y-axis of variables to a log scale.
    """

    def resample_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.set_index(time_col, inplace=True)
        resampled = df[var_columns].resample(time_window).mean()
        if group_col is not None:
            resampled[group_col] = (
                df[group_col].resample(time_window).apply(lambda x: x.mode().iloc[0] if not x.empty else pd.NA)
            )
        # Create a string version of the timestamp for labeling later
        resampled["time_str"] = resampled.index.strftime("%Y-%m-%d %H:%M")
        # Reset the index to get a simple integer index (0, 1, 2...) for plotting
        resampled = resampled.reset_index()
        return resampled

    my_palette = sns.color_palette("deep", 7).as_hex()
    n_vars = len(var_columns)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data_resampled = resample_data(data)
    data_resampled.dropna(inplace=True, how="all")

    for i, col in enumerate(var_columns):
        # --- Plotting against the integer index to remove gaps ---
        data_resampled[col] = data_resampled[col].interpolate(method="nearest")
        axes[i].plot(data_resampled.index, data_resampled[col], linestyle="-", color="gray", alpha=0.5)
        axes[i].axhline(y=0, linestyle="--", color="black", alpha=1)

        for data_group, group, index in group_iterator(data_resampled, group_col, preserve_index=True):
            axes[i].plot(
                data_resampled.index,
                data_group[col],
                marker="o",
                markersize=2.5,
                color=my_palette[index % 7],
                label=group,
                alpha=0.9,
            )

            # show data stats:
            print(f"{group}: mean: {data_group[col].mean()}, abs mean: {data_group[col].abs().mean()}")

        if col in vars_in_logscale:
            axes[i].set_yscale("symlog", linthresh=1)
        axes[i].grid(False)
        axes[i].legend(fontsize=7, frameon=False, loc="upper left")

    # --- Calculate and plot the vertical line's fractional position ---
    if vline_time:
        vline_dt = pd.to_datetime(vline_time)

        # Find where the vline timestamp would fit in our data's real timestamps
        # 'left' side gives us the index of the point just before our vline time
        insert_idx = data_resampled[time_col].searchsorted(vline_dt, side="left")

        # Handle edge cases
        if insert_idx == 0:
            vline_pos = 0.0
        elif insert_idx == len(data_resampled):
            vline_pos = len(data_resampled) - 1.0
        else:
            # Interpolate to find the fractional index position
            time_before = data_resampled.loc[insert_idx - 1, time_col]
            time_after = data_resampled.loc[insert_idx, time_col]

            time_span = time_after - time_before
            time_progress = vline_dt - time_before

            fraction = time_progress / time_span
            vline_pos = (insert_idx - 1) + fraction

        for ax in axes:
            ax.axvline(x=vline_pos, color="r", linestyle="--", linewidth=2, label=vline_label)

    # --- Customize X-axis to show readable time labels instead of integers ---
    # Use MaxNLocator to select a reasonable number of ticks to label
    ax = axes[-1]
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
    tick_positions = ax.get_xticks()
    # Filter out positions that are out of bounds of our data
    valid_ticks = [int(p) for p in tick_positions if 0 <= p < len(data_resampled)]
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels(data_resampled.loc[valid_ticks, "time_str"], rotation=45, ha="right", fontsize=7)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_heatmap(data: pd.DataFrame, x_label: str, y_label: str, title: str):

    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(
        data,
        # annot=True,  # Display the average values in each cell
        # fmt=".2f",  # Format the annotations to two decimal places
        cmap="viridis",  # Use a visually appealing color map (e.g., 'viridis', 'coolwarm', 'YlGnBu')
        linewidths=0.5,  # Add thin lines between cells for clarity
    )

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_dual_axis_sorted_swarm(
    data: pd.DataFrame,
    y_col_left: str,
    hue_col: str,
    value_cols: List[str],
    y_col_right: str = "",
) -> None:
    """
    Creates a horizontal swarm plot with dual y-axis labels and detailed annotations.

    For each category and hue, it calculates and displays the count and percentage
    of data points >= 0 (on the right) and < 0 (on the left).

    If y_col_left == y_col_right, only y_col_left is used to group data.

    Args:
        data (pd.DataFrame): The input DataFrame.
        y_col_left (str): Column for the left y-axis labels.
        hue_col (str): Column for coloring the points (hue).
        value_cols (List[str]): Numerical columns for the x-axis.
        y_col_right (str): Column for the right y-axis labels.
    """
    # prevent index error in swarm plot
    plot_data = data.copy().reset_index(drop=True)

    if y_col_right is not None and y_col_left != y_col_right:
        interaction_col = f"{y_col_left}__{y_col_right}"
        plot_data[interaction_col] = plot_data[y_col_left].astype(str) + " / " + plot_data[y_col_right].astype(str)
        label_map = plot_data[[interaction_col, y_col_left, y_col_right]].drop_duplicates().set_index(interaction_col)
    else:
        interaction_col = y_col_left

    hue_order = sorted(plot_data[hue_col].unique())

    # Loop through each value column to create a separate plot
    for value_col in value_cols:
        # --- Step 1: Pre-calculate all statistics for annotation ---
        def calculate_stats(group):
            total = len(group)
            pos_count = (group >= 0).sum()
            neg_count = (group < 0).sum()
            pos_pct = 100 * pos_count / total if total > 0 else 0
            neg_pct = 100 * neg_count / total if total > 0 else 0
            return pd.Series({"text_pos": f"{pos_count} ({pos_pct:.0f}%)", "text_neg": f"{neg_count} ({neg_pct:.0f}%)"})

        stats_df = plot_data.groupby([interaction_col, hue_col])[value_col].apply(calculate_stats).unstack()
        y_order = plot_data.groupby(interaction_col)[value_col].mean().sort_values(ascending=False).index
        fig, ax1 = plt.subplots(figsize=(12, min(len(y_order) * 0.8 + 1, 300)))
        sns.set_theme(style="whitegrid")

        sns.swarmplot(
            y=interaction_col,
            x=value_col,
            hue=hue_col,
            data=plot_data,
            order=y_order,
            hue_order=hue_order,
            palette="viridis",
            s=2,
            dodge=True,
            ax=ax1,
        )

        # --- Step 2: Add annotations ---
        # Get the integer position for each y-axis category
        y_positions = {label: i for i, label in enumerate(y_order)}
        num_hues = len(hue_order)
        # Calculate vertical dodge amount for text (to match swarm plot's dodge)
        dodge_amount = 0.4 if num_hues > 1 else 0

        for category in y_order:
            for i, hue in enumerate(hue_order):
                # Calculate the vertical position for this specific hue's text
                base_y = y_positions[category]
                y_offset = (i - (num_hues - 1) / 2) * dodge_amount
                text_y_pos = base_y + y_offset

                # Get the pre-calculated text strings
                try:
                    text_neg = stats_df.loc[(category, hue), "text_neg"]
                    text_pos = stats_df.loc[(category, hue), "text_pos"]
                except KeyError:
                    continue  # Skip if a category/hue combination has no data

                # Annotate on the left side (< 0)
                ax1.annotate(
                    text_neg,
                    xy=(0, text_y_pos),
                    xycoords=("axes fraction", "data"),  # type: ignore[arg-type]
                    xytext=(-10, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="crimson",
                )

                # Annotate on the right side (>= 0)
                ax1.annotate(
                    text_pos,
                    xy=(1, text_y_pos),
                    xycoords=("axes fraction", "data"),  # type: ignore[arg-type]
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="darkgreen",
                )

        # --- Step 3: Set up dual-axis labels (as before) ---
        if y_col_left != y_col_right:
            original_labels = [label.get_text() for label in ax1.get_yticklabels()]
            left_labels = label_map.loc[original_labels, y_col_left]
            right_labels = label_map.loc[original_labels, y_col_right]

            ax1.set_yticklabels(left_labels)
            ax2 = ax1.twinx()
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_yticks(ax1.get_yticks())
            ax2.set_yticklabels(right_labels)
            ax2.set_ylabel(y_col_right, fontsize=12)

        ax1.axvline(x=0, color="black", linestyle="--", linewidth=1.5, zorder=0)

        ax1.set_title(f'"{value_col}" by Group with Counts and Percentages', fontsize=16, pad=20)
        ax1.set_xlabel(value_col, fontsize=12)
        ax1.set_ylabel(y_col_left, fontsize=12)
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(handles[:num_hues], labels[:num_hues], title=hue_col, bbox_to_anchor=(1.2, 1), loc="upper left")

        fig.tight_layout()
        plt.show()


def plot_scatter_pairs(
    data: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str, bool, bool]]] = None,
    correlation_pairs: Optional[List[Tuple[str, str, float, float]]] = None,
    max_per_row: int = 5,
    fig_size_per_plot: Tuple[int, int] = (4, 4),
    alpha: float = 0.7,
    add_regression_line: bool = False,
    add_correlation_annotation: bool = False,
    significance_level: float = 0.05,
    save_path: Optional[Union[str, Path]] = None,
    s: int = 2,
    remove_pairwise_zeros: bool = False,
    zeros_eps: float = 1e-10,
    zeros_removal_threshold: float = 0.3,
    df_for_zero_check: Optional[pd.DataFrame] = None,
) -> None:
    """
    Plots scatter plots for every unique pair of numeric columns in the DataFrame.

    Parameters:
    - data: DataFrame containing numeric columns (used for x/y values; may be power-transformed).
    - pairs: List of (col_x, col_y, logx, logy). Ignored if correlation_pairs is provided.
    - correlation_pairs: List of (c1, c2, r, p) for correlation scatter plots with optional
        regression line and r/p annotation.
    - max_per_row: Maximum number of plots per row.
    - fig_size_per_plot: Size of each subplot (width, height).
    - alpha: Marker transparency.
    - add_regression_line: If True (and correlation_pairs provided), add linear regression line.
    - add_correlation_annotation: If True (and correlation_pairs provided), add r and p text.
    - significance_level: p-value threshold for significance marker (*) in annotation.
    - save_path: If provided, save figure to path instead of plt.show().
    - s: Marker size for scatter points.
    - remove_pairwise_zeros: If True, drop rows where col has zeros when zero frac > threshold
        (same logic as correlation_matrix_with_pvalues). Ensures scatter/regression match r/p.
    - zeros_eps: Values <= this treated as zero.
    - zeros_removal_threshold: Remove pairwise zeros only when zero proportion > this (0–1).
    - df_for_zero_check: Use for zero detection when data is power-transformed (required; raw data).
    """
    numeric_cols = data.select_dtypes(include="number").columns.tolist()

    use_correlation_mode = correlation_pairs is not None and len(correlation_pairs) > 0
    if use_correlation_mode and correlation_pairs is not None:
        plot_items: List[Tuple[str, str, Optional[float], Optional[float]]] = [
            (c1, c2, r, p) for c1, c2, r, p in correlation_pairs
        ]
    else:
        if pairs is None or len(pairs) == 0:
            pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    pairs.append((numeric_cols[i], numeric_cols[j], False, False))
        plot_items = [(col_x, col_y, None, None) for col_x, col_y, _logx, _logy in pairs]

    num_plots = len(plot_items)
    if num_plots == 0:
        print("No numeric column pairs to plot.")
        return

    nrows = (num_plots - 1) // max_per_row + 1
    ncols = min(num_plots, max_per_row)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_size_per_plot[0] * ncols, fig_size_per_plot[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    df_zero = df_for_zero_check if df_for_zero_check is not None else data
    for ax_idx, item in enumerate(plot_items):
        col_x, col_y = item[0], item[1]
        r, p = item[2], item[3]
        ax = axes_flat[ax_idx]
        valid = data[[col_x, col_y]].dropna()
        valid_zero = df_zero[[col_x, col_y]].reindex(valid.index).dropna()
        if remove_pairwise_zeros and len(valid) >= 3 and len(valid_zero) >= 3:
            zero_frac_c1 = (valid_zero[col_x] <= zeros_eps).mean()
            zero_frac_c2 = (valid_zero[col_y] <= zeros_eps).mean()
            keep = ((valid_zero[col_x] > zeros_eps) | (zero_frac_c1 <= zeros_removal_threshold)) & (
                (valid_zero[col_y] > zeros_eps) | (zero_frac_c2 <= zeros_removal_threshold)
            )
            valid = valid[keep]
        if len(valid) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        else:
            x_vals = valid[col_x].values
            y_vals = valid[col_y].values
            ax.scatter(x_vals, y_vals, alpha=alpha, s=s)
            if use_correlation_mode and add_regression_line and r is not None and p is not None:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, "r-", lw=2)
            if use_correlation_mode and add_correlation_annotation and r is not None and p is not None:
                sig = "*" if p < significance_level else ""
                ax.text(
                    0.05,
                    0.95,
                    f"r = {r:.3f}{sig}\np = {p:.2e}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"{col_x} vs {col_y}")
        if not use_correlation_mode and pairs is not None and ax_idx < len(pairs):
            _, _, logx, logy = pairs[ax_idx]
            if logx:
                ax.set_xscale("symlog")
            if logy:
                ax.set_yscale("symlog")

    for i in range(num_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
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


class DataVisualizer:
    """
    A comprehensive class for data visualization that organizes all plotting functions
    and allows sharing of data and configuration across methods.

    Most of the plots can also be created with seaborn.FacetGrid. This class is more flexiable to
    arrange layouts based on different columns of input data.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, "DataProfiler"],
    ):
        """
        Initialize the DataVisualizer with data and configuration.

        Parameters:
            data: The main DataFrame or DataProfiler instance
        """
        # Always create a DataProfiler for consistent analytics access
        if isinstance(data, DataProfiler):
            self.data_profiler = data
        else:
            self.data_profiler = DataProfiler(data)

        self.layout_cols = ListProperty("layout_cols", immutable=False)

        # attributes defined in create_figure:
        self.figure: plt.Figure = Figure()
        self.figure_size: Tuple[float, float] = (8, 6)
        self.group_col: str = ""
        self.axes_keys: List[Tuple[str, str, Any]] = []
        self.axes: List[Axes] = []
        self.n_cols: int = 1
        self.n_rows: int = 1
        self.plot_numeric_cols: List[str] = []
        self.plot_non_numeric_cols: List[str] = []

    @property
    def data(self) -> pd.DataFrame:
        """Access the DataFrame through the profiler."""
        return self.data_profiler.df

    def _convert_to_long_format(self, x_col: str, y_col: str) -> pd.DataFrame:
        """
        Convert data to long format for plotting.

        Parameters:
            x_col: Column to use as x-axis grouping
            y_col: Column to visualize on y-axis

        Returns:
            DataFrame in long format ready for plotting
        """
        cols_to_keep = [x_col, y_col] + ([self.group_col] if self.group_col else [])
        df_long = self.data[cols_to_keep].copy().dropna(subset=[y_col])

        df_long = df_long.melt(
            id_vars=[x_col] + ([self.group_col] if self.group_col else []),
            value_vars=[y_col],
            var_name="variable",
            value_name="value",
        )

        return df_long

    def _update_palette(self, group_col: Optional[str] = None, palette: str = "pastel"):

        num_groups_for_palette = len(self.data[group_col].unique()) if group_col else 1
        self.palette = sns.color_palette(palette, n_colors=num_groups_for_palette)

    def _adjust_figure_size(self, x_col: str) -> None:
        """
        Adjust the figure size based on the number of layout columns and the number of rows and columns.
        """

        max_x_unique = self.data[x_col].nunique()
        subplot_width_factor = max(max_x_unique / 4, 0.75)
        if self.group_col:
            num_groups = self.data[self.group_col].nunique()
            subplot_width_factor *= 1 + (num_groups - 1) * 0.2

        if self.figure is not None:
            self.figure.set_size_inches(
                self.figure_size[0] * self.n_cols * subplot_width_factor, self.figure_size[1] * self.n_rows
            )

    def _check_layout_cols(self, layout_cols: Optional[Union[str, List[str]]]) -> Tuple[List[str], List[str]]:
        """
        get numeric and non-numeric columns from layout_cols.
        """
        if layout_cols is None or len(layout_cols) == 0:
            return [], []

        numeric_cols = self.data_profiler.check_numeric_columns(include_boolean=False, subset_cols=layout_cols)
        numeric_cols.sort()
        non_numeric_cols = [col for col in layout_cols if col not in numeric_cols]
        return numeric_cols, non_numeric_cols

    def _get_n_plots(
        self, layout_cols: Optional[Union[str, List[str]]], numeric_cols: List[str], non_numeric_cols: List[str]
    ) -> int:
        """
        Get the number of sub plots in the figure.
        """
        if not layout_cols:
            return 1
        else:
            unique_counts = []
            for col in non_numeric_cols:
                n_unique = self.data[col].nunique(dropna=True)
                unique_counts.append(n_unique)
                if n_unique > 10:
                    logger.warning(
                        f"Column '{col}' has {n_unique} unique values, which may result in too many subplots."
                    )

            n_numeric = len(numeric_cols)
            n_plots = max(n_numeric, 1)
            if unique_counts:
                for n_unique in unique_counts:
                    n_plots *= n_unique

            logger.info(f"Number of plots in current figure: {n_plots}")
            return n_plots

    def _get_axes_keys(self, non_numeric_cols: List[str], numeric_cols: List[str]) -> List[Tuple[str, str, Any]]:
        """
        Get the keys for the axes.
        """
        if len(non_numeric_cols) == 0:
            non_numeric_cols = [""]

        if len(numeric_cols) == 0:
            numeric_cols = [""]

        axes_keys = []
        for col in non_numeric_cols:
            if col == "":
                unique_values = [""]
            else:
                unique_values = self.data[col].unique()
            for value in unique_values:
                for n_col in numeric_cols:
                    axes_keys.append((col, n_col, value))

        return axes_keys

    def _get_axes_data(self, axes_key: Tuple[str, str, Any], extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get the data for the axes.
        """
        non_numeric_col, numeric_col, value = axes_key
        cols = (
            ([numeric_col] if numeric_col else [])
            + ([self.group_col] if self.group_col else [])
            + (extra_cols if extra_cols else [])
        )
        if non_numeric_col == "":
            return self.data[cols]
        else:
            return self.data[self.data[non_numeric_col] == value][cols]

    def _get_n_rows(self, n_plots: int, n_cols: int) -> int:
        """
        Get the number of rows in the figure.
        """
        return (n_plots + n_cols - 1) // n_cols

    def create_figure(
        self,
        layout_cols: Optional[Union[str, List[str]]] = None,
        group_col: str = "",
        n_cols: int = 3,
        fig_title: Optional[str] = None,
        fig_size: Tuple[float, float] = (8, 6),
        palette: str = "pastel",
    ) -> Tuple[plt.Figure, List[Axes]]:
        """
        Create figure and subplots for visualization.

        Parameters:
            x_cols: List of columns to use as x-axis groupings (if None, uses self.x_cols)
            y_cols: List of columns to visualize on y-axis (if None, uses self.y_cols)
            group_col: Column name to use for grouping/hue (optional). If provided, different groups will be colored differently.
            n_cols: Number of subplots per row.
            fig_title: Optional figure title.
            log_scale: Whether to use logarithmic scale on y-axis.
            fig_size: Tuple specifying the default figure size (width, height) in inches.
            palette: Color palette for plots (string or list of colors).

        Returns:
            Tuple of (figure, axes, axes_map) for further customization.
            axes_map is a dict mapping (x_col, y_col) to the corresponding axis.
        """

        self.layout_cols = layout_cols  # type: ignore[assignment]

        self.group_col = group_col
        self._update_palette(group_col, palette)

        numeric_cols, non_numeric_cols = self._check_layout_cols(layout_cols)
        n_plots = self._get_n_plots(layout_cols, numeric_cols, non_numeric_cols)
        n_cols = min(n_cols, n_plots)
        n_rows = self._get_n_rows(n_plots, n_cols)
        axes_keys = self._get_axes_keys(non_numeric_cols, numeric_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows),
        )

        if n_plots == 1:
            axes = [axes]
        else:
            axes = list(np.array(axes).flatten())

        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < n_plots:
                ax.set_xlabel("", fontsize=12)
                ax.set_ylabel("", fontsize=12)
                ax.tick_params(axis="x", rotation=0, labelsize=10)
                for label in ax.get_xticklabels():
                    label.set_fontsize(12)
            else:
                ax.set_visible(False)

        if len(axes) > n_plots:
            axes = axes[:n_plots]

        if fig_title:
            fig.suptitle(fig_title, fontsize=16, y=0.98)

        self.figure = fig
        self.figure_size = fig_size
        self.axes_keys = axes_keys
        self.axes = axes
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.plot_numeric_cols = numeric_cols
        self.plot_non_numeric_cols = non_numeric_cols

        return fig, axes

    def _add_plot_to_axes(
        self,
        plot_func: Callable[..., Any],
        share_legend: bool = True,
        **kwargs,
    ) -> None:
        """
        Apply plot_fun to all axes in axes_map for the specified x_cols and y_cols.
        If x_cols or y_cols is None, use self.x_cols or self.y_cols.
        """
        x_col = kwargs.pop("x_col", None)
        # value_cols is used for correlation heatmap
        value_cols = kwargs.pop("value_cols", [])
        if x_col:
            self._adjust_figure_size(x_col)

        extra_cols = ([x_col] if x_col else []) + value_cols
        for i, key in enumerate(self.axes_keys):
            if i > 0 and share_legend:
                show_legend = False
            else:
                show_legend = True

            ax = self.axes[i]
            data_subset = self._get_axes_data(key, extra_cols)
            y_col = key[1]

            # Call the plot function with the correct arguments
            if plot_func.__name__ == "add_boxplot_to_axis":
                plot_func(data_subset, ax, x_col, y_col, self.group_col, self.palette, show_legend, **kwargs)
            elif plot_func.__name__ == "add_stripplot_to_axis":
                plot_func(data_subset, ax, x_col, y_col, self.group_col, self.palette, show_legend=False, **kwargs)
            elif plot_func.__name__ == "add_histogram_to_axis":
                plot_func(data_subset, ax, y_col, self.group_col, **kwargs)
            elif plot_func.__name__ == "add_correlation_heatmap_to_axis":
                title = f"{key[0]}={key[2]}" if key[0] != "" else ""
                plot_func(data_subset, ax, title=title, **kwargs)
            else:
                raise NotImplementedError(f"Plot function {plot_func.__name__} not implemented")

    def add_boxplot(self, **kwargs) -> None:
        logger.info(f"Adding boxplot to {len(self.axes)} axes")
        self._add_plot_to_axes(self.add_boxplot_to_axis, **kwargs)

    @staticmethod
    def add_boxplot_to_axis(
        data: pd.DataFrame,
        axis: Axes,
        x_col: str,
        y_col: str,
        group_col: str,
        palette: List[Tuple[float, float, float]],
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """
        Add a boxplot to the specified matplotlib axis.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            axis (Axes): The matplotlib axis to plot on.
            x_col (str): Column name to use for x-axis grouping.
            y_col (str): Column name to use for y-axis values.
            group_col (str): Column name to use for hue/grouping (can be None).
            palette (List[Tuple[float, float, float]]): List of colors for the plot.
            show_legend (bool, optional): Whether to display the legend. Defaults to True.
            **kwargs: Additional keyword arguments passed to seaborn.boxplot.

        Returns:
            Axes: The axis with the boxplot added.
        """

        sns.boxplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=group_col if group_col else None,
            ax=axis,
            palette=palette if group_col else None,
            boxprops=dict(linewidth=1.5, facecolor=(0, 0, 0, 0)),
            medianprops=dict(linewidth=2),
            whis=1.5,
            notch=True,
            legend=show_legend,
            **kwargs,
        )
        # Remove the frame of the legend if it exists
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)

        # Get the list of boxes and median lines
        boxes = [child for child in axis.get_children() if isinstance(child, mpatches.PathPatch)]
        box_lines = [
            child
            for child in axis.get_children()
            if isinstance(child, mlines.Line2D) and child.get_label() == "_nolegend_"
        ]

        if group_col:
            group_length = len(data[group_col].unique())
        else:
            group_length = 1
        x_col_length = len(data[x_col].unique())

        for i, box in enumerate(boxes):
            hue_idx = i // x_col_length % group_length
            color = palette[hue_idx]
            box.set_edgecolor(color)

        # Set median line color to match box edge
        for i, median in enumerate(box_lines):
            hue_idx = i // (x_col_length * 6) % group_length
            color = palette[hue_idx]
            median.set_color(color)
            median.set_linewidth(2)

        return axis

    def add_stripplot(
        self,
        **kwargs,
    ) -> None:
        logger.info(f"Adding stripplot to {len(self.axes)} axes")
        self._add_plot_to_axes(self.add_stripplot_to_axis, **kwargs)

    @staticmethod
    def add_stripplot_to_axis(
        data: pd.DataFrame,
        axis: Axes,
        x_col: str,
        y_col: str,
        group_col: str,
        palette: List[Tuple[float, float, float]],
        size: int = 5,
        alpha: float = 0.7,
        jitter: float = 0.25,
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """
        Add strip plot to the specified axes or all current axes.

        Parameters:
            axes: The axes to add strip plots to (if None, uses all current axes)
            x_col: Column to use as x-axis grouping (if None, uses current plot data)
            y_col: Column to visualize on y-axis (if None, uses current plot data)
            size: Size of the points
            alpha: Transparency level
            jitter: Amount of jitter for the points

        Returns:
            The axes with strip plots added
        """

        non_nans = data[y_col].count()
        sns.stripplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=group_col if group_col else None,
            dodge=True if group_col else False,
            ax=axis,
            palette=palette,
            size=size / 3 if non_nans > 1e4 else size,
            legend=show_legend,
            jitter=jitter,
            alpha=0.5 if non_nans > 1e4 else alpha,
            **kwargs,
        )

        return axis

    def add_histogram(
        self,
        **kwargs,
    ):
        logger.info(f"Adding histogram to {len(self.axes)} axes")
        self._add_plot_to_axes(self.add_histogram_to_axis, **kwargs)

    def add_histogram_to_axis(
        self,
        data: pd.DataFrame,
        axis: Axes,
        y_col: str,
        group_col: Optional[str] = None,
        show_distribution_stats: bool = False,
        **kwargs,
    ) -> Optional[Axes]:
        """
        Add histogram to the specified axes or all current axes.

        Parameters:
            y_col: Column to visualize (if None, uses current plot data or all y_cols)
            bins: Number of histogram bins
            alpha: Transparency level
            add_kde: Whether to add KDE plot
            show_distribution_stats: Whether to add distribution statistics (uses profiler analytics)

        Returns:
            The axes with histograms added
        """

        if group_col:
            data = data[[y_col, group_col]].copy()
        else:
            data = data[y_col].to_frame()

        max_value_length = int(1e5)
        if len(data) > max_value_length:
            logger.warning(f"Randomly selecting {max_value_length} rows from input dataframe to create histogram")
            data = data.sample(n=max_value_length, random_state=42)

        values = data[[y_col]].dropna().values.flatten()
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            logger.warning(f"No values found for column: {y_col}")
            return None

        if len(np.unique(values)) == 1:
            logger.warning(f"Only one unique value found for column: {y_col}. Set kde to False.")
            kwargs["kde"] = False

        group_col = None if group_col == "" else group_col
        sns.histplot(data, x=y_col, hue=group_col, palette=self.palette, ax=axis, **kwargs)

        if show_distribution_stats:
            legend_labels = []
            for data_group, group, _ in group_iterator(data, group_col):
                feature_stats = self.data_profiler.check_df_distribution_stats(data_group, [y_col])

                # Distribution statistics using profiler analytics
                stat = next((s for s in feature_stats if s.get("column") == y_col), None)
                if stat and not pd.isna(stat.get("skewness")):
                    skewness = stat["skewness"]
                    dip_stat = stat["dip_stat"]
                    p_value = stat["dip_p_values"]

                    group_lablel = "" if group is None else f"{group}: "
                    if p_value < 0.05:
                        legend_labels.append(f"{group_lablel}skewness: {skewness:.3f}\nunimodality: {dip_stat:.3f}*")
                    else:
                        legend_labels.append(f"{group_lablel}skewness: {skewness:.3f}\nunimodality: {dip_stat:.3f}")

            axis.legend(labels=legend_labels, frameon=False)

        # axis.set_title(y_col)
        axis.set_xlabel(y_col)
        axis.set_ylabel(None)  # type: ignore[arg-type]
        axis.grid(True)

        return axis

    def _add_histogram_original_and_transformed_to_axis(
        self,
        axis: Axes,
        col: str,
        bins: int = 50,
        log_scale_cols: Optional[Set[str]] = None,
        pt_log_scale_cols: Optional[Set[str]] = None,
        max_points: int = 100_000,
        random_state: int = 42,
    ) -> Tuple[List[Any], List[str], float, float]:
        """
        Draw original and power-transformed (Yeo-Johnson) distributions on the same axes.

        - Original: bottom x-axis, left y-axis, counts; symlog y when col in log_scale_cols.
        - Power-transformed: top x-axis, right y-axis, counts; symlog y when col in pt_log_scale_cols.
        (Both sets use the same rule: |skewness| >= threshold, computed in plot_numeric_distribution_pages.)

        Returns:
            (handles, labels, pt_skewness, pt_kurtosis) for figure-level legend and stats box.
        """
        df = self.data_profiler.df
        empty_return: Tuple[List[Any], List[str], float, float] = ([], [], np.nan, np.nan)
        if col not in df.columns:
            return empty_return
        log_scale_cols = log_scale_cols or set()
        raw = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(raw) > max_points:
            raw = raw.sample(n=max_points, random_state=random_state)
        values_orig = np.asarray(raw, dtype=float)
        values_orig = values_orig[np.isfinite(values_orig)]
        if len(values_orig) < 2:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes, fontsize=8)
            return empty_return

        # Power-transform (Yeo-Johnson); handle constant or near-constant columns
        try:
            pt = power_transform(values_orig.reshape(-1, 1), method="yeo-johnson")
            values_pt = pt.ravel()
        except Exception:
            values_pt = values_orig.copy()
        # Skew and kurtosis of power-transformed values (bias=False for consistency)
        pt_skew = skew(values_pt, bias=False) if len(values_pt) >= 2 else np.nan
        pt_kurt = scipy_kurtosis(values_pt, bias=False) if len(values_pt) >= 2 else np.nan

        # Original: bottom x, left y, counts; symlog y if highly skewed
        sns.histplot(
            values_orig,
            ax=axis,
            bins=bins,
            kde=True,
            stat="count",
            color="steelblue",
            alpha=0.45,
            label="original",
            edgecolor="white",
            linewidth=0.3,
        )
        if col in log_scale_cols:
            axis.set_yscale("symlog", linthresh=1)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.xaxis.tick_bottom()
        axis.yaxis.tick_left()
        axis.yaxis.set_label_position("left")

        # Transformed: top x, right y (twin axes). Use density, no log scale.
        ax_right = axis.twinx()
        ax_top_right = ax_right.twiny()
        n_bins_pt = min(bins, max(20, len(np.unique(values_pt)) // 2))
        sns.histplot(
            values_pt,
            ax=ax_top_right,
            bins=n_bins_pt,
            kde=True,
            stat="count",
            color="coral",
            alpha=0.45,
            label="power-transformed",
            edgecolor="white",
            linewidth=0.3,
        )
        ax_top_right.set_xlabel("")
        ax_right.set_ylabel("")
        pt_log = pt_log_scale_cols or set()
        if col in pt_log:
            ax_right.set_yscale("symlog", linthresh=1)
        ax_top_right.xaxis.tick_top()
        ax_right.yaxis.tick_right()
        ax_right.yaxis.set_label_position("right")
        axis.tick_params(axis="both", labelsize=7)
        ax_right.tick_params(axis="y", labelsize=7)
        ax_top_right.tick_params(axis="x", labelsize=7)
        # Return combined handles/labels for a single figure-level legend
        lines_orig, labels_orig = axis.get_legend_handles_labels()
        lines_pt, labels_pt = ax_top_right.get_legend_handles_labels()
        leg_pt = ax_top_right.get_legend()
        if leg_pt is not None:
            leg_pt.remove()
        axis.grid(True, alpha=0.3)
        axis.set_title(col, fontsize=8, pad=4)
        return (lines_orig + lines_pt, labels_orig + labels_pt, float(pt_skew), float(pt_kurt))

    def plot_numeric_distribution_pages(
        self,
        output_dir: Union[str, Path],
        layout_cols: Optional[List[str]] = None,
        cols_per_page: int = 4,
        rows_per_page: int = 5,
        bins: int = 50,
        skewness_log_scale_threshold: float = 2.0,
        dpi: int = 120,
        fig_title_prefix: str = "",
        show_distribution_stats_legend: bool = False,
        overlay_power_transform: bool = False,
    ) -> List[Path]:
        """
        Create paginated distribution plots (histograms) for numeric columns.

        One figure per page; each subplot shows a histogram with optional KDE.
        Subplot titles include skewness and kurtosis; log y-scale is applied
        for columns whose absolute skewness exceeds the given threshold.

        If overlay_power_transform is True, each subplot shows both the original
        and the power-transformed (Yeo-Johnson) distribution: original uses
        counts on the left y-axis and bottom x-axis (symlog y when highly skewed);
        power-transformed uses counts on the right y-axis and top x-axis (no log).

        Parameters:
            output_dir: Directory to save PNG files (created if missing).
            layout_cols: Columns to plot; default is profiler's numeric columns.
            cols_per_page: Subplot columns per page.
            rows_per_page: Subplot rows per page.
            bins: Histogram bins per subplot.
            skewness_log_scale_threshold: Use log y-scale when |skewness| >= this.
            dpi: Figure save DPI.
            fig_title_prefix: Prefix for each page suptitle (e.g. dataset name).
            show_distribution_stats_legend: If True, add legend with skew/dip per group.
            overlay_power_transform: If True, overlay power-transformed distribution
                on each subplot and use top x-ticks and right y-ticks.

        Returns:
            List of saved file paths.
        """
        out_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        numeric_cols = layout_cols or self.data_profiler.check_numeric_columns(include_boolean=False)
        if not numeric_cols:
            logger.warning("No numeric columns to plot.")
            return []

        stats_by_col = self.data_profiler.get_distribution_stats_dict()
        # Same rule for both: log y-scale when |skewness| >= threshold (raw from profiler, PT computed below)
        log_scale_cols = self.data_profiler.get_columns_for_log_scale(threshold=skewness_log_scale_threshold)
        pt_log_scale_cols: Set[str] = set()
        if overlay_power_transform:
            for col in numeric_cols:
                if col not in self.data_profiler.df.columns:
                    continue
                vals = self.data_profiler.df[col].replace([np.inf, -np.inf], np.nan).dropna()
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2:
                    continue
                try:
                    pt = power_transform(vals.reshape(-1, 1), method="yeo-johnson")
                    pt_skew = skew(pt.ravel(), bias=False)
                    if not pd.isna(pt_skew) and abs(pt_skew) >= skewness_log_scale_threshold:
                        pt_log_scale_cols.add(col)
                except Exception:
                    pass
        subplots_per_page = cols_per_page * rows_per_page
        saved_paths: List[Path] = []
        page = 0
        use_overlay = overlay_power_transform

        for start in range(0, len(numeric_cols), subplots_per_page):
            page += 1
            cols_this_page = numeric_cols[start : start + subplots_per_page]

            fig, axes = self.create_figure(
                layout_cols=cols_this_page,
                group_col="",
                n_cols=cols_per_page,
                fig_size=(4, 3),
                palette="pastel",
            )
            legend_handles, legend_labels = None, None
            pt_stats_list: List[Tuple[float, float]] = []
            if use_overlay:
                for i, col in enumerate(cols_this_page):
                    handles, labels, pt_skew, pt_kurt = self._add_histogram_original_and_transformed_to_axis(
                        self.axes[i],
                        col,
                        bins=bins,
                        log_scale_cols=log_scale_cols,
                        pt_log_scale_cols=pt_log_scale_cols,
                    )
                    if legend_handles is None and handles:
                        legend_handles, legend_labels = handles, labels
                    pt_stats_list.append((pt_skew, pt_kurt))
            else:
                self.add_histogram(
                    show_distribution_stats=show_distribution_stats_legend,
                    bins=bins,
                    kde=True,
                    stat="density",
                )

            for i, col in enumerate(cols_this_page):
                ax = self.axes[i]
                stat = stats_by_col.get(col, {})
                skew_val = stat.get("skewness", np.nan)
                kurt_val = stat.get("kurtosis", np.nan)
                skew_str = f"{skew_val:.3f}" if not pd.isna(skew_val) else "n/a"
                kurt_str = f"{kurt_val:.3f}" if not pd.isna(kurt_val) else "n/a"
                title_line = f"Skew: {skew_str}  Kurtosis: {kurt_str}"
                if col in log_scale_cols and not use_overlay:
                    ax.set_yscale("symlog")
                    title_line += "  (log y)"
                # Place stats in upper-right (original + power-transformed skew/kurtosis)
                if use_overlay:
                    if i < len(pt_stats_list):
                        pt_skew, pt_kurt = pt_stats_list[i]
                        pt_skew_str = f"{pt_skew:.3f}" if not pd.isna(pt_skew) else "n/a"
                        pt_kurt_str = f"{pt_kurt:.3f}" if not pd.isna(pt_kurt) else "n/a"
                        title_line += f"\nPT Skew: {pt_skew_str}  PT Kurt: {pt_kurt_str}"
                    ax.text(
                        0.98,
                        0.98,
                        title_line,
                        transform=ax.transAxes,
                        fontsize=6,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
                    )
                else:
                    ax.set_title(title_line, fontsize=8)
                ax.tick_params(axis="both", labelsize=7)

            # One legend per figure, just right of the plot area
            if use_overlay and legend_handles and legend_labels:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper left",
                    bbox_to_anchor=(0.98, 1.0),
                    fontsize=8,
                    frameon=True,
                )
            prefix = f"{fig_title_prefix} — " if fig_title_prefix else ""
            fig.suptitle(f"{prefix}numeric distributions (page {page})", fontsize=12, y=1.02)
            fig.tight_layout(rect=(0, 0, 0.90, 0.98))
            path = out_path / f"overview_page_{page}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)
            logger.info("Saved %s", path)

        return saved_paths

    def add_correlation_heatmap(
        self,
        method: str = "pearson",
        value_cols: List[str] = [],
        **kwargs,
    ) -> None:
        """Add correlation heatmap to the current figure."""
        logger.info(f"Adding correlation heatmap to {len(self.axes)} axes")

        if not hasattr(self, "figure") or self.figure is None:
            raise ValueError("No figure available. Create figure first.")

        if len(value_cols) == 0:
            value_cols = self.data_profiler.processed_numerical_columns

        # Use the static method to add the heatmap
        self._add_plot_to_axes(self.add_correlation_heatmap_to_axis, method=method, value_cols=value_cols, **kwargs)

    @staticmethod
    def add_correlation_heatmap_to_axis(
        heatmap_data: pd.DataFrame, axis: Axes, method: str = "pearson", title: str = "Heatmap", **kwargs
    ) -> Axes:
        """
        Add correlation heatmap to the specified axes or all current axes.

        Parameters:
            axes: The axes to add heatmaps to (if None, uses all current axes)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            title: Plot title

        Returns:
            The axes with heatmaps added
        """

        # Only drop columns (or rows) that are all NA, otherwise use available non-NaN pairs for correlation.
        completely_nan_cols = heatmap_data.columns[heatmap_data.isna().all()].tolist()
        completely_nan_rows = heatmap_data.index[heatmap_data.isna().all(axis=1)].tolist()
        if completely_nan_cols:
            print(f"Dropping columns with all NA values for heatmap: {completely_nan_cols}")
            heatmap_data = heatmap_data.drop(columns=completely_nan_cols)
        if completely_nan_rows:
            print(f"Dropping rows with all NA values for heatmap: {completely_nan_rows}")
            heatmap_data = heatmap_data.drop(index=completely_nan_rows)

        if heatmap_data.empty:
            raise ValueError("No numeric columns left after dropping all-NA columns/rows for correlation heatmap.")

        # Pandas' corr() computes correlations using pairwise complete observations (ignoring NaNs).
        correlation_matrix = heatmap_data.corr(method=method)

        heatmap = sns.heatmap(
            correlation_matrix,
            linewidths=0.5,
            ax=axis,
            **kwargs,
        )

        if title != "":
            axis.set_title(title, fontsize=16, pad=20)
        axis.set_xlabel(None)  # type: ignore[arg-type]
        axis.set_ylabel(None)  # type: ignore[arg-type]
        # Ensure all x and y tick labels are displayed
        axis.set_xticks(range(len(correlation_matrix.columns)))
        axis.set_xticklabels(correlation_matrix.columns, rotation=90, ha="right", fontsize=10)
        axis.set_yticks(range(len(correlation_matrix.index)))
        axis.set_yticklabels(correlation_matrix.index, rotation=0, fontsize=10)

        # Add more space at the bottom so x tick labels are fully displayed
        fig = axis.get_figure()
        if fig is not None:
            bottom_pad = 0.25  # adjust as needed
            fig.subplots_adjust(bottom=bottom_pad)

        return axis

    def add_scatter(
        self,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        alpha: float = 0.7,
        s: int = 2,
    ) -> None:
        """
        Add scatter plot to the current axes.

        Parameters:
            x_col: Column for x-axis (if None, uses current plot data)
            y_col: Column for y-axis (if None, uses current plot data)
            alpha: Transparency level
            s: Point size
        """
        if not hasattr(self, "axes_map") or not self.axes_map:
            raise ValueError("No axes available. Create figure first.")

        # Get data
        if x_col is not None and y_col is not None:
            x_data = self.data[x_col]
            y_data = self.data[y_col]
        else:
            raise ValueError("x_col and y_col must be provided.")

        # Add scatter plot to all axes
        for ax in self.axes_map.values():
            if ax is not None:
                ax.scatter(x_data, y_data, alpha=alpha, s=s)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} vs {y_col}")

    def add_statistical_annotations(
        self,
        post_hoc_table: pd.DataFrame = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
    ) -> None:
        """
        Add statistical annotations to box plots or strip plots.

        Parameters:
            post_hoc_table: DataFrame with columns ['variable', 'comparison', 'p_value']
            x_col: Column used for x-axis grouping (if None, uses current plot data)
            y_col: Column used for y-axis (if None, uses current plot data)
        """
        if not hasattr(self, "axes_map") or not self.axes_map:
            raise ValueError("No axes available. Create figure first.")

        if post_hoc_table is None:
            print("Warning: No post_hoc_table provided. Skipping annotations.")
            return

        # Prepare data if x_col and y_col provided
        if x_col is not None and y_col is not None:
            df_long = self._convert_to_long_format(x_col, y_col)
        else:
            raise ValueError("x_col and y_col must be provided.")

        group_col = self.group_col

        # Get annotation pairs and p_values for this y_col
        annotation_pairs = post_hoc_table.loc[post_hoc_table["variable"] == y_col, "comparison"].to_list()
        p_values = post_hoc_table.loc[post_hoc_table["variable"] == y_col, "p_value"].to_list()

        if not annotation_pairs:
            return

        for ax in self.axes_map.values():
            if ax is not None:
                try:
                    annotator = Annotator(
                        ax,
                        annotation_pairs,
                        data=df_long,
                        x=x_col,
                        y="value",
                        hue=group_col if group_col else None,
                    )
                    annotator.configure(
                        text_format="star",
                        loc="inside",
                        verbose=False,
                        line_offset=0.1,
                        line_height=0.02,
                        text_offset=1,
                    )
                    annotator.set_custom_annotations(p_values)
                    annotator.annotate()

                except ValueError as e:
                    print(f"Warning: Could not add annotation to plot for {y_col}. Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred during annotating the plot for {y_col}: {e}")

    def add_legend(self, fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Add legend to the figure.

        Parameters:
            fig: The figure to add legend to (if None, uses current figure)

        Returns:
            The figure with legend added
        """
        if fig is None:
            if not hasattr(self, "figure") or self.figure is None:
                raise ValueError("No figure available. Create a plot first.")
            fig = self.figure

        if not self.group_col:
            return fig

        # Create palette
        num_groups_for_palette = len(self.data[self.group_col].unique())
        palette = sns.color_palette(self.palette, n_colors=num_groups_for_palette)

        handles: list = []
        labels: list = []

        # Try to get handles and labels from any axis
        for ax in self.axes:
            if ax is not None and ax.get_legend():
                handles, labels = ax.get_legend_handles_labels()
                break

        # If no legend was found, manually create proxy artists
        if not handles and self.group_col:
            unique_groups = self.data[self.group_col].unique()
            for i, group in enumerate(unique_groups):
                handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i], markersize=8))
                labels.append(group)

        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 0.95), title=self.group_col)

        return fig

    def display(self):
        """Display the current figure."""
        if self.figure:
            plt.show()
        else:
            print("No figure to display. Create a plot first.")

    def save(self, filename: str, dpi: int = 300, bbox_inches: str = "tight"):
        """Save the current figure."""
        if self.figure:
            logger.info(f"saving figure to {filename}")
            self.figure.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        else:
            print("No figure to save. Create a plot first.")


class DataProfiler:
    """
    Data profiling for distributions, correlations, and numeric preparation.

    - Numeric preparation: use prepare_numeric_df() to get a numeric-only DataFrame
      from sources like Parquet where some columns are stored as fixed_len_byte_array.
    - Distribution stats: check_distribution_stats() / check_df_distribution_stats()
      return skewness, kurtosis, and dip test per column; get_distribution_stats_dict()
      for lookup by column name; get_columns_for_log_scale() for plot log-scale decisions.
    - Skewed columns: get_skewed_columns(), transform_skewed_columns() for transforms.
    """

    def __init__(
        self, df: Union[pd.DataFrame, pd.Series], skewness_threshold: float = 1.5, output_path: str = "."
    ) -> None:
        """
        Initialize the DataProfiler with data and configuration.

        Parameters:
            df: The DataFrame or Series to profile
            skewness_threshold: Threshold for determining skewed distributions
            output_path: Path to save output files
        """
        self.df = df.copy() if isinstance(df, pd.DataFrame) else df.to_frame()
        self.skewness_threshold = abs(skewness_threshold)
        self.output_path = output_path
        self._pos_skewed_col_suffix = "_log"
        self._neg_skewed_col_suffix = "_exp"
        self._transform_skewed_columns: Dict[str, str] = {}

        # Initialize profiling results
        self._original_numeric_columns: List[str] = self.check_numeric_columns(include_boolean=True)

        self.count_missing_columns(verbose=True)
        self.count_nan_for_columns()
        self.count_rows_with_nan()

    def count_nan_for_columns(self) -> Dict[str, int]:
        res = {col: self.df[col].isna().sum() for col in self.df.columns}
        logger.info(f"\n\n Number of nans in each column: \n\n {res}")
        return res

    def count_rows_with_nan(self) -> int:
        res = self.df.isna().any(axis=1).sum()
        logger.info(f"\n\n Number of rows with nans: \n\n {res}/{self.df.shape[0]}")
        return res

    @property
    def processed_numerical_columns(self):
        """Get processed numerical columns for analysis. If a column is power transformed, it will be replaced with the transformed column."""
        res = [
            col if col not in self._transform_skewed_columns else self._transform_skewed_columns[col]
            for col in self._original_numeric_columns
        ]
        return res

    def check_numeric_columns(
        self, include_boolean: bool = True, subset_cols: Optional[Union[str, List[str]]] = None, refresh: bool = False
    ) -> List[str]:
        """Check and return numeric columns in the DataFrame. This includes the orignal and power transformed columns."""

        if subset_cols:
            numerical_columns = self.check_df_numerical_columns(self.df[subset_cols], include_boolean)
        else:
            numerical_columns = self.check_df_numerical_columns(self.df, include_boolean)
        return numerical_columns

    @staticmethod
    def check_df_numerical_columns(data: Union[pd.DataFrame, pd.Series], include_boolean: bool = True) -> List[str]:
        """Check which columns are numerical in a DataFrame."""
        if isinstance(data, pd.Series):
            data = data.to_frame()

        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

        if not include_boolean:
            numeric_cols = [col for col in numeric_cols if data[col].dtype != "bool"]

        return numeric_cols

    @staticmethod
    def try_convert_series_to_numeric(series: pd.Series) -> Optional[pd.Series]:
        """
        Convert a series to numeric when possible.

        Handles Parquet fixed_len_byte_array (e.g. object/bytes), decimal.Decimal,
        and other types that support float conversion.

        Returns:
            A numeric series, or None if conversion fails entirely.
        """
        try:
            out = pd.to_numeric(series, errors="coerce")
            if out.notna().any():
                return out
            return None
        except (TypeError, ValueError):
            pass

        def _to_float(x: Any) -> float:
            if pd.isna(x):
                return np.nan
            if hasattr(x, "__float__"):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return np.nan
            if isinstance(x, bytes):
                try:
                    return float(x.decode("utf-8").strip())
                except (ValueError, UnicodeDecodeError):
                    return np.nan
            return np.nan

        try:
            out = series.apply(_to_float)
            return out if out.notna().any() else None
        except Exception:
            return None

    @staticmethod
    def prepare_numeric_df(
        df: pd.DataFrame,
        exclude_columns: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Build a numeric-only DataFrame suitable for profiling and distribution plots.

        Includes columns that are already numeric and columns that can be converted
        to numeric (e.g. from Parquet fixed_len_byte_array / object dtypes).
        Excluded columns are omitted from the result.

        Parameters:
            df: Input DataFrame (e.g. from read_parquet).
            exclude_columns: Column names to exclude (e.g. IDs: user_id, session_group).

        Returns:
            (numeric_column_names, dataframe_with_only_numeric_columns).
            Column order is preserved; converted columns are float.
        """
        exclude = set(exclude_columns or [])
        native_numeric = [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]
        other = [c for c in df.columns if c not in exclude and c not in native_numeric]
        converted: Dict[str, pd.Series] = {}
        for c in other:
            s = DataProfiler.try_convert_series_to_numeric(df[c])
            if s is not None:
                converted[c] = s
        numeric_cols = native_numeric + list(converted.keys())
        plot_df = df[native_numeric].copy()
        for c, s in converted.items():
            plot_df[c] = s
        return numeric_cols, plot_df

    def check_distribution_stats(self, refresh: bool = False) -> List[Dict]:
        """Check distribution statistics for numerical columns."""
        if not hasattr(self, "_distribution_stats") or refresh:
            numerical_cols = self.check_numeric_columns(include_boolean=True)
            self._distribution_stats = self.check_df_distribution_stats(self.df, numerical_cols)

        return self._distribution_stats

    @staticmethod
    def check_df_distribution_stats(
        data: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        include_kurtosis: bool = True,
    ) -> List[Dict]:
        """
        Compute distribution statistics for numerical columns.

        Parameters:
            data: DataFrame containing the columns.
            numeric_columns: Columns to analyze; default is all numeric columns.
            include_kurtosis: If True, add kurtosis (bias=False) to each stat dict.

        Returns:
            List of dicts with keys: column, skewness, dip_stat, dip_p_values,
            and optionally kurtosis. Inf values are dropped before computation.
        """
        distribution_stats: List[Dict] = []
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

        for col in numeric_columns:
            if col not in data.columns:
                continue
            values = data[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 2:
                continue

            skewness = skew(values, bias=False)
            dip_statistic_unimodal, p_value_unimodal = diptest(values)
            stat: Dict[str, Any] = {
                "column": col,
                "skewness": skewness,
                "dip_stat": dip_statistic_unimodal,
                "dip_p_values": p_value_unimodal,
            }
            if include_kurtosis:
                stat["kurtosis"] = scipy_kurtosis(values, bias=False)
            distribution_stats.append(stat)

        return distribution_stats

    def get_skewed_columns(self, refresh: bool = False) -> Tuple[List[str], List[str]]:
        """Get positively and negatively skewed columns."""
        stats = self.check_distribution_stats(refresh=refresh)

        pos_skewed = [stat["column"] for stat in stats if stat["skewness"] > self.skewness_threshold]
        neg_skewed = [stat["column"] for stat in stats if stat["skewness"] < -self.skewness_threshold]

        return pos_skewed, neg_skewed

    def get_distribution_stats_dict(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Return distribution statistics keyed by column name.

        Convenient for lookup when annotating plots (e.g. skewness, kurtosis per column).
        Keys include: column, skewness, kurtosis, dip_stat, dip_p_values.
        """
        stats_list = self.check_distribution_stats(refresh=refresh)
        return {s["column"]: s for s in stats_list}

    def get_columns_for_log_scale(
        self,
        threshold: float = 2.0,
        refresh: bool = False,
    ) -> Set[str]:
        """
        Return column names whose absolute skewness is at least threshold.

        Useful for deciding which distribution plots should use log y-scale.
        """
        stats = self.get_distribution_stats_dict(refresh=refresh)
        return {col for col, s in stats.items() if not pd.isna(s.get("skewness")) and abs(s["skewness"]) >= threshold}

    def transform_skewed_columns(self, pos_suffix: Optional[str] = None, neg_suffix: Optional[str] = None) -> None:
        """Transform skewed columns using log or exponential transformations."""
        pos_skewed, neg_skewed = self.get_skewed_columns()

        if pos_suffix is None:
            pos_suffix = self._pos_skewed_col_suffix
        else:
            self._pos_skewed_col_suffix = pos_suffix

        if neg_suffix is None:
            neg_suffix = self._neg_skewed_col_suffix
        else:
            self._neg_skewed_col_suffix = neg_suffix

        # Transform positively skewed columns (yeo-johnson transformation)
        for col in pos_skewed:
            self.df[f"{col}{pos_suffix}"] = power_transform(self.df[col].values.reshape(-1, 1), method="yeo-johnson")
            self._transform_skewed_columns[col] = f"{col}{pos_suffix}"

        # Transform negatively skewed columns (yeo-johnson transformation)
        for col in neg_skewed:
            self.df[f"{col}{neg_suffix}"] = power_transform(self.df[col].values.reshape(-1, 1), method="yeo-johnson")
            self._transform_skewed_columns[col] = f"{col}{neg_suffix}"

    def get_correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        """Get correlation matrix for numerical columns."""
        numerical_df = self.df[self.processed_numerical_columns]
        return numerical_df.corr(method=method)

    def get_feature_statistics(self, refresh: bool = False) -> List[Dict]:
        """Get comprehensive feature statistics."""
        return self.check_distribution_stats(refresh=refresh)

    def augment_columns(self, columns: List[str], col_name: str, method: Union[List[str], str] = "pca"):
        self.df = self.augment_df_columns(self.df, columns, col_name, method)

    @staticmethod
    def augment_df_columns(
        data: pd.DataFrame, columns: List[str], col_name: str, method: Union[List[str], str] = "pca"
    ) -> pd.DataFrame:

        if isinstance(method, str):
            method = [method]

        if len(columns) == 1:
            logger.warning("Only one columns is selected to augment data")

        scaler = StandardScaler()
        data = data.replace([np.inf, -np.inf], np.nan)
        df_scaled = scaler.fit_transform(data[columns].dropna(axis=1, how="any"))

        if "mean" in method:
            data[f"{col_name}_mean"] = df_scaled.mean(axis=1)
        if "pca" in method:
            logger.info(f"run pca on the following columns of dataframe:\n {columns}")
            pca = PCA(n_components=None)
            principal_components = pca.fit_transform(df_scaled)
            explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
            print(f"Cumulative Explained Variance for '{col_name}' PCA components:")
            for i, cum_var in enumerate(explained_variance_ratio_cumsum):
                print(f"  PC{i + 1}: {cum_var:.4f}")

            for i in range(principal_components.shape[1]):
                data[f"{col_name}_pca{i + 1}"] = principal_components[:, i]

        return data

    def show_group_stats(self, group_cols: List[str], var_columns: List[str], **kwargs) -> pd.DataFrame:
        return self.show_df_group_stats(self.df, group_cols, var_columns, **kwargs)

    @staticmethod
    def show_df_group_stats(
        data: pd.DataFrame,
        group_cols: List[str],
        var_columns: List[str],
        transpose: bool = False,
        agg_stats: List[str] = ["mean", "median", "std", "min", "max", "count"],
    ) -> pd.DataFrame:
        """
        Display group statistics for specified columns.
        """
        if not group_cols or not var_columns:
            print("Please provide both group_cols and var_columns.")
            return

        # Create groupby object
        grouped = data.groupby(group_cols)[var_columns]

        # Calculate statistics
        stats = grouped.agg(agg_stats)

        for var in var_columns:
            stats_var = stats[var]
            logger.info(
                f"Group Statistics: {var}:\n {stats_var.sort_values(by=stats_var.columns[0], ascending=False).reset_index().to_markdown(index=False)}"
            )

        if transpose:
            stats = stats.T

        logger.info(f"Group Statistics:\n {stats.reset_index().to_markdown(index=False)}")

        return stats

    def count_missing_columns(self, verbose: bool = True) -> dict:
        return self.count_df_missing_columns(self.df, verbose)

    @staticmethod
    def count_df_missing_columns(df: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Count missing values in each column of the DataFrame.
        """
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        missing_info = {
            "column": missing_counts.index.tolist(),
            "missing_count": missing_counts.values.tolist(),
            "missing_percentage": missing_percentages.values.tolist(),
        }

        if verbose:
            print("Missing Values Summary:")
            for col, count, pct in zip(
                missing_info["column"], missing_info["missing_count"], missing_info["missing_percentage"]
            ):
                if count > 0:
                    print(f"{col}: {count} ({pct:.2f}%)")

        return missing_info


class Anova:

    def __init__(self, df: DataFrame, between_vars: Union[List[str], str], var_columns: Union[List[str], str]):

        between_vars = convert_to_list(between_vars)
        var_columns = convert_to_list(var_columns)

        self.data = df[between_vars + var_columns].copy()
        self.data = self.data.replace([np.inf, -np.inf], np.nan)

        self.between_vars = between_vars
        self.var_columns = var_columns
        self.anova_report: Dict = defaultdict(list)
        self.post_hoc_report: Dict = defaultdict(list)

        self._check_between_var_levels()

    def _check_between_var_levels(self) -> None:
        unique_counts = self.data[self.between_vars].nunique()
        for col in self.between_vars:
            if unique_counts[col] <= 1:
                logger.warning(f"Remove between variable '{col}' which has only {unique_counts[col]} unique values")
                self.between_vars.remove(col)

    def run_anova(self, transpose_report: bool = False, p_thresh: float = 0.05) -> pd.DataFrame:

        for col in self.var_columns:
            if (not pd.api.types.is_numeric_dtype(self.data[col])) or pd.api.types.is_bool_dtype(self.data[col]):
                logger.info(f"skip non numeric columns: {col}.")
                continue

            anova_output = pg.anova(dv=col, between=self.between_vars, data=self.data, detailed=True)
            logger.info(f"anova result for {col}\n{anova_output}")

            if "p-unc" not in anova_output or all(anova_output["p-unc"] > p_thresh):
                logger.info(f"skip non-significant column: {col}")
                continue

            self.anova_report["variable"].append(col)
            for index, row in anova_output.iterrows():
                source = row["Source"]
                if source == "Residual" or source == "Within":
                    break
                self.anova_report[f"{source}-p_value"].append(row["p-unc"])
                self.anova_report[f"{source}-eta2"].append(row["np2"])

        res = self.anova_table.copy()
        if transpose_report:
            res = res.transpose()
        logger.info(f"anova result:\n{res.to_markdown(index=False)}")
        return res

    @property
    def anova_table(self) -> pd.DataFrame:
        res = pd.DataFrame(self.anova_report)
        if len(res) == 0:
            return res

        if len(self.between_vars) > 1:
            eta_label = f"{self.between_vars[0]} * {self.between_vars[1]}-eta2"
        else:
            eta_label = f"{self.between_vars[0]}-eta2"
        res.sort_values(by=eta_label, inplace=True, ascending=False)
        return res

    def _perform_and_report_post_hoc(
        self, data_df: DataFrame, dv_col: str, between_factor: str, method: str, p_thresh: float
    ):
        """
        Helper function to perform a post-hoc test and append results to the report dictionary.
        """

        data_df[between_factor] = data_df[between_factor].apply(self.tuple_to_string)
        logger.info(f"  Performing {method} for {between_factor} on {dv_col}")
        if method == "Sidak":
            post_hoc_res = pg.pairwise_ttests(
                dv=dv_col, between=between_factor, data=data_df, padjust="sidak", effsize="hedges"
            )
            p_col = "p-sidak"
            # Filter for significant results for printing and reporting
            post_hoc_res.rename(columns={"p-corr": p_col}, inplace=True)
            report_cols = ["A", "B", p_col, "hedges", "BF10"]
        elif method == "Games-Howell":
            # Games-Howell is only meaningful for factors with > 2 levels.
            if len(data_df[between_factor].unique()) <= 2:
                logger.info(f"  Skipping explicit Games-Howell for {between_factor} on {dv_col} (only 2 levels).")
                return
            post_hoc_res = pg.pairwise_gameshowell(dv=dv_col, between=between_factor, data=data_df, effsize="hedges")
            p_col = "p-gameshowell"
            post_hoc_res.rename(columns={"pval": p_col}, inplace=True)
            report_cols = ["A", "B", p_col, "hedges"]
        elif method == "Tukey":
            # Tukey's HSD typically assumes equal variances (homoscedasticity) and balanced groups,
            # though pingouin's implementation (pairwise_tukey) can handle unequal N.
            # If Levene's test for homogeneity of variance (which pingouin runs in anova output if detailed=True)
            # indicates heterogeneity (p < .05), Games-Howell is generally preferred.
            if len(data_df[between_factor].unique()) <= 2:
                logger.info(f"  Skipping Tukey's HSD for {between_factor} on {dv_col} (only 2 levels).")
                return
            post_hoc_res = pg.pairwise_tukey(dv=dv_col, between=between_factor, data=data_df, effsize="hedges")
            p_col = "p-tukey"
            report_cols = ["A", "B", p_col, "hedges"]

        else:
            raise ValueError(f"Unknown post-hoc method: {method}")

        # Filter for significant results for printing and reporting
        sig_res = post_hoc_res[post_hoc_res[p_col] < p_thresh]
        if sig_res.empty:
            logger.info(f"no significant results found on post hoc analysis for {dv_col}")
        else:
            logger.info(f"post hoc result for {dv_col}: \n{sig_res[report_cols].to_markdown(index=False)}")

        for _, row in sig_res.iterrows():
            self.post_hoc_report["variable"].append(dv_col)
            self.post_hoc_report["factor"].append(between_factor)
            self.post_hoc_report["comparison"].append((self.string_to_tuple(row["A"]), self.string_to_tuple(row["B"])))
            self.post_hoc_report["method"].append(method)
            self.post_hoc_report["p_value"].append(row[p_col])
            self.post_hoc_report["effect_size"].append(row["hedges"])

    @staticmethod
    def tuple_to_string(element):
        if isinstance(element, tuple):
            return ",".join(map(str, element))
        else:
            return element

    @staticmethod
    def string_to_tuple(element):
        return tuple(element.split(",")) if "," in element else (element,)

    def run_post_hoc_analysis(
        self,
        var_columns: List[str],
        p_thresh: float = 0.05,
        method: str = "Tukey",
        effects: str = "main",
        group_var: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        run post hoc analysis based on anova results
        :param var_columns:
        :param p_thresh:
        :param method:
        :param effects:
        :param group_var: an element from the between_vars, for interaction effects, only compare groups in other
            between_var for each level of group_var.
        :return:
        """

        logger.info("--- Running Post-Hoc Analysis ---")
        anova_table = self.anova_table

        # clear previous post-hoc result:
        self.post_hoc_report = defaultdict(list)

        if len(anova_table) == 0:
            logger.warning("Run anova before post-hoc analysis!")

        for col in var_columns:
            # Check if the variable was processed by ANOVA and if any main effect or interaction was significant
            if col not in self.anova_report["variable"]:
                logger.info(
                    f"Skip post-hoc for {col} as it was not included in the ANOVA summary (likely no significant effects)."
                )
                continue

            anova_row = anova_table[anova_table["variable"] == col].iloc[0]
            if effects == "main":
                for var in self.between_vars:
                    if anova_row[f"{var}-p_value"] < p_thresh:
                        logger.info(f"\nSignificance detected for {var} on {col} (p={anova_row[f'{var}-p_value']:.4f})")
                        self._perform_and_report_post_hoc(self.data, col, var, method, p_thresh)

            elif effects == "interaction":
                for var1, var2 in itertools.combinations(self.between_vars, 2):
                    interaction_label = f"{var1} * {var2}"
                    if anova_row[f"{interaction_label}-p_value"] < p_thresh:
                        # run post-hoc for each cluster group separately:
                        for data_interaction, cluster_index, _ in group_iterator(self.data, group_col=group_var):
                            data_interaction["interaction_group"] = list(
                                zip(data_interaction[var1].astype(str), data_interaction[var2].astype(str))
                            )
                            self._perform_and_report_post_hoc(
                                data_interaction, col, "interaction_group", method, p_thresh
                            )
            else:
                raise ValueError(f"Unknown effect: {effects}")

        res_post_hoc = pd.DataFrame(self.post_hoc_report)
        if not res_post_hoc.empty:
            print("\n--- Summary Post-Hoc Report (Significant Comparisons Only) ---")
            for res_post_hoc_group, _, _ in group_iterator(res_post_hoc, group_col="variable"):
                print(res_post_hoc_group.sort_values(by="comparison").to_markdown(index=False))
        else:
            print("\nNo significant post-hoc effects were found.")
        return res_post_hoc

    def show_box_plot(
        self,
        x_col: str,
        group_col: str = "",
        plots_per_figure: int = 5,
        output_path: str = ".",
        fig_title: str = "boxplot",
        stripplot_kws: Dict[str, Any] = {},
    ) -> Optional[DataVisualizer]:
        post_hoc_table = pd.DataFrame(self.post_hoc_report)

        if len(post_hoc_table) == 0:
            logger.warning("No post-hoc results found. Skip boxplot.")
            return None

        for layout_cols, i, num_chunks in batch_iterator(
            post_hoc_table["variable"].drop_duplicates(), plots_per_figure
        ):
            viz = DataVisualizer(self.data)
            viz.create_figure(
                layout_cols=layout_cols.to_list(),
                group_col=group_col,
                n_cols=1,
                fig_title=f"{fig_title}: {i}/{num_chunks}",
            )
            viz.add_boxplot(x_col=x_col)
            viz.add_stripplot(x_col=x_col, **stripplot_kws)
            viz.figure.subplots_adjust(left=0.15, bottom=0.1, top=0.95, right=0.97, wspace=0.25, hspace=0.1)
            viz.display()
            viz.save(f"{output_path}/{fig_title}_{i}.png")
        return viz


if __name__ == "__main__":

    # Generate test data with additional categorical group columns
    data = pd.DataFrame(
        {
            "normal": np.random.normal(loc=0, scale=1, size=5000),
            "positive_skewed": np.random.exponential(scale=1, size=5000),
            "negative_skewed": -np.random.exponential(scale=1, size=5000),
            "bimodal": np.hstack(
                (np.random.normal(loc=-1, scale=1, size=2500), np.random.normal(loc=3, scale=1, size=2500))
            ),
        }
    )
    # Add random categorical group columns
    np.random.seed(42)
    data["group1"] = np.random.choice(["A", "B", "C"], size=5000)
    data["group2"] = np.random.choice(["X", "Y"], size=5000)
    viz = DataVisualizer(data)

    # Test add_histogram without group_col
    viz.create_figure(layout_cols=["normal", "positive_skewed", "negative_skewed", "bimodal"], n_cols=2)
    viz.add_histogram(show_distribution_stats=True, kde=True)
    viz.figure.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95, wspace=0.25, hspace=0.25)
    viz.display()

    # Test add_histogram
    viz.create_figure(
        layout_cols=["normal", "positive_skewed", "negative_skewed", "bimodal"], group_col="group2", n_cols=2
    )
    viz.add_histogram(show_distribution_stats=True, kde=True)
    viz.figure.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95, wspace=0.25, hspace=0.25)
    viz.display()

    # Test add_boxplot and add_stripplot
    fig, axes = viz.create_figure(
        layout_cols=["normal", "positive_skewed", "negative_skewed", "bimodal"],
        group_col="group2",
        n_cols=2,
        fig_title="Boxplot by group1",
    )
    viz.add_boxplot(x_col="group1", showfliers=False)
    viz.add_stripplot(x_col="group1")
    viz.display()

    # Test add_heatmap
    viz.create_figure(layout_cols=["group1"], n_cols=2, fig_title="Correlation Heatmap (Test Data)")
    viz.add_correlation_heatmap()
    viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.9, right=0.97, wspace=0.25, hspace=0.4)
    viz.display()
