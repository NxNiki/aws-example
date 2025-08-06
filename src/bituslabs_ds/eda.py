import itertools
import logging
import math
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.cm as cm
import matplotlib.legend_handler as lh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from diptest import diptest
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator

from bituslabs_ds.config import DEFAULT_MAX_JOBS
from bituslabs_ds.utils import batch_iterator, convert_to_list, df_power_transform, group_iterator, keep_numeric_columns

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
                        warnings.warn(
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
            warnings.warn(f"Error processing '{file}': {e}", UserWarning)
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
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    y_col_right: Optional[str] = None,
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
                    xycoords=("axes fraction", "data"),
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
                    xycoords=("axes fraction", "data"),
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


def plot_multiple_box_swarm(
    data: pd.DataFrame,
    x_cols: List[str],
    y_cols: List[str],
    group_col: Optional[str] = None,
    n_cols: int = 3,
    fig_size: Tuple[float, float] = (8, 6),
    fig_title: Optional[str] = None,
    log_scale: bool = False,
    post_hoc_table: Optional[pd.DataFrame] = None,
):
    """
    For each combination of y_col and x_col, plot a box + swarm plot. All x_cols and group_col will be plotted in one
    subplot. Adds statistical test results (Tukey HSD) to the plot for pairwise comparisons within group_col
    using statannotations for visual display.

    Parameters:
        data (pd.DataFrame): Original DataFrame.
        x_cols (List[str]): Columns to use as x-axis groupings (categorical).
        y_cols (List[str]): Numeric value columns to visualize.
        group_col (Optional[str]): Column to use as hue (optional).
        n_cols (int): Subplots per row.
        fig_size (Tuple[float, float]): Figure size per plot.
        fig_title (Optional[str]): Optional figure title.
        log_scale (bool): Whether to use logarithmic scale on y-axis.
        post_hoc_table (Optional[pd.DataFrame]): table of the post-hoc results. should have columns: variable,
            comparison, and p_value
    """
    n_plots = len(y_cols) * len(x_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Calculate subplot width based on unique x_cols values to ensure readability
    # Max unique values across all x_cols
    max_x_unique = max(data[col].nunique() for col in x_cols)
    # Adjust subplot_width dynamically. A base of 0.75 for small number of categories,
    # scaling up for more categories.
    subplot_width_factor = max(max_x_unique / 4, 0.75)  # Ensure a minimum width

    # If group_col is present, each x-tick will have multiple dodged groups,
    # so we need more horizontal space.
    if group_col:
        num_groups = data[group_col].nunique()
        # Increase width for more groups, 0.2 is an arbitrary scaling factor
        subplot_width_factor *= 1 + (num_groups - 1) * 0.2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_size[0] * n_cols * subplot_width_factor, fig_size[1] * n_rows)
    )

    # Ensure axes is always iterable, even for a single subplot
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Determine the number of unique groups for palette creation
    # This is used for both plotting and manual-edge coloring
    num_groups_for_palette = len(data[group_col].unique()) if group_col else 1
    palette = sns.color_palette("pastel", n_colors=num_groups_for_palette)

    plot_idx = 0
    for x_col in x_cols:
        for y_col in y_cols:
            cols_to_keep = [x_col, y_col] + ([group_col] if group_col else [])
            # Drop NaNs in y_col before melting and statistical tests
            df_long = data[cols_to_keep].copy().dropna(subset=[y_col])

            # Melt the DataFrame for seaborn plotting
            df_long = df_long.melt(
                id_vars=[x_col] + ([group_col] if group_col else []),
                value_vars=[y_col],
                var_name="variable",  # This column is not strictly needed after melt for single y_col
                value_name="value",
            )

            ax = axes[plot_idx]
            plot_idx += 1

            # Plot strip plot
            non_nans = df_long["value"].count()
            sns.stripplot(
                data=df_long,
                x=x_col,
                y="value",
                hue=group_col if group_col else None,
                dodge=True if group_col else False,
                ax=ax,
                palette=palette,
                size=2 if non_nans > 1e5 else 4,
                legend=False,  # We will add a single global legend
                jitter=0.25,
                alpha=0.1 if non_nans > 1e5 else 0.7,
            )

            # Plot boxplot
            sns.boxplot(
                data=df_long,
                x=x_col,
                y="value",
                hue=group_col if group_col else None,
                ax=ax,
                palette=palette,
                boxprops=dict(linewidth=1.5, facecolor=(0, 0, 0, 0)),  # Transparent face color
                medianprops=dict(linewidth=2),
                whis=1.5,
                notch=True,
                showfliers=False,  # Do not show outliers, swarm plot handles individual points
            )

            if log_scale:
                ax.set_yscale("symlog", linthresh=1)  # Use symlog for better visualization of data around zero

            ax.set_xlabel("", fontsize=12)
            ax.set_ylabel("", fontsize=12)
            ax.set_title(y_col, fontsize=14)
            ax.tick_params(axis="x", rotation=0, labelsize=10)  # Use labelsize for tick labels
            for label in ax.get_xticklabels():
                label.set_fontsize(12)  # Ensure x-tick labels are readable

            if group_col and ax.get_legend():
                ax.legend_.remove()

            # Manually update each box-edge color to match hue color
            if group_col:
                # Get the order of hues as they appear in the plot
                hue_order_in_plot = df_long[group_col].unique()
                for i, artist in enumerate(ax.artists):
                    # Determine which hue color to apply based on the artist's index
                    hue_idx = i % len(hue_order_in_plot)
                    color = palette[hue_idx]
                    artist.set_edgecolor(color)
                    # Also set the color of the median line
                    if len(artist.get_children()) > 0:
                        line = artist.get_children()[0]
                        line.set_color(color)
            else:
                # If no group_col, set edge color to the first color in palette
                for artist in ax.artists:
                    artist.set_edgecolor(palette[0])
                    if len(artist.get_children()) > 0:
                        line = artist.get_children()[0]
                        line.set_color(palette[0])

            # Add statistical test results if group_col is provided using statannotations
            if post_hoc_table is not None:
                annotation_pairs = post_hoc_table.loc[post_hoc_table["variable"] == y_col, "comparison"].to_list()
                p_values = post_hoc_table.loc[post_hoc_table["variable"] == y_col, "p_value"].to_list()
                try:
                    # Initialize Annotator
                    # Note: x and hue parameters for Annotator should refer to the columns
                    # that define the groups being compared. In this case, x_col is the main
                    # grouping, and group_col is the hue within each x_col category.
                    # For statannotations, when comparing within a single x_col category,
                    # the 'x' parameter should be the column defining the groups being compared (group_col),
                    # and the 'data' should be the subset for that x_col category.
                    annotator = Annotator(
                        ax,
                        annotation_pairs,
                        data=df_long,
                        x=x_col,  # This is the column defining the groups for comparison
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
                    print(f"Warning: Could not perform add annotation to box plot. Error: {e}")
                    print("This might happen if there's not enough data or groups for comparison in this subset.")
                except Exception as e:
                    print(f"An unexpected error occurred during annotating the box plot: {e}")

    # Clean up any unused axes (subplots) that were created but not plotted on
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    if group_col:
        # Create a single legend for the entire figure
        handles, labels = [], []
        # Attempt to get handles and labels from the first subplot's legend if it exists
        # This is a robust way to get legend entries with colors
        for ax_item in axes:
            if ax_item and ax_item.get_legend():
                handles, labels = ax_item.get_legend_handles_labels()
                break

        # If no legend was found (e.g., due to legend=False in strip plot/boxplot and then removed),
        # manually create proxy artists for the global legend
        if not handles and group_col:
            unique_groups = data[group_col].unique()
            for i, group in enumerate(unique_groups):
                # Create a proxy artist (a line with a marker) for the legend entry
                handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i], markersize=8))
                labels.append(group)

        # Only add the global legend if there are handles to show
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 0.95), title=group_col)

    plt.tight_layout(pad=1)  # Adjust subplot parameters for a tight layout
    # Adjust top margin to make space for the suptitle
    fig.subplots_adjust(top=0.94, hspace=0.25, wspace=0.2)

    if fig_title:
        fig.suptitle(fig_title, fontsize=16, y=0.98)  # Add a main title to the figure
    plt.show()


def plot_scatter_pairs(
    data: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str, bool, bool]]] = None,
    max_per_row: int = 5,
    fig_size_per_plot: Tuple[int, int] = (4, 4),
    alpha: float = 0.7,
) -> None:
    """
    Plots scatter plots for every unique pair of numeric columns in the DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing numeric columns.
    - pairs (List[Tuple[str, str]]): List of pairs of numeric column names.
    - max_per_row (int): Maximum number of plots per row.
    - fig_size_per_plot (Tuple[int, int]): Size of each subplot (width, height).
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
        nrows=nrows, ncols=ncols, figsize=(fig_size_per_plot[0] * ncols, fig_size_per_plot[1] * nrows)
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


class DataVisualizer:
    """
    A comprehensive class for data visualization that organizes all plotting functions
    and allows sharing of data and configuration across methods.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, "DataProfiler"],
        x_cols: Optional[List[str]] = None,
        y_cols: Optional[List[str]] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        fig_size: Tuple[float, float] = (8, 6),
        palette: str = "pastel",
    ):
        """
        Initialize the DataVisualizer with data and configuration.

        Parameters:
            data: The main DataFrame or DataProfiler instance
            x_cols: Columns to use as x-axis groupings (categorical)
            y_cols: Numeric value columns to visualize
            group_col: Column to use as hue/grouping
            time_col: Column containing time information
            fig_size: Default figure size
            palette: Color palette for plots
        """
        # Always create a DataProfiler for consistent analytics access
        if isinstance(data, DataProfiler):
            self.profiler = data
        else:
            self.profiler = DataProfiler(data)

        self.x_cols = x_cols or []
        self.y_cols = y_cols or []
        self.group_col = group_col
        self.time_col = time_col
        self.fig_size = fig_size
        self.palette = palette

        # Store plot configurations and results
        self.current_figure = None
        self.current_axes = None
        self.plot_configs: List[Dict[str, Any]] = []
        self.current_plot_data = None

    @property
    def data(self) -> pd.DataFrame:
        """Access the DataFrame through the profiler."""
        return self.profiler.df

    def set_columns(
        self,
        x_cols: Optional[List[str]] = None,
        y_cols: Optional[List[str]] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        """Update column configuration."""
        if x_cols is not None:
            self.x_cols = x_cols
        if y_cols is not None:
            self.y_cols = y_cols
        if group_col is not None:
            self.group_col = group_col
        if time_col is not None:
            self.time_col = time_col

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

    def create_figure(
        self,
        x_cols: Optional[List[str]] = None,
        y_cols: Optional[List[str]] = None,
        n_cols: int = 3,
        fig_title: Optional[str] = None,
        log_scale: bool = False,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create figure and subplots for visualization.

        Parameters:
            x_cols: List of columns to use as x-axis groupings (if None, uses self.x_cols)
            y_cols: List of columns to visualize on y-axis (if None, uses self.y_cols)
            n_cols: Number of subplots per row
            fig_title: Optional figure title
            log_scale: Whether to use logarithmic scale on y-axis

        Returns:
            Tuple of (figure, axes) for further customization
        """
        # Use instance variables if not provided
        x_cols = x_cols or self.x_cols
        y_cols = y_cols or self.y_cols

        if not x_cols or not y_cols:
            raise ValueError("x_cols and y_cols must be provided or set in __init__")

        n_plots = len(y_cols) * len(x_cols)
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Calculate subplot width based on unique x_cols values
        max_x_unique = max(self.data[col].nunique() for col in x_cols)
        subplot_width_factor = max(max_x_unique / 4, 0.75)

        if self.group_col:
            num_groups = self.data[self.group_col].nunique()
            subplot_width_factor *= 1 + (num_groups - 1) * 0.2

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(self.fig_size[0] * n_cols * subplot_width_factor, self.fig_size[1] * n_rows)
        )

        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Set up basic plot properties for all axes
        for ax in axes:
            if log_scale:
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xlabel("", fontsize=12)
            ax.set_ylabel("", fontsize=12)
            ax.tick_params(axis="x", rotation=0, labelsize=10)
            for label in ax.get_xticklabels():
                label.set_fontsize(12)

        if fig_title:
            fig.suptitle(fig_title, fontsize=16, y=0.98)

        self.current_figure = fig
        self.current_axes = axes

        return fig, axes

    def add_boxplot(
        self,
        axes: Optional[np.ndarray] = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        Add box plot to the specified axes or all current axes.

        Parameters:
            axes: The axes to add box plots to (if None, uses all current axes)
            x_col: Column to use as x-axis grouping (if None, uses current plot data)
            y_col: Column to visualize on y-axis (if None, uses current plot data)
            alpha: Transparency level

        Returns:
            The axes with box plots added
        """
        if axes is None:
            if self.current_axes is None:
                raise ValueError("No axes available. Create figure first.")
            axes = self.current_axes

        # Prepare data if x_col and y_col provided
        if x_col is not None and y_col is not None:
            df_long = self._convert_to_long_format(x_col, y_col)
        elif self.current_plot_data is not None:
            df_long = self.current_plot_data["df_long"]
            x_col = self.current_plot_data["x_col"]
        else:
            raise ValueError("No plot data available. Provide x_col and y_col or set plot data first.")

        group_col = self.group_col

        # Create palette
        num_groups_for_palette = len(self.data[group_col].unique()) if group_col else 1
        palette = sns.color_palette(self.palette, n_colors=num_groups_for_palette)

        for ax in axes:
            if ax is None:
                continue

            # Plot boxplot
            sns.boxplot(
                data=df_long,
                x=x_col,
                y="value",
                hue=group_col if group_col else None,
                ax=ax,
                palette=palette,
                boxprops=dict(linewidth=1.5, facecolor=(0, 0, 0, 0)),
                medianprops=dict(linewidth=2),
                whis=1.5,
                notch=True,
                showfliers=False,
            )

            # Update box-edge colors
            if group_col:
                hue_order_in_plot = df_long[group_col].unique()
                for i, artist in enumerate(ax.artists):
                    hue_idx = i % len(hue_order_in_plot)
                    color = palette[hue_idx]
                    artist.set_edgecolor(color)
                    if len(artist.get_children()) > 0:
                        line = artist.get_children()[0]
                        line.set_color(color)
            else:
                for artist in ax.artists:
                    artist.set_edgecolor(palette[0])
                    if len(artist.get_children()) > 0:
                        line = artist.get_children()[0]
                        line.set_color(palette[0])

        return axes

    def add_stripplot(
        self,
        axes: Optional[np.ndarray] = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        size: int = 4,
        alpha: float = 0.7,
        jitter: float = 0.25,
    ) -> np.ndarray:
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
        if axes is None:
            if self.current_axes is None:
                raise ValueError("No axes available. Create figure first.")
            axes = self.current_axes

        # Prepare data if x_col and y_col provided
        if x_col is not None and y_col is not None:
            df_long = self._convert_to_long_format(x_col, y_col)
        elif self.current_plot_data is not None:
            df_long = self.current_plot_data["df_long"]
            x_col = self.current_plot_data["x_col"]
        else:
            raise ValueError("No plot data available. Provide x_col and y_col or set plot data first.")

        group_col = self.group_col

        # Create palette
        num_groups_for_palette = len(self.data[group_col].unique()) if group_col else 1
        palette = sns.color_palette(self.palette, n_colors=num_groups_for_palette)

        for ax in axes:
            if ax is None:
                continue

            # Plot strip plot
            non_nans = df_long["value"].count()
            sns.stripplot(
                data=df_long,
                x=x_col,
                y="value",
                hue=group_col if group_col else None,
                dodge=True if group_col else False,
                ax=ax,
                palette=palette,
                size=2 if non_nans > 1e5 else size,
                legend=False,
                jitter=jitter,
                alpha=0.1 if non_nans > 1e5 else alpha,
            )

        return axes

    def add_histogram(
        self,
        axes: Optional[np.ndarray] = None,
        y_col: Optional[str] = None,
        bins: int = 50,
        alpha: float = 0.5,
        add_kde: bool = False,
        show_distribution_stats: bool = False,
    ) -> np.ndarray:
        """
        Add histogram to the specified axes or all current axes.

        Parameters:
            axes: The axes to add histograms to (if None, uses all current axes)
            y_col: Column to visualize (if None, uses current plot data)
            bins: Number of histogram bins
            alpha: Transparency level
            add_kde: Whether to add KDE plot
            distribution_stats: Whether to add distribution statistics (uses profiler analytics)

        Returns:
            The axes with histograms added
        """
        if axes is None:
            if self.current_axes is None:
                raise ValueError("No axes available. Create figure first.")
            axes = self.current_axes

        # Get data
        if y_col is not None:
            values = self.data[y_col].dropna()
        elif self.current_plot_data is not None:
            values = self.current_plot_data["df_long"]["value"].dropna()
        else:
            raise ValueError("No plot data available. Provide y_col or set plot data first.")

        for ax in axes:
            if ax is None:
                continue

            if len(values) == 0:
                print("No values found for column", y_col)
                continue

            # Distribution statistics using profiler analytics
            if show_distribution_stats and y_col:
                feature_stats = self.profiler.check_distribution_stats(refresh=True)
                stat = next((s for s in feature_stats if s.get("column") == y_col), None)
                if stat and not pd.isna(stat.get("skewness")):
                    skewness = stat["skewness"]
                    dip_stat = stat["dip_stat"]
                    p_value = stat["dip_p_values"]

                    if p_value < 0.05:
                        legend_label = f"skewness: {skewness:.3f}\nunimodality: {dip_stat:.3f}*"
                    else:
                        legend_label = f"skewness: {skewness:.3f}\nunimodality: {dip_stat:.3f}"
                else:
                    legend_label = ""
            else:
                if len(values.unique()) <= 2:
                    legend_label = ""
                else:
                    skewness = skew(values, bias=False)
                    dip_statistic_unimodal, p_value_unimodal = diptest(values)

                    if p_value_unimodal < 0.05:
                        legend_label = f"skewness: {skewness:.3f}\nunimodality: {dip_statistic_unimodal:.3f}*"
                    else:
                        legend_label = f"skewness: {skewness:.3f}\nunimodality: {dip_statistic_unimodal:.3f}"

            n_bins = max(min(bins, values.nunique()), 50)
            counts, bin_edges, patches = ax.hist(
                values, bins=n_bins, alpha=alpha, edgecolor="black", label=legend_label
            )

            if add_kde:
                ax2 = ax.twinx()
                sns.kdeplot(values, bw_method="silverman", color="red", linestyle="--", ax=ax2)
                ax2.set_ylabel("Density")
                ax2.grid(False)

            # Annotate with column name at max bin
            if len(counts) > 0:
                max_bin_index = counts.argmax()
                x_pos = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
                y_pos = counts[max_bin_index]
                ax.text(x_pos, y_pos, f"{counts.max():.2e}", fontsize=9, ha="center", va="bottom")

            ax.set_title(y_col if y_col else "Distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

            custom_handler_mapping = {patches[0]: NoSymbolHandler()}
            ax.legend(handler_map=custom_handler_mapping, frameon=False)
            ax.grid(True)

        return axes

    def add_heatmap(
        self, axes: Optional[np.ndarray] = None, method: str = "pearson", title: str = "Correlation Heatmap"
    ) -> np.ndarray:
        """
        Add correlation heatmap to the specified axes or all current axes.

        Parameters:
            axes: The axes to add heatmaps to (if None, uses all current axes)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            title: Plot title

        Returns:
            The axes with heatmaps added
        """
        if axes is None:
            if self.current_axes is None:
                raise ValueError("No axes available. Create figure first.")
            axes = self.current_axes

        numeric_data = self.data.select_dtypes(include=["number"])
        correlation_matrix = numeric_data.corr(method=method)

        for ax in axes:
            if ax is None:
                continue

            heatmap = sns.heatmap(
                correlation_matrix,
                cmap="viridis",
                linewidths=0.5,
                ax=ax,
            )

            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel("Variables", fontsize=12)
            ax.set_ylabel("Variables", fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)

        return axes

    def add_scatter(
        self,
        axes: Optional[np.ndarray] = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        alpha: float = 0.7,
        s: int = 2,
    ) -> np.ndarray:
        """
        Add scatter plot to the specified axes or all current axes.

        Parameters:
            axes: The axes to add scatter plots to (if None, uses all current axes)
            x_col: Column for x-axis (if None, uses current plot data)
            y_col: Column for y-axis (if None, uses current plot data)
            alpha: Transparency level
            s: Point size

        Returns:
            The axes with scatter plots added
        """
        if axes is None:
            if self.current_axes is None:
                raise ValueError("No axes available. Create figure first.")
            axes = self.current_axes

        # Get data
        if x_col is not None and y_col is not None:
            x_data = self.data[x_col]
            y_data = self.data[y_col]
        elif self.current_plot_data is not None:
            df_long = self.current_plot_data["df_long"]
            x_col = self.current_plot_data["x_col"]
            x_data = df_long[x_col]
            y_data = df_long["value"]
        else:
            raise ValueError("No plot data available. Provide x_col and y_col or set plot data first.")

        for ax in axes:
            if ax is None:
                continue

            ax.scatter(x_data, y_data, alpha=alpha, s=s)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")

        return axes

    def add_statistical_annotations(
        self,
        axes: Optional[np.ndarray] = None,
        post_hoc_table: pd.DataFrame = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
    ) -> np.ndarray:
        """
        Add statistical annotations to box plots or strip plots.

        Parameters:
            axes: The axes to add annotations to (if None, uses all current axes)
            post_hoc_table: DataFrame with columns ['variable', 'comparison', 'p_value']
            x_col: Column used for x-axis grouping (if None, uses current plot data)
            y_col: Column used for y-axis (if None, uses current plot data)

        Returns:
            The axes with annotations added
        """
        if axes is None:
            if self.current_axes is None:
                raise ValueError("No axes available. Create figure first.")
            axes = self.current_axes

        if post_hoc_table is None:
            print("Warning: No post_hoc_table provided. Skipping annotations.")
            return axes

        # Prepare data if x_col and y_col provided
        if x_col is not None and y_col is not None:
            df_long = self._convert_to_long_format(x_col, y_col)
        elif self.current_plot_data is not None:
            df_long = self.current_plot_data["df_long"]
            x_col = self.current_plot_data["x_col"]
            y_col = self.current_plot_data["y_col"]
        else:
            raise ValueError("No plot data available. Provide x_col and y_col or set plot data first.")

        group_col = self.group_col

        # Get annotation pairs and p_values for this y_col
        annotation_pairs = post_hoc_table.loc[post_hoc_table["variable"] == y_col, "comparison"].to_list()
        p_values = post_hoc_table.loc[post_hoc_table["variable"] == y_col, "p_value"].to_list()

        if not annotation_pairs:
            return axes

        for ax in axes:
            if ax is None:
                continue

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

        return axes

    def add_legend(self, fig: Optional[plt.Figure] = None, axes: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Add legend to the figure.

        Parameters:
            fig: The figure to add legend to (if None, uses current figure)
            axes: The axes to get legend handles from (if None, uses all current axes)

        Returns:
            The figure with legend added
        """
        if fig is None:
            if self.current_figure is None:
                raise ValueError("No figure available. Create a plot first.")
            fig = self.current_figure

        if axes is None:
            if self.current_axes is None:
                raise ValueError("No axes available. Create a plot first.")
            axes = self.current_axes

        if not self.group_col:
            return fig

        # Create palette
        num_groups_for_palette = len(self.data[self.group_col].unique())
        palette = sns.color_palette(self.palette, n_colors=num_groups_for_palette)

        handles, labels = [], []

        # Try to get handles and labels from any axis
        for ax in axes:
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
        if self.current_figure:
            plt.show()
        else:
            print("No figure to display. Create a plot first.")

    def save(self, filename: str, dpi: int = 300, bbox_inches: str = "tight"):
        """Save the current figure."""
        if self.current_figure:
            self.current_figure.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        else:
            print("No figure to save. Create a plot first.")


class DataProfiler:
    """
    A comprehensive data profiling class that analyzes data distributions,
    correlations, and provides analytical insights.
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
        self.skewness_threshold = skewness_threshold
        self.output_path = output_path

        # Initialize profiling results
        self._original_numeric_columns: List[str] = self.check_numeric_columns(include_boolean=True)
        self.count_missing_columns(verbose=True)

    @property
    def processed_numerical_columns(self):
        """Get processed numerical columns for analysis."""
        return self.check_numeric_columns(include_boolean=True, refresh=True)

    def check_numeric_columns(self, include_boolean: bool = True, refresh: bool = False) -> List[str]:
        """Check and return numeric columns in the DataFrame."""
        if not hasattr(self, "_processed_numerical_columns") or refresh:
            self._processed_numerical_columns = self.check_df_numerical_columns(self.df, include_boolean)
        return self._processed_numerical_columns

    @staticmethod
    def check_df_numerical_columns(data: Union[pd.DataFrame, pd.Series], include_boolean: bool = True) -> List[str]:
        """Check which columns are numerical in a DataFrame."""
        if isinstance(data, pd.Series):
            data = data.to_frame()

        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

        if not include_boolean:
            numeric_cols = [col for col in numeric_cols if data[col].dtype != "bool"]

        return numeric_cols

    def check_distribution_stats(self, refresh: bool = False) -> List[Dict]:
        """Check distribution statistics for numerical columns."""
        if not hasattr(self, "_distribution_stats") or refresh:
            numerical_cols = self.check_numeric_columns(include_boolean=True)
            self._distribution_stats = []

            for col in numerical_cols:
                values = self.df[col].dropna()
                if len(values) == 0:
                    continue

                skewness = skew(values, bias=False)
                dip_statistic_unimodal, p_value_unimodal = diptest(values)

                self._distribution_stats.append(
                    {
                        "column": col,
                        "skewness": skewness,
                        "dip_stat": dip_statistic_unimodal,
                        "dip_p_values": p_value_unimodal,
                        "is_skewed": abs(skewness) > self.skewness_threshold,
                        "is_unimodal": p_value_unimodal >= 0.05,
                    }
                )

        return self._distribution_stats

    def get_skewed_columns(self, refresh: bool = False) -> Tuple[List[str], List[str]]:
        """Get positively and negatively skewed columns."""
        stats = self.check_distribution_stats(refresh=refresh)

        pos_skewed = [stat["column"] for stat in stats if stat["is_skewed"] and stat["skewness"] > 0]
        neg_skewed = [stat["column"] for stat in stats if stat["is_skewed"] and stat["skewness"] < 0]

        return pos_skewed, neg_skewed

    def transform_skewed_columns(self, pos_suffix: str = "_log", neg_suffix: str = "_exp") -> None:
        """Transform skewed columns using log or exponential transformations."""
        pos_skewed, neg_skewed = self.get_skewed_columns()

        # Transform positively skewed columns (log transformation)
        for col in pos_skewed:
            self.df[f"{col}{pos_suffix}"] = np.log1p(self.df[col])

        # Transform negatively skewed columns (exponential transformation)
        for col in neg_skewed:
            self.df[f"{col}{neg_suffix}"] = np.exp(self.df[col])

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
        df_scaled = scaler.fit_transform(data[columns].dropna(axis=1, how="any"))

        if "mean" in method:
            data[f"{col_name}_mean"] = df_scaled.mean(axis=1)
        if "pca" in method:
            pca = PCA(n_components=None)
            principal_components = pca.fit_transform(df_scaled)
            explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
            print(f"Cumulative Explained Variance for '{col_name}' PCA components:")
            for i, cum_var in enumerate(explained_variance_ratio_cumsum):
                print(f"  PC{i + 1}: {cum_var:.4f}")

            for i in range(principal_components.shape[1]):
                data[f"{col_name}_pca{i + 1}"] = principal_components[:, i]

        return data

    def show_group_stats(self, group_cols: List[str], var_columns: List[str], transpose: bool = False):
        self.show_df_group_stats(self.df, group_cols, var_columns, transpose)

    @staticmethod
    def show_df_group_stats(
        data: pd.DataFrame, group_cols: List[str], var_columns: List[str], transpose: bool = False
    ) -> None:
        """
        Display group statistics for specified columns.
        """
        if not group_cols or not var_columns:
            print("Please provide both group_cols and var_columns.")
            return

        # Create groupby object
        grouped = data.groupby(group_cols)[var_columns]

        # Calculate statistics
        stats = grouped.agg(["count", "mean", "std", "min", "max"])

        if transpose:
            stats = stats.T

        print("Group Statistics:")
        print(stats)
        print("\n" + "=" * 50 + "\n")

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
                if source == "Residual":
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
        if len(self.anova_report) == 0:
            res = pd.DataFrame({})
        else:
            res = pd.DataFrame(self.anova_report)
            interaction_eta_label = f"{self.between_vars[0]} * {self.between_vars[1]}-eta2"
            res.sort_values(by=interaction_eta_label, inplace=True, ascending=False)
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
            raise Exception("Run anova before post-hoc analysis!")

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

    def show_box_plot(self, x_cols: str, group_col: str) -> None:
        post_hoc_table = pd.DataFrame(self.post_hoc_report)
        for y_cols, i, num_chunks in batch_iterator(post_hoc_table["variable"].drop_duplicates(), 5):
            plot_multiple_box_swarm(
                self.data,
                x_cols=[x_cols],
                y_cols=list(y_cols),
                n_cols=1,
                group_col=group_col,
                log_scale=True,
                fig_title=f"variables with sig anova: {i}/{num_chunks}",
                post_hoc_table=post_hoc_table,
            )


if __name__ == "__main__":

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

    viz = DataVisualizer(data, y_cols=["normal", "positive_skewed", "negative_skewed", "bimodal"])
    viz.add_histogram(y_col="normal", show_distribution_stats=True)
    viz.display()
