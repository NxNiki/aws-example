import logging
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import leaves_list, linkage
from statannotations.Annotator import Annotator

from bituslabs_ds.config import DEFAULT_MAX_JOBS
from bituslabs_ds.utils import group_iterator, keep_numeric_columns

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def read_csv_cols(
    files: List[str],
    columns: List[str],
    filters: Optional[Dict[str, Any]] = None,
    sampling: Optional[Union[int, float]] = None,
    max_workers: int = DEFAULT_MAX_JOBS,
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


def read_excel_sheets(file_path: str, sheet_name_col: str = "sheet_name"):
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    sheet_names = list(all_sheets.keys())

    if len(sheet_names) == 1:
        df = all_sheets[sheet_names[0]]
        print(f"Only one sheet '{sheet_names[0]}' shape: {df.shape}")
    else:
        df = pd.concat(all_sheets.values(), keys=sheet_names)
        df = df.reset_index(level=0).rename(columns={"level_0": sheet_name_col})

        for sheet_name, sheet_df in all_sheets.items():
            print(f"Sheet '{sheet_name}' shape: {sheet_df.shape}")

        print(f"Combined data shape: {df.shape}")

    cols_to_drop = df.columns[df.isna().all()].tolist()
    if len(cols_to_drop) > 0:
        print("Drop Columns with all NaNs:", cols_to_drop)
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
    # Make a copy to avoid modifying the original DataFrame unexpectedly
    data_out = data.copy()

    # Split the column into a new temporary DataFrame.
    # `expand=True` automatically creates the necessary number of columns
    # and fills missing parts with None.
    split_df = data_out[column].str.split(sep, expand=True)

    # Dynamically create the new column names, e.g., 'pattern_1', 'pattern_2', etc.
    new_col_names = [f"{column}_{i+1}" for i in range(split_df.shape[1])]

    # Assign these new names to the columns of our temporary DataFrame
    split_df.columns = new_col_names

    # Join the new split columns back to the original DataFrame
    result_df = pd.concat([data_out, split_df], axis=1)

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


def plot_correlation(data: pd.DataFrame, output_path: str = ".", title: str = "Correlation Heatmap") -> None:

    corr = data.corr()
    Z = linkage(corr.values, method="average")
    g = sns.clustermap(
        corr,
        row_cluster=True,
        col_cluster=True,
        row_linkage=Z,
        col_linkage=Z,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        figsize=(10, 8),
        linewidths=0.75,
        dendrogram_ratio=(0.1, 0.15),
        cbar_pos=(0, 0.2, 0.02, 0.5),
    )

    g.figure.suptitle(title, fontsize=16, y=0.95)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.ax_row_dendrogram.set_visible(False)
    g.gs.update(left=0.05)

    pos = g.cax.get_position()
    new_pos = (pos.x0, pos.y0 - 0.5, pos.width * 0.5, pos.height * 2)
    g.cax.set_position(new_pos)

    os.makedirs(f"{output_path}/figures", exist_ok=True)
    g.savefig(f"{output_path}/figures/{title}.png")
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
        # Calculate vertical dodge amount for text (to match swarmplot's dodge)
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


def plot_df_distribution(
    data: Union[pd.DataFrame, pd.Series],
    bins: int = 30,
    alpha: float = 0.5,
    log: bool = False,
    figure_name: Optional[str] = None,
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

    data = data.copy()
    if isinstance(data, pd.Series):
        data = data.to_frame()

    boolean_cols = [col for col in data.columns if pd.api.types.is_bool_dtype(data[col])]
    data[boolean_cols] = data[boolean_cols].astype(int)
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
        if len(values) == 0:
            print("No values found for column", col)
            continue

        n_bins = max(min(bins, values.nunique()), 30)
        counts, bin_edges, _ = ax.hist(values, bins=n_bins, alpha=alpha, edgecolor="black")

        # Annotate with column name at max bin
        if len(counts) > 0:
            max_bin_index = counts.argmax()
            x_pos = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
            y_pos = counts[max_bin_index]
            ax.text(x_pos, y_pos, f"{counts.max():.2e}", fontsize=9, ha="center", va="bottom")

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
    if figure_name is not None:
        plt.savefig(figure_name)
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

            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=14)
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

                    # Set custom annotations with the p-values from Tukey HSD
                    annotator.set_custom_annotations(p_values)

                    # Configure and apply annotations
                    annotator.configure(
                        text_format="star",
                        loc="inside",
                        verbose=False,
                        # line_offset=0.1,
                        line_height=0.02,
                        text_offset=1,
                    )
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
    fig.subplots_adjust(top=0.92, hspace=0.2, wspace=0.2)

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
