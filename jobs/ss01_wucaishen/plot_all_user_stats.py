"""
Script to load CSV file and create barplots with standard error based on selected columns, grouped by another column.

Usage:
    python plot_all_user_stats.py

The script will:
    - Automatically detect all numeric columns (excluding group_by_column and user_id)
    - Filter out rows where ai_group == "newBee"
    - Create barplots with standard error bars if values vary within groups, otherwise create simple barplots
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURATION - Modify these variables for your use case
# ============================================================================

# Path to your data file (CSV or Parquet)
CSV_FILE_PATH = "/Users/niuxin/Documents/aws-example/jobs/output_ss01_wucaishen/all_user_stats.parquet"

# Column name to group by (e.g., "category", "group", "date")
GROUP_BY_COLUMN = "ai_group"

# Value to filter out from group_by_column (set to None to skip filtering)
FILTER_OUT_VALUE = "newBee"

# Columns to exclude from automatic detection (in addition to group_by_column and user_id)
EXCLUDE_COLUMNS: List[str] = []

# Output directory for saving plots (None = don't save, just display)
OUTPUT_DIR: Optional[str] = "/Users/niuxin/Documents/aws-example/jobs/output_ss01_wucaishen/all_user_stats_plot"

# Aggregation function for grouped data when using barplots (e.g., "sum", "mean", "count", "median")
AGGREGATION = "mean"

# Figure size for plots
FIG_SIZE = (10, 5)

# ============================================================================


def load_and_validate_data(file_path: str, group_by_col: str, exclude_cols: List[str]) -> pd.DataFrame:
    """
    Load CSV or Parquet file, filter data, and automatically detect numeric columns to plot.

    Args:
        file_path: Path to CSV or Parquet file
        group_by_col: Column name to group by
        exclude_cols: List of columns to exclude from plotting

    Returns:
        Tuple of (DataFrame with loaded and filtered data, list of numeric column names)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect file type and load accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".parquet":
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows from parquet file: {file_path}")
    elif file_ext == ".csv":
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from CSV file: {file_path}")
    else:
        # Try to auto-detect: first try parquet, then CSV
        try:
            df = pd.read_parquet(file_path)
            print(f"Loaded {len(df)} rows from parquet file: {file_path}")
        except Exception:
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded {len(df)} rows from CSV file: {file_path}")
            except Exception as e:
                raise ValueError(
                    f"Could not read file {file_path}. "
                    f"Supported formats: CSV (.csv) and Parquet (.parquet). Error: {e}"
                )

    # Validate group_by_col exists
    if group_by_col not in df.columns:
        raise ValueError(
            f"Group by column '{group_by_col}' not found in file.\n" f"Available columns: {list(df.columns)}"
        )

    # Filter out specified value from group_by_column
    if FILTER_OUT_VALUE is not None and group_by_col in df.columns:
        initial_count = len(df)
        df = df[df[group_by_col] != FILTER_OUT_VALUE]
        filtered_count = len(df)
        print(f"Filtered out {initial_count - filtered_count} rows where {group_by_col} == '{FILTER_OUT_VALUE}'")
        print(f"Remaining rows: {filtered_count}")

    # Get all columns except user_id and group_by_col
    exclude_set = {group_by_col, "user_id"} | set(exclude_cols)
    all_plot_cols = [col for col in df.columns if col not in exclude_set]

    # Convert all plot columns to numeric (coerce errors to NaN)
    numeric_cols = []
    for col in all_plot_cols:
        # Try to convert to numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Check if column has any valid numeric values
        if df[col].notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No numeric columns found to plot")

    print(f"\nDetected {len(numeric_cols)} numeric columns to plot (out of {len(all_plot_cols)} total columns):")
    for col in numeric_cols:
        print(f"  - {col}")

    return df, numeric_cols


def check_all_values_same_within_groups(df: pd.DataFrame, group_by_col: str, plot_col: str) -> bool:
    """
    Check if all values within each group are the same.

    Args:
        df: DataFrame with data
        group_by_col: Column name to group by
        plot_col: Column name to check

    Returns:
        True if all values within each group are the same, False otherwise
    """
    grouped = df.groupby(group_by_col)[plot_col]
    # Check if each group has only one unique value (excluding NaN)
    for group_name, group_data in grouped:
        unique_values = group_data.dropna().unique()
        if len(unique_values) > 1:
            return False
    return True


def create_barplot(
    df: pd.DataFrame,
    group_by_col: str,
    plot_col: str,
    aggregation: str = "mean",
    output_dir: Optional[str] = None,
    fig_size: tuple = (12, 6),
    group_order: Optional[List[str]] = None,
    color_map: Optional[dict] = None,
):
    """
    Create a barplot for a single column, grouped by another column.

    Args:
        df: DataFrame with data
        group_by_col: Column name to group by
        plot_col: Column name to plot
        aggregation: Aggregation function ("sum", "mean", "count", "median", etc.)
        output_dir: Directory to save plot (None = just display)
        fig_size: Figure size tuple
        group_order: Consistent order for groups across all plots
        color_map: Consistent color mapping for groups across all plots
    """
    # Aggregate data
    if aggregation == "count":
        grouped_data = df.groupby(group_by_col).size().reset_index(name=plot_col)
    else:
        grouped_data = df.groupby(group_by_col)[plot_col].agg(aggregation).reset_index()

    # Use consistent group order if provided
    if group_order is not None:
        # Reorder grouped_data to match group_order
        grouped_data = grouped_data.set_index(group_by_col).reindex(group_order).reset_index()
        grouped_data = grouped_data.dropna()  # Remove groups that don't exist in this data

    # Create the plot
    plt.figure(figsize=fig_size)

    # Use consistent color mapping if provided
    if color_map is not None and group_order is not None:
        palette = [color_map.get(group, "gray") for group in grouped_data[group_by_col]]
    else:
        palette = sns.color_palette("viridis", len(grouped_data))

    sns.barplot(
        x=group_by_col,
        y=plot_col,
        data=grouped_data,
        palette=palette,
        order=group_order if group_order is not None else None,
    )
    plt.title(f"{plot_col} by {group_by_col} ({aggregation})", fontsize=14, fontweight="bold")
    plt.xlabel(group_by_col, fontsize=12)
    plt.ylabel(plot_col, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = plot_col.replace(" ", "_").replace("/", "_")
        safe_group_name = group_by_col.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"barplot_{safe_col_name}_by_{safe_group_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def bootstrap_statistics(data: pd.Series, n_bootstrap: int = 500) -> dict:
    """
    Calculate bootstrap statistics (mean, SE, percentiles) from resampling.

    Args:
        data: Series of values to bootstrap
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with 'mean', 'se', 'p5', 'p95'
    """
    # Remove NaN values
    data_clean = data.dropna()
    if len(data_clean) == 0:
        return {"mean": np.nan, "se": np.nan, "p5": np.nan, "p95": np.nan}

    # Perform bootstrap resampling
    bootstrap_means = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        # Resample with replacement
        sample = data_clean.sample(n=len(data_clean), replace=True)
        bootstrap_means[i] = sample.mean()

    # Calculate statistics
    mean = np.mean(bootstrap_means)
    se = np.std(bootstrap_means)  # Standard error from bootstrap
    p5 = np.percentile(bootstrap_means, 5)
    p95 = np.percentile(bootstrap_means, 95)

    return {"mean": mean, "se": se, "p5": p5, "p95": p95}


def create_barplot_with_error(
    df: pd.DataFrame,
    group_by_col: str,
    plot_col: str,
    aggregation: str = "mean",
    output_dir: Optional[str] = None,
    fig_size: tuple = (10, 5),
    group_order: Optional[List[str]] = None,
    color_map: Optional[dict] = None,
    n_bootstrap: int = 500,
):
    """
    Create a barplot with bootstrap-based error bars (SE and 5%/95% percentiles).

    Args:
        df: DataFrame with data
        group_by_col: Column name to group by
        plot_col: Column name to plot
        aggregation: Aggregation function for the mean (not used, kept for compatibility)
        output_dir: Directory to save plot (None = just display)
        fig_size: Figure size tuple
        group_order: Consistent order for groups across all plots
        color_map: Consistent color mapping for groups across all plots
        n_bootstrap: Number of bootstrap iterations (default: 500)
    """
    # Calculate bootstrap statistics for each group
    grouped = df.groupby(group_by_col)[plot_col]
    bootstrap_stats = []

    for group_name, group_data in grouped:
        stats = bootstrap_statistics(group_data, n_bootstrap=n_bootstrap)
        stats[group_by_col] = group_name
        bootstrap_stats.append(stats)

    grouped_stats = pd.DataFrame(bootstrap_stats)

    # Use consistent group order if provided
    if group_order is not None:
        # Reorder grouped_stats to match group_order
        grouped_stats = grouped_stats.set_index(group_by_col).reindex(group_order).reset_index()
        grouped_stats = grouped_stats.dropna()  # Remove groups that don't exist in this data

    # Calculate error bar ranges (from 5th to 95th percentile)
    # Error bars will show distance from mean to percentiles
    grouped_stats["lower_error"] = grouped_stats["mean"] - grouped_stats["p5"]
    grouped_stats["upper_error"] = grouped_stats["p95"] - grouped_stats["mean"]

    # Create the plot
    plt.figure(figsize=fig_size)

    # Use consistent color mapping if provided
    if color_map is not None and group_order is not None:
        colors = [color_map.get(group, "gray") for group in grouped_stats[group_by_col]]
    else:
        colors = sns.color_palette("viridis", len(grouped_stats))

    # Create barplot with error bars using numeric positions for consistent ordering
    x_positions = range(len(grouped_stats))

    # Use asymmetric error bars for percentile-based confidence intervals
    bars = plt.bar(
        x_positions,
        grouped_stats["mean"],
        yerr=[grouped_stats["lower_error"], grouped_stats["upper_error"]],
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.title(f"{plot_col} by {group_by_col} (Mean, Bootstrap SE, 5%-95% CI)", fontsize=14, fontweight="bold")
    plt.xlabel(group_by_col, fontsize=12)
    plt.ylabel(plot_col, fontsize=12)
    # Set x-axis labels in consistent order
    plt.xticks(x_positions, grouped_stats[group_by_col], rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = plot_col.replace(" ", "_").replace("/", "_")
        safe_group_name = group_by_col.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"barplot_{safe_col_name}_by_{safe_group_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function to run the plotting script."""
    # Load and validate data
    try:
        df, valid_plot_cols = load_and_validate_data(CSV_FILE_PATH, GROUP_BY_COLUMN, EXCLUDE_COLUMNS)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    if not valid_plot_cols:
        print("Error: No valid numeric columns to plot")
        return

    print(f"\nGrouping by: {GROUP_BY_COLUMN}")
    print(f"Number of unique groups: {df[GROUP_BY_COLUMN].nunique()}")

    # Define consistent group order (alphabetical) and color mapping
    unique_groups = sorted(df[GROUP_BY_COLUMN].unique())
    print(f"Groups (in consistent order): {unique_groups}\n")

    # Create consistent color mapping for all groups
    colors = sns.color_palette("viridis", len(unique_groups))
    color_map = dict(zip(unique_groups, colors))

    # Create plots for each column
    barplot_count = 0
    barplot_error_count = 0

    for col in valid_plot_cols:
        # Check if all values within each group are the same
        all_same = check_all_values_same_within_groups(df, GROUP_BY_COLUMN, col)

        if all_same:
            print(f"Creating barplot for: {col} (all values same within groups)")
            create_barplot(
                df=df,
                group_by_col=GROUP_BY_COLUMN,
                plot_col=col,
                aggregation=AGGREGATION,
                output_dir=OUTPUT_DIR,
                fig_size=FIG_SIZE,
                group_order=unique_groups,
                color_map=color_map,
            )
            barplot_count += 1
        else:
            print(f"Creating barplot with bootstrap error bars for: {col} (values vary within groups, n_bootstrap=500)")
            create_barplot_with_error(
                df=df,
                group_by_col=GROUP_BY_COLUMN,
                plot_col=col,
                aggregation=AGGREGATION,
                output_dir=OUTPUT_DIR,
                fig_size=FIG_SIZE,
                group_order=unique_groups,
                color_map=color_map,
                n_bootstrap=500,
            )
            barplot_error_count += 1

    print(f"\nAll plots completed!")
    print(f"  Simple barplots created: {barplot_count}")
    print(f"  Barplots with error bars created: {barplot_error_count}")


if __name__ == "__main__":
    main()
