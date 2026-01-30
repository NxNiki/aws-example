"""
Script to load CSV file and create multi-group barplots (clustered barplots) with standard error using seaborn's built-in error bars.

Usage:
    python plot_all_user_stats.py

The script will:
    - Automatically detect all numeric columns (excluding group_by_column and user_id)
    - Filter out rows where ai_group == "newBee"
    - Create grouped barplots (first group: x axis, second or combined as hue/legend)
    - For group_by_column of >2, combine all but the first as legend label using ':'-join
    - Use seaborn's standard error (SE) error bars on the barplots
"""

import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURATION - Modify these variables for your use case
# ============================================================================

CSV_FILE_PATH = "/Users/niuxin/Documents/aws-example/jobs/output_ss01_wucaishen/all_user_stats.parquet"
GROUP_BY_COLUMN = ["ai_group", "math_policy"]
FILTER_OUT_VALUE = None  # "newBee"
EXCLUDE_COLUMNS: List[str] = []
OUTPUT_DIR: Optional[str] = "/Users/niuxin/Documents/aws-example/jobs/output_ss01_wucaishen/all_user_stats_plot"
AGGREGATION = "mean"
FIG_SIZE = (10, 5)

# ============================================================================


def _ensure_group_is_list(group_by_col: Union[str, List[str]]):
    if isinstance(group_by_col, str):
        return [group_by_col]
    return group_by_col


def load_and_validate_data(
    file_path: str, group_by_col: Union[str, List[str]], exclude_cols: List[str]
) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".parquet":
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows from parquet file: {file_path}")
    elif file_ext == ".csv":
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from CSV file: {file_path}")
    else:
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

    group_by_cols = _ensure_group_is_list(group_by_col)
    for col in group_by_cols:
        if col not in df.columns:
            raise ValueError(f"Group by column '{col}' not found in file.\nAvailable columns: {list(df.columns)}")

    filter_col = group_by_cols[0]
    if FILTER_OUT_VALUE is not None and filter_col in df.columns:
        initial_count = len(df)
        df = df[df[filter_col] != FILTER_OUT_VALUE]
        filtered_count = len(df)
        print(f"Filtered out {initial_count - filtered_count} rows where {filter_col} == '{FILTER_OUT_VALUE}'")
        print(f"Remaining rows: {filtered_count}")

    exclude_set = set(group_by_cols) | {"user_id"} | set(exclude_cols)
    all_plot_cols = [col for col in df.columns if col not in exclude_set]

    numeric_cols = []
    for col in all_plot_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No numeric columns found to plot")

    print(f"\nDetected {len(numeric_cols)} numeric columns to plot (out of {len(all_plot_cols)} total columns):")
    for col in numeric_cols:
        print(f"  - {col}")

    return df, numeric_cols


def check_all_values_same_within_groups(df: pd.DataFrame, group_by_col: Union[str, List[str]], plot_col: str) -> bool:
    """Returns True if all values within each group are the same, else False."""
    group_by_cols = _ensure_group_is_list(group_by_col)
    grouped = df.groupby(group_by_cols)[plot_col]
    for _, group_data in grouped:
        unique_values = group_data.dropna().unique()
        if len(unique_values) > 1:
            return False
    return True


def seaborn_multi_group_barplot(
    df: pd.DataFrame,
    group_by_cols: List[str],
    plot_col: str,
    output_dir: Optional[str],
    fig_size: tuple,
    legend_palette: Optional[dict] = None,
):
    """
    Shows/saves a grouped barplot using seaborn, using SE error bars, handling >2 group_by_cols by combining those after the first into a legend label.
    """
    group_cols = _ensure_group_is_list(group_by_cols)
    plt.figure(figsize=fig_size)
    hue = None

    # Build hue column if needed (for multi-group plotting)
    if len(group_cols) > 1:
        if len(group_cols) == 2:
            hue = group_cols[1]
            hue_legend = group_cols[1]
            df_to_plot = df
        else:
            # Combine all except the first for legend
            legend_col = "_legend"
            df_to_plot = df.copy()
            df_to_plot[legend_col] = df_to_plot[group_cols[1:]].astype(str).agg(":".join, axis=1)
            hue = legend_col
            hue_legend = ":".join(group_cols[1:])
    else:
        df_to_plot = df
        hue = None
        hue_legend = None

    order = sorted(df_to_plot[group_cols[0]].dropna().unique())  # clean up x order
    hue_order = sorted(df_to_plot[hue].dropna().unique()) if hue is not None else None

    ax = sns.barplot(
        x=group_cols[0],
        y=plot_col,
        hue=hue,
        data=df_to_plot,
        order=order,
        hue_order=hue_order,
        palette=legend_palette,
        estimator="mean",
        errorbar="se",
        capsize=0.08,
        edgecolor="black",
    )

    title_str = f"{plot_col} by {group_cols[0]}"
    if hue_legend:
        ax.legend(title=hue_legend, bbox_to_anchor=(1.01, 1), loc="upper left")
        title_str += f" + {hue_legend}"
    else:
        ax.legend_.remove() if ax.legend_ else None

    plt.title(title_str, fontsize=14, fontweight="bold")
    plt.xlabel(group_cols[0], fontsize=12)
    plt.ylabel(plot_col, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = plot_col.replace(" ", "_").replace("/", "_")
        safe_group_name = "_".join(group_cols).replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"barplot_{safe_col_name}_by_{safe_group_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    try:
        df, valid_plot_cols = load_and_validate_data(CSV_FILE_PATH, GROUP_BY_COLUMN, EXCLUDE_COLUMNS)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    if not valid_plot_cols:
        print("Error: No valid numeric columns to plot")
        return

    group_by_cols = _ensure_group_is_list(GROUP_BY_COLUMN)
    print(f"\nGrouping by: {group_by_cols}")

    # Build legend palette for consistent coloring
    legend_palette = None
    if len(group_by_cols) > 1:
        if len(group_by_cols) == 2:
            legend_labels = sorted(df[group_by_cols[1]].dropna().unique())
        else:
            legend_labels = sorted(df[group_by_cols[1:]].astype(str).agg(":".join, axis=1).unique())
        legend_palette = dict(zip(legend_labels, sns.color_palette("viridis", len(legend_labels))))

    plot_count = 0

    for col in valid_plot_cols:
        print(f"Creating grouped barplot for: {col} (with SE error bars via Seaborn)")
        seaborn_multi_group_barplot(
            df=df,
            group_by_cols=group_by_cols,
            plot_col=col,
            output_dir=OUTPUT_DIR,
            fig_size=FIG_SIZE,
            legend_palette=legend_palette,
        )
        plot_count += 1

    print(f"\nAll plots completed!")
    print(f"  Grouped barplots created: {plot_count}")


if __name__ == "__main__":
    main()
