"""
Script to load CSV file and create multi-group barplots (clustered barplots) and boxplots using seaborn, with 500 bootstrap error bars, and save them. 
Additionally, for each x tick (level of the x group), it calculates and displays the percentage change of the metrics between the first and second level 
of the group variable (hue/legend) above the bars and boxplot boxes.

Usage:
    python plot_all_user_stats.py

Features:
    - Automatically detects all numeric columns except those excluded (group_by_column and user_id)
    - Filters out rows where ai_group is in FILTER_OUT_VALUE
    - For grouping, supports >2 columns by combining all but the first as legend (":"-joined)
    - Barplot uses seaborn with bootstrap error bars (n_boot=500, ci=68%)
    - Percentage change between the first and second level of the hue (legend) variable is calculated and shown above each bar/box-group
    - Boxplots for each numeric column and grouping
    - Barplots and boxplots saved in different subfolders
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt

# Resolve project root for portable paths
try:
    from bituslabs_ds.config import LOCAL_ROOT
except ImportError:
    LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURATION - Modify these variables for your use case
# ============================================================================

CSV_FILE_PATH = str(Path(LOCAL_ROOT) / "jobs" / "output_ss01_wucaishen" / "all_user_stats.parquet")
GROUP_BY_COLUMN = ["ai_group", "math_policy"]
FILTER_OUT_VALUE = ["rollerCoaster", "dropTower"]
EXCLUDE_COLUMNS: List[str] = []
OUTPUT_DIR: Optional[str] = str(Path(LOCAL_ROOT) / "jobs" / "output_ss01_wucaishen" / "all_user_stats_plot")
BARPLOT_SUBDIR = "barplots"
BOXPLOT_SUBDIR = "boxplots"
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
    # Modified filtering to support list of ai_groups for removal
    if FILTER_OUT_VALUE is not None and filter_col in df.columns:
        initial_count = len(df)
        df = df[~df[filter_col].isin(FILTER_OUT_VALUE)]
        filtered_count = len(df)
        print(f"Filtered out {initial_count - filtered_count} rows where {filter_col} in {FILTER_OUT_VALUE}")
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


def annotate_pct_change(ax, bar_coords, pct_changes, is_boxplot=False):
    """
    Annotates each group (x) with the corresponding percentage change string, at the tallest bar/box position + offset.
    """
    for xc, pct in zip(bar_coords, pct_changes):
        if pct is not None:
            text = f"{pct:+.1f}%"
            y = bar_coords[xc] if not is_boxplot else bar_coords[xc]
            ax.text(
                xc,
                y,
                text,
                color="darkred",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
            )


def compute_pct_changes_means(df, group_by_cols, plot_col):
    """
    For each x-group (first group col),
    computes the mean metric for each hue (legend), sorted by hue_order, and returns percent change from first to second.
    Returns:
        pct_changes: list (float or None) per x category (NaN if not enough data)
        mean_by_group: dict of dict, mean_by_group[x][hue] -> mean value
    """
    x_col = group_by_cols[0]
    hue_col = group_by_cols[1]
    means = df.groupby([x_col, hue_col])[plot_col].mean().unstack(hue_col)
    cols = means.columns.tolist()
    pct_changes = []
    for idx, row in means.iterrows():
        if len(cols) >= 2 and pd.notna(row[cols[0]]) and pd.notna(row[cols[1]]) and row[cols[0]] != 0:
            pct = (row[cols[1]] - row[cols[0]]) / abs(row[cols[0]]) * 100.0
            pct_changes.append(pct)
        else:
            pct_changes.append(None)
    # Means for annotation positioning (max y of bars)
    max_means = means.max(axis=1)
    return pct_changes, means.index.tolist(), max_means


def seaborn_multi_group_barplot(
    df: pd.DataFrame,
    group_by_cols: List[str],
    plot_col: str,
    output_dir: Optional[str],
    fig_size: tuple,
    legend_palette: Optional[dict] = None,
    n_boot: int = 500,
):
    """
    Shows/saves a grouped barplot using seaborn, using 500 bootstrap error bars, handling >2 group_by_cols by combining those after the first into a legend label.
    Annotates each x-group with percentage change from first to second legend (hue).
    """
    group_cols = _ensure_group_is_list(group_by_cols)
    plt.figure(figsize=fig_size)
    hue = None

    if len(group_cols) <= 1:
        # Skipping annotation for <2 group groups
        hue_legend = None
        hue = None
        df_to_plot = df
    elif len(group_cols) == 2:
        hue = group_cols[1]
        hue_legend = group_cols[1]
        df_to_plot = df
    else:
        legend_col = "_legend"
        df_to_plot = df.copy()
        df_to_plot[legend_col] = df_to_plot[group_cols[1:]].astype(str).agg(":".join, axis=1)
        hue = legend_col
        hue_legend = ":".join(group_cols[1:])

    order = sorted(df_to_plot[group_cols[0]].dropna().unique())
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
        ci=95,
        n_boot=n_boot,
        capsize=0.08,
        edgecolor="black",
        errorbar=None,
    )

    # ----------- Annotate percentage changes ------------
    if len(group_cols) >= 2:
        pct_changes, idx, max_y = compute_pct_changes_means(df_to_plot, group_cols, plot_col)
        xtick_locs = []
        # Map xtick labels in order (possibly with categorical codes, so map from tick label to bar midpoint)
        for xi, x in enumerate(order):
            xtick_locs.append(ax.get_xticks()[xi])
        # Place annotation above the tallest bar for each x
        ylocs = []
        for xi, x in enumerate(order):
            # Find highest bar y for group x (but may have multiple hues)
            bars_y = []
            for child in ax.patches:
                if isinstance(child, plt.Rectangle):
                    if abs(child.get_x() + child.get_width() / 2 - xtick_locs[xi]) < child.get_width():
                        bars_y.append(child.get_height())
            if bars_y:
                ylocs.append(max(bars_y) * 1.07)
            else:
                ylocs.append(max_y[x] * 1.07 if x in max_y else max_y.max() * 1.07)
        # Annotate with percent change (%)
        for xc, y, pct in zip(xtick_locs, ylocs, pct_changes):
            if pct is not None:
                text = f"{pct:+.1f}%"
                ax.text(
                    xc,
                    y,
                    text,
                    color="darkred",
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
                )

    title_str = f"{plot_col} by {group_cols[0]}"
    if hue_legend:
        ax.legend(title=hue_legend, bbox_to_anchor=(1.01, 1), loc="upper left")
        title_str += f" + {hue_legend}"
    else:
        ax.legend_.remove() if ax.legend_ else None

    plt.title(f"{title_str} (Bootstrap error & %Δ)", fontsize=14, fontweight="bold")
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


def compute_pct_changes_boxplot(df, group_by_cols, plot_col):
    """
    For each x-group (first group col), computes the median per hue for the plot_col, returns percent change from first to second legend (hue).
    Returns pct_changes (float/None per x) and max_y (for position of box).
    """
    x_col = group_by_cols[0]
    hue_col = group_by_cols[1]
    medians = df.groupby([x_col, hue_col])[plot_col].median().unstack(hue_col)
    cols = medians.columns.tolist()
    pct_changes = []
    for idx, row in medians.iterrows():
        if len(cols) >= 2 and pd.notna(row[cols[0]]) and pd.notna(row[cols[1]]) and row[cols[0]] != 0:
            pct = (row[cols[1]] - row[cols[0]]) / abs(row[cols[0]]) * 100.0
            pct_changes.append(pct)
        else:
            pct_changes.append(None)
    # Max for annotation Y placement
    max_medians = medians.max(axis=1)
    return pct_changes, medians.index.tolist(), max_medians


def seaborn_multi_group_boxplot(
    df: pd.DataFrame,
    group_by_cols: List[str],
    plot_col: str,
    output_dir: Optional[str],
    fig_size: tuple,
    legend_palette: Optional[dict] = None,
):
    """
    Shows/saves a grouped boxplot using seaborn, handling >2 group_by_cols by combining those after the first into a legend label.
    Annotates each x-group with percentage change from first to second legend (hue).
    """
    group_cols = _ensure_group_is_list(group_by_cols)
    plt.figure(figsize=fig_size)
    hue = None

    if len(group_cols) <= 1:
        # Skipping annotation for <2 group groups
        hue_legend = None
        hue = None
        df_to_plot = df
    elif len(group_cols) == 2:
        hue = group_cols[1]
        hue_legend = group_cols[1]
        df_to_plot = df
    else:
        legend_col = "_legend"
        df_to_plot = df.copy()
        df_to_plot[legend_col] = df_to_plot[group_cols[1:]].astype(str).agg(":".join, axis=1)
        hue = legend_col
        hue_legend = ":".join(group_cols[1:])

    order = sorted(df_to_plot[group_cols[0]].dropna().unique())
    hue_order = sorted(df_to_plot[hue].dropna().unique()) if hue is not None else None

    ax = sns.boxplot(
        x=group_cols[0],
        y=plot_col,
        hue=hue,
        data=df_to_plot,
        order=order,
        hue_order=hue_order,
        palette=legend_palette,
        showfliers=False,
    )

    # ----------- Annotate percentage changes (based on the group medians) ------------
    if len(group_cols) >= 2:
        pct_changes, idx, max_y = compute_pct_changes_boxplot(df_to_plot, group_cols, plot_col)
        xtick_locs = []
        for xi, x in enumerate(order):
            xtick_locs.append(ax.get_xticks()[xi])
        ylocs = []
        for xi, x in enumerate(order):
            # get the maximum box top for this group
            boxes_y = []
            for artist in ax.artists:
                box_x = artist.get_x() + artist.get_width() / 2
                if abs(box_x - xtick_locs[xi]) < artist.get_width():
                    boxes_y.append(artist.get_ydata().max())
            if boxes_y:
                ylocs.append(max(boxes_y) * 1.07)
            else:
                ylocs.append(max_y[x] * 1.07 if x in max_y else max_y.max() * 1.07)
        for xc, y, pct in zip(xtick_locs, ylocs, pct_changes):
            if pct is not None:
                text = f"{pct:+.1f}%"
                ax.text(
                    xc,
                    y,
                    text,
                    color="darkred",
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
                )

    title_str = f"{plot_col} by {group_cols[0]}"
    if hue_legend:
        ax.legend(title=hue_legend, bbox_to_anchor=(1.01, 1), loc="upper left")
        title_str += f" + {hue_legend}"
    else:
        ax.legend_.remove() if ax.legend_ else None

    plt.title(f"{title_str} (Boxplot & %Δ)", fontsize=14, fontweight="bold")
    plt.xlabel(group_cols[0], fontsize=12)
    plt.ylabel(plot_col, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_col_name = plot_col.replace(" ", "_").replace("/", "_")
        safe_group_name = "_".join(group_cols).replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"boxplot_{safe_col_name}_by_{safe_group_name}.png")
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

    legend_palette = None
    if len(group_by_cols) > 1:
        if len(group_by_cols) == 2:
            legend_labels = sorted(df[group_by_cols[1]].dropna().unique())
        else:
            legend_labels = sorted(df[group_by_cols[1:]].astype(str).agg(":".join, axis=1).unique())
        legend_palette = dict(zip(legend_labels, sns.color_palette("viridis", len(legend_labels))))

    barplot_output_dir = os.path.join(OUTPUT_DIR, BARPLOT_SUBDIR) if OUTPUT_DIR else None
    boxplot_output_dir = os.path.join(OUTPUT_DIR, BOXPLOT_SUBDIR) if OUTPUT_DIR else None

    plot_count = 0

    for col in valid_plot_cols:
        print(f"Creating grouped barplot for: {col} (500 bootstrap error bars, percentage change annotation)")
        seaborn_multi_group_barplot(
            df=df,
            group_by_cols=group_by_cols,
            plot_col=col,
            output_dir=barplot_output_dir,
            fig_size=FIG_SIZE,
            legend_palette=legend_palette,
            n_boot=500,
        )
        print(f"Creating grouped boxplot for: {col} (percentage change annotation)")
        seaborn_multi_group_boxplot(
            df=df,
            group_by_cols=group_by_cols,
            plot_col=col,
            output_dir=boxplot_output_dir,
            fig_size=FIG_SIZE,
            legend_palette=legend_palette,
        )
        plot_count += 1

    print(f"\nAll plots completed!")
    print(f"  Grouped barplots created: {plot_count}")
    print(f"  Grouped boxplots created: {plot_count}")


if __name__ == "__main__":
    main()
