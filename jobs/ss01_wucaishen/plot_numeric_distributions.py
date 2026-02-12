"""
Plot distributions of all numeric columns in ss01_features_grouped.parquet.

Usage:
    python plot_numeric_distributions.py

Output:
    Figures saved under jobs/output_ss01_wucaishen/distribution_plots/
    - overview_*.png: grid of histograms (multiple pages if many columns)
    - Optional: individual column plots in a subfolder if desired
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
JOBS_DIR = Path(__file__).resolve().parent.parent
FEATURES_FILE = JOBS_DIR / "output_ss01_wucaishen" / "ss01_features_grouped.parquet"
OUTPUT_DIR = JOBS_DIR / "output_ss01_wucaishen" / "distribution_plots"

# Columns to exclude from distribution plots (IDs / categorical codes)
EXCLUDE_COLUMNS = {"user_id", "session_group", "agg_group"}

# Layout
COLS_PER_PAGE = 4
ROWS_PER_PAGE = 5
SUBPLOTS_PER_PAGE = COLS_PER_PAGE * ROWS_PER_PAGE
FIG_SIZE_PER_SUB = (4, 3)
HIST_BINS = 50
DPI = 120


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names, excluding EXCLUDE_COLUMNS."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in EXCLUDE_COLUMNS]


def plot_distribution(ax, series: pd.Series, name: str) -> None:
    """Draw histogram (+ KDE when meaningful) for one column."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(name, fontsize=9)
        return

    try:
        sns.histplot(
            clean, ax=ax, bins=HIST_BINS, kde=True, stat="density", color="steelblue", edgecolor="white", linewidth=0.3
        )
    except Exception:
        sns.histplot(
            clean,
            ax=ax,
            bins=min(HIST_BINS, max(10, len(clean) // 5)),
            stat="density",
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_title(name, fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    df = pd.read_parquet(FEATURES_FILE)
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        raise ValueError("No numeric columns found to plot.")

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    print(f"Plotting distributions for {len(numeric_cols)} numeric columns.")
    print(f"Output directory: {OUTPUT_DIR}")

    n_cols = len(numeric_cols)
    page = 0
    for start in range(0, n_cols, SUBPLOTS_PER_PAGE):
        page += 1
        end = min(start + SUBPLOTS_PER_PAGE, n_cols)
        cols_this_page = numeric_cols[start:end]
        n_plots = len(cols_this_page)
        n_rows = (n_plots + COLS_PER_PAGE - 1) // COLS_PER_PAGE

        fig, axes = plt.subplots(
            n_rows,
            COLS_PER_PAGE,
            figsize=(COLS_PER_PAGE * FIG_SIZE_PER_SUB[0], n_rows * FIG_SIZE_PER_SUB[1]),
            squeeze=False,
        )
        axes_flat = axes.flat

        for i, col in enumerate(cols_this_page):
            ax = axes_flat[i]
            plot_distribution(ax, df[col], col)

        for j in range(n_plots, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"ss01_features_grouped — numeric distributions (page {page})", fontsize=12, y=1.01)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"overview_page_{page}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
