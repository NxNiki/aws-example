"""
Plot distributions of all numeric columns in the ss01 features_grouped dataset.

Reads the ss01 features_grouped partitioned parquet dataset from S3 (written by
jobs/etl/redshift/ss01_wucaishen/etl_feature_engineer.py via ETLScheduler). Uses DataProfiler
and DataVisualizer from bituslabs_ds.eda. Numeric preparation (including Parquet
fixed_len_byte_array conversion), skewness, kurtosis, and log y-scale for highly
skewed features are handled by the EDA module.

Usage:
    python plot_numeric_distributions.py

Output:
    Figures saved under jobs/output_ss01_wucaishen/distribution_plots/
    - overview_page_*.png
"""

from pathlib import Path

import awswrangler as wr

from bituslabs_ds.eda import DataProfiler, DataVisualizer

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
JOBS_DIR = Path(__file__).resolve().parent.parent
FEATURES_DATASET_URI = (
    "s3://bituslabs-team-ai/etl-results/jobs/output_ss01_feature_engineer/features_grouped_binsize_40/"
)
OUTPUT_DIR = JOBS_DIR / "output_ss01_wucaishen" / "distribution_plots"

EXCLUDE_COLUMNS = {"user_id", "session_group", "agg_group"}

COLS_PER_PAGE = 4
ROWS_PER_PAGE = 5
HIST_BINS = 50
DPI = 120
SKEWNESS_LOG_SCALE_THRESHOLD = 2.0
FIG_TITLE_PREFIX = "ss01_features_grouped"


def main() -> None:
    df = wr.s3.read_parquet(path=FEATURES_DATASET_URI, dataset=True)
    numeric_cols, plot_df = DataProfiler.prepare_numeric_df(df, exclude_columns=EXCLUDE_COLUMNS)
    if not numeric_cols:
        raise ValueError("No numeric columns found to plot.")
    numeric_cols = sorted(numeric_cols)

    profiler = DataProfiler(plot_df)
    viz = DataVisualizer(profiler)

    log_scale_cols = profiler.get_columns_for_log_scale(threshold=SKEWNESS_LOG_SCALE_THRESHOLD)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    print(f"Plotting distributions for {len(numeric_cols)} numeric columns.")
    print(f"Log y-scale applied to {len(log_scale_cols)} columns (|skew| >= {SKEWNESS_LOG_SCALE_THRESHOLD}).")
    print(f"Output directory: {OUTPUT_DIR}")

    saved = viz.plot_numeric_distribution_pages(
        output_dir=OUTPUT_DIR,
        layout_cols=numeric_cols,
        cols_per_page=COLS_PER_PAGE,
        rows_per_page=ROWS_PER_PAGE,
        bins=HIST_BINS,
        skewness_log_scale_threshold=SKEWNESS_LOG_SCALE_THRESHOLD,
        dpi=DPI,
        fig_title_prefix=FIG_TITLE_PREFIX,
        show_distribution_stats_legend=False,
        overlay_power_transform=True,
    )

    for path in saved:
        print(f"Saved {path}")
    print("Done.")


if __name__ == "__main__":
    main()
