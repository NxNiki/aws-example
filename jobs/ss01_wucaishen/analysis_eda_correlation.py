"""
EDA: correlation analysis on ss01 daily_stats from S3.

Data source:
  Reads all parquet files from s3://bituslabs-team-ai/etl-results/jobs/output_ss01_wucaishen/daily_stats/.

Data processing (in order):
  1. Drop duplicates on (activity_date, ai_group) to obtain one row per group-day.
  2. Pairwise zero removal (optional): per pair, if col A (or B) has >threshold zeros, drop those rows independently.
  3. Power transform (optional): apply Yeo-Johnson to numeric columns before correlation.

Analysis:
  - Correlation: Pearson or Spearman (configurable via USE_SPEARMAN).
  - Correlation heatmap with annotations; significant pairs (p < alpha) marked with *.
  - Scatter plots with regression line and r/p for pairs with significant correlation.

Outputs:
  - distributions/: histogram pages (original and power-transformed overlaid when POWER_TRANSFORM=True).
  - Correlation heatmap, correlation matrix CSVs, scatter plots.
  - CCA heatmap: regularized CCA (rCCA) X→Y coefficients, * for significant (p < 0.05).
  - Per ai_group: same layout under by_ai_group/<group>/.

Uses bituslabs_ds.s3_utils for S3 reads, bituslabs_ds.eda (DataProfiler, DataVisualizer) for distributions
and heatmap/scatter, bituslabs_ds.utils for power transform.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bituslabs_ds.eda import (
    DataProfiler,
    DataVisualizer,
    correlation_matrix_with_pvalues,
    get_significant_correlation_pairs,
    plot_scatter_pairs,
)
from bituslabs_ds.s3_utils import read_files
from bituslabs_ds.utils import df_power_transform

# CCA: X (predictors) and Y (targets)
CCA_X_VARS = [
    "rtp",
    "rtp_bg",
    "rtp_fg",
    "user_rtp_median",
    "user_rtp_ultilization_ratio",
    "hit_rate",
    "hit_rate_bg",
    "hit_rate_fg",
    "fg_ratio",
    "total_payout_per_user",
    "active_user_no_fg_ratio",
    "active_user_0_rtp_ratio",
    "active_user_rtp_less_0_3_ratio",
    "active_user_rtp_less_0_5_ratio",
]
CCA_Y_VARS = [
    "total_num_bets_per_user",
    "total_bet_per_user",
    "retention_rate_day1",
    "retention_rate_day3",
]

# S3 path for daily_stats parquet files
S3_DAILY_STATS = "s3://bituslabs-team-ai/etl-results/jobs/output_ss01_wucaishen_old/daily_stats/"

# Variables for correlation analysis (retention_* matches ETL; retentation_* if present as variant)
CORRELATION_VARS = [
    "rtp",
    "rtp_bg",
    "rtp_fg",
    "user_rtp_median",
    "user_rtp_ultilization_ratio",
    "hit_rate",
    "hit_rate_bg",
    "hit_rate_fg",
    "fg_ratio",
    "total_payout_per_user",
    "active_user_no_fg_ratio",
    "active_user_0_rtp_ratio",
    "active_user_rtp_less_0_3_ratio",
    "active_user_rtp_less_0_5_ratio",
    "total_num_bets_per_user",
    "total_bet_per_user",
    "retention_rate_day1",
    "retention_rate_day3",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_ss01_eda_correlation"
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold
MIN_CORRELATION_ABS = 0.15  # minimum |r| to consider for scatter plots
POWER_TRANSFORM = True  # apply Yeo-Johnson before correlation
REMOVE_PAIRWISE_ZEROS = True  # drop zeros per pair when computing correlation and scatter plots
ZEROS_REMOVAL_THRESHOLD = 0.3  # remove pairwise zeros only when zero proportion exceeds this (0–1)
USE_SPEARMAN = False  # use Spearman (rank) correlation; more robust to zeros/skew
CCA_REGULARIZATION_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]  # L2 regularization values to tune
CCA_CV_FOLDS = 5


def load_daily_stats(s3_path: str) -> pd.DataFrame:
    """Load all parquet files from S3 path and return concatenated DataFrame."""
    result = read_files(s3_path, lazy_load=False)
    df: pd.DataFrame = result if isinstance(result, pd.DataFrame) else pd.concat(result, ignore_index=True)
    if df.empty:
        raise ValueError(f"No data loaded from {s3_path}")
    return df


def apply_power_transform(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply Yeo-Johnson power transform using utils.df_power_transform. Returns DataFrame with same column names."""
    try:
        transformed = df_power_transform(df[numeric_cols].copy(), col_names=numeric_cols, suffix="_pt")
    except Exception as e:
        print(f"  Power transform failed: {e}. Using original data.")
        return df[numeric_cols].copy()
    pt_cols = [c + "_pt" for c in numeric_cols if c + "_pt" in transformed.columns]
    if not pt_cols:
        print("  No columns power-transformed. Using original data.")
        return df[numeric_cols].copy()
    result = transformed[pt_cols].copy()
    result.columns = [c.replace("_pt", "") for c in result.columns]
    skipped = [c for c in numeric_cols if c not in result.columns]
    if skipped:
        result = pd.concat([result, df[skipped]], axis=1)
    return result


def prepare_for_correlation(df: pd.DataFrame, keep_ai_group: bool = False) -> pd.DataFrame:
    """Aggregate to one row per (activity_date, ai_group) and keep numeric columns for correlation."""
    group_cols = ["activity_date", "ai_group"]
    if "activity_date" not in df.columns or "ai_group" not in df.columns:
        group_cols = [c for c in df.columns if c in group_cols]
    available = [c for c in CORRELATION_VARS if c in df.columns]
    missing = [c for c in CORRELATION_VARS if c not in df.columns]
    if missing:
        print(f"Warning: Columns not found in data (skipped): {missing}")
    if not available:
        raise ValueError(f"None of the correlation variables found. Available columns: {list(df.columns)[:30]}...")
    select_cols = group_cols + available if group_cols else available
    sub = df[select_cols].copy()
    if group_cols:
        sub = sub.drop_duplicates(subset=group_cols).reset_index(drop=True)
    sub[available] = sub[available].astype(float)
    return sub if (keep_ai_group and "ai_group" in sub.columns) else sub[available]


# Distribution plot config (matches plot_numeric_distributions.py)
COLS_PER_PAGE = 4
ROWS_PER_PAGE = 5
HIST_BINS = 50
DIST_DPI = 120
SKEWNESS_LOG_SCALE_THRESHOLD = 2.0


def plot_distributions_via_visualizer(
    df: pd.DataFrame,
    output_dir: Path,
    fig_title_prefix: str,
    overlay_power_transform: bool = False,
) -> list:
    """Use DataProfiler and DataVisualizer to plot numeric distributions (reuses plot_numeric_distributions logic)."""
    numeric_cols, plot_df = DataProfiler.prepare_numeric_df(df, exclude_columns=set())
    if not numeric_cols:
        return []
    numeric_cols = sorted(numeric_cols)
    profiler = DataProfiler(plot_df)
    viz = DataVisualizer(profiler)
    saved = viz.plot_numeric_distribution_pages(
        output_dir=output_dir,
        layout_cols=numeric_cols,
        cols_per_page=COLS_PER_PAGE,
        rows_per_page=ROWS_PER_PAGE,
        bins=HIST_BINS,
        skewness_log_scale_threshold=SKEWNESS_LOG_SCALE_THRESHOLD,
        dpi=DIST_DPI,
        fig_title_prefix=fig_title_prefix,
        show_distribution_stats_legend=False,
        overlay_power_transform=overlay_power_transform,
    )
    return saved


def run_correlation_analysis(
    df_numeric: pd.DataFrame, output_dir: Path, title_suffix: str = "", scatter_subdir: str = ""
) -> None:
    """Run full correlation pipeline: distribution plots, pairwise zeros (optional), power transform (optional), heatmap, scatter plots."""
    if df_numeric.empty or len(df_numeric) < 3:
        print(f"  Skipping (insufficient rows: {len(df_numeric)})")
        return
    cols = [c for c in df_numeric.columns if pd.api.types.is_numeric_dtype(df_numeric[c])]
    df_raw = df_numeric[cols].copy()

    dist_dir = output_dir / "distributions"
    prefix = "original_and_power_transformed" if POWER_TRANSFORM else "original"
    saved = plot_distributions_via_visualizer(
        df_raw, dist_dir, fig_title_prefix=prefix, overlay_power_transform=POWER_TRANSFORM
    )
    for p in saved:
        print(f"  Saved {p}")

    if not REMOVE_PAIRWISE_ZEROS:
        df_raw = None
    if POWER_TRANSFORM and cols:
        df_numeric = apply_power_transform(df_numeric, cols)
    method = "spearman" if USE_SPEARMAN else "pearson"
    corr_df, pval_df = correlation_matrix_with_pvalues(
        df_numeric,
        method=method,
        remove_pairwise_zeros=REMOVE_PAIRWISE_ZEROS,
        zeros_removal_threshold=ZEROS_REMOVAL_THRESHOLD,
        df_for_zero_check=df_raw,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(output_dir / "correlation_matrix.csv")
    pval_df.to_csv(output_dir / "correlation_pvalues.csv")

    method_label = " (Spearman)" if USE_SPEARMAN else ""
    zeros_label = " (pairwise zeros removed)" if REMOVE_PAIRWISE_ZEROS else ""
    pt_label = " (power-transformed)" if POWER_TRANSFORM else ""
    title = f"Correlation matrix{method_label}{zeros_label}{pt_label} {title_suffix}".strip()
    plot_heatmap(
        df_numeric,
        output_dir,
        title=title,
        method=method,
        pval_df=pval_df,
        significance_level=SIGNIFICANCE_LEVEL,
    )

    pairs = get_significant_correlation_pairs(
        corr_df,
        pval_df,
        significance_level=SIGNIFICANCE_LEVEL,
        min_abs_corr=MIN_CORRELATION_ABS,
    )
    print(f"  Found {len(pairs)} significant pairs (p < {SIGNIFICANCE_LEVEL}, |r| >= {MIN_CORRELATION_ABS})")
    for c1, c2, r, p in pairs:
        print(f"    {c1} vs {c2}: r = {r:.3f}, p = {p:.2e}")

    scatter_dir = (output_dir / scatter_subdir) if scatter_subdir else output_dir
    if pairs:
        scatter_dir.mkdir(parents=True, exist_ok=True)
        plot_scatter_pairs(
            df_numeric,
            correlation_pairs=pairs,
            add_regression_line=True,
            add_correlation_annotation=True,
            significance_level=SIGNIFICANCE_LEVEL,
            save_path=scatter_dir / "scatter_significant_pairs.png",
            alpha=0.5,
            s=20,
            remove_pairwise_zeros=REMOVE_PAIRWISE_ZEROS,
            zeros_eps=1e-10,
            zeros_removal_threshold=ZEROS_REMOVAL_THRESHOLD,
            df_for_zero_check=df_raw,
        )
    else:
        print("  No significant pairs for scatter plots.")

    # Scatter plot for significant X–Y pairs only (from correlation analysis, CCA var subsets)
    x_vars = [c for c in CCA_X_VARS if c in corr_df.columns]
    y_vars = [c for c in CCA_Y_VARS if c in corr_df.columns]
    xy_pairs = []
    for xi in x_vars:
        for yj in y_vars:
            r = corr_df.loc[xi, yj] if xi in corr_df.index and yj in corr_df.columns else np.nan
            p = pval_df.loc[xi, yj] if xi in pval_df.index and yj in pval_df.columns else np.nan
            if np.isfinite(r) and np.isfinite(p) and p < SIGNIFICANCE_LEVEL and abs(r) >= MIN_CORRELATION_ABS:
                xy_pairs.append((xi, yj, float(r), float(p)))
    if xy_pairs:
        scatter_dir.mkdir(parents=True, exist_ok=True)
        plot_scatter_pairs(
            df_numeric,
            correlation_pairs=xy_pairs,
            add_regression_line=True,
            add_correlation_annotation=True,
            significance_level=SIGNIFICANCE_LEVEL,
            save_path=scatter_dir / "scatter_significant_x_y_pairs.png",
            alpha=0.5,
            s=20,
            remove_pairwise_zeros=REMOVE_PAIRWISE_ZEROS,
            zeros_eps=1e-10,
            zeros_removal_threshold=ZEROS_REMOVAL_THRESHOLD,
            df_for_zero_check=df_raw,
        )

    run_cca_analysis(df_numeric, output_dir, title_suffix=title_suffix, df_for_zero_check=df_raw)


def plot_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    title: str = "Correlation matrix",
    method: str = "pearson",
    pval_df: pd.DataFrame | None = None,
    significance_level: float = 0.05,
) -> None:
    """Create and save correlation heatmap using eda.DataVisualizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(df.columns)
    figsize = (max(14, n * 1.0), max(12, n * 0.9))
    fig, ax = plt.subplots(figsize=figsize)
    kwargs = dict(annot=True, fmt=".3f", cmap="RdBu_r", center=0)
    if pval_df is not None:
        corr_mat = df.corr(method=method)
        pval_aligned = pval_df.reindex(index=corr_mat.index, columns=corr_mat.columns)
        annot_arr = pd.DataFrame("", index=corr_mat.index, columns=corr_mat.columns)
        for i in corr_mat.index:
            for j in corr_mat.columns:
                r, p = corr_mat.loc[i, j], pval_aligned.loc[i, j]
                if pd.notna(r):
                    annot_arr.loc[i, j] = f"{r:.3f}*" if (pd.notna(p) and p < significance_level) else f"{r:.3f}"
        kwargs["annot"] = annot_arr
        kwargs["fmt"] = ""
    DataVisualizer.add_correlation_heatmap_to_axis(df, ax, method=method, title=title, **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {output_dir / 'correlation_heatmap.png'}")


def run_cca_analysis(
    df_numeric: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    df_for_zero_check: pd.DataFrame | None = None,
) -> None:
    """
    Regularized CCA (rCCA): X (game metrics) -> Y (engagement/retention).
    L2 regularization helps when X is highly correlated.
    When REMOVE_PAIRWISE_ZEROS, drops rows where any X/Y var with >threshold zeros is zero (uses raw data for detection).
    Heatmap: rows=X, cols=Y, values=canonical coefficients, * for significant (p < alpha).
    """
    x_vars = [c for c in CCA_X_VARS if c in df_numeric.columns]
    y_vars = [c for c in CCA_Y_VARS if c in df_numeric.columns]
    if not x_vars or not y_vars:
        print("  CCA skipped: missing X or Y variables")
        return
    subset = df_numeric[x_vars + y_vars].dropna(how="any")
    # Row-wise zero removal: drop rows where any var with >threshold zeros is zero
    if REMOVE_PAIRWISE_ZEROS and df_for_zero_check is not None and len(subset) >= 3:
        df_z = df_for_zero_check[x_vars + y_vars].reindex(subset.index).dropna(how="any")
        if len(df_z) >= 3:
            zeros_eps = 1e-10
            keep = pd.Series(True, index=subset.index)
            for col in x_vars + y_vars:
                zero_frac = (df_z[col] <= zeros_eps).mean()
                if zero_frac > ZEROS_REMOVAL_THRESHOLD:
                    keep &= df_z[col] > zeros_eps
            if keep.any():
                subset = subset.loc[keep]
    if len(subset) < 10:
        print("  CCA skipped: too few complete rows")
        return
    X = subset[x_vars].astype(float).values
    Y = subset[y_vars].astype(float).values

    n_components = min(len(x_vars), len(y_vars), len(subset) - 1)
    if n_components < 1:
        return

    try:
        from cca_zoo.linear import rCCA
        from cca_zoo.model_selection import GridSearchCV
    except ImportError:
        print("  CCA skipped: cca-zoo not installed. Run: poetry add cca-zoo --group ds")
        return

    param_grid = {"c": CCA_REGULARIZATION_GRID}
    cv_folds = min(CCA_CV_FOLDS, len(subset) // 2)  # need enough samples per fold
    if cv_folds < 2:
        cv_folds = 2
    search = GridSearchCV(
        rCCA(latent_dimensions=n_components),
        param_grid=param_grid,
        cv=cv_folds,
        verbose=0,
    )
    search.fit((X, Y))
    cca = search.best_estimator_
    best_c = search.best_params_.get("c", "?")
    print(f"  CCA best c={best_c} (mean CV score={search.best_score_:.4f})")

    wx = cca.weights_[0]  # (n_X, n_components)
    wy = cca.weights_[1]  # (n_Y, n_components)
    X_scores = X @ wx
    Y_scores = Y @ wy
    r_vals = np.array([np.corrcoef(X_scores[:, k], Y_scores[:, k])[0, 1] for k in range(n_components)])
    r_vals = np.nan_to_num(r_vals, nan=0)
    coef = wx @ np.diag(r_vals) @ np.linalg.pinv(wy)  # (n_X, n_Y)
    coef_df = pd.DataFrame(coef, index=x_vars, columns=y_vars)

    # P-values and correlations: Pearson for each (X_i, Y_j) on the same complete-case data
    pval_df = pd.DataFrame(index=x_vars, columns=y_vars, dtype=float)
    corr_xy_df = pd.DataFrame(index=x_vars, columns=y_vars, dtype=float)
    for xi in x_vars:
        for yj in y_vars:
            vx = subset[xi].values
            vy = subset[yj].values
            if len(vx) >= 3:
                r, p = stats.pearsonr(vx, vy)
                pval_df.loc[xi, yj] = p
                corr_xy_df.loc[xi, yj] = r
            else:
                pval_df.loc[xi, yj] = np.nan
                corr_xy_df.loc[xi, yj] = np.nan
    pval_df = pval_df.astype(float)
    corr_xy_df = corr_xy_df.astype(float)

    # Significant X–Y pairs for scatter
    cca_pairs = []
    for xi in x_vars:
        for yj in y_vars:
            p = pval_df.loc[xi, yj]
            r = corr_xy_df.loc[xi, yj]
            if pd.notna(p) and p < SIGNIFICANCE_LEVEL and pd.notna(r):
                cca_pairs.append((xi, yj, float(r), float(p)))
    output_dir.mkdir(parents=True, exist_ok=True)
    if cca_pairs:
        plot_scatter_pairs(
            subset,
            correlation_pairs=cca_pairs,
            add_regression_line=True,
            add_correlation_annotation=True,
            significance_level=SIGNIFICANCE_LEVEL,
            save_path=output_dir / "cca_scatter_significant_pairs.png",
            alpha=0.5,
            s=20,
        )
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(output_dir / "cca_cv_results.csv", index=False)
    coef_df.to_csv(output_dir / "cca_coefficients.csv")
    pval_df.to_csv(output_dir / "cca_pvalues.csv")

    annot = pd.DataFrame("", index=x_vars, columns=y_vars)
    for i in x_vars:
        for j in y_vars:
            c = coef_df.loc[i, j]
            p = pval_df.loc[i, j]
            sig = "*" if (pd.notna(p) and p < SIGNIFICANCE_LEVEL) else ""
            annot.loc[i, j] = f"{c:.3f}{sig}"

    fig, ax = plt.subplots(figsize=(max(10, len(y_vars) * 2.5), max(8, len(x_vars) * 0.8)))
    vmax = np.nanmax(np.abs(coef_df.values)) or 1
    im = ax.imshow(coef_df.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(y_vars)))
    ax.set_yticks(np.arange(len(x_vars)))
    ax.set_xticklabels(y_vars, rotation=45, ha="right")
    ax.set_yticklabels(x_vars)
    for row_idx in range(len(x_vars)):
        for col_idx in range(len(y_vars)):
            ax.text(
                float(col_idx), float(row_idx), str(annot.iloc[row_idx, col_idx]), ha="center", va="center", fontsize=9
            )
    ax.set_xlabel("Y (engagement / retention)")
    ax.set_ylabel("X (game metrics)")
    ax.set_title(f"Regularized CCA (c={best_c}): X → Y coefficients {title_suffix}".strip() + "\n* p < 0.05")
    plt.colorbar(im, ax=ax, label="Coefficient")
    plt.tight_layout()
    fig.savefig(output_dir / "cca_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CCA heatmap saved to {output_dir / 'cca_heatmap.png'}")


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {S3_DAILY_STATS}...")
    df_raw = load_daily_stats(S3_DAILY_STATS)
    print(f"Loaded {len(df_raw)} rows")

    df_prep = prepare_for_correlation(df_raw, keep_ai_group=True)
    # Save the exact data used for correlation & CCA (after duplicate removal, before power transform / zero removal)
    df_prep.to_csv(output_dir / "data_for_correlation_and_cca.csv", index=False)

    numeric_cols = [c for c in CORRELATION_VARS if c in df_prep.columns]
    print(f"Aggregated to {len(df_prep)} group-day rows. Variables: {numeric_cols}")

    # Overall analysis (all ai_groups combined)
    print("\n--- Overall correlation analysis ---")
    run_correlation_analysis(df_prep[numeric_cols], output_dir, scatter_subdir="scatter_plots")

    # Per ai_group analysis
    if "ai_group" in df_prep.columns:
        by_group_dir = output_dir / "by_ai_group"
        for ai_group in sorted(df_prep["ai_group"].dropna().unique()):
            subset = df_prep.loc[df_prep["ai_group"] == ai_group, numeric_cols]
            group_dir = by_group_dir / str(ai_group).replace("/", "_")
            print(f"\n--- ai_group = {ai_group} (n={len(subset)}) ---")
            run_correlation_analysis(subset, group_dir, title_suffix=f"(ai_group={ai_group})")


if __name__ == "__main__":
    main()
