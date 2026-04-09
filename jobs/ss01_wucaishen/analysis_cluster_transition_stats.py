"""
Analyze cluster transition stats: merge cluster labels from S3 with features,
compute transition labels (from_cluster:to_cluster), run two-way ANOVA
(current cluster × next cluster) and plot boxplots/barplots with p-values.
"""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats

from bituslabs_ds.s3_utils import read_local_cache
from jobs.ss01_wucaishen.cluster_transition_data import ensure_merged_parquet

# columns to show in boxplots and barplots regardless of significance:
STATS_COLUMNS = [
    "fg_rounds",
    "delta_t_seconds_avg",
    "delta_t_seconds_nogap_avg",
    "bet_amount_avg",
    "accum_pos_delta_bet_amount",
    "accum_neg_delta_bet_amount",
    "accum_pos_delta_bet_amount_ratio",
    "accum_neg_delta_bet_amount_ratio",
    "payout_avg",
    "payout_spike_ratio",
    "payout_drawdown_ratio",
]

# For features file: columns with index >= 6 are used for ANOVA (after merge keys / id columns)
FEATURE_START_INDEX = 6

# Output
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_ss01_cluster_transition"
BOXPLOT_DIR_NAME = "boxplots"
BARPLOT_DIR_NAME = "barplots"
ANOVA_SIGNIFICANCE_LEVEL = 0.05
BOOTSTRAP_N_SAMPLES = 500
BOOTSTRAP_RANDOM_SEED = 42
# Use Kruskal-Wallis (non-parametric) instead of ANOVA when True
USE_KRUSKAL_WALLIS = False


def _is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """True if column is numeric (handles pandas nullable Int64, etc.)."""
    if col not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[col])


# Factor names for two-way ANOVA: current cluster (from) and next cluster (to)
FACTOR_CURRENT = "cluster_label"
FACTOR_NEXT = "next_cluster"


def _two_way_anova_one_feature(merged: pd.DataFrame, col: str) -> dict[str, float] | None:
    """Run two-way ANOVA (current cluster × next cluster) for one feature.
    Returns dict with p_current, p_next, p_interaction; or None if invalid.
    """
    df = merged[[FACTOR_CURRENT, FACTOR_NEXT, col]].dropna()
    col_series = pd.Series(df[col])
    if col_series.nunique() < 2 or len(df) < 10:
        return None
    try:
        aov = pg.anova(
            dv=col,
            between=[FACTOR_CURRENT, FACTOR_NEXT],
            data=df,
            detailed=True,
        )
    except Exception:
        return None
    out = {}
    for _, row in aov.iterrows():
        src = str(row["Source"]).strip()
        p = float(row["p-unc"]) if "p-unc" in row else np.nan
        if src == FACTOR_CURRENT:
            out["p_current"] = p
        elif src == FACTOR_NEXT:
            out["p_next"] = p
        elif "*" in src or "interaction" in src.lower():
            out["p_interaction"] = p
    return out if len(out) >= 3 else None


def _simple_effects_next_within_current(merged: pd.DataFrame, col: str) -> dict[int, float]:
    """One-way ANOVA on next_cluster within each level of current cluster. Returns {current_level: p_value}."""
    result = {}
    for current in sorted(merged[FACTOR_CURRENT].dropna().unique()):
        subset = merged.loc[merged[FACTOR_CURRENT] == current, [FACTOR_NEXT, col]].dropna()
        groups = [g[col].values for _, g in subset.groupby(FACTOR_NEXT) if len(g) >= 2]
        if len(groups) < 2:
            result[int(current)] = np.nan
            continue
        _, p_val = stats.f_oneway(*groups)
        result[int(current)] = float(p_val)
    return result


def _two_way_kruskal_one_feature(merged: pd.DataFrame, col: str) -> dict[str, float] | None:
    """Run Kruskal-Wallis for main effects (current, next) and overall transition (as interaction proxy).
    Returns dict with p_current, p_next, p_interaction; or None if invalid.
    """
    df = merged[[FACTOR_CURRENT, FACTOR_NEXT, "transition", col]].dropna()
    col_series = pd.Series(df[col])
    if col_series.nunique() < 2 or len(df) < 10:
        return None
    out = {}
    try:
        groups_current = [pd.Series(g[col]).to_numpy() for _, g in df.groupby(FACTOR_CURRENT) if len(g) >= 2]
        if len(groups_current) >= 2:
            _, p = stats.kruskal(*groups_current)
            out["p_current"] = float(p)
        groups_next = [pd.Series(g[col]).to_numpy() for _, g in df.groupby(FACTOR_NEXT) if len(g) >= 2]
        if len(groups_next) >= 2:
            _, p = stats.kruskal(*groups_next)
            out["p_next"] = float(p)
        groups_tr = [pd.Series(g[col]).to_numpy() for _, g in df.groupby("transition") if len(g) >= 2]
        if len(groups_tr) >= 2:
            _, p = stats.kruskal(*groups_tr)
            out["p_interaction"] = float(p)
    except Exception:
        return None
    return out if len(out) >= 3 else None


def _simple_effects_kruskal_next_within_current(merged: pd.DataFrame, col: str) -> dict[int, float]:
    """Kruskal-Wallis on next_cluster within each level of current cluster. Returns {current_level: p_value}."""
    result = {}
    for current in sorted(merged[FACTOR_CURRENT].dropna().unique()):
        subset = merged.loc[merged[FACTOR_CURRENT] == current, [FACTOR_NEXT, col]].dropna()
        groups = [g[col].values for _, g in subset.groupby(FACTOR_NEXT) if len(g) >= 2]
        if len(groups) < 2:
            result[int(current)] = np.nan
            continue
        try:
            _, p_val = stats.kruskal(*groups)
            result[int(current)] = float(p_val)
        except Exception:
            result[int(current)] = np.nan
    return result


def two_way_anova_and_simple_effects(
    merged: pd.DataFrame,
    feature_columns: list[str],
    use_kruskal: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], str]:
    """Two-way ANOVA or Kruskal-Wallis (current × next cluster) plus simple-effect tests.
    Returns (results_df, anova_by_feature, test_name). test_name is 'ANOVA' or 'Kruskal-Wallis'.
    """
    rows = []
    anova_by_feature = {}
    for col in feature_columns:
        if col not in merged.columns or not _is_numeric_column(merged, col):
            continue
        if use_kruskal:
            two_way = _two_way_kruskal_one_feature(merged, col)
            simple = _simple_effects_kruskal_next_within_current(merged, col)
        else:
            two_way = _two_way_anova_one_feature(merged, col)
            simple = _simple_effects_next_within_current(merged, col)
        if two_way is None:
            continue
        row = {
            "feature": col,
            "p_current": two_way.get("p_current", np.nan),
            "p_next": two_way.get("p_next", np.nan),
            "p_interaction": two_way.get("p_interaction", np.nan),
        }
        for k, p in simple.items():
            row[f"p_next_within_{k}"] = p
        rows.append(row)
        anova_by_feature[col] = {
            "p_current": row["p_current"],
            "p_next": row["p_next"],
            "p_interaction": row["p_interaction"],
            **{f"p_next_within_{k}": v for k, v in simple.items()},
        }
    test_name = "Kruskal-Wallis" if use_kruskal else "ANOVA"
    return pd.DataFrame(rows), anova_by_feature, test_name


def _transition_order(merged: pd.DataFrame) -> list:
    """Canonical order for transition labels (cluster_from:cluster_to)."""
    return sorted(
        merged["transition"].unique(),
        key=lambda x: (int(x.split(":")[0].replace("cluster", "")), int(x.split(":")[1])),
    )


def _fmt_p(p: float, alpha: float = ANOVA_SIGNIFICANCE_LEVEL) -> str:
    """Format p-value with * if significant."""
    s = f"{p:.2e}" if np.isfinite(p) else "n/a"
    return s + "*" if np.isfinite(p) and p < alpha else s


def _format_anova_text_main(anova_dict: dict[str, Any] | None, test_name: str = "ANOVA") -> str:
    """Format main effects and interaction only (for top-left box). Uses * for significant p."""
    if not anova_dict:
        return ""
    suffix = f", {test_name}" if test_name and test_name != "ANOVA" else ""
    return "\n".join(
        [
            f"Current (main{suffix}): p = {_fmt_p(anova_dict.get('p_current', np.nan))}",
            f"Next (main{suffix}): p = {_fmt_p(anova_dict.get('p_next', np.nan))}",
            f"Interaction{suffix}: p = {_fmt_p(anova_dict.get('p_interaction', np.nan))}",
        ]
    )


def _current_to_x_center(order: list[str]) -> dict[int, float]:
    """Map current cluster level -> x position (center of that group) for placing simple-effect text."""
    level_to_indices: dict[int, list[int]] = {}
    for i, tr in enumerate(order):
        from_part = tr.split(":")[0].replace("cluster", "")
        level = int(from_part)
        level_to_indices.setdefault(level, []).append(i)
    # Explicitly cast np.mean result to float for mypy compatibility
    return {lev: float(np.mean(idxs)) for lev, idxs in level_to_indices.items()}


def plot_transition_counts(merged: pd.DataFrame, output_dir: Path) -> None:
    """Plot the number of bets (sample size) for each transition condition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    order = _transition_order(merged)
    counts = merged["transition"].value_counts().reindex(order)
    vals = pd.Series(counts).fillna(0).to_numpy(dtype=int)
    # Percentage within current cluster: e.g. cluster0:0 / (cluster0:0 + cluster0:1 + cluster0:2)
    sum_by_current: dict[int, float] = {}
    for i, tr in enumerate(order):
        current = int(tr.split(":")[0].replace("cluster", ""))
        sum_by_current[current] = sum_by_current.get(current, 0) + vals[i]
    pcts = [
        100 * vals[i] / (sum_by_current[int(tr.split(":")[0].replace("cluster", ""))] or 1)
        for i, tr in enumerate(order)
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.8), 4))
    x_pos = np.arange(len(order))
    ax.bar(x_pos, np.where(vals <= 0, 1, vals))  # use 1 for log scale when count is 0
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Number of bets")
    ax.set_yscale("log")
    ax.set_xlabel("Transition (from:to)")
    ax.set_title("Sample size per transition condition", pad=20)
    # Add count and percentage (within current cluster) on top of each bar
    for i, (x, c) in enumerate(zip(x_pos, vals)):
        h = c if c > 0 else 1
        label = f"{int(c)} ({pcts[i]:.2f}%)"
        ax.text(x, h * 1.08, label, ha="center", va="bottom", fontsize=8, rotation=0)
    # More space on top for labels and between title and figure; remove top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "transition_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots_significant(
    merged: pd.DataFrame,
    significant_features: list[str],
    output_dir: Path,
    anova_by_feature: dict[str, dict[str, Any]] | None = None,
    test_name: str = "ANOVA",
) -> None:
    """Save one boxplot per significant feature (transition vs value), with test p-values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    order = _transition_order(merged)
    anova_by_feature = anova_by_feature or {}
    x_centers = _current_to_x_center(order)
    for col in significant_features:
        if col not in merged.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.8), 6))
        sns.boxplot(data=merged, x="transition", y=col, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{col} by cluster transition")
        # Main effects and interaction in top-left
        text = _format_anova_text_main(anova_by_feature.get(col), test_name=test_name)
        if text:
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8, verticalalignment="top", family="monospace")
        # Simple-effect p-values just above each current-cluster group (lower to avoid overlap with main-effect box)
        anova_dict = anova_by_feature.get(col)
        if anova_dict:
            y_lo, y_hi = ax.get_ylim()
            pad = (y_hi - y_lo) * 0.18
            ax.set_ylim(y_lo, y_hi + pad)
            y_annot = y_hi + pad * 0.08
            for level in sorted(x_centers.keys()):
                key = f"p_next_within_{level}"
                p = anova_dict.get(key, np.nan)
                if not np.isfinite(p):
                    continue
                x_c = x_centers[level]
                ax.text(x_c, y_annot, f"p={_fmt_p(p)}", ha="center", fontsize=8, family="monospace")
        plt.tight_layout()
        safe_name = col.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"boxplot_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _bootstrap_95ci(
    values: np.ndarray,
    n_samples: int = BOOTSTRAP_N_SAMPLES,
    random_state: int | None = BOOTSTRAP_RANDOM_SEED,
) -> tuple[float, float]:
    """Return (lower, upper) 95% CI of the mean using bootstrap."""
    # Coerce to float (handles Decimal from parquet)
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    boot_means = rng.choice(values, size=(n_samples, n), replace=True).mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def plot_barplots_significant(
    merged: pd.DataFrame,
    significant_features: list[str],
    output_dir: Path,
    anova_by_feature: dict[str, dict[str, Any]] | None = None,
    test_name: str = "ANOVA",
) -> None:
    """Save one bar plot (mean ± 95% bootstrap CI) per significant feature, with test p-values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    order = _transition_order(merged)
    anova_by_feature = anova_by_feature or {}
    x_centers = _current_to_x_center(order)
    for col in significant_features:
        if col not in merged.columns:
            continue
        means = np.empty(len(order), dtype=float)
        err_lower = np.empty(len(order), dtype=float)
        err_upper = np.empty(len(order), dtype=float)
        for idx, tr in enumerate(order):
            vals = merged.loc[merged["transition"] == tr, col].dropna().values
            mean_val = np.mean(vals) if len(vals) > 0 else np.nan
            if len(vals) > 0:
                lo, hi = _bootstrap_95ci(vals, n_samples=BOOTSTRAP_N_SAMPLES, random_state=BOOTSTRAP_RANDOM_SEED)
                means[idx] = mean_val
                err_lower[idx] = mean_val - lo
                err_upper[idx] = hi - mean_val
            else:
                means[idx] = mean_val
                err_lower[idx] = 0
                err_upper[idx] = 0
        yerr = np.array([err_lower, err_upper])
        fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.8), 6))
        x_pos = np.arange(len(order))
        ax.bar(x_pos, means, yerr=yerr, capsize=4, align="center")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel(col)
        ax.set_xlabel("Transition (from:to)")
        ax.set_title(f"{col} by cluster transition (mean ± 95% CI, {BOOTSTRAP_N_SAMPLES} bootstrap)")
        # Main effects and interaction in top-left
        text = _format_anova_text_main(anova_by_feature.get(col), test_name=test_name)
        if text:
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8, verticalalignment="top", family="monospace")
        # Simple-effect p-values just above each current-cluster group (lower to avoid overlap with main-effect box)
        anova_dict = anova_by_feature.get(col)
        if anova_dict:
            y_lo, y_hi = ax.get_ylim()
            pad = (y_hi - y_lo) * 0.18
            ax.set_ylim(y_lo, y_hi + pad)
            y_annot = y_hi + pad * 0.08
            for level in sorted(x_centers.keys()):
                key = f"p_next_within_{level}"
                p = anova_dict.get(key, np.nan)
                if not np.isfinite(p):
                    continue
                x_c = x_centers[level]
                ax.text(x_c, y_annot, f"p={_fmt_p(p)}", ha="center", fontsize=8, family="monospace")
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        safe_name = col.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"barplot_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Merged data: use cache if present, else build from S3 cluster labels + features
    merged_path = output_dir / "merged_with_transitions.parquet"
    ensure_merged_parquet(merged_path)
    merged = cast(pd.DataFrame, read_local_cache(str(merged_path), lazy_load=False))
    print(f"Merged shape: {merged.shape}, transitions: {merged['transition'].nunique()}")

    # 3) Stats for requested columns
    stats_cols = [c for c in STATS_COLUMNS if c in merged.columns]
    if stats_cols:
        trans_stats = merged.groupby("transition")[stats_cols].agg(["count", "mean", "std"]).round(6)
        trans_stats.to_csv(output_dir / "transition_stats.csv")
        print("Transition stats saved to transition_stats.csv")

    # 3b) Plot sample size (number of bets) per transition condition
    plot_transition_counts(merged, output_dir)

    # 4) Two-way ANOVA or Kruskal-Wallis (current cluster × next cluster) and simple-effect tests
    # Feature columns = merged columns minus transition metadata (order preserved from features file)
    transition_meta = {"cluster_label", "next_cluster", "transition"}
    feature_file_cols = [c for c in merged.columns if c not in transition_meta]
    anova_cols = [c for i, c in enumerate(feature_file_cols) if i >= FEATURE_START_INDEX and c in merged.columns]
    anova_cols = [c for c in anova_cols if _is_numeric_column(merged, c)]

    use_kruskal = USE_KRUSKAL_WALLIS
    anova_res, anova_by_feature, test_name = two_way_anova_and_simple_effects(
        merged, anova_cols, use_kruskal=use_kruskal
    )
    csv_name = "anova_two_way_and_simple_effects_kruskal.csv" if use_kruskal else "anova_two_way_and_simple_effects.csv"
    anova_res.to_csv(output_dir / csv_name, index=False)
    # Significant if any main effect or interaction is significant
    significant = anova_res.loc[
        (anova_res["p_current"] < ANOVA_SIGNIFICANCE_LEVEL)
        | (anova_res["p_next"] < ANOVA_SIGNIFICANCE_LEVEL)
        | (anova_res["p_interaction"] < ANOVA_SIGNIFICANCE_LEVEL),
        "feature",
    ].tolist()
    print(f"{test_name}: {len(significant)} features with some p < {ANOVA_SIGNIFICANCE_LEVEL}")
    if significant:
        print("Significant features:", significant[:20] if len(significant) > 20 else significant)

    # Variables with significant interaction effects
    interaction_sig = anova_res.loc[anova_res["p_interaction"] < ANOVA_SIGNIFICANCE_LEVEL, "feature"].tolist()
    print(f"Features with significant interaction (p_interaction < {ANOVA_SIGNIFICANCE_LEVEL}):")
    if interaction_sig:
        print(interaction_sig[:50] if len(interaction_sig) > 50 else interaction_sig)
    else:
        print("None")

    # 5) Boxplots and bar plots with test p-values, in separate folders
    significant = list(set(significant + STATS_COLUMNS))
    if significant:
        box_dir = output_dir / BOXPLOT_DIR_NAME
        bar_dir = output_dir / BARPLOT_DIR_NAME
        plot_boxplots_significant(merged, significant, box_dir, anova_by_feature, test_name=test_name)
        plot_barplots_significant(merged, significant, bar_dir, anova_by_feature, test_name=test_name)
        print(f"Boxplots saved under {box_dir}")
        print(f"Bar plots (mean ± 95% bootstrap CI) saved under {bar_dir}")


if __name__ == "__main__":
    main()
