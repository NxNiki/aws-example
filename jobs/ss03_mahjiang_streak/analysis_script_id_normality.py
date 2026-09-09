"""
SS03 script_id distribution normality check.

Tests whether per-script usage counts (``num_bets`` / ``num_users`` from the
script_id distribution export) follow a normal distribution.

Input: the parquet written by
``jobs/etl/redshift/ss03_mahjiang_streak/etl_script_id_distribution_oneoff.py``.

The analysis runs separately per ``(math_table_id, bet_type)`` — usage levels
differ across math tables, and FREE (free-game) spins concentrate on a subset
of scripts, so pooling either dimension would manufacture artificial
multimodality. Within a (math table, bet type) cell the distribution can
still be bimodal (two script pools separated by an empty gap), so the script
splits the data at the largest gap in the sorted metric values and tests each
pool separately, alongside the pooled data:

- D'Agostino-Pearson K^2 (skew + kurtosis based; valid for large n)
- Anderson-Darling (tail sensitive)
- Shapiro-Wilk on a seeded random subsample of 5000 (the exact test is only
  calibrated up to n=5000)
- Skewness / excess kurtosis as effect sizes: at n=10000 the tests reject
  tiny deviations, so judge practical normality from these and the QQ plots.

Each math table (and each pool within it) also gets a uniformity check of
the script_ids themselves:

- chi-square goodness-of-fit of the per-script bet counts against equal
  expected counts (i.e. every script served with equal probability), plus
  the dispersion index (variance/mean; ~1 under uniform-random assignment,
  Poisson-like counts)
- a per-script scatter (``*_uniformity.png``): counts in script_id order
  with the uniform expectation and a +/-3*sqrt(mean) Poisson band, so any
  over/under-served script or ordering pattern is visible directly.

Outputs one PNG per (math_table, bet_type, group) (histogram + fitted normal
overlay, QQ plot) and a JSON summary under
``jobs/output_ss03_mahjiang_streak/script_id_normality/``.

Run manually:

    poetry run python jobs/ss03_mahjiang_streak/analysis_script_id_normality.py
    poetry run python jobs/ss03_mahjiang_streak/analysis_script_id_normality.py --metric num_users
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from bituslabs_ds.config import LOCAL_ROOT, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PARQUET = Path(LOCAL_ROOT).parent / "data_ss03" / "script_id_distribution_all_math_tables.parquet"
DEFAULT_OUTPUT_DIR = Path(LOCAL_ROOT) / "jobs" / "output_ss03_mahjiang_streak" / "script_id_normality"

SHAPIRO_MAX_N = 5000
SHAPIRO_SEED = 42
# A gap this many times wider than the typical spacing between sorted unique
# values marks a genuine break between modes rather than sampling noise.
GAP_RATIO_THRESHOLD = 20.0


def split_modes(values: np.ndarray) -> dict[str, np.ndarray]:
    """Split values at the largest gap between sorted unique values.

    Returns ``{"pooled": ..., "low_pool": ..., "high_pool": ...}`` when a
    dominant gap exists, otherwise just ``{"pooled": ...}``.
    """
    unique_sorted = np.unique(values)
    if len(unique_sorted) < 3:
        return {"pooled": values}

    gaps = np.diff(unique_sorted)
    max_gap_idx = int(np.argmax(gaps))
    median_gap = float(np.median(gaps))
    if median_gap <= 0 or gaps[max_gap_idx] / median_gap < GAP_RATIO_THRESHOLD:
        return {"pooled": values}

    cut = (unique_sorted[max_gap_idx] + unique_sorted[max_gap_idx + 1]) / 2
    low, high = values[values < cut], values[values >= cut]
    # A gap isolating a handful of outliers is not a second mode.
    min_pool = max(20, len(values) // 100)
    if min(len(low), len(high)) < min_pool:
        return {"pooled": values}

    logger.info(
        "Bimodal split detected: gap %.0f (median gap %.2f), cutting at %.1f",
        gaps[max_gap_idx],
        median_gap,
        cut,
    )
    return {"pooled": values, "low_pool": low, "high_pool": high}


def normality_tests(values: np.ndarray) -> dict:
    n = len(values)
    mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
    skew = float(stats.skew(values))
    excess_kurtosis = float(stats.kurtosis(values))

    k2_stat, k2_p = stats.normaltest(values)

    ad: Any = stats.anderson(values, dist="norm")
    ad_crit_5pct = float(ad.critical_values[list(ad.significance_level).index(5.0)])

    rng = np.random.default_rng(SHAPIRO_SEED)
    sample = values if n <= SHAPIRO_MAX_N else rng.choice(values, size=SHAPIRO_MAX_N, replace=False)
    sw_stat, sw_p = stats.shapiro(sample)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "dagostino_k2": {"stat": float(k2_stat), "p_value": float(k2_p)},
        "anderson_darling": {
            "stat": float(ad.statistic),
            "crit_5pct": ad_crit_5pct,
            "reject_at_5pct": bool(ad.statistic > ad_crit_5pct),
        },
        "shapiro_wilk": {"n_subsample": len(sample), "stat": float(sw_stat), "p_value": float(sw_p)},
    }


def uniformity_test(values: np.ndarray) -> dict:
    """Chi-square test that every script is served with equal probability.

    Under uniform random script assignment the per-script counts are
    Binomial(N, 1/k) ~ Poisson, so chisquare(values) against equal expected
    counts is the standard goodness-of-fit test; the dispersion index
    (variance/mean) should be ~1.
    """
    k = len(values)
    mean = float(np.mean(values))
    chi2_stat, chi2_p = stats.chisquare(values)
    dispersion = float(np.var(values, ddof=1) / mean) if mean > 0 else float("nan")
    return {
        "n_scripts": k,
        "chi2_stat": float(chi2_stat),
        "dof": k - 1,
        "p_value": float(chi2_p),
        "dispersion_index": dispersion,
        "reject_at_5pct": bool(chi2_p < 0.05),
    }


def plot_uniformity(table_df: pd.DataFrame, cell: str, metric: str, uni: dict, output_dir: Path) -> Path:
    """Per-script counts in script_id order — a script served unusually often/rarely shows as a point outside the band."""
    d = table_df.sort_values("snapshot_script_id")
    y = d[metric].to_numpy(dtype=float)
    mean = float(y.mean())
    band = 3 * np.sqrt(mean)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(np.arange(len(y)), y, s=2, alpha=0.4, color="steelblue", linewidths=0)
    ax.axhline(mean, color="red", lw=1.5, label=f"uniform expectation ({mean:.1f})")
    ax.axhline(mean + band, color="orange", lw=1, ls="--", label="±3·√mean (Poisson band)")
    ax.axhline(max(mean - band, 0), color="orange", lw=1, ls="--")
    ax.set_xlabel("script_id (lexicographic order)")
    ax.set_ylabel(f"{metric} per script")
    ax.set_title(
        f"{metric} per script_id — {cell} | uniform chi² p={uni['p_value']:.3g}, "
        f"dispersion={uni['dispersion_index']:.3f}"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    path = output_dir / f"{metric}_{cell}_uniformity.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_group(values: np.ndarray, label: str, metric: str, result: dict, output_dir: Path) -> Path:
    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 5))

    bins = min(60, max(10, len(values) // 20))
    ax_hist.hist(values, bins=bins, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    x = np.linspace(values.min(), values.max(), 400)
    ax_hist.plot(x, stats.norm.pdf(x, result["mean"], result["std"]), "r-", lw=2, label="fitted normal")
    ax_hist.set_title(f"{metric} — {label} (n={result['n']})")
    ax_hist.set_xlabel(metric)
    ax_hist.set_ylabel("density")
    ax_hist.legend()

    stats.probplot(values, dist="norm", plot=ax_qq)
    ax_qq.set_title(
        f"QQ plot — skew={result['skewness']:.3f}, ex.kurt={result['excess_kurtosis']:.3f}\n"
        f"K² p={result['dagostino_k2']['p_value']:.2e}, "
        f"AD={result['anderson_darling']['stat']:.2f} (crit 5%={result['anderson_darling']['crit_5pct']:.2f})"
    )

    fig.tight_layout()
    path = output_dir / f"{metric}_{label}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    parser = argparse.ArgumentParser(description="Normality check of SS03 per-script usage counts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PARQUET, help="Script distribution parquet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNGs + JSON.")
    parser.add_argument("--metric", choices=["num_bets", "num_users"], default="num_bets")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # normaltest needs n >= 20 for a meaningful result; skip tiny math tables.
    MIN_N = 20

    summary: dict[str, dict] = {}
    for group_key, table_df in df.groupby(["math_table_id", "bet_type"]):
        math_table, bet_type = cast("tuple[Any, Any]", group_key)
        cell = f"{math_table}_{bet_type}"
        values = table_df[args.metric].to_numpy(dtype=float)
        if len(values) < MIN_N:
            logger.info("%s: only %d scripts, skipping normality tests", cell, len(values))
            summary[cell] = {"n": len(values), "skipped": f"fewer than {MIN_N} scripts"}
            continue

        cell_uniformity = uniformity_test(values)
        uniformity_png = plot_uniformity(table_df, cell, args.metric, cell_uniformity, args.output_dir)

        table_summary: dict[str, dict] = {}
        for group, group_values in split_modes(values).items():
            result = normality_tests(group_values)
            result["uniformity"] = uniformity_test(group_values)
            if group == "pooled":
                result["uniformity"]["figure"] = str(uniformity_png)
            png = plot_group(group_values, f"{cell}_{group}", args.metric, result, args.output_dir)
            result["figure"] = str(png)
            table_summary[group] = result

            k2_p = result["dagostino_k2"]["p_value"]
            ad = result["anderson_darling"]
            uni = result["uniformity"]
            logger.info(
                "%s/%s: n=%d mean=%.1f std=%.1f skew=%.3f ex.kurt=%.3f | K² p=%.3g | "
                "AD=%.2f vs crit(5%%)=%.2f -> %s | Shapiro(sub-%d) p=%.3g | "
                "uniform chi² p=%.3g dispersion=%.3f -> %s",
                cell,
                group,
                result["n"],
                result["mean"],
                result["std"],
                result["skewness"],
                result["excess_kurtosis"],
                k2_p,
                ad["stat"],
                ad["crit_5pct"],
                "REJECT normality" if ad["reject_at_5pct"] else "consistent with normal",
                result["shapiro_wilk"]["n_subsample"],
                result["shapiro_wilk"]["p_value"],
                uni["p_value"],
                uni["dispersion_index"],
                "REJECT uniformity" if uni["reject_at_5pct"] else "consistent with uniform",
            )
        summary[cell] = table_summary

    summary_path = args.output_dir / f"{args.metric}_normality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
