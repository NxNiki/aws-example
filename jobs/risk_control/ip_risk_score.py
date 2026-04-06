"""
Compute an IP-level risk score from Fish Hunter IP stats Parquet (per ``ip`` + ``currency_type``).

Pipeline:
  1. Yeo–Johnson power transform (``sklearn.preprocessing.PowerTransformer``).
  2. Min–max scale each feature to [0, 1].
  3. Linear score = sum(weight[f] * feature[f]) + ``LINEAR_BIAS``; weights in ``FEATURE_WEIGHTS``.
  4. Map the linear score to approximately standard normal via ``QuantileTransformer``
     (``output_distribution='normal'``) so the final ``risk_score`` is easier to interpret and threshold.

Plots (saved under ``--plot-dir``): risk score distribution; feature boxplots for risk vs non-risk using
**raw** feature values (median-imputed, before power transform), split by ``risk_score`` threshold.

When ``--input`` is ``s3://...``, risk-flagged rows are also written to Parquet beside the input key
(same folder), default filename ``risk_ips.parquet``, including ``linear_risk_score`` (weighted linear
score before quantile-normalization) and ``risk_score``. Disable with ``--no-s3-risk-export``.

**Profit hard filter:** a row is risk only if it passes the score quantile/threshold *and*
``total_profit`` in CNY equivalent is at least ``PROFIT_THRESHOLD_CNY`` (default 5000). Conversion uses
``CURRENCY_TO_CNY_RATE`` (multiply local-currency profit by the factor to get CNY); edit that dict for
your settlement rates. Unknown currencies log a warning and use ``1.0`` (treat as CNY) until you add them.

Requires: pandas, numpy, scikit-learn, matplotlib, awswrangler for S3 I/O (optional: seaborn for boxplots).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
except ImportError as e:
    raise ImportError("ip_risk_score.py requires scikit-learn. Install with: poetry install --with ml") from e

try:
    import seaborn as sns
except ImportError:
    sns = None

from bituslabs_ds.config import DEFAULT_ETL_OUTPUT, LOCAL_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edit weights here: positive => higher normalized feature value increases risk_score (before normal map).
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS: dict[str, float] = {
    "num_user_id": 0.75,
    "max_user_concurrency": 0.95,
    "rtp": 0.1,
    "total_profit": 0.5,
    "min_bet": -0.05,
    "max_bet": -0.05,
    "avg_user_id_duration_hours": -0.1,
    "std_user_id_duration_hours": -0.1,
    "num_bets": 0.25,
    "num_unique_bets": -0.5,
    "num_unique_fish_value": -0.25,
    "mean_of_user_mean_bet_interval_sec": 0,
    "std_of_user_mean_bet_interval_sec": 0,
    "mean_of_user_std_bet_interval_sec": 0,
    "std_of_user_std_bet_interval_sec": 0,
}

# Added after the weighted sum (before quantile-normalization).
LINEAR_BIAS: float = 0.0

# Hard filter: require CNY-equivalent total_profit >= this (after score-based risk flag).
PROFIT_THRESHOLD_CNY: float = 5000.0

# Multiply ``total_profit`` in the row's ``currency_type`` by this factor to get CNY equivalent.
# Rates are illustrative; replace with your book / FX policy (uppercase keys match stripped currency codes).
CURRENCY_TO_CNY_RATE: dict[str, float] = {
    "CNY": 1.0,
    "RMB": 1.0,
    "USD": 7.2,
    "USDT": 7.2,
    "HKD": 0.92,
    "TWD": 0.22,
    "EUR": 7.8,
    "JPY": 0.048,
    "VND": 0.00029,
    "THB": 0.20,
    "KRW": 0.0053,
    "SGD": 5.35,
    "MYR": 1.52,
    "PHP": 0.13,
    "IDR": 0.00045,
}

# Default Parquet from ``etl_get_ip_stats_fishhunter`` output.
DEFAULT_INPUT_URI = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/ip_stats/ip_stats_fishhunter.parquet"


def load_table(uri: str) -> pd.DataFrame:
    if uri.startswith("s3://"):
        import awswrangler as wr

        return wr.s3.read_parquet(path=uri)
    return pd.read_parquet(uri)


def s3_sibling_parquet_uri(input_uri: str, filename: str) -> str:
    """``s3://bucket/prefix/a.parquet`` + ``b.parquet`` → ``s3://bucket/prefix/b.parquet``."""
    u = input_uri.rstrip("/")
    rest = u[5:]  # after "s3://"
    if "/" not in rest:
        return f"{u}/{filename}"
    parent, _ = u.rsplit("/", 1)
    return f"{parent}/{filename}"


def profit_cny_equivalent(
    df: pd.DataFrame,
    rates: dict[str, float],
    profit_col: str = "total_profit",
    currency_col: str = "currency_type",
) -> pd.Series:
    """``total_profit`` in local currency × rate → CNY equivalent (same index as ``df``)."""
    if profit_col not in df.columns or currency_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    cur = df[currency_col].astype(str).str.strip().str.upper()
    rate_ser = cur.map(rates)
    missing = rate_ser.isna()
    if missing.any():
        unknown = sorted({str(x) for x in cur[missing].unique() if str(x) and str(x).upper() != "NAN"})
        if unknown:
            logger.warning(
                "CURRENCY_TO_CNY_RATE missing for %s; using 1.0 (treat profit as CNY). "
                "Add keys to CURRENCY_TO_CNY_RATE in ip_risk_score.py.",
                unknown[:25],
            )
        rate_ser = rate_ser.fillna(1.0)
    tp = np.asarray(pd.to_numeric(df[profit_col], errors="coerce"), dtype=np.float64)
    rate_arr = np.asarray(rate_ser, dtype=np.float64)
    out = tp * rate_arr
    return pd.Series(out, index=df.index, dtype=np.float64)


def save_risk_ips_to_s3_sibling(
    risk_df: pd.DataFrame,
    input_uri: str,
    filename: str,
    feature_cols: list[str],
) -> None:
    """Write risk-only rows to S3 next to ``input_uri`` (same prefix / folder)."""
    import awswrangler as wr

    if risk_df.empty:
        logger.info("No risk rows; skipping S3 risk export.")
        return
    out_uri = s3_sibling_parquet_uri(input_uri, filename)
    lead = [c for c in ("ip", "currency_type") if c in risk_df.columns]
    scores = [c for c in ("linear_risk_score", "risk_score", "profit_cny_equiv") if c in risk_df.columns]
    cols = lead + scores + [c for c in feature_cols if c in risk_df.columns]
    extra = [c for c in risk_df.columns if c not in cols and c != "is_risk"]
    want_cols = cols + extra
    export = risk_df.reindex(columns=want_cols).copy()
    if "risk_score" in export.columns:
        export = export.sort_values(by="risk_score", ascending=False)
    wr.s3.to_parquet(df=export, path=out_uri, index=False)
    logger.info("Wrote %s risk IP rows to %s (linear_risk_score = pre–quantile linear score)", len(export), out_uri)


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    if not mask.all():
        n_drop = int((~mask).sum())
        logger.warning("Dropping %s rows with non-finite feature values", n_drop)
    return X.loc[mask].reset_index(drop=True), mask


def power_transform_features(X: pd.DataFrame) -> tuple[np.ndarray, PowerTransformer]:
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    Xt = np.asarray(pt.fit_transform(X.to_numpy(dtype=float)), dtype=np.float64)
    return Xt, pt


def minmax_01(X: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    return scaler.fit_transform(X), scaler


def linear_combination(X01: np.ndarray, columns: list[str], weights: dict[str, float], bias: float) -> np.ndarray:
    w = np.array([weights[c] for c in columns], dtype=float)
    return X01 @ w + bias


def to_approx_normal(scores: np.ndarray, random_state: int) -> tuple[np.ndarray, QuantileTransformer]:
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, max(10, len(scores))),
        random_state=random_state,
    )
    z = qt.fit_transform(scores.reshape(-1, 1)).ravel()
    return z, qt


def risk_flags(risk_score: np.ndarray, threshold: float | None, quantile: float | None) -> np.ndarray:
    if threshold is not None:
        logger.info("Risk threshold (fixed): score > %.6f", threshold)
        return risk_score > threshold
    q = 0.9 if quantile is None else float(quantile)
    thr = float(np.quantile(risk_score, q))
    logger.info("Risk threshold at quantile %.4f: score > %.6f", q, thr)
    return risk_score > thr


def plot_score_distribution(scores: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=min(80, max(20, len(scores) // 50)), density=True, alpha=0.75, color="steelblue")
    xs = np.linspace(scores.min(), scores.max(), 200)
    ax.plot(
        xs,
        1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * xs**2),
        color="darkred",
        lw=2,
        label="N(0,1) PDF (reference)",
    )
    ax.set_xlabel("risk_score (quantile-normalized)")
    ax.set_ylabel("density")
    ax.set_title("Risk score distribution vs standard normal PDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_features_by_risk(
    X_raw: pd.DataFrame,
    is_risk: np.ndarray,
    out_path: Path,
) -> None:
    """Compare feature distributions by risk flag using raw-scale inputs (same row order as ``is_risk``).

    ``X_raw`` is typically the matrix used before power transform (e.g. median-imputed numeric columns).
    """
    if is_risk.sum() == 0 or (~is_risk).sum() == 0:
        logger.warning(
            "Skipping feature comparison plot: one of risk / non-risk groups is empty "
            "(risk=%s, non-risk=%s). Adjust threshold or quantile.",
            int(is_risk.sum()),
            int((~is_risk).sum()),
        )
        return
    n_feat = X_raw.shape[1]
    n_cols = min(4, n_feat)
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    group = np.where(is_risk, "risk (above threshold)", "non-risk")
    plot_df = X_raw.copy()
    plot_df["_group"] = group

    for i, col in enumerate(X_raw.columns):
        ax = axes[i]
        if sns is not None:
            sns.boxplot(data=plot_df, x="_group", y=col, ax=ax, order=["non-risk", "risk (above threshold)"])
            ax.set_xlabel("")
        else:
            d0 = plot_df.loc[plot_df["_group"] == "non-risk", col]
            d1 = plot_df.loc[plot_df["_group"] == "risk (above threshold)", col]
            ax.boxplot([d0.dropna(), d1.dropna()], labels=["non-risk", "risk"])
        ax.set_title(col, fontsize=8)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("raw value")

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Raw features: risk vs non-risk IPs (median-imputed; by risk_score threshold)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="IP risk score from stats Parquet (power → min–max → weighted sum → normal score)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_URI,
        help=f"Parquet path (local or s3://). Default: {DEFAULT_INPUT_URI}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write scored Parquet (local path recommended).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(Path(LOCAL_ROOT) / "jobs/risk_control/risk_score_plots"),
        help="Directory for PNG plots.",
    )
    parser.add_argument(
        "--risk-quantile",
        type=float,
        default=0.95,
        help="Classify as risk if risk_score > this quantile (used when --risk-threshold not set).",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=None,
        help="If set, risk when risk_score > this value (overrides --risk-quantile).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="RNG seed for QuantileTransformer.",
    )
    parser.add_argument(
        "--risk-s3-filename",
        type=str,
        default="risk_ips.parquet",
        help="When --input is s3://, write risk-flagged rows (with linear_risk_score) beside the input object under this name.",
    )
    parser.add_argument(
        "--no-s3-risk-export",
        action="store_true",
        help="Do not write risk IPs to S3 even if --input is on S3.",
    )
    parser.add_argument(
        "--profit-threshold-cny",
        type=float,
        default=None,
        help=f"Minimum total_profit in CNY equivalent for final risk flag (default: {PROFIT_THRESHOLD_CNY}).",
    )
    parser.add_argument(
        "--no-profit-filter",
        action="store_true",
        help="Disable the CNY profit hard filter (risk = score threshold only).",
    )
    args = parser.parse_args()

    feature_cols = list(FEATURE_WEIGHTS.keys())

    logger.info("Loading %s", args.input)
    df = load_table(args.input)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input missing columns: {missing_cols}. Present: {list(df.columns)}")

    id_cols = [c for c in ("ip", "currency_type") if c in df.columns]
    X_df, row_ok = build_feature_matrix(df, feature_cols)
    keep_cols = id_cols + feature_cols
    df_ok = df.loc[row_ok, keep_cols].reset_index(drop=True)

    X_pow, _pt = power_transform_features(X_df)
    X_01, _mm = minmax_01(X_pow)
    linear = linear_combination(X_01, feature_cols, FEATURE_WEIGHTS, LINEAR_BIAS)
    risk_score, _qt = to_approx_normal(linear, args.random_state)
    is_risk_score = risk_flags(
        risk_score, args.risk_threshold, args.risk_quantile if args.risk_threshold is None else None
    )

    profit_cny = profit_cny_equivalent(df_ok, CURRENCY_TO_CNY_RATE)
    thresh_cny = PROFIT_THRESHOLD_CNY if args.profit_threshold_cny is None else float(args.profit_threshold_cny)
    if args.no_profit_filter:
        meets_profit = pd.Series(True, index=df_ok.index)
    else:
        meets_profit = profit_cny >= thresh_cny
    is_risk = is_risk_score & meets_profit.to_numpy(dtype=bool)

    out_df = df_ok.copy()
    out_df["linear_risk_score"] = linear
    out_df["risk_score"] = risk_score
    out_df["profit_cny_equiv"] = profit_cny
    out_df["meets_profit_threshold_cny"] = meets_profit.to_numpy(dtype=bool)
    out_df["is_risk_score"] = is_risk_score
    out_df["is_risk"] = is_risk

    if not args.no_profit_filter:
        logger.info(
            "Score-based risk rows: %s; after CNY profit filter (>=%s): %s",
            int(np.sum(is_risk_score)),
            thresh_cny,
            int(np.sum(is_risk)),
        )

    risk_rows = out_df.loc[out_df["is_risk"]]
    logger.info("Final risk-flagged rows: %s", len(risk_rows))
    if "ip" not in risk_rows.columns:
        logger.warning("No ip column in output; skipping risk IP listing.")
    elif risk_rows.empty:
        print("No IPs identified as risk (empty risk set).")
    else:
        sort_cols = ["risk_score"] if "risk_score" in risk_rows.columns else []
        r = risk_rows.sort_values(sort_cols, ascending=False) if sort_cols else risk_rows

        def _fmt_cell(v: object) -> str:
            if v is None:
                return ""
            if isinstance(v, (float, np.floating)):
                fv = float(v)
                if np.isnan(fv):
                    return ""
                return f"{fv:.6g}"
            return str(v)

        lead = ["ip"] + (["currency_type"] if "currency_type" in r.columns else [])
        scores = [c for c in ("linear_risk_score", "risk_score", "profit_cny_equiv") if c in r.columns]
        header_parts = lead + scores + feature_cols
        print(f"\nRisk-flagged IP rows ({len(r)}): tab-separated (raw feature values)")
        print("\t".join(header_parts))
        for _, row in r.iterrows():
            parts = [str(row["ip"])]
            if "currency_type" in r.columns:
                parts.append(str(row["currency_type"]))
            for c in scores:
                parts.append(_fmt_cell(row[c]))
            for c in feature_cols:
                parts.append(_fmt_cell(row[c]))
            print("\t".join(parts))

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_score_distribution(risk_score, plot_dir / "risk_score_distribution.png")

    plot_features_by_risk(X_df, is_risk, plot_dir / "features_risk_vs_nonrisk.png")

    logger.info("Wrote plots to %s", plot_dir)

    if args.input.startswith("s3://") and not args.no_s3_risk_export:
        save_risk_ips_to_s3_sibling(risk_rows, args.input, args.risk_s3_filename, feature_cols)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        logger.info("Wrote scored table (%s rows) to %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
