"""SS03 mahjong streak incentive stats from one merged extract, aggregate, plot RTP, export CSV.

Tune ``CUTOFF_DATE`` and ``HIST_BINS`` at the top of this module (not CLI).

When ``CUTOFF_DATE`` is a non-empty string (default ``2026-04-10``), the same aggregation runs
separately for rows with ``activity_date`` **before** the cutoff (``period`` = ``before``) and
**on or after** the cutoff (``period`` = ``after``). Set ``CUTOFF_DATE`` to ``None`` or ``""`` for
the original single-run summary (no ``period`` column) and the day×arm histogram.
Results are concatenated into one summary CSV when split is enabled.

Summary DataFrame columns (``analyze()[0]``, also ``incentive_stats_summary_before_after.csv``)
--------------------------------------------------------------------------------
period  *(only when ``CUTOFF_DATE`` is a non-empty date string)*
    ``before`` if ``activity_date`` < cutoff, ``after`` if ``activity_date`` >= cutoff.
activity_date
    Activity day from the merged incentive extract.
ab_group_id
    AB cohort id (UUID string); arm is inferred from the first two characters
    (``4a`` vs ``4f``) for thresholds.
num_users
    Count of distinct ``user_id`` for this date and partition.
num_users_with_fg
    Users with at least one **non-incentive** free-game row that day: ``bet_type != 'BASE'``
    and ``incentivized == 0`` (numeric ``0`` / false; missing treated as ``0``). Rows that are
    free game but ``incentivized == 1`` do **not** count toward this metric.
num_users_low_spin
    Users whose **base-game** spin total (``bet_type == 'BASE'`` ``spin_count`` summed)
    is **strictly less** than the arm spin threshold. Before ``CUTOFF_DATE``: 130 for ``4f*``,
    100 for ``4a*``. On or after the cutoff: **70** for both arms.
num_users_high_rtp
    Users with base-game spin total **greater than or equal** to the arm spin
    threshold **and** mean base-game ``rtp`` **greater than** the arm RTP cutoff
    (0.5 for ``4a*``, 0.6 for ``4f*``) — same RTP cutoffs before and after the cutoff.
num_users_incentivized
    Users with any detail row where ``incentivized == 1`` that day.
num_users_high_rtp_incentivized
    Users who satisfy ``high_rtp`` **and** have ``has_incentivized`` true (any
    incentivized row that day).
num_user_pass_incentive_thresh
    Users whose **base-game** spin total is **strictly greater** than the arm spin threshold
    **and** mean base-game ``rtp`` is **strictly greater** than the arm RTP cutoff (same
    thresholds as ``spin_thresh`` / ``rtp_thresh`` for that row). Unlike ``num_users_high_rtp``,
    this uses ``>`` on spin count, not ``>=``.
num_user_pass_incentive_thresh_ratio
    ``num_user_pass_incentive_thresh / num_users``; NaN if ``num_users`` is 0.
median_spin_below_thresh
    Median of base-game spin totals among users in ``num_users_low_spin`` only;
    missing (NaN) when there are no users below the spin threshold for that row.
mean_rtp_no_fg_not_incentivized
    Mean of per-user base-game ``rtp`` among users with **no** non-``'BASE'`` row and
    **no** ``incentivized == 1`` row that day. Rows with missing ``rtp`` are ignored in
    the mean; NaN if no such user has a defined ``rtp``.
avg_base_spin_no_fg_not_incentivized
    Mean of ``total_spin`` (base-game spin sum) over the same user subset as
    ``mean_rtp_no_fg_not_incentivized``. NaN when that subset is empty.
ratio_users_with_fg
    ``num_users_with_fg / num_users``; NaN if ``num_users`` is 0.
num_user_with_fg_ratio
    Same as ``ratio_users_with_fg``: ``num_users_with_fg / num_users``.
ratio_users_low_spin
    ``num_users_low_spin / num_users``; NaN if ``num_users`` is 0.
ratio_users_high_rtp
    ``num_users_high_rtp / num_users``; NaN if ``num_users`` is 0.
ratio_users_incentivized
    ``num_users_incentivized / num_users``; NaN if ``num_users`` is 0.
ratio_users_high_rtp_incentivized
    ``num_users_high_rtp_incentivized / num_users``; NaN if ``num_users`` is 0.

Per-user working frame ``user_day`` (``analyze()[1]``)
------------------------------------------------------
total_spin
    Sum of ``spin_count`` over rows with ``bet_type == 'BASE'`` only (base game).
has_any_non_base
    True if any row that day has ``bet_type != 'BASE'`` (any free game, regardless of
    ``incentivized``). Used for ``no_fg_not_incentivized``.
has_fg
    True if any row that day has ``bet_type != 'BASE'`` **and** ``incentivized == 0`` (non-incentive
    free game only). Drives ``num_users_with_fg``.
has_incentivized
    True if any row that day has ``incentivized == 1``.
spin_thresh / rtp_thresh
    Arm-specific cutoffs: ``spin_thresh`` is 130/100 (``4f``/``4a``) before ``CUTOFF_DATE`` and
    70/70 on or after; ``rtp_thresh`` is 0.6 for ``4f`` and 0.5 for ``4a`` in both periods.
rtp
    Mean of ``rtp`` over ``bet_type == 'BASE'`` rows only (missing if no base-game rows).
low_spin
    ``total_spin < spin_thresh``.
high_rtp
    ``total_spin >= spin_thresh`` and ``rtp > rtp_thresh`` (requires finite ``rtp``).
high_rtp_incentivized
    ``high_rtp`` and ``has_incentivized`` (high RTP bucket and got incentivized).
pass_incentive_thresh
    ``total_spin > spin_thresh`` and ``rtp > rtp_thresh`` (strict on both; drives
    ``num_user_pass_incentive_thresh``).
no_fg_not_incentivized
    True when the user had no non-``'BASE'`` row at all (``~has_any_non_base``) and no
    incentivized row that day (``~has_incentivized``).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET
from bituslabs_ds.s3_utils import upload_file_to_s3

# file1 = "/Users/niuxin/Documents/data_ss03/ab_cny_bet_rtp_by_user_mathtable.parquet"
file1 = "/Users/niuxin/Documents/data_ss03/merged_user_id_20260401-202604016_with_ab.csv"

OUTPUT_DIR = Path(LOCAL_ROOT) / "jobs" / "output" / "ss03_mahjiang_streak"
S3_ANALYSIS_PREFIX = "ds-analysis/ss03_mahjiang_streak"

# Before/after split: ISO date string, or None / "" for no split (legacy summary + day×arm plot).
CUTOFF_DATE: str | None = "2026-04-10"
HIST_BINS = 72

# Spin thresholds by AB prefix (``4f`` / ``4a``): before cutoff vs on/after ``CUTOFF_DATE``. RTP unchanged.
# ``activity_date`` / ``CUTOFF_DATE`` are treated as naive calendar days (e.g. Beijing-time date strings);
# no timezone conversion — compare with ``>=`` / ``<`` on normalized datetimes.
_SPIN_BEFORE_4F, _SPIN_BEFORE_4A = 130, 100
_SPIN_AFTER_4F, _SPIN_AFTER_4A = 70, 70
_RTP_4F, _RTP_4A = 0.6, 0.5

DEFAULT_LOCAL_FIGURE = OUTPUT_DIR / "base_game_rtp_before_after.png"
DEFAULT_S3_FIGURE_KEY = f"{S3_ANALYSIS_PREFIX}/base_game_rtp_before_after.png"

DEFAULT_LOCAL_RATIO_INCENTIVE_FIGURE = OUTPUT_DIR / "ratio_users_incentivized_by_day.png"
DEFAULT_S3_RATIO_INCENTIVE_FIGURE_KEY = f"{S3_ANALYSIS_PREFIX}/ratio_users_incentivized_by_day.png"

DEFAULT_LOCAL_SUMMARY_CSV = OUTPUT_DIR / "incentive_stats_summary_before_after.csv"
DEFAULT_S3_SUMMARY_KEY = f"{S3_ANALYSIS_PREFIX}/incentive_stats_summary_before_after.csv"

DEFAULT_LOCAL_HTML_REPORT = OUTPUT_DIR / "ss03_incentive_analysis_report.html"

# Columns not offered as Y-axis metrics in the HTML line chart (grouping / index fields).
LINE_PLOT_METRIC_EXCLUDE = frozenset({"period", "activity_date", "ab_group_id", "ab_arm"})


def _cutoff_date_display() -> str:
    """ISO date string for titles; empty if ``CUTOFF_DATE`` unset."""
    if CUTOFF_DATE is None:
        return ""
    s = str(CUTOFF_DATE).strip()
    if not s:
        return ""
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return s


def _aggregate_summary_for_ab_arm(summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``ab_group_id`` to first two characters; sum user counts; recompute ``ratio_users_*``."""
    if summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    if "period" in df.columns:
        df = df.drop(columns=["period"])
    df["activity_date"] = pd.to_datetime(df["activity_date"])
    df["ab_arm"] = df["ab_group_id"].astype(str).str[:2].str.lower()
    gcols = ["activity_date", "ab_arm"]
    num_cols = [c for c in df.columns if c.startswith("num_")]
    agg_dict: dict[str, str] = {c: "sum" for c in num_cols}
    for c in df.columns:
        if c.startswith(("median_", "mean_", "avg_")):
            agg_dict[c] = "mean"
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    out = cast(pd.DataFrame, df.groupby(gcols, as_index=False).agg(agg_dict))
    nu = out["num_users"].replace(0, np.nan) if "num_users" in out.columns else None
    if nu is not None:
        for c in num_cols:
            if c == "num_users" or not c.startswith("num_users_"):
                continue
            rest = c[len("num_users_") :]
            rc = f"ratio_users_{rest}"
            out[rc] = cast(pd.Series, out[c]) / nu
        if "num_user_pass_incentive_thresh" in out.columns:
            out["num_user_pass_incentive_thresh_ratio"] = cast(pd.Series, out["num_user_pass_incentive_thresh"]) / nu
        if "num_users_with_fg" in out.columns:
            out["num_user_with_fg_ratio"] = cast(pd.Series, out["num_users_with_fg"]) / nu
    return out


def _line_plot_metric_columns(plot_df: pd.DataFrame) -> list[str]:
    """Numeric columns suitable for the HTML line chart (excludes grouping keys)."""
    cols: list[str] = []
    for c in plot_df.columns:
        if c in LINE_PLOT_METRIC_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(plot_df[c]):
            cols.append(c)
    return sorted(cols)


def _prepare_stats_frame(path_stats: str) -> pd.DataFrame:
    if path_stats.lower().endswith(".parquet"):
        m = pd.read_parquet(path_stats)
    else:
        m = pd.read_csv(path_stats)
    m = m.copy()
    m["activity_date"] = cast(pd.Series, pd.to_datetime(m["activity_date"], errors="coerce")).dt.normalize()
    m["ab_group_id"] = m["ab_group_id"].astype(str).str.strip().str.strip('"')
    prefix = m["ab_group_id"].str[:2].str.lower()
    cutoff_raw = CUTOFF_DATE
    if cutoff_raw is not None and str(cutoff_raw).strip():
        cutoff_dt = pd.Timestamp(pd.to_datetime(cutoff_raw).normalize())
        # On or after cutoff calendar day → after spin thresholds (70/70); strictly before → before (130/100).
        is_after = (m["activity_date"] >= cutoff_dt).to_numpy(dtype=bool)
    else:
        is_after = np.zeros(len(m), dtype=bool)

    spin_before = np.select(
        [prefix == "4f", prefix == "4a"],
        [_SPIN_BEFORE_4F, _SPIN_BEFORE_4A],
        default=np.nan,
    )
    spin_after = np.select(
        [prefix == "4f", prefix == "4a"],
        [_SPIN_AFTER_4F, _SPIN_AFTER_4A],
        default=np.nan,
    )
    m["spin_thresh"] = np.where(is_after, spin_after, spin_before)
    m["rtp_thresh"] = np.select(
        [prefix == "4f", prefix == "4a"],
        [_RTP_4F, _RTP_4A],
        default=np.nan,
    )
    m = cast(pd.DataFrame, m.loc[m["spin_thresh"].notna()].copy())
    m["spin_count_base"] = np.where(m["bet_type"] == "BASE", m["spin_count"], 0)
    return m


def _aggregate_from_prepared(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate prepared frame into daily summary and per-user-day stats."""
    m = m.copy()
    _inc = cast(pd.Series, pd.to_numeric(cast(pd.Series, m["incentivized"]), errors="coerce")).fillna(0)
    m["row_non_base"] = m["bet_type"] != "BASE"
    m["row_fg_non_incentive"] = m["row_non_base"] & (_inc == 0)

    gcols = ["activity_date", "ab_group_id", "user_id"]
    user_day = cast(
        pd.DataFrame,
        m.groupby(gcols, as_index=False).agg(
            total_spin=("spin_count_base", "sum"),
            has_any_non_base=("row_non_base", "any"),
            has_fg=("row_fg_non_incentive", "any"),
            has_incentivized=("incentivized", lambda s: bool((cast(pd.Series, s) == 1).any())),
            spin_thresh=("spin_thresh", "first"),
            rtp_thresh=("rtp_thresh", "first"),
        ),
    )
    rtp_base = cast(
        pd.DataFrame,
        m.loc[m["bet_type"] == "BASE"].groupby(gcols, as_index=False).agg(rtp=("rtp", "mean")),
    )
    user_day = cast(pd.DataFrame, user_day.merge(rtp_base, on=gcols, how="left"))

    user_day["low_spin"] = user_day["total_spin"] < user_day["spin_thresh"]
    user_day["high_rtp"] = (user_day["total_spin"] >= user_day["spin_thresh"]) & (
        user_day["rtp"] > user_day["rtp_thresh"]
    )
    user_day["pass_incentive_thresh"] = (user_day["total_spin"] > user_day["spin_thresh"]) & (
        user_day["rtp"] > user_day["rtp_thresh"]
    )
    user_day["high_rtp_incentivized"] = user_day["high_rtp"] & user_day["has_incentivized"]
    user_day["no_fg_not_incentivized"] = (~user_day["has_any_non_base"]) & (~user_day["has_incentivized"])

    summary = user_day.groupby(["activity_date", "ab_group_id"], as_index=False).agg(
        num_users=("user_id", "count"),
        num_users_with_fg=("has_fg", "sum"),
        num_users_low_spin=("low_spin", "sum"),
        num_users_high_rtp=("high_rtp", "sum"),
        num_users_incentivized=("has_incentivized", "sum"),
        num_users_high_rtp_incentivized=("high_rtp_incentivized", "sum"),
        num_user_pass_incentive_thresh=("pass_incentive_thresh", "sum"),
        num_users_no_fg_not_incentivized=("no_fg_not_incentivized", "sum"),
    )
    med_below = cast(
        pd.DataFrame,
        user_day[user_day["low_spin"]]
        .groupby(["activity_date", "ab_group_id"], as_index=False)
        .agg(
            median_spin_below_thresh=("total_spin", "median"),
        ),
    )
    summary = cast(pd.DataFrame, summary.merge(med_below, on=["activity_date", "ab_group_id"], how="left"))

    pure = user_day[user_day["no_fg_not_incentivized"]]
    pure_stats = cast(
        pd.DataFrame,
        pure.groupby(["activity_date", "ab_group_id"], as_index=False).agg(
            mean_rtp_no_fg_not_incentivized=("rtp", "mean"),
            avg_base_spin_no_fg_not_incentivized=("total_spin", "mean"),
        ),
    )
    summary = cast(pd.DataFrame, summary.merge(pure_stats, on=["activity_date", "ab_group_id"], how="left"))

    summary = summary.sort_values(["activity_date", "ab_group_id"]).reset_index(
        drop=True
    )  # pyright: ignore[reportCallIssue]

    nu = summary["num_users"].replace(0, np.nan)
    summary["ratio_users_with_fg"] = summary["num_users_with_fg"] / nu
    summary["num_user_with_fg_ratio"] = summary["num_users_with_fg"] / nu
    summary["ratio_users_low_spin"] = summary["num_users_low_spin"] / nu
    summary["ratio_users_high_rtp"] = summary["num_users_high_rtp"] / nu
    summary["ratio_users_incentivized"] = summary["num_users_incentivized"] / nu
    summary["ratio_users_high_rtp_incentivized"] = summary["num_users_high_rtp_incentivized"] / nu
    summary["num_user_pass_incentive_thresh_ratio"] = summary["num_user_pass_incentive_thresh"] / nu
    summary["ratio_users_no_fg_not_incentivized"] = summary["num_users_no_fg_not_incentivized"] / nu

    return summary, user_day


def analyze(path_stats: str = file1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged extract and aggregate. See module docstring for column definitions.

    Uses module ``CUTOFF_DATE`` (``None`` or ``""`` means no before/after split).

    Returns ``(summary, user_day)``.
    """
    cutoff_date: str | None = CUTOFF_DATE
    if cutoff_date is not None and isinstance(cutoff_date, str) and not cutoff_date.strip():
        cutoff_date = None

    m = _prepare_stats_frame(path_stats)
    if cutoff_date is None:
        summary, user_day = _aggregate_from_prepared(m)
        return summary, user_day

    cutoff = pd.Timestamp(pd.to_datetime(cutoff_date).normalize())
    m_before = m.loc[m["activity_date"] < cutoff]
    m_after = m.loc[m["activity_date"] >= cutoff]

    summary_before, user_before = _aggregate_from_prepared(m_before)
    summary_after, user_after = _aggregate_from_prepared(m_after)
    summary_before = summary_before.copy()
    summary_after = summary_after.copy()
    summary_before.insert(0, "period", "before")
    summary_after.insert(0, "period", "after")
    summary = cast(
        pd.DataFrame,
        pd.concat([summary_before, summary_after], ignore_index=True),
    )

    user_before = user_before.copy()
    user_after = user_after.copy()
    user_before["period"] = "before"
    user_after["period"] = "after"
    user_day = cast(pd.DataFrame, pd.concat([user_before, user_after], ignore_index=True))

    return summary, user_day


def rtp_per_user_per_day(user_day: pd.DataFrame) -> pd.DataFrame:
    """One base-game RTP per user per day and AB prefix group (4a / 4f)."""
    base = user_day.dropna(subset=["rtp"]).copy()
    base["ab_group_id"] = base["ab_group_id"].str[:2].str.lower()
    return cast(
        pd.DataFrame,
        base.groupby(["activity_date", "ab_group_id", "user_id"], as_index=False).agg(
            rtp=("rtp", "mean"),
        ),
    )


def rtp_for_histogram(user_day: pd.DataFrame) -> pd.DataFrame:
    """Per-user-per-day RTP with 2-char arm; includes ``period`` when present (for before/after plot)."""
    base = user_day.dropna(subset=["rtp"]).copy()
    base["ab_group_id"] = base["ab_group_id"].str[:2].str.lower()
    gcols: list[str] = ["activity_date", "ab_group_id", "user_id"]
    if "period" in base.columns:
        gcols = ["period"] + gcols
    return cast(
        pd.DataFrame,
        base.groupby(gcols, as_index=False).agg(rtp=("rtp", "mean")),
    )


def save_rtp_histplot_by_day(plot_df: pd.DataFrame, out_path: Path, *, bins: int = HIST_BINS) -> None:
    """Facet by day and AB arm (legacy path when ``period`` is absent from ``plot_df``)."""
    if plot_df.empty:
        return
    df = plot_df.copy()
    df["activity_date"] = df["activity_date"].astype(str)
    ab_cats = sorted(df["ab_group_id"].astype(str).str.lower().unique())
    df["ab_group"] = pd.Categorical(
        df["ab_group_id"].astype(str).str.lower(),
        categories=ab_cats,
        ordered=True,
    )
    g = sns.displot(
        df,
        x="rtp",
        row="ab_group_id",
        col="activity_date",
        kind="hist",
        bins=bins,
        height=2.8,
        aspect=1.0,
        facet_kws={"margin_titles": True, "sharex": True, "sharey": False},
    )
    g.set_axis_labels("Base-game RTP (mean)", "Users")
    g.figure.suptitle(
        "Per-user base-game RTP (bet_type = 'BASE') by day and AB group (4a vs 4f)",
        y=1.02,
    )
    g.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")


def save_rtp_histplot_before_after(plot_df: pd.DataFrame, out_path: Path, *, bins: int = HIST_BINS) -> None:
    """Overlay before/after cutoff RTP distributions per AB arm; y-axis log-scaled (counts)."""
    if plot_df.empty:
        return
    df = plot_df.copy()
    period_cats = [p for p in ("before", "after") if p in set(df["period"].astype(str).unique())]
    df["period"] = pd.Categorical(df["period"].astype(str), categories=period_cats, ordered=True)
    ab_cats = sorted(df["ab_group_id"].astype(str).str.lower().unique())
    df["ab_group_id"] = pd.Categorical(
        df["ab_group_id"].astype(str).str.lower(),
        categories=ab_cats,
        ordered=True,
    )
    g = sns.displot(
        df,
        x="rtp",
        hue="period",
        col="ab_group_id",
        kind="hist",
        bins=bins,
        height=4.0,
        aspect=1.15,
        multiple="layer",
        element="bars",
        alpha=0.55,
        palette="Set2",
        facet_kws={"margin_titles": True, "sharex": True, "sharey": True},
    )
    g.set_axis_labels("Base-game RTP (mean)", "User-days (log scale)")
    for ax in g.axes.flat:
        ax.set_yscale("log")
        # Avoid log(0): counts are integers ≥ 1 when visible; keep a sensible floor
        ax.set_ylim(bottom=0.8)
    cd = _cutoff_date_display()
    subtitle = f"before: < {cd}; after: ≥ {cd}" if cd else ""
    title = "Per-user-day base-game RTP: before vs after cutoff (overlay), by AB arm"
    if subtitle:
        title += "\n" + subtitle
    g.figure.suptitle(title, y=1.04)
    g.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")


def _escape_json_for_html_script(s: str) -> str:
    """Avoid closing a ``</script>`` tag when embedding JSON in HTML."""
    return s.replace("</", "<\\/")


def save_analysis_html_report(
    summary: pd.DataFrame,
    rtp_figure_path: Path,
    out_path: Path,
    *,
    cutoff_date: str | None = None,
) -> Path:
    """Write a self-contained HTML page: RTP histogram image + interactive line chart (Plotly.js).

    Line chart Y-axis can be any numeric column from ``_aggregate_summary_for_ab_arm`` except
    grouping fields (``period``, ``activity_date``, ``ab_group_id``, ``ab_arm``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rtp_b64 = ""
    if rtp_figure_path.is_file():
        rtp_b64 = base64.b64encode(rtp_figure_path.read_bytes()).decode("ascii")

    plot_df = _aggregate_summary_for_ab_arm(summary)
    metrics = _line_plot_metric_columns(plot_df)
    data_json = plot_df.to_json(orient="records", date_format="iso", date_unit="s")
    if data_json is None or data_json == "null":
        data_json = "[]"
    metrics_json = json.dumps(metrics)
    cd = cutoff_date if cutoff_date is not None else CUTOFF_DATE
    cutoff_js: str | None = None
    if cd is not None and str(cd).strip():
        cutoff_js = pd.to_datetime(cd).strftime("%Y-%m-%dT00:00:00.000Z")
    cutoff_json = json.dumps(cutoff_js)

    rtp_name = html.escape(rtp_figure_path.name)
    data_safe = _escape_json_for_html_script(data_json)
    img_block = (
        f'<img alt="RTP histogram" src="data:image/png;base64,{rtp_b64}" />'
        if rtp_b64
        else "<p><em>RTP figure file was not found.</em></p>"
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SS03 incentive analysis</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 1rem auto; padding: 0 1rem; }}
section {{ margin-bottom: 2rem; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
select {{ margin-left: 0.5rem; padding: 0.35rem 0.5rem; min-width: 16rem; }}
h1 {{ font-size: 1.35rem; }}
h2 {{ font-size: 1.1rem; margin-top: 0; }}
</style>
</head>
<body>
<h1>SS03 incentive analysis</h1>
<section>
<h2>RTP distribution (base-game RTP)</h2>
<p>Matplotlib export: <code>{rtp_name}</code></p>
{img_block}
</section>
<section>
<h2>Daily metrics by AB arm</h2>
<p>One line per <code>ab_arm</code> (first two characters of <code>ab_group_id</code>). Choose a numeric column.</p>
<label for="metric-select">Metric</label>
<select id="metric-select" aria-label="Metric column"></select>
<div id="line-chart" style="width:100%; min-height:420px;"></div>
</section>
<script>
const RAW = {data_safe};
const METRICS = {metrics_json};
const CUTOFF = {cutoff_json};

/* ---- Daily metrics line chart ---- */
(function () {{
  const sel = document.getElementById("metric-select");
  const chartEl = document.getElementById("line-chart");
  METRICS.forEach(function (m) {{
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  }});
  const defaultMetric = METRICS.indexOf("ratio_users_incentivized") >= 0
    ? "ratio_users_incentivized"
    : METRICS[0];
  if (METRICS.length) {{
    sel.value = defaultMetric;
  }}
  function layoutFor(metric) {{
    const layout = {{
      title: metric,
      xaxis: {{ title: "activity_date" }},
      yaxis: {{ title: metric }},
      hovermode: "closest",
      legend: {{ orientation: "h" }},
    }};
    if (CUTOFF) {{
      layout.shapes = [{{
        type: "line",
        xref: "x",
        yref: "paper",
        x0: CUTOFF,
        x1: CUTOFF,
        y0: 0,
        y1: 1,
        line: {{ color: "#555", width: 2, dash: "dash" }},
      }}];
    }}
    return layout;
  }}
  function tracesFor(metric) {{
    const arms = Array.from(new Set(RAW.map(function (r) {{ return r.ab_arm; }}))).sort();
    return arms.map(function (arm) {{
      const rows = RAW.filter(function (r) {{ return r.ab_arm === arm; }}).sort(function (a, b) {{
        return String(a.activity_date).localeCompare(String(b.activity_date));
      }});
      return {{
        type: "scatter",
        mode: "lines+markers",
        name: arm,
        x: rows.map(function (r) {{ return r.activity_date; }}),
        y: rows.map(function (r) {{ return r[metric]; }}),
      }};
    }});
  }}
  function redraw() {{
    const metric = sel.value;
    Plotly.newPlot("line-chart", tracesFor(metric), layoutFor(metric), {{ responsive: true }});
  }}
  if (METRICS.length) {{
    sel.addEventListener("change", redraw);
    redraw();
  }} else {{
    chartEl.innerHTML = "<p><em>No numeric metrics after aggregating by day and AB arm.</em></p>";
  }}
}})();
</script>
</body>
</html>
"""
    out_path.write_text(page, encoding="utf-8")
    return out_path


def save_ratio_incentivized_lineplot(summary: pd.DataFrame, out_path: Path) -> bool:
    """Line chart: ``ratio_users_incentivized`` vs ``activity_date``, one series per AB arm (first 2 chars).

    Before/after cutoff rows are combined into one continuous series per arm (user-weighted ratio).
    When ``CUTOFF_DATE`` is set, a vertical line marks the cutoff.
    """
    plot_df = _aggregate_summary_for_ab_arm(summary)
    if plot_df.empty or "ratio_users_incentivized" not in plot_df.columns:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="activity_date",
        y="ratio_users_incentivized",
        hue="ab_arm",
        marker="o",
        markersize=4,
        ax=ax,
    )
    ax.set_xlabel("activity_date")
    ax.set_ylabel("ratio_users_incentivized")
    ax.set_title("Share of users with an incentivized row (by day and AB arm, first 2 chars of ab_group_id)")
    ax.grid(True, alpha=0.3)
    cutoff_raw = CUTOFF_DATE
    if cutoff_raw is not None and str(cutoff_raw).strip():
        cd = pd.to_datetime(cutoff_raw).normalize()
        ax.axvline(float(mdates.date2num(cd)), color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return True


def save_ratio_incentivized_local_and_s3(
    summary: pd.DataFrame,
    local_path: Path = DEFAULT_LOCAL_RATIO_INCENTIVE_FIGURE,
    s3_bucket: str = S3_BUCKET,
    s3_key: str = DEFAULT_S3_RATIO_INCENTIVE_FIGURE_KEY,
) -> tuple[Path, str | None, bool]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = save_ratio_incentivized_lineplot(summary, local_path)
    uri = upload_file_to_s3(str(local_path), s3_bucket, s3_key) if wrote else None
    return local_path, uri, wrote


def save_summary_local_and_s3(
    summary: pd.DataFrame,
    local_path: Path = DEFAULT_LOCAL_SUMMARY_CSV,
    s3_bucket: str = S3_BUCKET,
    s3_key: str = DEFAULT_S3_SUMMARY_KEY,
) -> tuple[Path, str | None]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(local_path, index=False)
    uri = upload_file_to_s3(str(local_path), s3_bucket, s3_key)
    return local_path, uri


def save_figure_local_and_s3(
    plot_df: pd.DataFrame,
    local_path: Path = DEFAULT_LOCAL_FIGURE,
    s3_bucket: str = S3_BUCKET,
    s3_key: str = DEFAULT_S3_FIGURE_KEY,
    *,
    bins: int = HIST_BINS,
    use_period_plot: bool = True,
) -> tuple[Path, str | None]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if use_period_plot and "period" in plot_df.columns:
        save_rtp_histplot_before_after(plot_df, local_path, bins=bins)
    else:
        save_rtp_histplot_by_day(plot_df, local_path, bins=bins)
    uri = upload_file_to_s3(str(local_path), s3_bucket, s3_key)
    return local_path, uri


def main() -> pd.DataFrame:
    summary, _ = analyze()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS03 incentive stats summary and RTP histogram.")
    parser.add_argument(
        "--stats-file",
        default=file1,
        help="Path to merged incentive stats file (parquet or csv).",
    )
    args = parser.parse_args()

    summary, user_day = analyze(args.stats_file)
    print(summary.to_string(index=False))

    csv_path, csv_uri = save_summary_local_and_s3(summary)
    print(f"Wrote summary CSV to {csv_path.resolve()}")
    if csv_uri:
        print(f"Uploaded summary CSV to {csv_uri}")
    else:
        print("Summary CSV S3 upload failed (see logs); local CSV was still saved.")

    plot_df = rtp_for_histogram(user_day)
    if plot_df.empty:
        print("No base-game RTP values to plot; skipping figure.")
        raise SystemExit(0)

    use_period = "period" in user_day.columns and "period" in plot_df.columns
    local_path, uri = save_figure_local_and_s3(
        plot_df,
        bins=HIST_BINS,
        use_period_plot=use_period,
    )
    print(f"Wrote histogram to {local_path.resolve()}")
    if uri:
        print(f"Uploaded histogram to {uri}")
    else:
        print("S3 upload failed or skipped (see logs); local file was still saved.")

    html_path = save_analysis_html_report(
        summary,
        local_path,
        DEFAULT_LOCAL_HTML_REPORT,
        cutoff_date=CUTOFF_DATE,
    )
    print(f"Wrote HTML report to {html_path.resolve()}")

    ratio_path, ratio_uri, ratio_wrote = save_ratio_incentivized_local_and_s3(summary)
    if ratio_wrote:
        print(f"Wrote ratio incentivized line plot to {ratio_path.resolve()}")
        if ratio_uri:
            print(f"Uploaded ratio incentivized line plot to {ratio_uri}")
        else:
            print("S3 upload failed or skipped (see logs); local ratio plot was still saved.")
    else:
        print("Skipped ratio incentivized line plot (no summary data).")
