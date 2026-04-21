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
    AB cohort id (UUID string) carried through as a grouping key.
ab_arm
    Lowercased 4-character prefix of ``ab_group_id`` (``4a04`` / ``4f1a``). Emitted by
    ``merge_ab_user_stats.py`` / the Redshift query; drives threshold lookup and UI labels.
num_users
    Count of distinct ``user_id`` for this date and partition.
num_users_with_fg
    Users with at least one **non-incentive** free-game row that day: ``bet_type != 'BASE'``
    and ``incentivized == 0`` (numeric ``0`` / false; missing treated as ``0``). Rows that are
    free game but ``incentivized == 1`` do **not** count toward this metric.
num_users_low_spin
    Users whose **base-game** spin total (``bet_type == 'BASE'`` ``spin_count`` summed)
    is **strictly less** than the arm spin threshold. Before ``CUTOFF_DATE``: 130 for ``4f1a``,
    100 for ``4a04``. On or after the cutoff: **70** for both arms.
num_users_high_rtp
    Users with base-game spin total **greater than or equal** to the arm spin
    threshold **and** mean base-game ``rtp`` **greater than** the arm RTP cutoff
    (0.5 for ``4a04``, 0.6 for ``4f1a``) — same RTP cutoffs before and after the cutoff.
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
    Arm-specific cutoffs: ``spin_thresh`` is 130/100 (``4f1a``/``4a04``) before ``CUTOFF_DATE`` and
    70/70 on or after; ``rtp_thresh`` is 0.6 for ``4f1a`` and 0.5 for ``4a04`` in both periods.
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
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT
from bituslabs_ds.reports import escape_json_for_html_script, load_template, render_page

# file1 = "/Users/niuxin/Documents/data_ss03/ab_cny_bet_rtp_by_user_mathtable.parquet"
file1 = "/Users/niuxin/Documents/data_ss03/merged_user_id_20260401-202604016_with_ab.csv"

OUTPUT_DIR = Path(LOCAL_ROOT) / "jobs" / "output" / "ss03_mahjiang_streak"

# Before/after split: ISO date string, or None / "" for no split.
CUTOFF_DATE: str | None = "2026-04-10"
HIST_BINS = 72

# Spin thresholds by 4-char ``ab_arm`` (``4f1a`` / ``4a04``): before cutoff vs on/after ``CUTOFF_DATE``.
# ``activity_date`` / ``CUTOFF_DATE`` are treated as naive calendar days (e.g. Beijing-time date strings);
# no timezone conversion — compare with ``>=`` / ``<`` on normalized datetimes.
_SPIN_BEFORE_4F1A, _SPIN_BEFORE_4A04 = 130, 100
_SPIN_AFTER_4F1A, _SPIN_AFTER_4A04 = 70, 70
_RTP_4F1A, _RTP_4A04 = 0.6, 0.5

DEFAULT_LOCAL_SUMMARY_CSV = OUTPUT_DIR / "incentive_stats_summary_before_after.csv"
DEFAULT_LOCAL_HTML_REPORT = OUTPUT_DIR / "ss03_incentive_analysis_report.html"
_JS_TEMPLATE_PATH = Path(__file__).with_suffix(".js")

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
    """Aggregate the summary by (``activity_date``, ``ab_arm``); recompute ``ratio_users_*``."""
    if summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    if "period" in df.columns:
        df = df.drop(columns=["period"])
    df["activity_date"] = pd.to_datetime(df["activity_date"])
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
    if "ab_arm" not in m.columns:
        raise ValueError(
            "Input stats file is missing the 'ab_arm' column. Re-run merge_ab_user_stats.py "
            "to regenerate the merged CSV with the new ab_arm column."
        )
    m["activity_date"] = cast(pd.Series, pd.to_datetime(m["activity_date"], errors="coerce")).dt.normalize()
    m["ab_group_id"] = m["ab_group_id"].astype(str).str.strip().str.strip('"')
    m["ab_arm"] = m["ab_arm"].astype(str).str.lower()
    arm = m["ab_arm"]
    cutoff_raw = CUTOFF_DATE
    if cutoff_raw is not None and str(cutoff_raw).strip():
        cutoff_dt = pd.Timestamp(pd.to_datetime(cutoff_raw).normalize())
        # On or after cutoff calendar day → after spin thresholds (70/70); strictly before → before (130/100).
        is_after = (m["activity_date"] >= cutoff_dt).to_numpy(dtype=bool)
    else:
        is_after = np.zeros(len(m), dtype=bool)

    spin_before = np.select(
        [arm == "4f1a", arm == "4a04"],
        [_SPIN_BEFORE_4F1A, _SPIN_BEFORE_4A04],
        default=np.nan,
    )
    spin_after = np.select(
        [arm == "4f1a", arm == "4a04"],
        [_SPIN_AFTER_4F1A, _SPIN_AFTER_4A04],
        default=np.nan,
    )
    m["spin_thresh"] = np.where(is_after, spin_after, spin_before)
    m["rtp_thresh"] = np.select(
        [arm == "4f1a", arm == "4a04"],
        [_RTP_4F1A, _RTP_4A04],
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
            ab_arm=("ab_arm", "first"),
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
        ab_arm=("ab_arm", "first"),
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
    """One base-game RTP per user per day and AB arm (``ab_arm`` from source ETL)."""
    base = user_day.dropna(subset=["rtp"]).copy()
    return cast(
        pd.DataFrame,
        base.groupby(["activity_date", "ab_arm", "user_id"], as_index=False).agg(
            rtp=("rtp", "mean"),
        ),
    )


def rtp_for_histogram(user_day: pd.DataFrame) -> pd.DataFrame:
    """Per-user-per-day RTP grouped by ``ab_arm``; includes ``period`` when present."""
    base = user_day.dropna(subset=["rtp"]).copy()
    gcols: list[str] = ["activity_date", "ab_arm", "user_id"]
    if "period" in base.columns:
        gcols = ["period"] + gcols
    return cast(
        pd.DataFrame,
        base.groupby(gcols, as_index=False).agg(rtp=("rtp", "mean")),
    )


def _prepare_rtp_histogram_payload(
    rtp_df: pd.DataFrame,
) -> tuple[list[dict], list[str], str | None]:
    """Shape per-user-day RTP for in-browser Plotly histograms.

    Returns ``(rows, groups, group_key)`` where:
      - ``rows`` is a list of ``{rtp, ab_arm, <group_key>?}`` records
      - ``groups`` is the display order for traces (e.g. ``["before", "after"]``)
      - ``group_key`` is either ``"period"``, ``"activity_date"``, or ``None``
    """
    if rtp_df.empty:
        return [], [], None
    df = rtp_df.copy()
    df = cast(pd.DataFrame, df.dropna(subset=["rtp"]))

    group_key: str | None
    groups: list[str]
    if "period" in df.columns:
        group_key = "period"
        present = set(df[group_key].astype(str).unique())
        groups = [p for p in ("before", "after") if p in present]
        cols = ["rtp", "ab_arm", "period"]
    elif "activity_date" in df.columns:
        group_key = "activity_date"
        df["activity_date"] = df["activity_date"].astype(str)
        groups = sorted(df["activity_date"].unique().tolist())
        cols = ["rtp", "ab_arm", "activity_date"]
    else:
        group_key = None
        groups = []
        cols = ["rtp", "ab_arm"]

    rows_json = df[cols].to_json(orient="records")
    rows = cast(list[dict], json.loads(rows_json or "[]"))
    return rows, groups, group_key


def save_analysis_html_report(
    summary: pd.DataFrame,
    rtp_df: pd.DataFrame,
    out_path: Path,
    *,
    cutoff_date: str | None = None,
    bins: int = HIST_BINS,
) -> Path:
    """Write a self-contained interactive HTML report (Plotly).

    Sections:
      1. Per-arm RTP histograms (overlaid before/after when ``period`` is present).
      2. Daily metrics line chart (one line per ``ab_arm``; metric selectable).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = _aggregate_summary_for_ab_arm(summary)
    metrics = _line_plot_metric_columns(plot_df)
    line_raw_json = plot_df.to_json(orient="records", date_format="iso", date_unit="s") or "[]"
    if line_raw_json == "null":
        line_raw_json = "[]"

    cd = cutoff_date if cutoff_date is not None else CUTOFF_DATE
    cutoff_js: str | None = None
    if cd is not None and str(cd).strip():
        cutoff_js = pd.to_datetime(cd).strftime("%Y-%m-%dT00:00:00.000Z")

    rtp_rows, rtp_groups, rtp_group_key = _prepare_rtp_histogram_payload(rtp_df)

    inline_js = (
        load_template(_JS_TEMPLATE_PATH)
        .replace('"__LINE_RAW__"', escape_json_for_html_script(line_raw_json))
        .replace('"__LINE_METRICS__"', json.dumps(metrics))
        .replace('"__LINE_CUTOFF__"', json.dumps(cutoff_js))
        .replace('"__RTP_DATA__"', escape_json_for_html_script(json.dumps(rtp_rows)))
        .replace('"__RTP_GROUPS__"', json.dumps(rtp_groups))
        .replace('"__RTP_GROUP_KEY__"', json.dumps(rtp_group_key))
        .replace('"__RTP_BINS__"', str(bins))
    )

    cutoff_label = _cutoff_date_display()
    rtp_subtitle = (
        f"<p>Overlaid histograms per AB arm. Before: <code>activity_date &lt; {cutoff_label}</code>; "
        f"after: <code>activity_date &ge; {cutoff_label}</code>. Y-axis is log-scaled.</p>"
        if rtp_group_key == "period" and cutoff_label
        else "<p>Per-arm histograms of per-user-day base-game RTP. Y-axis is log-scaled.</p>"
    )

    body = f"""<h1>SS03 incentive analysis</h1>
<section>
<h2>Per-user-day base-game RTP distribution</h2>
{rtp_subtitle}
<div id="rtp-hist-container"></div>
</section>
<section>
<h2>Daily metrics by AB arm</h2>
<p>One line per <code>ab_arm</code> (4-char prefix of <code>ab_group_id</code>, e.g. <code>4a04</code>/<code>4f1a</code>). Choose a numeric column.</p>
<label for="metric-select">Metric</label>
<select id="metric-select" aria-label="Metric column"></select>
<div id="line-chart" style="width:100%; min-height:420px;"></div>
</section>"""

    page = render_page(
        title="SS03 incentive analysis",
        body=body,
        inline_scripts=inline_js,
    )
    out_path.write_text(page, encoding="utf-8")
    return out_path


def main() -> pd.DataFrame:
    summary, _ = analyze()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS03 incentive stats summary CSV and interactive HTML report.")
    parser.add_argument(
        "--stats-file",
        default=file1,
        help="Path to merged incentive stats file (parquet or csv).",
    )
    args = parser.parse_args()

    summary, user_day = analyze(args.stats_file)
    print(summary.to_string(index=False))

    DEFAULT_LOCAL_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEFAULT_LOCAL_SUMMARY_CSV, index=False)
    print(f"Wrote summary CSV to {DEFAULT_LOCAL_SUMMARY_CSV.resolve()}")

    rtp_df = rtp_for_histogram(user_day)
    html_path = save_analysis_html_report(
        summary,
        rtp_df,
        DEFAULT_LOCAL_HTML_REPORT,
        cutoff_date=CUTOFF_DATE,
        bins=HIST_BINS,
    )
    print(f"Wrote HTML report to {html_path.resolve()}")
