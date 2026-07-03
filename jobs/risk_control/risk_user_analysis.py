"""
Risk vs control analysis from ETL aggregate outputs.

Loads the per-user aggregate datasets written by ``etl_risk_user_aggregates.py``
and ``etl_control_user_aggregates.py`` (``user_summary`` + ``category_counts``,
partitioned by group — see ``aggregate_queries.py`` for the schema), then
writes one interactive HTML (dual charts) and one Markdown summary under
``<repo_parent>/data_fishhunter``. Raw bullet events never leave Redshift.

Optional: ``analyze_user_by_name`` can still build per-user markdown + plots
when called from code.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import awswrangler as wr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from bituslabs_ds.config import DEFAULT_ETL_OUTPUT, LOCAL_ROOT, setup_logging
from bituslabs_ds.utils import get_ip_location, get_ip_locations

logger = logging.getLogger(__name__)

S3_RISK_AGG_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/risk_user_agg"
S3_CONTROL_AGG_PREFIX = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/control_user_agg"
DATA_ROOT = LOCAL_ROOT.parent / "data_fishhunter"
CACHE_DIR = DATA_ROOT / "risk_control_cache"
REPORT_DIR = DATA_ROOT / "risk_control_reports"
PLOT_DIR = REPORT_DIR / "plots"
HTML_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _json_for_html_embed(obj: Any) -> str:
    """Serialize JSON for embedding in HTML; avoid breaking out of script/context."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _render_html_from_template(template_name: str, replacements: dict[str, str]) -> str:
    path = HTML_TEMPLATES_DIR / template_name
    text = path.read_text(encoding="utf-8")
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text


CITY_LEGEND_TOP_N = 20
ALL_USERS_MD_NAME = "risk_user_all_users.md"
ALL_USERS_HTML_NAME = "risk_user_all_users_stats.html"
CITY_COLOR_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]
CITY_PATTERN_PALETTE = ["", "/", "\\", "x", "-", "|", "+", "."]
MAX_STRATEGY_CATEGORIES = 40
MAX_FISH_VALUE_CATEGORIES = 60
MAX_MULTIPLIER_BULLET_CATEGORIES = 60

NUMERIC_SORT_METRICS = {"bullet_level", "fish_value"}


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value).strip("_")


def _load_aggregates(
    prefix: str,
    cache_stem: str,
    group_col: str,
    *,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (user_summary, category_counts) for one pipeline side.

    Reads the partitioned datasets with ``wr.s3.read_parquet(dataset=True)``
    (the group column lives in the hive partition path) and keeps a plain
    local parquet cache per dataset under ``CACHE_DIR``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for name in ("user_summary", "category_counts"):
        cache_file = CACHE_DIR / f"{cache_stem}_{name}.parquet"
        if cache_file.exists() and not refresh_cache:
            df = pd.read_parquet(cache_file)
        else:
            df = wr.s3.read_parquet(path=f"{prefix}/{name}/", dataset=True)
            df.to_parquet(cache_file, index=False)
        frames[name] = df

    summary, counts = frames["user_summary"], frames["category_counts"]
    if group_col not in summary.columns or group_col not in counts.columns:
        raise ValueError(f"Missing '{group_col}' column under {prefix}. Please rerun the aggregate ETL.")

    summary[group_col] = summary[group_col].astype(str)
    summary["user_name"] = summary["user_name"].astype(str)
    for col in ("first_event_ts", "last_event_ts"):
        summary[col] = pd.to_datetime(summary[col], errors="coerce")
    for col in ("avg_bet_interval_s", "median_bet_interval_s"):
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0.0)
    duration = (summary["last_event_ts"] - summary["first_event_ts"]).dt.total_seconds() / 3600.0
    summary["account_duration_hours"] = duration.fillna(0.0)

    counts[group_col] = counts[group_col].astype(str)
    counts["user_name"] = counts["user_name"].astype(str)
    counts["category"] = counts["category"].astype(str)
    counts["n"] = pd.to_numeric(counts["n"], errors="raise").astype(int)
    return summary, counts


def _user_metric_counts(cat_df: pd.DataFrame, user_name: str, metric: str) -> pd.Series:
    """Per-user category counts for one metric (index=category, values=n).

    Numeric metrics sort by category value ascending (like the old
    ``value_counts().sort_index()``); the rest sort by count descending.
    """
    rows = cat_df.loc[(cat_df["user_name"] == user_name) & (cat_df["metric"] == metric)]
    s = pd.Series(rows["n"].to_numpy(), index=rows["category"].to_numpy(), dtype="int64")
    if metric in NUMERIC_SORT_METRICS:
        return s.sort_index(key=lambda idx: idx.astype(int))
    return s.sort_values(ascending=False)


def _time_stats_from_summary(row: pd.Series) -> dict[str, float | int]:
    return {
        "account_duration_hours": float(row["account_duration_hours"]),
        "average_bet_interval_seconds": float(row["avg_bet_interval_s"]),
        "median_bet_interval_seconds": float(row["median_bet_interval_s"]),
        "num_bet_sessions": int(row["n_bet_sessions"]),
    }


def _location_to_city(location: str) -> str:
    if location in {"N/A", "Timeout/Failed", "", "UNKNOWN"}:
        return "Unknown"
    parts = [p for p in str(location).split(" ") if p and p != "None"]
    return parts[-1] if parts else "Unknown"


def _city_or_ip_label(ip: str, location: str) -> str:
    city = _location_to_city(location)
    if city == "Unknown":
        return f"IP {ip}"
    return city


def _city_style_map(cities: list[str]) -> dict[str, dict[str, str]]:
    unique_cities = sorted(set(cities))
    styles: dict[str, dict[str, str]] = {}
    for i, city in enumerate(unique_cities):
        styles[city] = {
            "color": CITY_COLOR_PALETTE[i % len(CITY_COLOR_PALETTE)],
            "pattern": CITY_PATTERN_PALETTE[i % len(CITY_PATTERN_PALETTE)],
        }
    return styles


def _ip_locations(ips: list[str]) -> dict[str, str]:
    """Bulk ip->location map (batch endpoint; 'UNKNOWN' maps to 'N/A')."""
    loc_map = get_ip_locations(ip for ip in ips if ip != "UNKNOWN")
    if "UNKNOWN" in ips:
        loc_map["UNKNOWN"] = "N/A"
    return loc_map


def _ip_counts_to_city_counts(ip_counts: pd.Series) -> pd.Series:
    """Aggregate per-ip counts into per-city counts (count-desc)."""
    loc_map = _ip_locations([str(ip) for ip in ip_counts.index])
    cities = pd.Series([_city_or_ip_label(str(ip), loc_map[str(ip)]) for ip in ip_counts.index], index=ip_counts.index)
    return ip_counts.groupby(cities).sum().sort_values(ascending=False)


def _ordered_user_names(summary_df: pd.DataFrame, user_names: list[str] | None) -> list[str]:
    if user_names is None:
        return sorted(summary_df["user_name"].dropna().unique().tolist())
    requested = [str(u) for u in user_names]
    present = set(summary_df["user_name"].dropna().astype(str).unique().tolist())
    return [u for u in requested if u in present]


def _counts_matrix(
    cat_df: pd.DataFrame,
    user_names: list[str],
    metric: str,
) -> dict[str, Any]:
    df = cast(pd.DataFrame, cat_df.loc[cat_df["user_name"].isin(user_names) & (cat_df["metric"] == metric)].copy())
    if df.empty:
        return {"labels": [], "traces": [], "showlegend": True}

    df["_metric"] = df["category"]
    if metric == "bullet_level":
        categories = sorted(df["_metric"].unique().tolist(), key=lambda x: int(x))
    elif metric == "strategy_name":
        top_metric = df.groupby("_metric")["n"].sum().sort_values(ascending=False).head(MAX_STRATEGY_CATEGORIES)
        categories = [str(x) for x in top_metric.index.tolist()]
        df = cast(pd.DataFrame, df.loc[df["_metric"].isin(categories)].copy())
    elif metric == "fish_value":
        top_metric = df.groupby("_metric")["n"].sum().sort_values(ascending=False).head(MAX_FISH_VALUE_CATEGORIES)
        categories = sorted([str(x) for x in top_metric.index.tolist()], key=lambda x: int(x))
        df = cast(pd.DataFrame, df.loc[df["_metric"].isin(categories)].copy())
    elif metric == "multiplier_x_bullet":
        top_metric = (
            df.groupby("_metric")["n"].sum().sort_values(ascending=False).head(MAX_MULTIPLIER_BULLET_CATEGORIES)
        )
        categories = [str(x) for x in top_metric.index.tolist()]
        df = cast(pd.DataFrame, df.loc[df["_metric"].isin(categories)].copy())
    elif metric == "ip":
        ip_values = [str(v) for v in df["category"].unique().tolist()]
        loc_map = _ip_locations(ip_values)
        city_map = {ip: _city_or_ip_label(ip, loc_map[ip]) for ip in ip_values}
        df["_city"] = df["category"].apply(lambda ip: city_map.get(str(ip), "Unknown"))
        city_totals = df.groupby("_city")["n"].sum().sort_values(ascending=False)
        top_cities = [str(city) for city in city_totals.head(CITY_LEGEND_TOP_N).index.tolist()]
        df["_metric"] = df["_city"].apply(lambda city: city if str(city) in top_cities else "Others")
        categories = top_cities.copy()
        if bool((df["_metric"] == "Others").any()):
            categories.append("Others")
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    pivot_raw = df.pivot_table(index="_metric", columns="user_name", values="n", aggfunc="sum", fill_value=0)
    pivot_idx = cast(pd.DataFrame, cast(Any, pivot_raw).reindex(index=categories, fill_value=0))
    pivot = cast(pd.DataFrame, cast(Any, pivot_idx).reindex(user_names, axis=1, fill_value=0))

    traces: list[dict[str, Any]] = []
    city_styles: dict[str, dict[str, str]] = {}
    if metric == "ip":
        city_styles = _city_style_map(categories)

    for label in categories:
        y_vals = [int(v) for v in pivot.loc[label].tolist()]
        trace: dict[str, Any] = {"name": label, "x": user_names, "y": y_vals, "type": "bar"}
        if metric == "ip":
            style = city_styles.get(label, {"color": "#BAB0AC", "pattern": ""})
            trace["marker"] = {"color": style["color"], "pattern": {"shape": style["pattern"]}}
            trace["legendgroup"] = label
            trace["hovertemplate"] = "city=%{fullData.name}<br>" "user=%{x}<br>" "count=%{y}<extra></extra>"
        traces.append(trace)

    return {"labels": categories, "traces": traces, "showlegend": True}


def _plot_user_distributions(
    counts_by_metric: dict[str, pd.Series],
    target_name: str,
    user_dir: Path,
) -> Path:
    user_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(target_name)
    plot_path = user_dir / f"{safe_name}_distributions.png"

    fig, axes = plt.subplots(5, 1, figsize=(16, 28), dpi=120)
    fig.subplots_adjust(hspace=0.45)

    def _plot_count_bar(ax: Any, counts: pd.Series, *, title: str, color: str, x_label: str, rotate_x: int = 0) -> None:
        ax.bar(counts.index.astype(str), counts.values, color=color)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        if rotate_x:
            ax.tick_params(axis="x", rotation=rotate_x)

    _plot_count_bar(
        axes[0],
        counts_by_metric["bullet_level"],
        title="Bullet Level Count",
        color="#3266ad",
        x_label="Bullet Level",
    )
    _plot_count_bar(
        axes[1],
        counts_by_metric["strategy_name"],
        title="Strategy Name Count",
        color="#6dac4f",
        x_label="Strategy Name",
        rotate_x=20,
    )

    city_counts = _ip_counts_to_city_counts(counts_by_metric["ip"])
    top_city_counts = city_counts.head(CITY_LEGEND_TOP_N)
    other_city_count = int(city_counts.iloc[CITY_LEGEND_TOP_N:].sum()) if len(city_counts) > CITY_LEGEND_TOP_N else 0

    city_stack_rows: list[dict[str, Any]] = [
        {"city": str(city), "count": int(count)} for city, count in top_city_counts.items()
    ]
    if other_city_count > 0:
        city_stack_rows.append({"city": "Others", "count": other_city_count})

    city_stack_df = pd.DataFrame(city_stack_rows)
    city_styles = _city_style_map(city_stack_df["city"].tolist() if not city_stack_df.empty else [])
    left = 0
    for _, row in city_stack_df.iterrows():
        count = int(row["count"])
        city = str(row["city"])
        style = city_styles.get(city, {"color": "#BAB0AC", "pattern": ""})
        axes[2].barh(
            [target_name],
            [count],
            left=left,
            color=style["color"],
            hatch=style["pattern"],
            edgecolor="#222222",
            linewidth=0.3,
            height=0.55,
        )
        left += count
    axes[2].set_title("IP Location Count (Top Cities + Others)")
    axes[2].set_xlabel("Count")
    legend_handles = [
        mpatches.Patch(
            facecolor=style["color"],
            hatch=style["pattern"],
            edgecolor="#222222",
            linewidth=0.3,
            label=city,
        )
        for city, style in city_styles.items()
    ]
    if legend_handles:
        axes[2].legend(handles=legend_handles, title="City", loc="upper right")

    _plot_count_bar(
        axes[3],
        counts_by_metric["fish_value"],
        title="Fish Value Count",
        color="#e07b39",
        x_label="Fish Value",
    )

    combo_counts = counts_by_metric["multiplier_x_bullet"]
    axes[4].bar(combo_counts.index.astype(str), combo_counts.values, color="#b05dc4")
    axes[4].set_title("Multiplier x Bullet Level Count")
    axes[4].set_ylabel("Count")
    axes[4].tick_params(axis="x", rotation=45)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(f"Risk User Distribution Plots: {target_name}", fontsize=16)
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def analyze_user_by_name(
    target_name: str,
    summary_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    report_root: Path = REPORT_DIR,
) -> dict[str, Any] | None:
    """Build one markdown report + plots for a target user from aggregates."""
    user_rows = summary_df.loc[summary_df["user_name"] == target_name]
    if user_rows.empty:
        logger.warning("No aggregate data found for user_name=%s", target_name)
        return None
    row = user_rows.iloc[0]

    report_root.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    counts_by_metric = {
        m: _user_metric_counts(cat_df, target_name, m)
        for m in ("bullet_level", "strategy_name", "ip", "fish_value", "multiplier_x_bullet")
    }

    safe_name = _sanitize_filename(target_name)
    user_dir = PLOT_DIR / safe_name
    plot_path = _plot_user_distributions(counts_by_metric, target_name, user_dir)

    user_id = str(int(row["user_id"]))
    total_orders = int(row["n_orders"])
    total_bet = float(row["total_bet"])
    total_payout = float(row["total_payout"])
    total_profit = float(row["total_profit"])
    rtp = (total_payout / total_bet) * 100 if total_bet > 0 else 0.0
    time_stats = _time_stats_from_summary(row)

    ip_counts = counts_by_metric["ip"]
    top_ip_table_rows = []
    for ip, count in ip_counts.head(20).items():
        top_ip_table_rows.append((str(ip), get_ip_location(str(ip)), int(count)))

    plot_rel = plot_path.relative_to(report_root)
    report_path = report_root / f"{safe_name}.md"
    lines = [
        f"# Risk User Report: {target_name}",
        "",
        "## Summary",
        f"- user_id: `{user_id}`",
        f"- total_orders: `{total_orders}`",
        f"- total_bet: `{total_bet:,.2f}`",
        f"- total_payout: `{total_payout:,.2f}`",
        f"- total_profit: `{total_profit:,.2f}`",
        f"- actual_rtp: `{rtp:.2f}%`",
        f"- account_duration_hours: `{float(time_stats['account_duration_hours']):.2f}`",
        f"- average_bet_interval_seconds: `{float(time_stats['average_bet_interval_seconds']):.2f}`",
        f"- num_bet_sessions: `{int(time_stats['num_bet_sessions'])}`",
        "",
        "## Distribution Plot",
        f"![{target_name} plots]({plot_rel.as_posix()})",
        "",
        "## Bullet Level Counts",
        "| bullet_level | count |",
        "|---|---:|",
    ]
    lines.extend([f"| {idx} | {int(val)} |" for idx, val in counts_by_metric["bullet_level"].items()])
    lines.extend(
        [
            "",
            "## Strategy Name Counts",
            "| strategy_name | count |",
            "|---|---:|",
        ]
    )
    lines.extend([f"| {idx} | {int(val)} |" for idx, val in counts_by_metric["strategy_name"].items()])
    lines.extend(
        [
            "",
            "## Fish Value Counts",
            "| fish_value | count |",
            "|---|---:|",
        ]
    )
    lines.extend([f"| {idx} | {int(val)} |" for idx, val in counts_by_metric["fish_value"].items()])
    lines.extend(
        [
            "",
            "## Multiplier x Bullet Level Counts",
            "| multiplier_x_bullet | count |",
            "|---|---:|",
        ]
    )
    lines.extend([f"| {label} | {int(count)} |" for label, count in counts_by_metric["multiplier_x_bullet"].items()])
    lines.extend(
        [
            "",
            "## Top IPs with Location",
            "| ip | location | count |",
            "|---|---|---:|",
        ]
    )
    lines.extend([f"| `{ip}` | {loc} | {cnt} |" for ip, loc, cnt in top_ip_table_rows])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "name": target_name,
        "user_id": user_id,
        "total_orders": total_orders,
        "report_path": str(report_path),
        "plot_path": str(plot_path),
    }


def _sort_names_by_profit(summary_df: pd.DataFrame, user_names: list[str]) -> list[str]:
    if not user_names:
        return []
    filtered = summary_df.loc[summary_df["user_name"].isin(user_names)]
    if filtered.empty:
        return []
    ordered = filtered.sort_values(by="total_profit", ascending=False)
    return [str(name) for name in ordered["user_name"].tolist()]


def _build_group_plot_bundle(
    summary_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    user_names: list[str],
) -> dict[str, Any]:
    metrics = [
        ("bullet_level", "Bullet Level Counts"),
        ("strategy_name", "Strategy Name Counts"),
        ("ip", "IP Counts (Top IPs + OTHERS)"),
        ("fish_value", "Fish Value Counts"),
        ("multiplier_x_bullet", "Multiplier x Bullet Level Counts"),
    ]
    per_user = summary_df.set_index("user_name").reindex(user_names)
    bet = per_user["total_bet"].fillna(0.0)
    payout = per_user["total_payout"].fillna(0.0)
    rtp_vals = [(float(p) / float(b)) * 100 if float(b) > 0 else 0.0 for b, p in zip(bet, payout)]
    profit_vals = [float(v) for v in per_user["total_profit"].fillna(0.0)]
    account_duration_vals = [float(v) for v in per_user["account_duration_hours"].fillna(0.0)]
    avg_bet_interval_vals = [float(v) for v in per_user["avg_bet_interval_s"].fillna(0.0)]
    median_bet_interval_vals = [float(v) for v in per_user["median_bet_interval_s"].fillna(0.0)]
    num_sessions_vals = [int(v) for v in per_user["n_bet_sessions"].fillna(0)]

    line_payload = {
        "none": {"name": "None", "y": []},
        "rtp": {"name": "RTP %", "y": rtp_vals},
        "total_profit": {"name": "Total Profit", "y": profit_vals},
        "account_duration_hours": {"name": "Account Duration (Hours)", "y": account_duration_vals},
        "average_bet_interval_seconds": {"name": "Average Bet Interval (Seconds)", "y": avg_bet_interval_vals},
        "median_bet_interval_seconds": {"name": "Median Bet Interval (Seconds)", "y": median_bet_interval_vals},
        "num_bet_sessions": {"name": "Number of Bet Sessions", "y": num_sessions_vals},
    }
    payload: dict[str, Any] = {}
    for metric_key, metric_title in metrics:
        matrix = _counts_matrix(cat_df, user_names, metric_key)
        payload[metric_key] = {
            "title": metric_title,
            "data": matrix["traces"],
            "layout": {
                "barmode": "stack",
                "xaxis": {"title": "User Name", "tickangle": -35},
                "yaxis": {"title": "Count"},
                "showlegend": bool(matrix["showlegend"]),
                "margin": {"l": 60, "r": 30, "t": 70, "b": 140},
            },
        }
    return {"payload": payload, "line_payload": line_payload, "user_names": user_names}


def _append_markdown_group_section(
    lines: list[str],
    summary_df: pd.DataFrame,
    user_names: list[str],
    section_title: str,
) -> None:
    group_summary = summary_df.loc[summary_df["user_name"].isin(user_names)]
    lines.extend(
        [
            "",
            f"## {section_title}",
            "",
            f"- users_included: `{len(user_names)}`",
            f"- total_rows: `{int(group_summary['n_orders'].sum())}`",
            "",
        ]
    )
    if group_summary.empty:
        lines.append("- no matching data rows found")
        return
    ordered = group_summary.sort_values(by="total_profit", ascending=False)
    lines.append("| user_name | user_id | total_orders | total_bet | total_payout | total_profit | rtp | sessions |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in ordered.iterrows():
        total_bet = float(row["total_bet"])
        rtp = (float(row["total_payout"]) / total_bet) * 100 if total_bet > 0 else 0.0
        lines.append(
            f"| {str(row['user_name'])} | {str(int(row['user_id']))} | {int(row['n_orders'])} | "
            f"{total_bet:,.2f} | {float(row['total_payout']):,.2f} | {float(row['total_profit']):,.2f} | "
            f"{rtp:.2f}% | {int(row['n_bet_sessions'])} |"
        )


def _discover_groups(df: pd.DataFrame, group_col: str) -> list[str]:
    """Return ``groupN`` values present in ``group_col``, sorted by N ascending."""
    if group_col not in df.columns or df.empty:
        return []
    unique = {str(g) for g in df[group_col].dropna().unique().tolist()}
    return sorted(
        unique,
        key=lambda g: (0, int(g[len("group") :])) if g.startswith("group") and g[len("group") :].isdigit() else (1, g),
    )


def generate_risk_control_reports(
    *,
    refresh_cache: bool = False,
    risk_agg_prefix: str = S3_RISK_AGG_PREFIX,
    control_agg_prefix: str = S3_CONTROL_AGG_PREFIX,
) -> list[dict[str, Any]]:
    risk_summary, risk_counts = _load_aggregates(
        risk_agg_prefix, "risk", "risk_user_group", refresh_cache=refresh_cache
    )
    control_summary, control_counts = _load_aggregates(
        control_agg_prefix, "control", "control_user_group", refresh_cache=refresh_cache
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    risk_groups = _discover_groups(risk_summary, "risk_user_group")
    control_groups = _discover_groups(control_summary, "control_user_group")

    def _group_payload(
        summary_df: pd.DataFrame,
        counts_df: pd.DataFrame,
        groups: list[str],
        group_col: str,
    ) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for group in groups:
            g_summary = summary_df.loc[summary_df[group_col] == group]
            g_counts = counts_df.loc[counts_df[group_col] == group]
            names = _sort_names_by_profit(g_summary, _ordered_user_names(g_summary, None))
            payload[group] = _build_group_plot_bundle(g_summary, g_counts, names)
        return payload

    risk_group_payload = _group_payload(risk_summary, risk_counts, risk_groups, "risk_user_group")
    control_group_payload = _group_payload(control_summary, control_counts, control_groups, "control_user_group")

    html_path = REPORT_DIR / ALL_USERS_HTML_NAME
    md_path = REPORT_DIR / ALL_USERS_MD_NAME

    bootstrap = {
        "risk_payload": risk_group_payload,
        "control_payload": control_group_payload,
        "risk_groups": risk_groups,
        "control_groups": control_groups,
    }
    html = _render_html_from_template(
        "risk_vs_control_stats.html",
        {"__BOOTSTRAP_JSON__": _json_for_html_embed(bootstrap)},
    )
    html_path.write_text(html, encoding="utf-8")

    lines = [
        "# Risk vs Control User Report",
        "",
        "- note: group labels come from ETL aggregate partitions `risk_user_group` and `control_user_group`",
        f"- risk_groups: `{', '.join(risk_groups) or '(none)'}`",
        f"- control_groups: `{', '.join(control_groups) or '(none)'}`",
        f"- html_stats_report: `{html_path.name}`",
        "",
        f"[Open interactive chart report]({html_path.name})",
        "",
    ]
    for group in risk_groups:
        names = risk_group_payload[group]["user_names"]
        _append_markdown_group_section(lines, risk_summary, names, f"Risk {group}")
    for group in control_groups:
        names = control_group_payload[group]["user_names"]
        _append_markdown_group_section(lines, control_summary, names, f"Control {group}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Generated combined markdown report: %s", md_path)
    logger.info("Generated combined html report: %s", html_path)
    summary: dict[str, Any] = {
        "markdown_report": str(md_path),
        "html_report": str(html_path),
    }
    for group in risk_groups:
        summary[f"risk_{group}_users"] = len(risk_group_payload[group]["user_names"])
    for group in control_groups:
        summary[f"control_{group}_users"] = len(control_group_payload[group]["user_names"])
    return [summary]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate risk vs control dual-chart HTML and Markdown reports.")
    parser.add_argument(
        "--refresh-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload aggregates from S3 instead of using the local cache (default: false).",
    )
    parser.add_argument(
        "--risk-agg-prefix",
        default=S3_RISK_AGG_PREFIX,
        help="S3 prefix of the risk aggregate datasets (user_summary/ + category_counts/).",
    )
    parser.add_argument(
        "--control-agg-prefix",
        default=S3_CONTROL_AGG_PREFIX,
        help="S3 prefix of the control aggregate datasets (user_summary/ + category_counts/).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )
    generate_risk_control_reports(
        refresh_cache=bool(args.refresh_cache),
        risk_agg_prefix=str(args.risk_agg_prefix),
        control_agg_prefix=str(args.control_agg_prefix),
    )


if __name__ == "__main__":
    main()
