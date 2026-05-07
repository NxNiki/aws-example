"""
Risk vs control analysis from ETL parquet outputs.

Loads risk and control bullet aggregates, then writes one interactive HTML
(dual charts) and one Markdown summary under ``<repo_parent>/data_fishhunter``.

Optional: ``analyze_user_by_name`` can still build per-user markdown + plots
when called from code.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from bituslabs_ds.config import DEFAULT_ETL_OUTPUT, LOCAL_ROOT, setup_logging
from bituslabs_ds.s3_utils import read_files
from bituslabs_ds.utils import get_ip_location

logger = logging.getLogger(__name__)

S3_RISK_USER_STATS_URI = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/risk_user_stats/risk_user_stats.parquet"
S3_CONTROL_USER_STATS_URI = (
    f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/control_user_stats/control_user_stats.parquet"
)
DATA_ROOT = LOCAL_ROOT.parent / "data_fishhunter"
CACHE_DIR = DATA_ROOT / "risk_control_cache"
REPORT_DIR = DATA_ROOT / "risk_control_reports"
PLOT_DIR = REPORT_DIR / "plots"
LOCAL_CACHE_FILE = CACHE_DIR / "risk_user_stats.parquet"
LOCAL_CONTROL_CACHE_FILE = CACHE_DIR / "control_user_stats.parquet"
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


IP_STACK_TOP_N = 20
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


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value).strip("_")


def _value_counts(
    df: pd.DataFrame,
    column: str,
    *,
    numeric_int: bool = False,
    fill_unknown: bool = False,
    sort_index: bool = False,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="int64")
    s = pd.Series(df[column])
    if fill_unknown:
        s = s.fillna("UNKNOWN")
    else:
        s = s.dropna()
    if numeric_int:
        s = pd.Series(pd.to_numeric(s, errors="coerce")).dropna().astype(int)
    counts = pd.Series(s).value_counts()
    return counts.sort_index() if sort_index else counts


def _combo_counts(df: pd.DataFrame) -> pd.Series:
    if "multiplier" not in df.columns or "bullet_level" not in df.columns:
        return pd.Series(dtype="int64")
    raw_counts = df.dropna(subset=["multiplier", "bullet_level"]).groupby(["multiplier", "bullet_level"]).size()
    return cast(pd.Series, cast(Any, raw_counts).sort_values(ascending=False))


def _plot_count_bar(
    ax: Any,
    counts: pd.Series,
    *,
    title: str,
    color: str,
    x_label: str,
    y_label: str = "Count",
    rotate_x: int = 0,
) -> None:
    ax.bar(counts.index.astype(str), counts.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if rotate_x:
        ax.tick_params(axis="x", rotation=rotate_x)


def _sum_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    s = pd.Series(pd.to_numeric(df[column], errors="coerce")).fillna(0)
    return float(s.sum())


def _user_time_stats(df_user: pd.DataFrame, *, session_gap_seconds: int = 30 * 60) -> dict[str, float | int]:
    if "event_timestamp" not in df_user.columns:
        return {
            "account_duration_hours": 0.0,
            "average_bet_interval_seconds": 0.0,
            "median_bet_interval_seconds": 0.0,
            "num_bet_sessions": 0,
        }

    ts = pd.to_datetime(pd.Series(df_user["event_timestamp"]), errors="coerce").dropna().sort_values()
    if ts.empty:
        return {
            "account_duration_hours": 0.0,
            "average_bet_interval_seconds": 0.0,
            "median_bet_interval_seconds": 0.0,
            "num_bet_sessions": 0,
        }

    duration_hours = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0) if len(ts) > 1 else 0.0
    diffs = ts.diff().dt.total_seconds().dropna()
    valid_diffs = diffs[(diffs >= 0) & (diffs <= session_gap_seconds)]
    avg_interval = float(valid_diffs.mean()) if len(valid_diffs) > 0 else 0.0
    median_interval = float(valid_diffs.median()) if len(valid_diffs) > 0 else 0.0
    num_sessions = int(1 + (diffs > session_gap_seconds).sum())

    return {
        "account_duration_hours": duration_hours,
        "average_bet_interval_seconds": avg_interval,
        "median_bet_interval_seconds": median_interval,
        "num_bet_sessions": num_sessions,
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


def _combo_label_rows(combo_counts: pd.Series) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for idx, c in combo_counts.items():
        if not isinstance(idx, tuple) or len(idx) != 2:
            continue
        m, b = idx
        rows.append((f"{m}x_L{int(float(b))}", int(c)))
    return rows


def _ordered_user_names(events_df: pd.DataFrame, user_names: list[str] | None) -> list[str]:
    if user_names is None:
        return sorted(events_df["user_name"].dropna().unique().tolist())
    requested = [str(u) for u in user_names]
    present = set(events_df["user_name"].dropna().astype(str).unique().tolist())
    return [u for u in requested if u in present]


def _counts_matrix(
    events_df: pd.DataFrame,
    user_names: list[str],
    metric: str,
) -> dict[str, Any]:
    df = cast(pd.DataFrame, events_df.loc[events_df["user_name"].isin(user_names)].copy())
    if df.empty:
        return {"labels": [], "traces": [], "showlegend": True}
    if metric == "bullet_level":
        values = (
            pd.Series(pd.to_numeric(pd.Series(df["bullet_level"]), errors="coerce")).dropna().astype(int).astype(str)
        )
        df = df.loc[values.index].copy()
        df["_metric"] = values
        categories = sorted(df["_metric"].unique().tolist(), key=lambda x: int(x))
        showlegend = True
    elif metric == "strategy_name":
        df["_metric"] = df["strategy_name"].fillna("UNKNOWN").astype(str)
        top_metric = df["_metric"].value_counts().head(MAX_STRATEGY_CATEGORIES)
        categories = [str(x) for x in top_metric.index.tolist()]
        df = cast(pd.DataFrame, df.loc[df["_metric"].isin(categories)].copy())
        showlegend = True
    elif metric == "fish_value":
        values = pd.Series(pd.to_numeric(pd.Series(df["fish_value"]), errors="coerce")).dropna().astype(int).astype(str)
        df = df.loc[values.index].copy()
        df["_metric"] = values
        top_metric = df["_metric"].value_counts().head(MAX_FISH_VALUE_CATEGORIES)
        categories = sorted([str(x) for x in top_metric.index.tolist()], key=lambda x: int(x))
        df = cast(pd.DataFrame, df.loc[df["_metric"].isin(categories)].copy())
        showlegend = True
    elif metric == "multiplier_x_bullet":
        combo_df = df.dropna(subset=["multiplier", "bullet_level"]).copy()
        combo_df["_metric"] = (
            combo_df["multiplier"].astype(str)
            + "x_L"
            + pd.Series(pd.to_numeric(pd.Series(combo_df["bullet_level"]), errors="coerce"))
            .fillna(0)
            .astype(int)
            .astype(str)
        )
        df = combo_df
        top_metric = df["_metric"].value_counts().head(MAX_MULTIPLIER_BULLET_CATEGORIES)
        categories = [str(x) for x in top_metric.index.tolist()]
        df = cast(pd.DataFrame, df.loc[df["_metric"].isin(categories)].copy())
        showlegend = True
    elif metric == "ip":
        df["_ip_value"] = df["ip"].fillna("UNKNOWN").astype(str)
        ip_values = [str(v) for v in df["_ip_value"].dropna().astype(str).unique().tolist()]
        loc_map = {ip: get_ip_location(ip) if ip != "UNKNOWN" else "N/A" for ip in ip_values}
        city_map = {ip: _city_or_ip_label(ip, loc_map[ip]) for ip in ip_values}
        df["_city"] = df["_ip_value"].apply(lambda ip: city_map.get(str(ip), "Unknown"))
        city_totals = df["_city"].value_counts()
        top_cities = [str(city) for city in city_totals.head(CITY_LEGEND_TOP_N).index.tolist()]
        df["_metric"] = df["_city"].apply(lambda city: city if str(city) in top_cities else "Others")
        categories = top_cities.copy()
        has_others = bool((df["_metric"] == "Others").any())
        if has_others:
            categories.append("Others")
        showlegend = True
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    pivot_raw = cast(pd.DataFrame, cast(Any, df.groupby(["_metric", "user_name"]).size().unstack(fill_value=0)))
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

    return {"labels": categories, "traces": traces, "showlegend": showlegend}


def _load_events_df(
    *,
    s3_uri: str = S3_RISK_USER_STATS_URI,
    local_cache_file: Path = LOCAL_CACHE_FILE,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = cast(
        pd.DataFrame,
        read_files(
            files=s3_uri,
            local_cache_path=str(local_cache_file),
            reload=refresh_cache,
            lazy_load=False,
        ),
    )

    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    if "data_date" in df.columns:
        df["data_date"] = pd.to_datetime(df["data_date"], errors="coerce").dt.date

    for col in ("bet", "payout", "profit", "bullet_level", "multiplier", "fish_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _plot_user_distributions(
    df_user: pd.DataFrame,
    target_name: str,
    user_dir: Path,
) -> Path:
    user_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(target_name)
    plot_path = user_dir / f"{safe_name}_distributions.png"

    fig, axes = plt.subplots(5, 1, figsize=(16, 28), dpi=120)
    fig.subplots_adjust(hspace=0.45)

    bullet_counts = _value_counts(df_user, "bullet_level", numeric_int=True, sort_index=True)
    _plot_count_bar(
        axes[0],
        bullet_counts,
        title="Bullet Level Count",
        color="#3266ad",
        x_label="Bullet Level",
    )

    strategy_counts = _value_counts(df_user, "strategy_name", fill_unknown=True)
    _plot_count_bar(
        axes[1],
        strategy_counts,
        title="Strategy Name Count",
        color="#6dac4f",
        x_label="Strategy Name",
        rotate_x=20,
    )

    ip_values = pd.Series(df_user["ip"]).fillna("UNKNOWN").astype(str)
    unique_ips = [str(v) for v in ip_values.unique().tolist()]
    loc_map = {ip: get_ip_location(ip) if ip != "UNKNOWN" else "N/A" for ip in unique_ips}
    city_series = ip_values.apply(lambda ip: _city_or_ip_label(str(ip), loc_map.get(str(ip), "N/A")))
    city_counts = city_series.value_counts()
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

    fish_counts = _value_counts(df_user, "fish_value", numeric_int=True, sort_index=True)
    _plot_count_bar(
        axes[3],
        fish_counts,
        title="Fish Value Count",
        color="#e07b39",
        x_label="Fish Value",
    )

    combo_counts = _combo_counts(df_user)
    combo_rows = _combo_label_rows(combo_counts)
    combo_labels = [label for label, _ in combo_rows]
    combo_values = [count for _, count in combo_rows]
    axes[4].bar(combo_labels, combo_values, color="#b05dc4")
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
    events_df: pd.DataFrame,
    report_root: Path = REPORT_DIR,
) -> dict[str, Any] | None:
    """Build one markdown report + plots for a target user."""
    df_user = cast(pd.DataFrame, events_df.loc[events_df["user_name"] == target_name].copy())
    if df_user.empty:
        logger.warning("No events found for user_name=%s", target_name)
        return None

    report_root.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_filename(target_name)
    user_dir = PLOT_DIR / safe_name
    plot_path = _plot_user_distributions(df_user, target_name, user_dir)

    if "user_id" in df_user.columns:
        user_id = str(pd.Series(df_user["user_id"]).iloc[0])
    else:
        user_id = "N/A"
    total_orders = int(len(df_user))
    total_bet = _sum_numeric(df_user, "bet")
    total_payout = _sum_numeric(df_user, "payout")
    total_profit = _sum_numeric(df_user, "profit")
    rtp = (total_payout / total_bet) * 100 if total_bet > 0 else 0.0
    time_stats = _user_time_stats(df_user)

    bullet_counts = _value_counts(df_user, "bullet_level", numeric_int=True, sort_index=True)
    strategy_counts = _value_counts(df_user, "strategy_name", fill_unknown=True)
    fish_counts = _value_counts(df_user, "fish_value", numeric_int=True, sort_index=True)
    combo_counts = _combo_counts(df_user)
    ip_counts = _value_counts(df_user, "ip", fill_unknown=True)

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
    lines.extend([f"| {idx} | {int(val)} |" for idx, val in bullet_counts.items()])
    lines.extend(
        [
            "",
            "## Strategy Name Counts",
            "| strategy_name | count |",
            "|---|---:|",
        ]
    )
    lines.extend([f"| {idx} | {int(val)} |" for idx, val in strategy_counts.items()])
    lines.extend(
        [
            "",
            "## Fish Value Counts",
            "| fish_value | count |",
            "|---|---:|",
        ]
    )
    lines.extend([f"| {idx} | {int(val)} |" for idx, val in fish_counts.items()])
    lines.extend(
        [
            "",
            "## Multiplier x Bullet Level Counts",
            "| multiplier_x_bullet | count |",
            "|---|---:|",
        ]
    )
    lines.extend([f"| {label} | {count} |" for label, count in _combo_label_rows(combo_counts)])
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


def _sort_names_by_profit(events_df: pd.DataFrame, user_names: list[str]) -> list[str]:
    if not user_names:
        return []
    filtered_df = cast(pd.DataFrame, events_df.loc[events_df["user_name"].isin(user_names)].copy())
    if filtered_df.empty:
        return []
    profit_series = pd.Series(pd.to_numeric(pd.Series(filtered_df["profit"]), errors="coerce")).fillna(0)
    grouped_profit = cast(pd.Series, cast(Any, profit_series.groupby(filtered_df["user_name"])).sum())
    profit_order = cast(pd.Series, cast(Any, grouped_profit).sort_values(ascending=False))
    return [str(name) for name in profit_order.index.tolist()]


def _build_group_plot_bundle(events_df: pd.DataFrame, user_names: list[str]) -> dict[str, Any]:
    metrics = [
        ("bullet_level", "Bullet Level Counts"),
        ("strategy_name", "Strategy Name Counts"),
        ("ip", "IP Counts (Top IPs + OTHERS)"),
        ("fish_value", "Fish Value Counts"),
        ("multiplier_x_bullet", "Multiplier x Bullet Level Counts"),
    ]
    per_user = (
        events_df.groupby("user_name")[["bet", "payout", "profit"]].sum(min_count=1).reindex(user_names).fillna(0)
    )
    rtp_vals = []
    profit_vals = []
    account_duration_vals = []
    avg_bet_interval_vals = []
    median_bet_interval_vals = []
    num_sessions_vals = []
    for _, row in per_user.iterrows():
        bet = float(row["bet"])
        payout = float(row["payout"])
        profit = float(row["profit"])
        rtp_vals.append((payout / bet) * 100 if bet > 0 else 0.0)
        profit_vals.append(profit)
    for user_name in user_names:
        df_user = cast(pd.DataFrame, events_df.loc[events_df["user_name"] == user_name].copy())
        time_stats = _user_time_stats(df_user)
        account_duration_vals.append(float(time_stats["account_duration_hours"]))
        avg_bet_interval_vals.append(float(time_stats["average_bet_interval_seconds"]))
        median_bet_interval_vals.append(float(time_stats["median_bet_interval_seconds"]))
        num_sessions_vals.append(int(time_stats["num_bet_sessions"]))

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
        matrix = _counts_matrix(events_df, user_names, metric_key)
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
    events_df: pd.DataFrame,
    user_names: list[str],
    section_title: str,
) -> None:
    lines.extend(
        [
            "",
            f"## {section_title}",
            "",
            f"- users_included: `{len(user_names)}`",
            f"- total_rows: `{len(events_df)}`",
            "",
        ]
    )
    user_blocks: list[dict[str, Any]] = []
    for user_name in user_names:
        df_user = cast(pd.DataFrame, events_df.loc[events_df["user_name"] == user_name].copy())
        if df_user.empty:
            continue
        user_id = str(pd.Series(df_user["user_id"]).iloc[0]) if "user_id" in df_user.columns else "N/A"
        total_orders = int(len(df_user))
        total_bet = _sum_numeric(df_user, "bet")
        total_payout = _sum_numeric(df_user, "payout")
        total_profit = _sum_numeric(df_user, "profit")
        rtp = (total_payout / total_bet) * 100 if total_bet > 0 else 0.0
        time_stats = _user_time_stats(df_user)
        user_blocks.append(
            {
                "user_name": user_name,
                "user_id": user_id,
                "total_orders": total_orders,
                "total_bet": total_bet,
                "total_payout": total_payout,
                "total_profit": total_profit,
                "rtp_pct": rtp,
                "account_duration_hours": float(time_stats["account_duration_hours"]),
                "average_bet_interval_seconds": float(time_stats["average_bet_interval_seconds"]),
                "num_bet_sessions": int(time_stats["num_bet_sessions"]),
            }
        )
    summary_df = pd.DataFrame(user_blocks)
    if summary_df.empty:
        lines.append("- no matching data rows found")
        return
    summary_df = summary_df.sort_values(by="total_profit", ascending=False)
    lines.append("| user_name | user_id | total_orders | total_bet | total_payout | total_profit | rtp | sessions |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {str(row['user_name'])} | {str(row['user_id'])} | {int(row['total_orders'])} | "
            f"{float(row['total_bet']):,.2f} | {float(row['total_payout']):,.2f} | {float(row['total_profit']):,.2f} | "
            f"{float(row['rtp_pct']):.2f}% | {int(row['num_bet_sessions'])} |"
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


def generate_risk_control_reports(*, refresh_cache: bool = False) -> list[dict[str, Any]]:
    risk_df = _load_events_df(
        s3_uri=S3_RISK_USER_STATS_URI,
        local_cache_file=LOCAL_CACHE_FILE,
        refresh_cache=refresh_cache,
    )
    control_df = _load_events_df(
        s3_uri=S3_CONTROL_USER_STATS_URI,
        local_cache_file=LOCAL_CONTROL_CACHE_FILE,
        refresh_cache=refresh_cache,
    )

    if "risk_user_group" not in risk_df.columns:
        raise ValueError("Missing 'risk_user_group' column in ETL parquet. Please rerun etl_get_risk_user_stats.py.")
    if "control_user_group" not in control_df.columns:
        raise ValueError(
            "Missing 'control_user_group' column in ETL parquet. Please rerun etl_get_control_user_stats.py."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    risk_groups = _discover_groups(risk_df, "risk_user_group")
    control_groups = _discover_groups(control_df, "control_user_group")

    risk_group_payload: dict[str, dict[str, Any]] = {}
    for group in risk_groups:
        risk_group_df = cast(pd.DataFrame, risk_df.loc[risk_df["risk_user_group"] == group].copy())
        risk_group_names = _sort_names_by_profit(risk_group_df, _ordered_user_names(risk_group_df, None))
        risk_group_payload[group] = _build_group_plot_bundle(risk_group_df, risk_group_names)

    control_group_payload: dict[str, dict[str, Any]] = {}
    for group in control_groups:
        control_group_df = cast(pd.DataFrame, control_df.loc[control_df["control_user_group"] == group].copy())
        control_group_names = _sort_names_by_profit(control_group_df, _ordered_user_names(control_group_df, None))
        control_group_payload[group] = _build_group_plot_bundle(control_group_df, control_group_names)

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
        "- note: group labels come from ETL parquet columns `risk_user_group` and `control_user_group`",
        f"- risk_groups: `{', '.join(risk_groups) or '(none)'}`",
        f"- control_groups: `{', '.join(control_groups) or '(none)'}`",
        f"- html_stats_report: `{html_path.name}`",
        "",
        f"[Open interactive chart report]({html_path.name})",
        "",
    ]
    for group in risk_groups:
        group_df = cast(pd.DataFrame, risk_df.loc[risk_df["risk_user_group"] == group].copy())
        group_names = _sort_names_by_profit(group_df, _ordered_user_names(group_df, None))
        _append_markdown_group_section(lines, group_df, group_names, f"Risk {group}")
    for group in control_groups:
        group_df = cast(pd.DataFrame, control_df.loc[control_df["control_user_group"] == group].copy())
        group_names = _sort_names_by_profit(group_df, _ordered_user_names(group_df, None))
        _append_markdown_group_section(lines, group_df, group_names, f"Control {group}")
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
        help="Reload parquet from S3 instead of using local cache (default: false).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )
    generate_risk_control_reports(refresh_cache=bool(args.refresh_cache))


if __name__ == "__main__":
    main()
