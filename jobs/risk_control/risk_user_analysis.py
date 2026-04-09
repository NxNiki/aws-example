"""
Generate per-user risk reports from ``etl_get_risk_user_stats`` output.

This script:
1) loads ETL data from local cache (or S3 on cache miss),
2) builds a markdown report for each user via ``analyze_user_by_name``,
3) saves 5 count-based plots per user:
   - bullet_level
   - strategy_name
   - ip (single stacked bar; no legend)
   - fish_value
   - multiplier x bullet_level

Cache/report directory is outside repo:
``<repo_parent>/data_fishhunter``.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

from bituslabs_ds.config import DEFAULT_ETL_OUTPUT, LOCAL_ROOT, setup_logging
from bituslabs_ds.s3_utils import read_files
from bituslabs_ds.utils import get_ip_location

logger = logging.getLogger(__name__)

S3_RISK_USER_STATS_URI = f"{DEFAULT_ETL_OUTPUT}/jobs/output_risk_control/risk_user_stats/risk_user_stats.parquet"
DATA_ROOT = LOCAL_ROOT.parent / "data_fishhunter"
CACHE_DIR = DATA_ROOT / "risk_control_cache"
REPORT_DIR = DATA_ROOT / "risk_control_reports"
PLOT_DIR = REPORT_DIR / "plots"
LOCAL_CACHE_FILE = CACHE_DIR / "risk_user_stats.parquet"
IP_STACK_TOP_N = 20
ALL_USERS_MD_NAME = "risk_user_all_users.md"
ALL_USERS_HTML_NAME = "risk_user_all_users_stats.html"


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
    top_ips: list[str] = []
    city_map: dict[str, str] = {}

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
        categories = sorted(df["_metric"].unique().tolist())
        showlegend = True
    elif metric == "fish_value":
        values = pd.Series(pd.to_numeric(pd.Series(df["fish_value"]), errors="coerce")).dropna().astype(int).astype(str)
        df = df.loc[values.index].copy()
        df["_metric"] = values
        categories = sorted(df["_metric"].unique().tolist(), key=lambda x: int(x))
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
        categories = sorted(df["_metric"].unique().tolist())
        showlegend = True
    elif metric == "ip":
        df["_metric"] = df["ip"].fillna("UNKNOWN").astype(str)
        ip_totals = df["_metric"].value_counts()
        top_ips = [str(ip) for ip in ip_totals.head(IP_STACK_TOP_N).index.tolist()]
        loc_map = {str(ip): get_ip_location(str(ip)) if str(ip) != "UNKNOWN" else "N/A" for ip in top_ips}
        for ip in top_ips:
            ip_key = str(ip)
            location = loc_map[ip_key]
            if location == "Timeout/Failed":
                city_map[ip_key] = "未知"
                continue
            parts = [p for p in location.split(" ") if p and p != "None"]
            city_map[ip_key] = parts[-1] if parts else "未知"
        df["_metric"] = df["_metric"].apply(lambda ip: f"{ip} ({loc_map[str(ip)]})" if str(ip) in loc_map else "OTHERS")
        categories = [f"{ip} ({loc_map[str(ip)]})" for ip in top_ips]
        if "OTHERS" in df["_metric"].values:
            categories.append("OTHERS")
        showlegend = False
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    pivot_raw = cast(pd.DataFrame, cast(Any, df.groupby(["_metric", "user_name"]).size().unstack(fill_value=0)))
    pivot_idx = cast(pd.DataFrame, cast(Any, pivot_raw).reindex(index=categories, fill_value=0))
    pivot = cast(pd.DataFrame, cast(Any, pivot_idx).reindex(user_names, axis=1, fill_value=0))

    traces: list[dict[str, Any]] = []
    ip_label_enabled: set[str] = set()
    if metric == "ip":
        if len(top_ips) > 5:
            ip_label_enabled = {str(ip) for ip in top_ips[:5]}
        else:
            ip_label_enabled = {str(ip) for ip in top_ips}

    for label in categories:
        y_vals = [int(v) for v in pivot.loc[label].tolist()]
        trace: dict[str, Any] = {"name": label, "x": user_names, "y": y_vals, "type": "bar"}
        if metric == "ip":
            if label == "OTHERS":
                trace["text"] = ["" for _ in y_vals]
            else:
                ip_key = label.split(" (", 1)[0]
                city = city_map.get(ip_key, "未知")
                show_city = ip_key in ip_label_enabled
                trace["text"] = [city if (show_city and val > 0) else "" for val in y_vals]
                trace["textposition"] = "inside"
                trace["insidetextanchor"] = "middle"
                trace["textfont"] = {"size": 10, "color": "white"}
                trace["cliponaxis"] = False
        traces.append(trace)

    return {"labels": categories, "traces": traces, "showlegend": showlegend}


def _write_all_users_html_report(
    events_df: pd.DataFrame,
    user_names: list[str],
    out_path: Path,
) -> None:
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
    for _, row in per_user.iterrows():
        bet = float(row["bet"])
        payout = float(row["payout"])
        profit = float(row["profit"])
        rtp_vals.append((payout / bet) * 100 if bet > 0 else 0.0)
        profit_vals.append(profit)

    line_payload = {
        "none": {"name": "None", "y": []},
        "rtp": {"name": "RTP %", "y": rtp_vals},
        "total_profit": {"name": "Total Profit", "y": profit_vals},
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

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Risk User All-Users Stats</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .toolbar {{ margin-bottom: 16px; }}
    #chart {{ width: 100%; height: 760px; }}
    label {{ font-weight: 600; margin-right: 8px; }}
    select {{ padding: 6px 8px; min-width: 320px; }}
  </style>
</head>
<body>
  <h1>Risk User Multi-User Stats</h1>
  <div class="toolbar">
    <label for="metricSelect">Select metric:</label>
    <select id="metricSelect">
      <option value="bullet_level">Bullet Level</option>
      <option value="strategy_name">Strategy Name</option>
      <option value="ip">IP</option>
      <option value="fish_value">Fish Value</option>
      <option value="multiplier_x_bullet">Multiplier x Bullet Level</option>
    </select>
    <label for="lineSelect" style="margin-left: 12px;">Right-axis line:</label>
    <select id="lineSelect">
      <option value="none">None</option>
      <option value="rtp">RTP %</option>
      <option value="total_profit">Total Profit</option>
    </select>
  </div>
  <div id="chart"></div>
  <script>
    const payload = {json.dumps(payload)};
    const linePayload = {json.dumps(line_payload)};
    const userNames = {json.dumps(user_names)};
    function render(metricKey, lineKey) {{
      const obj = payload[metricKey];
      const traces = [...obj.data];
      const layout = Object.assign({{}}, obj.layout, {{ title: obj.title }});
      if (lineKey !== 'none') {{
        const lineObj = linePayload[lineKey];
        traces.push({{
          x: userNames,
          y: lineObj.y,
          name: lineObj.name,
          type: 'scatter',
          mode: 'lines+markers',
          yaxis: 'y2',
          line: {{ width: 2 }}
        }});
        layout.yaxis2 = {{
          title: lineObj.name,
          overlaying: 'y',
          side: 'right',
          showgrid: false
        }};
      }}
      Plotly.react('chart', traces, layout, {{responsive: true}});
    }}
    const metricSelect = document.getElementById('metricSelect');
    const lineSelect = document.getElementById('lineSelect');
    metricSelect.addEventListener('change', () => render(metricSelect.value, lineSelect.value));
    lineSelect.addEventListener('change', () => render(metricSelect.value, lineSelect.value));
    render(metricSelect.value, lineSelect.value);
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _write_all_users_markdown_report(
    events_df: pd.DataFrame,
    user_names: list[str],
    out_path: Path,
    html_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
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
        ip_counts = _value_counts(df_user, "ip", fill_unknown=True)
        top_ip = str(ip_counts.index[0]) if len(ip_counts) > 0 else "N/A"
        top_ip_loc = get_ip_location(top_ip) if top_ip not in {"N/A", "UNKNOWN"} else "N/A"
        strategy_counts = _value_counts(df_user, "strategy_name", fill_unknown=True)
        top_strategy = str(strategy_counts.index[0]) if len(strategy_counts) > 0 else "N/A"
        rows.append(
            {
                "user_name": user_name,
                "user_id": user_id,
                "total_orders": total_orders,
                "total_bet": total_bet,
                "total_payout": total_payout,
                "total_profit": total_profit,
                "actual_rtp_pct": rtp,
                "top_strategy": top_strategy,
                "top_ip": top_ip,
                "top_ip_location": top_ip_loc,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="total_profit", ascending=False)

    lines = [
        "# Risk User Report (All Users)",
        "",
        f"- users_included: `{len(summary_df)}`",
        f"- total_rows: `{len(events_df)}`",
        f"- html_stats_report: `{html_path.name}`",
        "",
        f"[Open interactive chart report]({html_path.name})",
        "",
        "## User Summary",
        "| user_name | user_id | total_orders | total_bet | total_payout | total_profit | actual_rtp_pct | top_strategy | top_ip | top_ip_location |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["user_name"]),
                    str(row["user_id"]),
                    str(int(row["total_orders"])),
                    f"{float(row['total_bet']):,.2f}",
                    f"{float(row['total_payout']):,.2f}",
                    f"{float(row['total_profit']):,.2f}",
                    f"{float(row['actual_rtp_pct']):.2f}",
                    str(row["top_strategy"]),
                    str(row["top_ip"]),
                    str(row["top_ip_location"]),
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _load_events_df(*, refresh_cache: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = cast(
        pd.DataFrame,
        read_files(
            files=S3_RISK_USER_STATS_URI,
            local_cache_path=str(LOCAL_CACHE_FILE),
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

    ip_counts = _value_counts(df_user, "ip", fill_unknown=True)
    top_ip_counts = ip_counts.head(IP_STACK_TOP_N)
    other_ip_count = int(ip_counts.iloc[IP_STACK_TOP_N:].sum()) if len(ip_counts) > IP_STACK_TOP_N else 0
    stack_counts = top_ip_counts.tolist()
    if other_ip_count > 0:
        stack_counts.append(other_ip_count)

    left = 0
    for i, count in enumerate(stack_counts):
        color = plt.get_cmap("tab20")(i % 20)
        axes[2].barh([target_name], [count], left=left, color=color, height=0.55)
        left += count
    axes[2].set_title("IP Count (Stacked, No Legend)")
    axes[2].set_xlabel("Count")

    annotate_n = min(5, len(top_ip_counts))
    cursor = 0
    for i in range(annotate_n):
        ip = top_ip_counts.index[i]
        count = int(top_ip_counts.iloc[i])
        location = get_ip_location(str(ip))
        mid = cursor + count / 2
        axes[2].text(mid, 0, location, ha="center", va="center", fontsize=8, color="white")
        cursor += count

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


def generate_reports_for_users(
    user_names: list[str] | None = None,
    *,
    refresh_cache: bool = False,
) -> list[dict[str, Any]]:
    events_df = _load_events_df(refresh_cache=refresh_cache)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ordered_names = _ordered_user_names(events_df, user_names)
    if not ordered_names:
        logger.warning("No matching user_name records found; skipping report generation.")
        return []

    filtered_df = cast(pd.DataFrame, events_df.loc[events_df["user_name"].isin(ordered_names)].copy())
    profit_series = pd.Series(pd.to_numeric(pd.Series(filtered_df["profit"]), errors="coerce")).fillna(0)
    grouped_profit = cast(pd.Series, cast(Any, profit_series.groupby(filtered_df["user_name"])).sum())
    profit_order = cast(pd.Series, cast(Any, grouped_profit).sort_values(ascending=False))
    ordered_names = [str(name) for name in profit_order.index.tolist()]
    md_path = REPORT_DIR / ALL_USERS_MD_NAME
    html_path = REPORT_DIR / ALL_USERS_HTML_NAME

    _write_all_users_html_report(filtered_df, ordered_names, html_path)
    _write_all_users_markdown_report(filtered_df, ordered_names, md_path, html_path)

    logger.info("Generated all-user markdown report: %s", md_path)
    logger.info("Generated all-user html report: %s", html_path)
    return [{"markdown_report": str(md_path), "html_report": str(html_path), "users": len(ordered_names)}]


def main() -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )
    generate_reports_for_users(user_names=None, refresh_cache=False)


if __name__ == "__main__":
    main()
