"""
Weekly ops report (platform fct_platform_ops_daily_report) — shared logic and Dash fragments.

Data is loaded from S3 Parquet (see jobs/operation_daily_report/etl_weekly_report_all_games.py).
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table.DataTable import DataTable
from dash.development.base_component import Component

logger = logging.getLogger(__name__)


def _scalar_is_na(x: object) -> bool:
    """True if x is NA-like. Use instead of `if pd.isna(x)` — stubs allow array/Frame returns."""
    r = pd.isna(x)
    if isinstance(r, (bool, np.bool_)):
        return bool(r)
    return bool(np.asarray(r, dtype=bool).any())


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    column: str
    value_type: str


@dataclass(frozen=True)
class MetricSummaryRow:
    """One row of weekly comparison stats (output of compare_metric). Typed so UI code needs no casts."""

    metric_key: str
    metric_label: str
    column: str
    value_type: str
    previous_value: Optional[float]
    current_value: Optional[float]
    change: Optional[float]
    previous_valid_days: int
    current_valid_days: int
    series: pd.DataFrame
    current_week_start: pd.Timestamp


def _series_cell_float(series_row: pd.Series, col: str) -> Optional[float]:
    """Single cell from iterrows → optional float for format_value."""
    v = series_row[col]
    if _scalar_is_na(v):
        return None
    return float(v)


GAME_LABELS = {
    "FM01": "FM01",
    "SS01": "SS01",
    "SS03": "SS03",
}

GAME_EMAIL_TITLES = {
    "FM01": "捕鱼大师 - FM01",
    "SS01": "酒馆宝藏（老虎机）- SS01",
    "SS03": "麻将连庄（老虎机）- SS03",
}

METRICS_BY_GAME: Dict[str, List[MetricSpec]] = {
    "FM01": [
        MetricSpec("login_tried", "尝试登录玩家", "dau_login_tried_all", "count"),
        MetricSpec("betting_users", "投注玩家", "dpu_all", "count"),
        MetricSpec("conversion_rate", "投注转化率", "conversion_rate_all", "percent"),
        MetricSpec("new_active_users", "首次活跃玩家", "dau_login_tried_new", "count"),
        MetricSpec("new_betting_users", "首次投注玩家", "dpu_new", "count"),
        MetricSpec("new_active_pct", "首次活跃玩家占比", "new_active_pct_all", "percent"),
        MetricSpec("new_conversion_rate", "新玩家转化率", "conversion_rate_new", "percent"),
        MetricSpec("betting_amount", "流水", "total_betting_amount_all", "currency"),
        MetricSpec("ggr", "GGR", "platform_ggr_all", "currency"),
        MetricSpec("arpu", "ARPU（客单价）", "arpu_all", "currency"),
        MetricSpec("rtp", "RTP", "rtp_all", "percent"),
        MetricSpec("active_pid_amount", "活跃PID 数量（CNY, DAU）", "active_pid_amount_all", "count"),
        MetricSpec("top1_pid_pct", "Top1 PID DAU占比", "top1_pid_dau_pct_all", "percent"),
        MetricSpec("top5_pid_pct", "Top5 PID DAU占比", "top5_pid_dau_pct_all", "percent"),
        MetricSpec("retention_1d_all", "次日留存（全币种）", "retention_1d_all_currency", "percent"),
        MetricSpec("retention_1d_old", "次日留存（全币种，老用户）", "retention_1d_all_currency_old", "percent"),
        MetricSpec("retention_1d_new", "次日留存（全币种，新用户）", "retention_1d_all_currency_new", "percent"),
        MetricSpec("churn_3d_all", "三日流失率（全币种）", "non_retention_3d_all_currency", "percent"),
        MetricSpec("churn_3d_old", "三日流失率（全币种，老用户）", "non_retention_3d_all_currency_old", "percent"),
        MetricSpec("churn_3d_new", "三日流失率（全币种，新用户）", "non_retention_3d_all_currency_new", "percent"),
    ],
    "SS01": [
        MetricSpec("login_tried", "尝试登录玩家", "dau_login_tried_all", "count"),
        MetricSpec("betting_users", "投注玩家", "dpu_all", "count"),
        MetricSpec("conversion_rate", "投注转化率", "conversion_rate_all", "percent"),
        MetricSpec("new_active_pct", "首次活跃玩家占比", "new_active_pct_all", "percent"),
        MetricSpec("new_betting_pct", "首次投注玩家占比", "new_better_pct_all", "percent"),
        MetricSpec("betting_amount", "流水", "total_betting_amount_all", "currency"),
        MetricSpec("ggr", "GGR", "platform_ggr_all", "currency"),
        MetricSpec("arpu", "ARPU（客单价）", "arpu_all", "currency"),
        MetricSpec("rtp", "RTP", "rtp_all", "percent"),
        MetricSpec("active_pid_amount", "活跃PID 数量（CNY, DAU）", "active_pid_amount_all", "count"),
        MetricSpec("top1_pid_pct", "Top1 PID DAU占比", "top1_pid_dau_pct_all", "percent"),
        MetricSpec("top5_pid_pct", "Top5 PID DAU占比", "top5_pid_dau_pct_all", "percent"),
        MetricSpec("retention_1d_all", "次日留存（全币种）", "retention_1d_all_currency", "percent"),
        MetricSpec("retention_1d_old", "次日留存（全币种，老用户）", "retention_1d_all_currency_old", "percent"),
        MetricSpec("retention_1d_new", "次日留存（全币种，新用户）", "retention_1d_all_currency_new", "percent"),
        MetricSpec("churn_3d_all", "三日流失率（全币种）", "non_retention_3d_all_currency", "percent"),
        MetricSpec("churn_3d_old", "三日流失率（全币种，老用户）", "non_retention_3d_all_currency_old", "percent"),
        MetricSpec("churn_3d_new", "三日流失率（全币种，新用户）", "non_retention_3d_all_currency_new", "percent"),
    ],
    "SS03": [
        MetricSpec("login_tried", "尝试登录玩家", "dau_login_tried_all", "count"),
        MetricSpec("betting_users", "投注玩家", "dpu_all", "count"),
        MetricSpec("conversion_rate", "投注转化率", "conversion_rate_all", "percent"),
        MetricSpec("betting_amount", "流水", "total_betting_amount_all", "currency"),
        MetricSpec("ggr", "GGR", "platform_ggr_all", "currency"),
        MetricSpec("arpu", "ARPU（客单价）", "arpu_all", "currency"),
        MetricSpec("rtp", "RTP", "rtp_all", "percent"),
        MetricSpec("active_pid_amount", "活跃PID 数量（CNY, DAU）", "active_pid_amount_all", "count"),
        MetricSpec("top1_pid_pct", "Top1 PID DAU占比", "top1_pid_dau_pct_all", "percent"),
        MetricSpec("top5_pid_pct", "Top5 PID DAU占比", "top5_pid_dau_pct_all", "percent"),
        MetricSpec("retention_1d_all", "次日留存（全币种）", "retention_1d_all_currency", "percent"),
        MetricSpec("churn_3d_all", "三日流失率（全币种）", "non_retention_3d_all_currency", "percent"),
    ],
}

TEXT_TEMPLATE_CONFIG = {
    "FM01": [
        (
            "玩家数量",
            [
                "login_tried",
                "betting_users",
                "conversion_rate",
                "new_active_users",
                "new_betting_users",
                "new_active_pct",
                "new_conversion_rate",
            ],
        ),
        ("盈利情况", ["betting_amount", "ggr", "arpu", "rtp"]),
        ("PID情况", ["active_pid_amount", "top1_pid_pct", "top5_pid_pct"]),
        (
            "留存",
            [
                "retention_1d_all",
                "retention_1d_old",
                "retention_1d_new",
                "churn_3d_all",
                "churn_3d_old",
                "churn_3d_new",
            ],
        ),
    ],
    "SS01": [
        ("玩家数量", ["login_tried", "betting_users", "conversion_rate", "new_active_pct", "new_betting_pct"]),
        ("盈利情况", ["betting_amount", "ggr", "arpu", "rtp"]),
        ("PID情况", ["active_pid_amount", "top1_pid_pct", "top5_pid_pct"]),
        (
            "留存",
            [
                "retention_1d_all",
                "retention_1d_old",
                "retention_1d_new",
                "churn_3d_all",
                "churn_3d_old",
                "churn_3d_new",
            ],
        ),
    ],
    "SS03": [
        ("玩家数量", ["login_tried", "betting_users", "conversion_rate"]),
        ("盈利情况", ["betting_amount", "ggr", "arpu", "rtp"]),
        ("PID情况", ["active_pid_amount", "top1_pid_pct", "top5_pid_pct"]),
        ("留存", ["retention_1d_all", "churn_3d_all"]),
    ],
}

EMAIL_CHART_ROWS = [
    ["login_tried", "betting_users", "conversion_rate"],
    ["betting_amount", "ggr", "arpu"],
    ["rtp", "active_pid_amount", "retention_1d_all"],
]


def load_source_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return transform_source_data(df)


def transform_source_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "bj_date_key" not in df.columns:
        return df
    df["bj_date_key"] = pd.to_datetime(df["bj_date_key"])

    def parse_payload_value(payload: object, key: str) -> Optional[float]:
        if _scalar_is_na(payload):
            return None
        try:
            parsed = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(parsed, dict):
                return None
        except (TypeError, json.JSONDecodeError):
            return None
        value = parsed.get(key)
        return float(value) if value is not None else None

    if "payload" in df.columns:
        df["free_spin_rate"] = df["payload"].apply(lambda value: parse_payload_value(value, "free_spin_rate"))
        df["join_game_scene_users"] = df["payload"].apply(
            lambda value: parse_payload_value(value, "join_game_scene_users")
        )
    return df.sort_values(by=["game_id", "bj_date_key"]).reset_index(drop=True)


def dataframe_from_split_json(source_data_json: str) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(source_data_json), orient="split")
    if "bj_date_key" in df.columns:
        df["bj_date_key"] = pd.to_datetime(df["bj_date_key"])
    return df


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16, "color": "#475569"},
            }
        ],
        height=320,
        margin=dict(l=24, r=24, t=24, b=24),
    )
    return fig


def get_allowed_windows(game_df: pd.DataFrame) -> List[int]:
    total_days = game_df["bj_date_key"].nunique()
    allowed = [days for days in (7, 6, 5, 4, 3) if total_days >= days * 2]
    if allowed:
        return allowed
    fallback = total_days // 2
    return [fallback] if fallback >= 2 else [1]


def get_fixed_window_days(game_df: pd.DataFrame) -> int:
    return get_allowed_windows(game_df)[0]


WEEKLY_REPORT_DATA_SLICE_DAYS = 14
"""Inclusive calendar span ending at the selected end date (7 current + 7 previous for comparisons)."""


def weekly_report_end_date_bounds(game_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(min_date_iso, max_date_iso, default_date_iso)`` for the end-date picker (default = latest day in data)."""
    if game_df.empty or "bj_date_key" not in game_df.columns:
        return None, None, None
    uniq = pd.to_datetime(game_df["bj_date_key"], errors="coerce").dropna()
    if uniq.empty:
        return None, None, None
    min_ts = pd.Timestamp(uniq.min())
    max_ts = pd.Timestamp(uniq.max())
    if _scalar_is_na(min_ts) or _scalar_is_na(max_ts):
        return None, None, None
    assert isinstance(min_ts, pd.Timestamp)
    assert isinstance(max_ts, pd.Timestamp)
    min_d = cast(pd.Timestamp, min_ts.normalize())
    max_d = cast(pd.Timestamp, max_ts.normalize())
    min_iso = min_d.strftime("%Y-%m-%d")
    max_iso = max_d.strftime("%Y-%m-%d")
    return min_iso, max_iso, max_iso


def slice_weekly_report_by_end_date(game_df: pd.DataFrame, end_date_iso: Optional[str]) -> pd.DataFrame:
    """Keep rows whose `bj_date_key` falls in the last ``WEEKLY_REPORT_DATA_SLICE_DAYS`` days ending at ``end_date_iso``."""
    if game_df.empty or not end_date_iso or "bj_date_key" not in game_df.columns:
        return game_df
    end_raw = pd.Timestamp(end_date_iso)
    if _scalar_is_na(end_raw):
        return game_df
    assert isinstance(end_raw, pd.Timestamp)
    end = cast(pd.Timestamp, end_raw.normalize())
    start = end - pd.Timedelta(days=WEEKLY_REPORT_DATA_SLICE_DAYS - 1)
    g = game_df.copy()
    g["_d"] = pd.to_datetime(g["bj_date_key"], errors="coerce").dt.normalize()
    mask = (g["_d"] >= start) & (g["_d"] <= end)
    out = g.loc[mask].drop(columns=["_d"]).sort_values(by="bj_date_key")
    return out


def format_weekly_report_subject(end_date: pd.Timestamp | str) -> str:
    """Subject for the 7-day window ending on ``end_date``: 运营周报MMDDYYYY - MMDDYYYY."""
    end_ts = pd.Timestamp(end_date)
    if _scalar_is_na(end_ts):
        return "运营周报"
    assert isinstance(end_ts, pd.Timestamp)
    end = cast(pd.Timestamp, end_ts.normalize())
    start = cast(pd.Timestamp, end - pd.Timedelta(days=6))
    return f"运营周报{start:%m%d%Y} - {end:%m%d%Y}"


def format_value(value: Optional[float], value_type: str) -> str:
    if value is None or _scalar_is_na(value):
        return "N/A"
    if value_type == "currency":
        return f"¥{value:,.2f} CNY"
    if value_type == "percent":
        return f"{value * 100:.2f}%"
    return f"{value:,.0f}"


def compute_change(previous: Optional[float], current: Optional[float]) -> Optional[float]:
    if previous is None or current is None or _scalar_is_na(previous) or _scalar_is_na(current) or previous == 0:
        return None
    return (current - previous) / previous


def format_change(change: Optional[float]) -> str:
    if change is None or _scalar_is_na(change):
        return "**N/A**"
    direction = "增长" if change >= 0 else "下降"
    return f"**{direction} {abs(change) * 100:.2f}%**"


def is_reverse_metric(metric_key: Optional[str]) -> bool:
    return bool(metric_key and "churn" in metric_key)


def current_value_color(change: Optional[float], metric_key: Optional[str] = None) -> str:
    if change is None or _scalar_is_na(change):
        return "#0f172a"
    if abs(change) <= 0.01:
        return "#2563eb"
    if is_reverse_metric(metric_key):
        return "#16a34a" if change < 0 else "#dc2626"
    return "#16a34a" if change > 0 else "#dc2626"


def change_text_color(change: Optional[float], metric_key: Optional[str] = None) -> str:
    if change is None or _scalar_is_na(change) or abs(change) <= 0.10:
        return "#0f172a"
    if is_reverse_metric(metric_key):
        return "#16a34a" if change < 0 else "#dc2626"
    return "#16a34a" if change > 0 else "#dc2626"


def build_metric_sentence(metric: MetricSummaryRow) -> html.P:
    ch = metric.change
    return html.P(
        [
            f"本周日均{metric.metric_label} ",
            html.Span(
                format_value(metric.current_value, metric.value_type),
                style={
                    "fontWeight": "bold",
                    "color": current_value_color(ch, metric.metric_key),
                },
            ),
            "，上周 ",
            html.Span(
                format_value(metric.previous_value, metric.value_type),
                style={"fontWeight": "bold", "color": "#0f172a"},
            ),
            "，环比 ",
            html.Span(
                format_change(ch).replace("**", ""),
                style={
                    "fontWeight": "bold",
                    "color": change_text_color(ch, metric.metric_key),
                },
            ),
        ],
        style={"margin": "0 0 8px 0", "lineHeight": "1.7"},
    )


def build_weekly_insight(summary: Dict[str, MetricSummaryRow]) -> List[str]:
    candidates = []
    for metric in summary.values():
        change = metric.change
        if change is None or _scalar_is_na(change):
            continue
        candidates.append(metric)

    candidates.sort(key=lambda item: abs(item.change or 0.0), reverse=True)
    significant = [item for item in candidates if abs(item.change or 0.0) > 0.10]

    if not significant:
        return ["整体指标波动不大，报告期内表现基本稳定。"]

    insights = []
    for item in significant[:2]:
        ch = item.change
        direction = "增长" if (ch is not None and not _scalar_is_na(ch) and ch > 0) else "下降"
        insights.append(
            f"{item.metric_label}变动较大，本期为 {format_value(item.current_value, item.value_type)}，"
            f"较上期{direction} {abs(ch or 0.0) * 100:.2f}%。"
        )
    return insights


def split_periods(game_df: pd.DataFrame, column: str, window_days: int):
    """
    Split into two equal-length windows by date (not calendar week).

    Calendar-week splits break for lagged metrics (e.g. 次日留存): the latest days are often
    null in the warehouse while older days are populated, so "this week" can be all NaN and
    "last week" looks fine. Using the last ``window_days`` rows vs the prior ``window_days``
    rows matches ``get_fixed_window_days`` and aligns with charts and subject-line dates.
    """
    metric_df = game_df.loc[:, ["bj_date_key", column]].sort_values(by="bj_date_key").copy()
    metric_df = metric_df.drop_duplicates(subset=["bj_date_key"], keep="last")
    n = len(metric_df)
    if n == 0:
        return metric_df, pd.DataFrame(), pd.DataFrame(), cast(pd.Timestamp, pd.Timestamp("1970-01-01"))

    if n >= window_days * 2:
        current_period = metric_df.iloc[-window_days:].copy()
        previous_period = metric_df.iloc[-(window_days * 2) : -window_days].copy()
    else:
        split = n // 2
        if split == 0 or split == n:
            current_period = metric_df.copy()
            previous_period = pd.DataFrame(columns=metric_df.columns)
        else:
            previous_period = metric_df.iloc[:split].copy()
            current_period = metric_df.iloc[split:].copy()

    current_week_start = cast(pd.Timestamp, pd.Timestamp(current_period["bj_date_key"].iloc[0]))
    return metric_df, previous_period, current_period, current_week_start


def compare_metric(game_df: pd.DataFrame, metric: MetricSpec, window_days: int) -> MetricSummaryRow:
    metric_df, previous_period, current_period, current_week_start = split_periods(game_df, metric.column, window_days)
    pv_raw = previous_period[metric.column].mean(skipna=True)
    cv_raw = current_period[metric.column].mean(skipna=True)
    previous_valid = int(previous_period[metric.column].notna().sum())
    current_valid = int(current_period[metric.column].notna().sum())
    if previous_valid == 0:
        previous_value: Optional[float] = None
    else:
        previous_value = None if _scalar_is_na(pv_raw) else float(pv_raw)
    if current_valid == 0:
        current_value = None
    else:
        current_value = None if _scalar_is_na(cv_raw) else float(cv_raw)
    change = compute_change(previous_value, current_value)
    return MetricSummaryRow(
        metric_key=metric.key,
        metric_label=metric.label,
        column=metric.column,
        value_type=metric.value_type,
        previous_value=previous_value,
        current_value=current_value,
        change=change,
        previous_valid_days=previous_valid,
        current_valid_days=current_valid,
        series=metric_df,
        current_week_start=cast(pd.Timestamp, current_week_start),
    )


def build_summary(game_df: pd.DataFrame, game_id: str, window_days: int) -> Dict[str, MetricSummaryRow]:
    return {
        metric.key: compare_metric(game_df, metric, window_days)
        for metric in METRICS_BY_GAME[game_id]
        if metric.column in game_df.columns
    }


def build_period_daily_table(summary: Dict[str, MetricSummaryRow], window_days: int, period: str):
    summary_items = list(summary.values())
    if not summary_items:
        return [], []

    current_week_start = summary_items[0].current_week_start
    if period == "current":
        date_series = summary_items[0].series.loc[
            summary_items[0].series["bj_date_key"] >= current_week_start, "bj_date_key"
        ]
        selector = lambda df: df["bj_date_key"] >= current_week_start
    else:
        date_series = summary_items[0].series.loc[
            summary_items[0].series["bj_date_key"] < current_week_start, "bj_date_key"
        ]
        selector = lambda df: df["bj_date_key"] < current_week_start

    date_columns = []
    for idx, date_value in enumerate(date_series, start=1):
        d0 = pd.Timestamp(date_value)
        if _scalar_is_na(d0):
            date_columns.append({"id": f"day_{idx}", "name": ["—", "—"]})
        else:
            d0_ts = cast(pd.Timestamp, d0)
            date_columns.append(
                {
                    "id": f"day_{idx}",
                    "name": [d0_ts.strftime("%b %d, %Y"), d0_ts.strftime("%A")],
                }
            )

    rows = []
    for item in summary_items:
        sub = item.series.loc[selector(item.series)]
        row = {"metric": item.metric_label}
        for idx, (_, series_row) in enumerate(sub.iterrows(), start=1):
            row[f"day_{idx}"] = format_value(_series_cell_float(series_row, item.column), item.value_type)
        rows.append(row)

    columns = [{"id": "metric", "name": ["指标", "指标"]}] + date_columns
    return rows, columns


def build_email_summary_block(game_id: str, summary: Dict[str, MetricSummaryRow]) -> List[Component]:
    blocks: List[Component] = []
    for section_title, metric_keys in TEXT_TEMPLATE_CONFIG[game_id]:
        blocks.append(html.Div(section_title, style={"fontWeight": "bold", "marginTop": "12px", "marginBottom": "8px"}))
        for key in metric_keys:
            metric = summary.get(key)
            if metric:
                blocks.append(build_metric_sentence(metric))
    return blocks


def build_email_section(
    game_id: str, summary: Dict[str, MetricSummaryRow], game_df: pd.DataFrame, window_days: int
) -> html.Div:
    chart_rows = []
    for metric_keys in EMAIL_CHART_ROWS:
        row_components = []
        for metric_key in metric_keys:
            metric = next((spec for spec in METRICS_BY_GAME[game_id] if spec.key == metric_key), None)
            if metric is None:
                row_components.append(html.Div(style={"width": "500px", "flex": "0 0 500px"}))
                continue
            row_components.append(
                html.Div(
                    style={
                        "width": "500px",
                        "minWidth": "500px",
                        "flex": "0 0 500px",
                        "backgroundColor": "white",
                    },
                    children=[
                        dcc.Graph(
                            figure=build_chart(game_df, metric, window_days),
                            config={"displayModeBar": False, "responsive": False},
                            style={"height": "280px", "width": "500px"},
                        )
                    ],
                )
            )
        chart_rows.append(
            html.Div(
                style={
                    "display": "flex",
                    "gap": "12px",
                    "marginBottom": "12px",
                    "overflowX": "auto",
                    "paddingBottom": "6px",
                },
                children=row_components,
            )
        )

    table_data, table_columns = build_period_daily_table(summary, window_days, "current")

    return html.Div(
        style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "12px",
            "boxShadow": "0 1px 3px rgba(15,23,42,0.08)",
            "marginBottom": "20px",
        },
        children=[
            html.H3(f"【{GAME_EMAIL_TITLES[game_id]}】", style={"marginTop": "0"}),
            html.H4("Weekly Highlight：", style={"marginBottom": "8px"}),
            html.Ul(
                [html.Li(item, style={"marginBottom": "6px"}) for item in build_weekly_insight(summary)],
                style={"marginTop": "0", "marginBottom": "16px", "paddingLeft": "20px"},
            ),
            html.H4("Weekly Summary（指标效果：红色↓，绿色↑ ，蓝色→）", style={"marginBottom": "8px"}),
            html.Div(build_email_summary_block(game_id, summary)),
            html.H4("数据趋势图一览", style={"marginTop": "20px", "marginBottom": "8px"}),
            html.Div(children=chart_rows),
            html.H4("CNY 周数据一览", style={"marginTop": "20px", "marginBottom": "8px"}),
            DataTable(
                data=table_data,
                columns=cast(Any, table_columns),
                merge_duplicate_headers=True,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "8px",
                    "fontFamily": "Arial, sans-serif",
                    "fontSize": "12px",
                },
                style_header={"backgroundColor": "#e2e8f0", "fontWeight": "bold"},
                page_action="none",
            ),
        ],
    )


def build_metric_summary_card(metric_summary: MetricSummaryRow, window_days: int) -> html.Div:
    ch = metric_summary.change
    return html.Div(
        children=[
            html.H4(metric_summary.metric_label, style={"marginTop": "0", "marginBottom": "16px"}),
            html.Div("上期均值", style={"color": "#475569", "marginBottom": "4px"}),
            html.Div(
                format_value(metric_summary.previous_value, metric_summary.value_type),
                style={"fontSize": "24px", "fontWeight": "bold", "marginBottom": "16px"},
            ),
            html.Div("本期均值", style={"color": "#475569", "marginBottom": "4px"}),
            html.Div(
                format_value(metric_summary.current_value, metric_summary.value_type),
                style={
                    "fontSize": "24px",
                    "fontWeight": "bold",
                    "marginBottom": "16px",
                    "color": current_value_color(ch, metric_summary.metric_key),
                },
            ),
            html.Div("环比", style={"color": "#475569", "marginBottom": "4px"}),
            html.Div(
                format_change(ch).replace("**", ""),
                style={
                    "fontSize": "20px",
                    "fontWeight": "bold",
                    "marginBottom": "16px",
                    "color": change_text_color(ch, metric_summary.metric_key),
                },
            ),
            html.Div(
                "上期均值对应蓝色区间，本期均值对应黄色区间",
                style={"color": "#64748b", "fontSize": "13px"},
            ),
        ]
    )


def build_default_subject(game_df: pd.DataFrame) -> str:
    if game_df.empty or "bj_date_key" not in game_df.columns:
        return "运营周报"
    window_days = get_fixed_window_days(game_df)
    uniq = game_df["bj_date_key"].drop_duplicates().to_numpy()
    sorted_vals = np.sort(uniq)
    current_dates = pd.Series(sorted_vals[-window_days:])
    if current_dates.empty:
        return "运营周报"
    start_raw = current_dates.iloc[0]
    end_raw = current_dates.iloc[-1]
    if _scalar_is_na(start_raw) or _scalar_is_na(end_raw):
        return "运营周报"
    start_text = cast(pd.Timestamp, pd.Timestamp(start_raw)).strftime("%m%d")
    end_text = cast(pd.Timestamp, pd.Timestamp(end_raw)).strftime("%m%d")
    return f"运营周报{start_text} - {end_text}"


def build_chart(game_df: pd.DataFrame, metric: MetricSpec, window_days: int) -> go.Figure:
    plot_df, previous_period, current_period, _ = split_periods(game_df, metric.column, window_days)

    fig = go.Figure()
    game_label = GAME_LABELS.get(str(game_df["game_id"].iloc[0]), str(game_df["game_id"].iloc[0]))
    if not previous_period.empty:
        fig.add_trace(
            go.Scatter(
                x=previous_period["bj_date_key"],
                y=previous_period[metric.column],
                mode="lines+markers",
                name="历史区间",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=5),
                cliponaxis=False,
            )
        )
    if not current_period.empty:
        fig.add_trace(
            go.Scatter(
                x=current_period["bj_date_key"],
                y=current_period[metric.column],
                mode="lines+markers",
                name="最近一周",
                line=dict(color="#eab308", width=2),
                marker=dict(size=5),
                cliponaxis=False,
            )
        )
    if not previous_period.empty and not current_period.empty:
        fig.add_trace(
            go.Scatter(
                x=[previous_period["bj_date_key"].iloc[-1], current_period["bj_date_key"].iloc[0]],
                y=[previous_period[metric.column].iloc[-1], current_period[metric.column].iloc[0]],
                mode="lines",
                name="连接",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    prev_mean = previous_period[metric.column].mean(skipna=True)
    curr_mean = current_period[metric.column].mean(skipna=True)
    if not _scalar_is_na(prev_mean):
        fig.add_hline(y=prev_mean, line_dash="dash", line_color="#2563eb")
    if not _scalar_is_na(curr_mean):
        fig.add_hline(y=curr_mean, line_dash="dash", line_color="#eab308")

    all_values = pd.concat([previous_period[metric.column], current_period[metric.column]], ignore_index=True).dropna()
    yaxis_range = None
    if not all_values.empty:
        min_value = float(all_values.min())
        max_value = float(all_values.max())
        spread = max_value - min_value
        if spread > 0:
            lower_padding = spread * 0.12
            upper_padding = spread * 0.20
        else:
            base_padding = max(abs(max_value) * 0.10, 1)
            lower_padding = base_padding
            upper_padding = base_padding * 1.4
        yaxis_range = [min_value - lower_padding, max_value + upper_padding]

    yaxis_format = None
    if metric.value_type == "percent":
        yaxis_format = ".1%"

    fig.update_layout(
        title=dict(text=f"{game_label} | {metric.label}", font=dict(size=16)),
        template="plotly_white",
        height=280,
        width=500,
        autosize=False,
        margin=dict(l=24, r=16, t=72, b=28),
        legend=dict(orientation="h", y=1.18, x=1, xanchor="right", font=dict(size=11)),
    )
    if yaxis_format:
        fig.update_yaxes(tickformat=yaxis_format)
    fig.update_xaxes(tickfont=dict(size=10), title=None)
    fig.update_yaxes(tickfont=dict(size=10), title=None, range=yaxis_range, automargin=True)
    return fig


def build_draft_preview_single(game_df: pd.DataFrame, game_id: str) -> List[Component]:
    if game_df.empty:
        return [
            html.Div(
                "暂无数据。请确认 ETL 已运行且 weekly_report.files 指向正确的 S3 路径。",
                style={
                    "backgroundColor": "white",
                    "padding": "24px",
                    "borderRadius": "12px",
                    "color": "#475569",
                },
            )
        ]
    window_days = get_fixed_window_days(game_df)
    summary = build_summary(game_df, game_id, window_days)
    return [build_email_section(game_id, summary, game_df, window_days)]


def layout_weekly_report_panel(game_df: pd.DataFrame, game_id: str) -> html.Div:
    """Chart + summary card; expects callbacks on weekly-report-* ids."""
    if game_df.empty:
        return html.Div(
            style={"marginBottom": "20px"},
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "padding": "24px",
                        "borderRadius": "12px",
                        "boxShadow": "0 1px 3px rgba(15,23,42,0.08)",
                        "color": "#475569",
                    },
                    children=[
                        html.H3("趋势图预览", style={"marginTop": "0"}),
                        html.P("当前没有可用数据。", style={"marginBottom": "0"}),
                    ],
                )
            ],
        )

    metric_options = [{"label": m.label, "value": m.key} for m in METRICS_BY_GAME[game_id]]
    default_metric = metric_options[0]["value"] if metric_options else None
    min_end, max_end, default_end = weekly_report_end_date_bounds(game_df)
    control_row: List[Component] = [
        html.Div(
            [
                html.Label("指标", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="weekly-report-metric-select",
                    options=cast(Any, metric_options),
                    value=default_metric,
                    clearable=False,
                    style={"width": "360px"},
                ),
            ]
        ),
    ]
    if min_end and max_end and default_end:
        control_row.append(
            html.Div(
                [
                    html.Label("报告截止日（7 日窗口）", style={"fontWeight": "bold"}),
                    dcc.DatePickerSingle(
                        id="weekly-report-end-date",
                        min_date_allowed=min_end,
                        max_date_allowed=max_end,
                        date=default_end,
                        initial_visible_month=default_end,
                        display_format="YYYY-MM-DD",
                        style={"width": "200px"},
                    ),
                ]
            )
        )

    return html.Div(
        style={"marginBottom": "20px"},
        children=[
            html.H3("趋势图预览", style={"marginTop": "0", "marginBottom": "12px", "color": "#377EB8"}),
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"},
                children=control_row,
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1.1fr 0.35fr", "gap": "16px", "alignItems": "start"},
                children=[
                    html.Div(
                        style={
                            "backgroundColor": "white",
                            "padding": "20px",
                            "borderRadius": "12px",
                            "boxShadow": "0 1px 3px rgba(15,23,42,0.08)",
                        },
                        children=[dcc.Graph(id="weekly-report-chart")],
                    ),
                    html.Div(
                        id="weekly-report-summary-card",
                        style={
                            "backgroundColor": "white",
                            "padding": "20px",
                            "borderRadius": "12px",
                            "boxShadow": "0 1px 3px rgba(15,23,42,0.08)",
                        },
                    ),
                ],
            ),
        ],
    )
