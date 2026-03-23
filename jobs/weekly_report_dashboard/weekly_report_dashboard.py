import json
import os
import socket
import base64
import io
import tempfile
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import dash
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, dcc, html, dash_table


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


def env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = env_str(name)
    if value is None:
        return default
    return int(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


load_env_file(ENV_PATH)

DEFAULT_CSV_PATH = Path(
    env_str("DEFAULT_CSV_PATH", "/Users/stephaniechen/Documents/platform_public_fct_form_ops_daily_report_3.csv")
)
CREDENTIALS_PATH = Path(env_str("GMAIL_CREDENTIALS_PATH", str(BASE_DIR / "credentials.json")))
TOKEN_PATH = Path(env_str("GMAIL_TOKEN_PATH", str(BASE_DIR / "token.json")))
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
EMAIL_IMAGE_WIDTH = 300
EMAIL_IMAGE_HEIGHT = 210
REDSHIFT_CONFIG = {
    "host": env_str("REDSHIFT_HOST"),
    "port": env_int("REDSHIFT_PORT", 5439),
    "database": env_str("REDSHIFT_DATABASE"),
    "user": env_str("REDSHIFT_USER"),
    "password": env_str("REDSHIFT_PASSWORD"),
    "ssl": env_bool("REDSHIFT_SSL", True),
}
SSH_TUNNEL_CONFIG = {
    "host": env_str("SSH_TUNNEL_HOST"),
    "port": env_int("SSH_TUNNEL_PORT", 22),
    "user": env_str("SSH_TUNNEL_USER"),
    "private_key": env_str("SSH_PRIVATE_KEY_PATH"),
}


def redshift_enabled() -> bool:
    return all(
        [
            REDSHIFT_CONFIG["host"],
            REDSHIFT_CONFIG["port"],
            REDSHIFT_CONFIG["database"],
            REDSHIFT_CONFIG["user"],
            REDSHIFT_CONFIG["password"],
            SSH_TUNNEL_CONFIG["host"],
            SSH_TUNNEL_CONFIG["port"],
            SSH_TUNNEL_CONFIG["user"],
            SSH_TUNNEL_CONFIG["private_key"],
        ]
    )


def gmail_enabled() -> bool:
    return CREDENTIALS_PATH.exists()


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    column: str
    value_type: str


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
        ("玩家数量", [
            "login_tried", "betting_users", "conversion_rate", "new_active_users",
            "new_betting_users", "new_active_pct", "new_conversion_rate",
        ]),
        ("盈利情况", ["betting_amount", "ggr", "arpu", "rtp"]),
        ("PID情况", ["active_pid_amount", "top1_pid_pct", "top5_pid_pct"]),
        ("留存", [
            "retention_1d_all", "retention_1d_old", "retention_1d_new",
            "churn_3d_all", "churn_3d_old", "churn_3d_new",
        ]),
    ],
    "SS01": [
        ("玩家数量", ["login_tried", "betting_users", "conversion_rate", "new_active_pct", "new_betting_pct"]),
        ("盈利情况", ["betting_amount", "ggr", "arpu", "rtp"]),
        ("PID情况", ["active_pid_amount", "top1_pid_pct", "top5_pid_pct"]),
        ("留存", [
            "retention_1d_all", "retention_1d_old", "retention_1d_new",
            "churn_3d_all", "churn_3d_old", "churn_3d_new",
        ]),
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
    df["bj_date_key"] = pd.to_datetime(df["bj_date_key"])

    def parse_payload_value(payload: object, key: str) -> Optional[float]:
        if pd.isna(payload):
            return None
        try:
            parsed = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return None
        value = parsed.get(key)
        return float(value) if value is not None else None

    df["free_spin_rate"] = df["payload"].apply(lambda value: parse_payload_value(value, "free_spin_rate"))
    df["join_game_scene_users"] = df["payload"].apply(lambda value: parse_payload_value(value, "join_game_scene_users"))
    return df.sort_values(["game_id", "bj_date_key"]).reset_index(drop=True)


def parse_uploaded_contents(contents: str, filename: Optional[str]) -> pd.DataFrame:
    if not contents:
        raise ValueError("未收到上传内容")
    if filename and not filename.lower().endswith(".csv"):
        raise ValueError("目前只支持上传 CSV 文件")

    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return transform_source_data(df)


def dataframe_from_store(source_data_json: str) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(source_data_json), orient="split")
    df["bj_date_key"] = pd.to_datetime(df["bj_date_key"])
    return df


def fetch_redshift_data(start_date: str, end_date: str, currency_type: str) -> pd.DataFrame:
    if not redshift_enabled():
        raise RuntimeError("Redshift 未配置。请先在 .env 中补齐连接信息。")
    try:
        import redshift_connector
        from sshtunnel import SSHTunnelForwarder
    except ImportError as exc:
        raise RuntimeError(f"缺少 Redshift 依赖：{exc}") from exc

    query = """
        with base as (
            select *
            from platform.public.fct_platform_ops_daily_report
            where bj_date_key between %s and %s
              and currency_type = %s
            order by bj_date_key
        )
        select *
        from base
    """

    with SSHTunnelForwarder(
        (SSH_TUNNEL_CONFIG["host"], SSH_TUNNEL_CONFIG["port"]),
        ssh_username=SSH_TUNNEL_CONFIG["user"],
        ssh_pkey=SSH_TUNNEL_CONFIG["private_key"],
        remote_bind_address=(REDSHIFT_CONFIG["host"], REDSHIFT_CONFIG["port"]),
    ) as tunnel:
        conn = redshift_connector.connect(
            host="127.0.0.1",
            port=tunnel.local_bind_port,
            database=REDSHIFT_CONFIG["database"],
            user=REDSHIFT_CONFIG["user"],
            password=REDSHIFT_CONFIG["password"],
            ssl=REDSHIFT_CONFIG["ssl"],
        )
        try:
            cursor = conn.cursor()
            cursor.execute(query, (start_date, end_date, currency_type))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    if df.empty:
        raise RuntimeError("Redshift 查询结果为空")
    return transform_source_data(df)


def get_allowed_windows(game_df: pd.DataFrame) -> List[int]:
    total_days = game_df["bj_date_key"].nunique()
    allowed = [days for days in (7, 6, 5, 4, 3) if total_days >= days * 2]
    if allowed:
        return allowed
    fallback = total_days // 2
    return [fallback] if fallback >= 2 else [1]


def format_value(value: Optional[float], value_type: str) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if value_type == "currency":
        return f"¥{value:,.2f} CNY"
    if value_type == "percent":
        return f"{value * 100:.2f}%"
    return f"{value:,.0f}"


def format_bold_value(value: Optional[float], value_type: str) -> str:
    return f"**{format_value(value, value_type)}**"


def compute_change(previous: Optional[float], current: Optional[float]) -> Optional[float]:
    if previous is None or current is None or pd.isna(previous) or pd.isna(current) or previous == 0:
        return None
    return (current - previous) / previous


def format_change(change: Optional[float]) -> str:
    if change is None or pd.isna(change):
        return "**N/A**"
    direction = "增长" if change >= 0 else "下降"
    return f"**{direction} {abs(change) * 100:.2f}%**"


def get_fixed_window_days(game_df: pd.DataFrame) -> int:
    return get_allowed_windows(game_df)[0]


def is_reverse_metric(metric_key: Optional[str]) -> bool:
    return bool(metric_key and "churn" in metric_key)


def current_value_color(change: Optional[float], metric_key: Optional[str] = None) -> str:
    if change is None or pd.isna(change):
        return "#0f172a"
    if abs(change) <= 0.01:
        return "#2563eb"
    if is_reverse_metric(metric_key):
        return "#16a34a" if change < 0 else "#dc2626"
    return "#16a34a" if change > 0 else "#dc2626"


def change_text_color(change: Optional[float], metric_key: Optional[str] = None) -> str:
    if change is None or pd.isna(change) or abs(change) <= 0.10:
        return "#0f172a"
    if is_reverse_metric(metric_key):
        return "#16a34a" if change < 0 else "#dc2626"
    return "#16a34a" if change > 0 else "#dc2626"


def build_metric_sentence(metric: Dict[str, object]) -> html.P:
    return html.P(
        [
            f"本周日均{metric['metric_label']} ",
            html.Span(
                format_value(metric["current_value"], metric["type"]),
                style={
                    "fontWeight": "bold",
                    "color": current_value_color(metric["change"], metric["metric_key"]),
                },
            ),
            "，上周 ",
            html.Span(
                format_value(metric["previous_value"], metric["type"]),
                style={"fontWeight": "bold", "color": "#0f172a"},
            ),
            "，环比 ",
            html.Span(
                format_change(metric["change"]).replace("**", ""),
                style={
                    "fontWeight": "bold",
                    "color": change_text_color(metric["change"], metric["metric_key"]),
                },
            ),
        ],
        style={"margin": "0 0 8px 0", "lineHeight": "1.7"},
    )


def build_weekly_insight(summary: Dict[str, Dict[str, object]]) -> List[str]:
    candidates = []
    for metric in summary.values():
        change = metric["change"]
        if change is None or pd.isna(change):
            continue
        candidates.append(metric)

    candidates.sort(key=lambda item: abs(item["change"]), reverse=True)
    significant = [item for item in candidates if abs(item["change"]) > 0.10]

    if not significant:
        return ["整体指标波动不大，报告期内表现基本稳定。"]

    insights = []
    for item in significant[:2]:
        direction = "增长" if item["change"] > 0 else "下降"
        insights.append(
            f"{item['metric_label']}变动较大，本期为 {format_value(item['current_value'], item['type'])}，"
            f"较上期{direction} {abs(item['change']) * 100:.2f}%。"
        )
    return insights


def split_periods(game_df: pd.DataFrame, column: str):
    metric_df = game_df[["bj_date_key", column]].sort_values("bj_date_key").copy()
    latest_date = metric_df["bj_date_key"].max()
    current_week_start = latest_date - pd.Timedelta(days=int(latest_date.weekday()))
    previous_period = metric_df[metric_df["bj_date_key"] < current_week_start].copy()
    current_period = metric_df[metric_df["bj_date_key"] >= current_week_start].copy()
    return metric_df, previous_period, current_period, current_week_start


def compare_metric(game_df: pd.DataFrame, metric: MetricSpec, window_days: int) -> Dict[str, object]:
    metric_df, previous_period, current_period, current_week_start = split_periods(game_df, metric.column)
    previous_value = previous_period[metric.column].mean(skipna=True)
    current_value = current_period[metric.column].mean(skipna=True)
    previous_valid = int(previous_period[metric.column].notna().sum())
    current_valid = int(current_period[metric.column].notna().sum())
    if previous_valid == 0:
        previous_value = None
    if current_valid == 0:
        current_value = None
    change = compute_change(previous_value, current_value)
    return {
        "metric_key": metric.key,
        "metric_label": metric.label,
        "column": metric.column,
        "type": metric.value_type,
        "previous_value": previous_value,
        "current_value": current_value,
        "change": change,
        "previous_valid_days": previous_valid,
        "current_valid_days": current_valid,
        "series": metric_df,
        "current_week_start": current_week_start,
    }


def build_summary(game_df: pd.DataFrame, game_id: str, window_days: int) -> Dict[str, Dict[str, object]]:
    return {
        metric.key: compare_metric(game_df, metric, window_days)
        for metric in METRICS_BY_GAME[game_id]
        if metric.column in game_df.columns
    }


def build_text_template(game_id: str, summary: Dict[str, Dict[str, object]], window_days: int, date_range_text: str) -> List[object]:
    sections: List[object] = [
        html.H3(f"{GAME_LABELS[game_id]} Weekly Summary", style={"marginTop": "0", "marginBottom": "8px"}),
        html.P(f"对比区间：{date_range_text}", style={"margin": "0 0 4px 0", "color": "#475569"}),
        html.P(f"当前按 {window_days} 天窗口计算。", style={"margin": "0 0 16px 0", "color": "#475569"}),
    ]

    if window_days != 7:
        sections.append(
            html.Blockquote(
                "当前 CSV 不足 14 天完整数据，已自动按可用的等长窗口计算，因此这里的“本周/上周”实际表示“本期/上期”。",
                style={"margin": "0 0 16px 0", "color": "#92400e", "borderLeft": "4px solid #f59e0b", "paddingLeft": "12px"},
            )
        )

    sections.append(html.H4("Weekly Insight", style={"marginBottom": "8px"}))
    sections.append(
        html.Ul(
            [html.Li(item, style={"marginBottom": "6px"}) for item in build_weekly_insight(summary)],
            style={"marginTop": "0", "marginBottom": "20px", "paddingLeft": "20px"},
        )
    )

    for section_title, metric_keys in TEXT_TEMPLATE_CONFIG[game_id]:
        sections.append(html.H4(section_title, style={"marginBottom": "10px"}))
        for key in metric_keys:
            metric = summary.get(key)
            if metric:
                sections.append(build_metric_sentence(metric))
        sections.append(html.Div(style={"height": "10px"}))

    return sections


def build_metric_table(summary: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for item in summary.values():
        rows.append({
            "指标": item["metric_label"],
            "上期均值": format_value(item["previous_value"], item["type"]),
            "本期均值": format_value(item["current_value"], item["type"]),
            "环比": format_change(item["change"]).replace("**", ""),
            "上期有效天数": item["previous_valid_days"],
            "本期有效天数": item["current_valid_days"],
        })
    return rows


def build_previous_period_daily_table(summary: Dict[str, Dict[str, object]], window_days: int):
    summary_items = list(summary.values())
    if not summary_items:
        return [], []

    date_series = summary_items[0]["series"]["bj_date_key"].iloc[:window_days]
    date_columns = []
    for idx, date_value in enumerate(date_series, start=1):
        date_columns.append(
            {
                "id": f"day_{idx}",
                "name": [date_value.strftime("%b %-d, %Y"), date_value.strftime("%A")],
            }
        )

    rows = []
    for item in summary_items:
        series = item["series"].iloc[:window_days]
        row = {"metric": item["metric_label"]}
        for idx, (_, series_row) in enumerate(series.iterrows(), start=1):
            row[f"day_{idx}"] = format_value(series_row[item["column"]], item["type"])
        rows.append(row)

    columns = [{"id": "metric", "name": ["指标", "指标"]}] + date_columns
    return rows, columns


def build_period_daily_table(summary: Dict[str, Dict[str, object]], window_days: int, period: str):
    summary_items = list(summary.values())
    if not summary_items:
        return [], []

    current_week_start = summary_items[0]["current_week_start"]
    if period == "current":
        date_series = summary_items[0]["series"].loc[
            summary_items[0]["series"]["bj_date_key"] >= current_week_start, "bj_date_key"
        ]
        selector = lambda df: df["bj_date_key"] >= current_week_start
    else:
        date_series = summary_items[0]["series"].loc[
            summary_items[0]["series"]["bj_date_key"] < current_week_start, "bj_date_key"
        ]
        selector = lambda df: df["bj_date_key"] < current_week_start

    date_columns = []
    for idx, date_value in enumerate(date_series, start=1):
        date_columns.append(
            {
                "id": f"day_{idx}",
                "name": [date_value.strftime("%b %-d, %Y"), date_value.strftime("%A")],
            }
        )

    rows = []
    for item in summary_items:
        series = item["series"].loc[selector(item["series"])]
        row = {"metric": item["metric_label"]}
        for idx, (_, series_row) in enumerate(series.iterrows(), start=1):
            row[f"day_{idx}"] = format_value(series_row[item["column"]], item["type"])
        rows.append(row)

    columns = [{"id": "metric", "name": ["指标", "指标"]}] + date_columns
    return rows, columns


def build_email_summary_block(game_id: str, summary: Dict[str, Dict[str, object]]) -> List[object]:
    blocks: List[object] = []
    for section_title, metric_keys in TEXT_TEMPLATE_CONFIG[game_id]:
        blocks.append(html.Div(section_title, style={"fontWeight": "bold", "marginTop": "12px", "marginBottom": "8px"}))
        for key in metric_keys:
            metric = summary.get(key)
            if metric:
                blocks.append(build_metric_sentence(metric))
    return blocks


def build_email_section(game_id: str, summary: Dict[str, Dict[str, object]], game_df: pd.DataFrame, window_days: int) -> html.Div:
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
            dash_table.DataTable(
                data=table_data,
                columns=table_columns,
                merge_duplicate_headers=True,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "8px", "fontFamily": "Arial, sans-serif", "fontSize": "12px"},
                style_header={"backgroundColor": "#e2e8f0", "fontWeight": "bold"},
                page_action="none",
            ),
        ],
    )


def build_draft_preview(source_df: pd.DataFrame) -> List[object]:
    email_sections = []
    for game_id in get_available_games(source_df):
        game_df = source_df[source_df["game_id"] == game_id].sort_values("bj_date_key")
        window_days = get_fixed_window_days(game_df)
        summary = build_summary(game_df, game_id, window_days)
        email_sections.append(build_email_section(game_id, summary, game_df, window_days))
    return email_sections


def build_metric_summary_card(metric_summary: Dict[str, object], window_days: int) -> html.Div:
    return html.Div(
        children=[
            html.H4(metric_summary["metric_label"], style={"marginTop": "0", "marginBottom": "16px"}),
            html.Div("上期均值", style={"color": "#475569", "marginBottom": "4px"}),
            html.Div(
                format_value(metric_summary["previous_value"], metric_summary["type"]),
                style={"fontSize": "24px", "fontWeight": "bold", "marginBottom": "16px"},
            ),
            html.Div("本期均值", style={"color": "#475569", "marginBottom": "4px"}),
            html.Div(
                format_value(metric_summary["current_value"], metric_summary["type"]),
                style={
                    "fontSize": "24px",
                    "fontWeight": "bold",
                    "marginBottom": "16px",
                    "color": current_value_color(metric_summary["change"], metric_summary["metric_key"]),
                },
            ),
            html.Div("环比", style={"color": "#475569", "marginBottom": "4px"}),
            html.Div(
                format_change(metric_summary["change"]).replace("**", ""),
                style={
                    "fontSize": "20px",
                    "fontWeight": "bold",
                    "marginBottom": "16px",
                    "color": change_text_color(metric_summary["change"], metric_summary["metric_key"]),
                },
            ),
            html.Div(
                "上期均值对应蓝色区间，本期均值对应黄色区间",
                style={"color": "#64748b", "fontSize": "13px"},
            ),
        ]
    )


def build_default_subject(source_df: pd.DataFrame) -> str:
    available_games = get_available_games(source_df)
    if not available_games:
        return "运营周报"
    game_df = source_df[source_df["game_id"] == available_games[0]].sort_values("bj_date_key")
    window_days = get_fixed_window_days(game_df)
    current_dates = game_df["bj_date_key"].drop_duplicates().sort_values().tail(window_days)
    if current_dates.empty:
        return "运营周报"
    start_text = current_dates.iloc[0].strftime("%m%d")
    end_text = current_dates.iloc[-1].strftime("%m%d")
    return f"运营周报{start_text} - {end_text}"


def html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_summary_lines_html(game_id: str, summary: Dict[str, Dict[str, object]]) -> str:
    parts = []
    for section_title, metric_keys in TEXT_TEMPLATE_CONFIG[game_id]:
        parts.append(f"<div style='font-weight:700;margin:12px 0 8px 0;'>{html_escape(section_title)}</div>")
        for key in metric_keys:
            metric = summary.get(key)
            if not metric:
                continue
            current_color = current_value_color(metric["change"], metric["metric_key"])
            change_color = change_text_color(metric["change"], metric["metric_key"])
            parts.append(
                "<div style='margin:0 0 8px 0;line-height:1.7;'>"
                f"本周日均{html_escape(metric['metric_label'])} "
                f"<span style='font-weight:700;color:{current_color};'>{html_escape(format_value(metric['current_value'], metric['type']))}</span>，"
                f"上周 <span style='font-weight:700;color:#0f172a;'>{html_escape(format_value(metric['previous_value'], metric['type']))}</span>，"
                f"环比 <span style='font-weight:700;color:{change_color};'>{html_escape(format_change(metric['change']).replace('**', ''))}</span>"
                "</div>"
            )
    return "".join(parts)


def render_period_table_html(summary: Dict[str, Dict[str, object]], window_days: int, period: str) -> str:
    rows, columns = build_period_daily_table(summary, window_days, period)
    header_cells = "".join(
        f"<th style='border:1px solid #cbd5e1;padding:6px 8px;background:#e2e8f0;text-align:left;'>{html_escape(' / '.join(col['name']))}</th>"
        for col in columns
    )
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>" + "".join(
                f"<td style='border:1px solid #e2e8f0;padding:6px 8px;text-align:left;'>{html_escape(row.get(col['id'], ''))}</td>"
                for col in columns
            ) + "</tr>"
        )
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:12px;'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def render_period_table_email_html(summary: Dict[str, Dict[str, object]], window_days: int, period: str) -> str:
    rows, columns = build_period_daily_table(summary, window_days, period)
    date_columns = columns[1:]
    body_rows = []

    for row in rows:
        day_lines = []
        for col in date_columns:
            day_lines.append(
                "<div style='margin-bottom:4px;'>"
                f"<span style='color:#475569;'>{html_escape(' / '.join(col['name']))}</span>"
                f": <span style='font-weight:600;color:#0f172a;'>{html_escape(row.get(col['id'], ''))}</span>"
                "</div>"
            )
        body_rows.append(
            "<tr>"
            f"<td style='border:1px solid #e2e8f0;padding:8px 10px;vertical-align:top;font-weight:700;width:180px;'>{html_escape(row['metric'])}</td>"
            f"<td style='border:1px solid #e2e8f0;padding:8px 10px;vertical-align:top;'>{''.join(day_lines)}</td>"
            "</tr>"
        )

    return (
        "<table style='border-collapse:collapse;width:100%;font-size:12px;table-layout:fixed;'>"
        "<thead><tr>"
        "<th style='border:1px solid #cbd5e1;padding:8px 10px;background:#e2e8f0;text-align:left;width:180px;'>指标</th>"
        "<th style='border:1px solid #cbd5e1;padding:8px 10px;background:#e2e8f0;text-align:left;'>本期每日值</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def render_chart_images_html(game_id: str, game_df: pd.DataFrame, window_days: int):
    parts = []
    attachments = []
    fig_specs = []
    for row_index, metric_keys in enumerate(EMAIL_CHART_ROWS, start=1):
        for col_index, metric_key in enumerate(metric_keys, start=1):
            metric = next((spec for spec in METRICS_BY_GAME[game_id] if spec.key == metric_key), None)
            fig_specs.append((row_index, col_index, metric_key, metric))

    figures = []
    file_paths = []
    cid_map = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for row_index, col_index, metric_key, metric in fig_specs:
            if metric is None:
                continue
            cid = f"{game_id.lower()}_{metric_key}_{row_index}_{col_index}@weeklyreport"
            cid_map[(row_index, col_index)] = cid
            figures.append(build_chart(game_df, metric, window_days))
            file_paths.append(os.path.join(temp_dir, f"{cid}.png"))

        export_error = None
        if figures:
            try:
                pio.write_images(
                    fig=figures,
                    file=file_paths,
                    format="png",
                    width=EMAIL_IMAGE_WIDTH,
                    height=EMAIL_IMAGE_HEIGHT,
                    scale=1,
                )
                for path, cid in zip(file_paths, cid_map.values()):
                    attachments.append((cid, Path(path).read_bytes()))
            except Exception as exc:
                export_error = str(exc)

        for row_index, metric_keys in enumerate(EMAIL_CHART_ROWS, start=1):
            row_parts = []
            for col_index, metric_key in enumerate(metric_keys, start=1):
                metric = next((spec for spec in METRICS_BY_GAME[game_id] if spec.key == metric_key), None)
                if metric is None:
                    row_parts.append("<td style='width:33%;vertical-align:top;'></td>")
                    continue
                cid = cid_map.get((row_index, col_index))
                if export_error or cid is None:
                    row_parts.append(
                        "<td style='padding:6px;vertical-align:top;'>"
                        f"<div style='font-weight:700;margin:0 0 6px 0;'>{html_escape(metric.label)}</div>"
                        f"<div style='width:{EMAIL_IMAGE_WIDTH}px;height:{EMAIL_IMAGE_HEIGHT}px;border:1px solid #e2e8f0;"
                        "display:flex;align-items:center;justify-content:center;color:#64748b;"
                        "text-align:center;padding:12px;box-sizing:border-box;'>"
                        f"{html_escape(metric.label)} 图表导出失败：{html_escape(export_error or 'unknown error')}"
                        "</div></td>"
                    )
                else:
                    row_parts.append(
                        "<td style='padding:6px;vertical-align:top;'>"
                        f"<div style='font-weight:700;margin:0 0 6px 0;'>{html_escape(metric.label)}</div>"
                        f"<img src='cid:{cid}' style='display:block;width:{EMAIL_IMAGE_WIDTH}px;height:{EMAIL_IMAGE_HEIGHT}px;border:1px solid #e2e8f0;' />"
                        "</td>"
                    )
            parts.append(f"<tr>{''.join(row_parts)}</tr>")
    html_block = (
        "<table style='border-collapse:separate;border-spacing:0 4px;'>"
        f"{''.join(parts)}"
        "</table>"
    )
    return html_block, attachments


def build_email_html(source_df: pd.DataFrame):
    sections = []
    inline_images = []
    for game_id in get_available_games(source_df):
        game_df = source_df[source_df["game_id"] == game_id].sort_values("bj_date_key")
        window_days = get_fixed_window_days(game_df)
        summary = build_summary(game_df, game_id, window_days)
        highlights = "".join(f"<li style='margin-bottom:6px;'>{html_escape(item)}</li>" for item in build_weekly_insight(summary))
        chart_html, chart_attachments = render_chart_images_html(game_id, game_df, window_days)
        inline_images.extend(chart_attachments)
        table_html = render_period_table_html(summary, window_days, "current")
        sections.append(
            "<div style='background:#ffffff;padding:20px;border-radius:12px;margin-bottom:20px;border:1px solid #e2e8f0;'>"
            f"<h2 style='margin:0 0 12px 0;'>【{html_escape(GAME_EMAIL_TITLES[game_id])}】</h2>"
            "<h3 style='margin:0 0 8px 0;'>Weekly Highlight：</h3>"
            f"<ul style='margin:0 0 16px 20px;padding:0;'>{highlights}</ul>"
            "<h3 style='margin:0 0 8px 0;'>Weekly Summary（指标效果：红色↓，绿色↑ ，蓝色→）</h3>"
            f"{render_summary_lines_html(game_id, summary)}"
            "<h3 style='margin:20px 0 8px 0;'>数据趋势图一览</h3>"
            f"{chart_html}"
            "<h3 style='margin:20px 0 8px 0;'>CNY 周数据一览</h3>"
            f"{table_html}"
            "</div>"
        )
    full_html = (
        "<html><body style='font-family:Arial,sans-serif;background:#f8fafc;color:#0f172a;padding:16px;'>"
        f"{''.join(sections)}"
        "</body></html>"
    )
    return full_html, inline_images


def get_gmail_service():
    if not gmail_enabled():
        raise RuntimeError("Gmail 未配置。请先提供 credentials.json 或设置 GMAIL_CREDENTIALS_PATH。")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(f"Gmail 依赖未安装：{exc}") from exc

    if not CREDENTIALS_PATH.exists():
        raise RuntimeError(f"缺少 OAuth 配置文件：{CREDENTIALS_PATH}")

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def create_gmail_draft(subject: str, html_body: str, inline_images):
    try:
        service = get_gmail_service()
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    message = MIMEMultipart("related")
    message["Subject"] = subject
    message.attach(MIMEText(html_body, "html", "utf-8"))

    for cid, image_bytes in inline_images:
        image_part = MIMEBase("image", "png")
        image_part.set_payload(image_bytes)
        encoders.encode_base64(image_part)
        image_part.add_header("Content-ID", f"<{cid}>")
        image_part.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        message.attach(image_part)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {"message": {"raw": raw}}
    return service.users().drafts().create(userId="me", body=body).execute()


def build_top_chart_panel(source_df: pd.DataFrame) -> html.Div:
    available_games = get_available_games(source_df)
    default_game = available_games[0] if available_games else None
    default_metric_options = [
        {"label": metric.label, "value": metric.key}
        for metric in METRICS_BY_GAME[default_game]
    ] if default_game else []

    return html.Div(
        style={"marginBottom": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"},
                children=[
                    html.Div([
                        html.Label("游戏", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="game-select",
                            options=[{"label": GAME_LABELS[game_id], "value": game_id} for game_id in available_games],
                            value=default_game,
                            clearable=False,
                            style={"width": "220px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("指标", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="metric-select",
                            options=default_metric_options,
                            value=default_metric_options[0]["value"] if default_metric_options else None,
                            clearable=False,
                            style={"width": "360px"},
                        ),
                    ]),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1.1fr 0.35fr", "gap": "16px", "alignItems": "start"},
                children=[
                    html.Div(
                        style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 1px 3px rgba(15,23,42,0.08)"},
                        children=[dcc.Graph(id="metric-chart")],
                    ),
                    html.Div(
                        id="metric-summary-card",
                        style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 1px 3px rgba(15,23,42,0.08)"},
                    ),
                ],
            ),
        ],
    )


def build_chart(game_df: pd.DataFrame, metric: MetricSpec, window_days: int) -> go.Figure:
    plot_df = game_df[["bj_date_key", metric.column]].sort_values("bj_date_key").copy()
    latest_date = plot_df["bj_date_key"].max()
    current_week_start = latest_date - pd.Timedelta(days=int(latest_date.weekday()))
    previous_period = plot_df[plot_df["bj_date_key"] < current_week_start].copy()
    current_period = plot_df[plot_df["bj_date_key"] >= current_week_start].copy()

    fig = go.Figure()
    if not previous_period.empty:
        fig.add_trace(go.Scatter(
            x=previous_period["bj_date_key"],
            y=previous_period[metric.column],
            mode="lines+markers",
            name="历史区间",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=5),
            cliponaxis=False,
        ))
    if not current_period.empty:
        fig.add_trace(go.Scatter(
            x=current_period["bj_date_key"],
            y=current_period[metric.column],
            mode="lines+markers",
            name="最近一周",
            line=dict(color="#eab308", width=2),
            marker=dict(size=5),
            cliponaxis=False,
        ))
    if not previous_period.empty and not current_period.empty:
        fig.add_trace(go.Scatter(
            x=[previous_period["bj_date_key"].iloc[-1], current_period["bj_date_key"].iloc[0]],
            y=[previous_period[metric.column].iloc[-1], current_period[metric.column].iloc[0]],
            mode="lines",
            name="连接",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        ))

    prev_mean = previous_period[metric.column].mean(skipna=True)
    curr_mean = current_period[metric.column].mean(skipna=True)
    if not pd.isna(prev_mean):
        fig.add_hline(y=prev_mean, line_dash="dash", line_color="#2563eb")
    if not pd.isna(curr_mean):
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
        title=dict(text=f"{GAME_LABELS[game_df['game_id'].iloc[0]]} | {metric.label}", font=dict(size=16)),
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


def get_available_port(start_port: int) -> int:
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


def get_default_source_df() -> pd.DataFrame:
    return load_source_data(DEFAULT_CSV_PATH)


def get_available_games(source_df: pd.DataFrame) -> List[str]:
    return [game_id for game_id in GAME_LABELS if game_id in source_df["game_id"].dropna().unique()]

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Weekly Report Dashboard"

def serve_layout():
    source_df = get_default_source_df()
    min_date = source_df["bj_date_key"].min().date().isoformat()
    max_date = source_df["bj_date_key"].max().date().isoformat()
    redshift_ready = redshift_enabled()
    gmail_ready = gmail_enabled()
    return html.Div(
        style={
            "maxWidth": "1600px",
            "margin": "0 auto",
            "padding": "24px",
            "fontFamily": "Arial, sans-serif",
            "backgroundColor": "#f8fafc",
        },
        children=[
            html.H2("Weekly Report Dashboard", style={"marginBottom": "8px"}),
            dcc.Store(id="source-data-store", data=source_df.to_json(date_format="iso", orient="split")),
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "marginBottom": "20px", "flexWrap": "wrap"},
                children=[
                    dcc.DatePickerSingle(
                        id="redshift-start-date",
                        date=min_date,
                        display_format="YYYY-MM-DD",
                    ),
                    dcc.DatePickerSingle(
                        id="redshift-end-date",
                        date=max_date,
                        display_format="YYYY-MM-DD",
                    ),
                    dcc.Input(
                        id="redshift-currency",
                        type="text",
                        value="CNY",
                        placeholder="Currency",
                        style={"width": "100px", "padding": "10px 12px", "borderRadius": "8px", "border": "1px solid #cbd5e1"},
                    ),
                    html.Button(
                        "Fetch from Redshift",
                        id="fetch-redshift-data",
                        disabled=not redshift_ready,
                        style={
                            "backgroundColor": "#0f766e" if redshift_ready else "#94a3b8",
                            "color": "white",
                            "border": "none",
                            "padding": "10px 16px",
                            "borderRadius": "8px",
                            "cursor": "pointer" if redshift_ready else "not-allowed",
                            "fontWeight": "bold",
                        },
                    ),
                    html.Div(
                        "Redshift 未配置，当前仅可使用默认 CSV。" if not redshift_ready else "",
                        style={"color": "#64748b", "fontSize": "14px"},
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "marginBottom": "20px", "flexWrap": "wrap"},
                children=[
                    dcc.Input(
                        id="draft-subject",
                        type="text",
                        value=build_default_subject(source_df),
                        style={
                            "width": "420px",
                            "padding": "10px 12px",
                            "borderRadius": "8px",
                            "border": "1px solid #cbd5e1",
                        },
                    ),
                    html.Button(
                        "Create Gmail Draft",
                        id="create-gmail-draft",
                        disabled=not gmail_ready,
                        style={
                            "backgroundColor": "#16a34a" if gmail_ready else "#94a3b8",
                            "color": "white",
                            "border": "none",
                            "padding": "10px 16px",
                            "borderRadius": "8px",
                            "cursor": "pointer" if gmail_ready else "not-allowed",
                            "fontWeight": "bold",
                        },
                    ),
                    html.Div(
                        id="gmail-draft-status",
                        children="" if gmail_ready else "Gmail 未配置，当前不可创建 Draft。",
                        style={"color": "#475569"},
                    ),
                ],
            ),
            html.Div(
                id="draft-progress",
                style={
                    "display": "none",
                    "width": "520px",
                    "height": "10px",
                    "backgroundColor": "#dbeafe",
                    "borderRadius": "999px",
                    "overflow": "hidden",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        style={
                            "width": "100%",
                            "height": "100%",
                            "backgroundColor": "#2563eb",
                            "borderRadius": "999px",
                        }
                    )
                ],
            ),
            dcc.ConfirmDialog(id="gmail-draft-dialog"),
            html.Div(
                style={"marginTop": "20px"},
                children=[
                    build_top_chart_panel(source_df),
                    html.H2("Weekly Draft Preview", style={"marginBottom": "12px"}),
                    html.Div(id="gmail-draft-preview", children=build_draft_preview(source_df)),
                ],
            ),
        ],
    )


app.layout = serve_layout


@app.callback(
    Output("source-data-store", "data"),
    Input("fetch-redshift-data", "n_clicks"),
    Input("redshift-start-date", "date"),
    Input("redshift-end-date", "date"),
    Input("redshift-currency", "value"),
    prevent_initial_call=True,
)
def update_uploaded_data(
    fetch_clicks: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
    currency_value: Optional[str],
):
    triggered = dash.callback_context.triggered_id
    if triggered == "fetch-redshift-data":
        try:
            df = fetch_redshift_data(
                (start_date or "").strip(),
                (end_date or "").strip(),
                (currency_value or "CNY").strip().upper(),
            )
        except Exception as exc:
            raise RuntimeError(f"Redshift 取数失败：{exc}") from exc
        return df.to_json(date_format="iso", orient="split")
    return dash.no_update


@app.callback(
    Output("metric-select", "options"),
    Output("metric-select", "value"),
    Input("source-data-store", "data"),
    Input("game-select", "value"),
)
def update_metric_options(source_data_json: str, current_game_id: Optional[str]):
    source_df = dataframe_from_store(source_data_json)
    available_games = get_available_games(source_df)
    game_value = current_game_id if current_game_id in available_games else available_games[0]
    metric_options = [{"label": metric.label, "value": metric.key} for metric in METRICS_BY_GAME[game_value]]
    return metric_options, metric_options[0]["value"]


@app.callback(
    Output("draft-subject", "value"),
    Input("source-data-store", "data"),
)
def update_draft_subject(source_data_json: str):
    source_df = dataframe_from_store(source_data_json)
    return build_default_subject(source_df)


@app.callback(
    Output("metric-chart", "figure"),
    Output("metric-summary-card", "children"),
    Input("source-data-store", "data"),
    Input("game-select", "value"),
    Input("metric-select", "value"),
)
def update_metric_chart(source_data_json: str, game_id: str, metric_key: str):
    source_df = dataframe_from_store(source_data_json)
    game_df = source_df[source_df["game_id"] == game_id].sort_values("bj_date_key")
    window_days = get_fixed_window_days(game_df)
    metric = next(spec for spec in METRICS_BY_GAME[game_id] if spec.key == metric_key)
    metric_summary = compare_metric(game_df, metric, window_days)
    return build_chart(game_df, metric, window_days), build_metric_summary_card(metric_summary, window_days)


@app.callback(
    Output("gmail-draft-preview", "children"),
    Input("source-data-store", "data"),
)
def update_dashboard(source_data_json: str):
    source_df = dataframe_from_store(source_data_json)
    return build_draft_preview(source_df)


@app.callback(
    Output("gmail-draft-status", "children"),
    Output("gmail-draft-dialog", "displayed"),
    Output("gmail-draft-dialog", "message"),
    Input("create-gmail-draft", "n_clicks"),
    Input("source-data-store", "data"),
    Input("draft-subject", "value"),
    prevent_initial_call=True,
    running=[
        (
            Output("draft-progress", "style"),
            {
                "display": "block",
                "width": "520px",
                "height": "10px",
                "backgroundColor": "#dbeafe",
                "borderRadius": "999px",
                "overflow": "hidden",
                "marginBottom": "20px",
            },
            {
                "display": "none",
                "width": "520px",
                "height": "10px",
                "backgroundColor": "#dbeafe",
                "borderRadius": "999px",
                "overflow": "hidden",
                "marginBottom": "20px",
            },
        ),
        (Output("create-gmail-draft", "disabled"), True, False),
    ],
)
def create_gmail_draft_callback(n_clicks: Optional[int], source_data_json: str, subject: Optional[str]):
    triggered = dash.callback_context.triggered_id
    if triggered != "create-gmail-draft":
        return dash.no_update, False, dash.no_update

    source_df = dataframe_from_store(source_data_json)
    draft_subject = subject.strip() if subject else build_default_subject(source_df)
    try:
        html_body, inline_images = build_email_html(source_df)
        draft = create_gmail_draft(draft_subject, html_body, inline_images)
    except Exception as exc:
        message = f"创建 Gmail Draft 失败：{exc}"
        return message, True, message

    draft_id = draft.get("id", "unknown")
    message = f"Gmail Draft 创建成功，draft id: {draft_id}"
    return message, True, message


if __name__ == "__main__":
    requested_port = int(os.environ.get("PORT", "8051"))
    strict_port = os.environ.get("STRICT_PORT", "0") == "1"
    debug_mode = os.environ.get("DASH_DEBUG", "1") == "1"
    app.run(
        debug=debug_mode,
        port=requested_port if strict_port else get_available_port(requested_port),
    )
