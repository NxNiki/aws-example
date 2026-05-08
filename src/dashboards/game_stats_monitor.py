import logging
import os
import re
import socket
import uuid
from datetime import datetime, timedelta
from math import inf
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.development.base_component import Component
from flask import request
from plotly.subplots import make_subplots

from bituslabs_ds.config import DASHBOARD_CONFIG_S3_PATH, LOCAL_ROOT, setup_logging
from bituslabs_ds.dashboard_utils import bootstrap_worker, load_config
from bituslabs_ds.s3_utils import list_s3_files, parse_s3_path, read_files, read_json_from_s3, write_json_to_s3
from dashboards.user_stats_aggregates import (
    ENRICH_PRODUCED_COLUMNS,
    ENRICH_USER_ROW_INPUT_COLUMNS,
    RETENTION_LOAD_EXTRA_DAYS,
    DataMetrics,
)
from dashboards.weekly_report import (
    METRICS_BY_GAME,
    build_chart,
    build_draft_preview_single,
    build_metric_summary_card,
    compare_metric,
    dataframe_from_split_json,
    empty_figure,
    format_weekly_report_subject,
    get_fixed_window_days,
    layout_weekly_report_panel,
    slice_weekly_report_by_end_date,
    transform_source_data,
    weekly_report_end_date_bounds,
)

# The AI chat agent runs as a separate service (FastAPI + Uvicorn).
# The dashboard calls it over HTTP — no langchain/langgraph deps needed here.
_CHAT_API_URL = os.environ.get("CHAT_API_URL", "http://localhost:8051")

# ==========================================
# Styling & Constants
# ==========================================

# Dash exposes `Dropdown.Options` as a TypedDict for this prop.
_CHAT_PROVIDER_DROPDOWN_OPTIONS: list[dcc.Dropdown.Options] = [
    {"label": "Gemini 2.5 Flash (fast)", "value": "gemini:gemini-2.5-flash"},
    {"label": "Gemini 2.5 Pro (powerful)", "value": "gemini:gemini-2.5-pro"},
    {"label": "GPT-4o-mini (fast)", "value": "openai:gpt-4o-mini"},
    {"label": "GPT-4o (balanced)", "value": "openai:gpt-4o"},
    {"label": "GPT-4.1-mini (balanced)", "value": "openai:gpt-4.1-mini"},
    {"label": "GPT-4.1 (powerful)", "value": "openai:gpt-4.1"},
]

# Per-figure export dimensions for the report-agent "Add to doc" button.
# Widths are wider than the on-screen render so axis ticks and category
# labels (especially the bar plots in Stats by Group) stay legible after
# Confluence scales the image to its column width. Heights match the
# dashboard so aspect ratios are preserved.
_FIGURE_EXPORT_DIMENSIONS: Dict[str, tuple[int, int]] = {
    "date-g1-plot": (1800, 430),
    "date-g2-plot": (1800, 430),
    "date-g3-plot": (1800, 430),
    "group-g1-plot": (1800, 430),
    "group-g2-plot": (1800, 430),
    "group-g3-plot": (1800, 430),
    "viz-p1-plot": (1800, 900),
    "viz-p2-plot": (1800, 900),
    "combined-plot": (1800, 950),
}


def _dropdown_options_from_strings(items: Sequence[str]) -> list[dcc.Dropdown.Options]:
    """Same string for label and value; matches Dash ``Dropdown.Options``."""
    return [{"label": n, "value": n} for n in items]


def _dropdown_option_rows(rows: Sequence[tuple[str, str]]) -> list[dcc.Dropdown.Options]:
    """Arbitrary label/value pairs; valid for ``dcc.Dropdown``, ``Checklist``, and ``RadioItems`` ``options``."""
    return [{"label": a, "value": b} for a, b in rows]


logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


def _json_sanitize(obj: Any) -> Any:
    """Convert obj to JSON-serializable types (for S3 save)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def _normalize_date_value(val: Any) -> Optional[str]:
    """Normalize date to YYYY-MM-DD string for Dash date pickers."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if "T" in s:
        s = s.split("T")[0]
    return s


def _normalize_dates_in_state(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize all date-like values in a state dict for saving (YYYY-MM-DD)."""
    if state is None:
        return None
    if not isinstance(state, dict):
        return state  # type: ignore[return-value]
    date_keys = {
        "start_date",
        "end_date",
        "date_range1_start",
        "date_range1_end",
        "date_range2_start",
        "date_range2_end",
        "date_range3_start",
        "date_range3_end",
    }
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if k in date_keys and v is not None:
            out[k] = _normalize_date_value(v)
        elif isinstance(v, dict):
            out[k] = _normalize_dates_in_state(v)
        else:
            out[k] = v
    return out


class Styles:
    COLORS: List[str] = [
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#FF7F00",
        "#984EA3",
        "#A65628",
        "#F781BF",
        "#999999",
    ]

    LINE_SHAPE: List[str] = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

    # Tab styles
    NAV_TAB_SELECTED: Dict[str, Any] = {
        "borderTop": "3px solid #377EB8",
        "borderBottom": "1px solid white",
        "backgroundColor": "white",
        "color": "#377EB8",
        "fontWeight": "bold",
        "padding": "12px",
    }

    NAV_TAB: Dict[str, Any] = {
        "padding": "12px",
        "backgroundColor": "#f9f9f9",
        "color": "#555",
        "border": "1px solid #d6d6d6",
    }

    # Card/panel styles
    BOX: Dict[str, Any] = {
        "border": "1px solid #e0e0e0",
        "borderRadius": "8px",
        "backgroundColor": "#ffffff",
        "padding": "24px",  # Increased slightly for breathing room
        "boxShadow": "0 2px 10px 0 rgba(0,0,0,0.05)",
        "marginBottom": "20px",
    }
    PANEL_HEADER: Dict[str, Any] = {
        "marginTop": "0",
        "marginBottom": "20px",  # Consistent space below header
        "color": "#377EB8",
        "fontWeight": "bold",
    }

    # Section wrappers
    SECTION: Dict[str, Any] = {
        "marginBottom": "20px",
        "padding": "18px",
        "backgroundColor": "white",
        "borderRadius": "8px",
        "border": "1px solid #e0e0e0",
    }
    SECTION_GROUP: Dict[str, Any] = {
        "marginBottom": "10px",
        "padding": "20px",
        "backgroundColor": "white",
        "borderRadius": "8px",
        "border": "1px solid #e0e0e0",
    }

    # Layout/flex
    FLEX_ROW: Dict[str, Any] = {
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "space-between",
        "alignItems": "flex-start",
    }
    FLEX_ROW_CENTER: Dict[str, Any] = {
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "center",
        "justifyContent": "flex-start",
    }

    # Grid Layouts
    CONTROL_PANEL_CONTAINER: Dict[str, Any] = {
        "width": "20%",
        "minWidth": "250px",  # Prevent squashing
        "display": "inline-block",
        "verticalAlign": "top",
        "paddingRight": "20px",
        "boxSizing": "border-box",
    }

    GRAPH_CONTAINER: Dict[str, Any] = {
        "width": "100%",
        "display": "inline-block",
        "verticalAlign": "top",
        "padding": "10px",
    }

    # Main layout
    NAV_CONTAINER: Dict[str, Any] = {
        "width": "100%",
        "minWidth": "330px",
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "flex-start",
        "alignItems": "flex-start",
        "borderBottom": "1px solid #eee",
        "marginBottom": "20px",
    }
    PAGE_CONTENT: Dict[str, Any] = {
        "padding": "0 20px 20px 20px",
        "backgroundColor": "#f4f7f6",
        "minHeight": "100vh",
    }

    # --- CONTROL STYLING CONSTANTS ---

    # 1. Standard Label
    CONTROL_LABEL: Dict[str, Any] = {
        "fontWeight": "bold",
        "marginBottom": "8px",  # Space between Label and Input
        "display": "block",  # Forces label to own line
        "fontSize": "14px",
    }

    # 2. Standard Wrapper for Label+Input pairs
    CONTROL_GROUP: Dict[str, Any] = {
        "marginTop": "0px",
        "marginBottom": "2px",
        "paddingTop": "2px",
        "paddingBottom": "0px",
    }

    # 3. Horizontal Rule Standard
    HR: Dict[str, Any] = {"marginTop": "10px", "marginBottom": "10px", "border": "0", "borderTop": "1px solid #eee"}

    # 4. Helper Utils
    DROPDOWN_NARROW: Dict[str, Any] = {"width": "150px"}
    DROPDOWN_WIDE: Dict[str, Any] = {"width": "100%"}  # Fill container

    INPUT_SMALL: Dict[str, Any] = {"width": "70px", "marginLeft": "4px", "marginRight": "10px"}
    INPUT_FULLWIDTH: Dict[str, Any] = {"width": "100%"}

    CHECKLIST_INLINE: Dict[str, Any] = {
        "display": "inline-block",
        "marginRight": "10px",
    }
    CHECKLIST_BLOCK: Dict[str, Any] = {
        "display": "block",
        "marginBottom": "6px",
    }


# ==========================================
# Dashboard Logic
# ==========================================


class GameStatsDashboard:

    DEFAULT_VISIBLE_GROUPS: int = 2

    config_files: list[dcc.Dropdown.Options]
    config: dict
    lfs_by_date: Dict[str, pl.LazyFrame]
    lf_bet: pl.LazyFrame
    df_bet_groups: List[str]
    df_date_group_col: str
    df_bet_group_col: str
    df_user_group_col: str
    df_user_group_dropdown_options: list[dcc.Dropdown.Options]
    sessions: List[str]
    bet_metrics: List[str]
    plot_metrics: Dict[str, List[str]]
    _weekly_report_df: Optional[pd.DataFrame]

    def __init__(self, config_dir: str, host_ip: str = "127.0.0.1"):
        self.host_ip: str = host_ip
        self.config_dir: str = config_dir

        self.config_files = self._find_config_files()
        self._reset_state()
        self.config = load_config(str(self.config_files[0]["value"]))

        self.app: Dash = Dash(__name__, suppress_callback_exceptions=True)

        @self.app.server.route("/health")
        def health():
            return "ok", 200

        @self.app.server.after_request
        def _emit_user_request_metric(response):
            """Emit CloudWatch metric for user requests (excludes /health for scale-to-zero)."""
            if request.path != "/health":
                try:
                    import boto3

                    cw = boto3.client("cloudwatch")
                    service_name = os.environ.get("DASHBOARD_SERVICE_NAME", "game-stats-dashboard")
                    cw.put_metric_data(
                        Namespace="Dashboard",
                        MetricData=[
                            {
                                "MetricName": "UserRequestCount",
                                "Dimensions": [{"Name": "Service", "Value": service_name}],
                                "Value": 1,
                                "Unit": "Count",
                            }
                        ],
                    )
                except Exception as e:
                    logger.debug("Could not emit UserRequestCount metric: %s", e)
            return response

        self._load_date_data()
        self._load_weekly_report_data()
        self._ensure_date_plot_groups()
        self._ensure_bet_groups()
        self._build_main_layout()
        self._register_callbacks()
        self._register_chat_callbacks()
        self._register_report_tab_callbacks()

    @property
    def date_col(self) -> str:
        return self.config["stats_by_date"]["date_col"]

    def _reset_state(self) -> None:
        self.config = {}
        self.lfs_by_date = {}
        self.lf_bet = pl.DataFrame().lazy()
        self.sessions = []
        self.df_bet_groups = []
        self.df_date_group_col = ""
        self.df_bet_group_col = ""
        self.df_user_group_col = ""
        self.df_user_group_cols: List[str] = []
        self.df_user_group_col_options: list[dcc.Dropdown.Options] = []
        self.df_user_group_dropdown_options = _dropdown_options_from_strings(["all"])
        self.bet_metrics = []
        self.plot_metrics = {}
        # When loading a saved config, we may want to reuse its group date ranges
        # when rebuilding the group layout instead of recomputing defaults.
        self._loaded_group_date_ranges: Optional[
            Tuple[
                Optional[Any],
                Optional[Any],
                Optional[Any],
                Optional[Any],
                Optional[Any],
                Optional[Any],
                Optional[Any],
                Optional[Any],
            ]
        ] = None
        self._weekly_report_df = None

    def _find_config_files(self) -> list[dcc.Dropdown.Options]:
        config_files: list[dcc.Dropdown.Options] = []
        config_dir_path = Path(self.config_dir)
        if not config_dir_path.exists() or not config_dir_path.is_dir():
            config_dir_path = Path(__file__).parent
        pattern = re.compile(r"^dashboard_config.*\.yaml$")
        for file_path in config_dir_path.glob("*.yaml"):
            if pattern.match(file_path.name):
                try:
                    temp_config = load_config(str(file_path))
                    title = temp_config.get("title", file_path.stem)
                    config_files.append({"label": title, "value": str(file_path.absolute())})
                except Exception as e:
                    logger.warning(f"Could not load config {file_path}: {e}")
                    config_files.append({"label": file_path.stem, "value": str(file_path.absolute())})
        return sorted(config_files, key=lambda x: str(x["label"]))

    def _load_date_data(self) -> None:
        for gran, file_paths in self.date_files_config.items():
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            valid_paths = [f for f in file_paths if f]
            if not valid_paths:
                self.lfs_by_date[gran] = pl.DataFrame().lazy()
                continue

            lf_date = read_files(valid_paths, lazy_load=True, return_as_list=False, expand_s3_prefixes=True)
            if isinstance(lf_date, pl.LazyFrame) and lf_date.collect_schema().len() > 0:
                if self.date_col in lf_date.collect_schema().names():
                    lf_date = lf_date.with_columns(pl.col(self.date_col).cast(pl.Datetime))
                self.lfs_by_date[gran] = lf_date
            else:
                self.lfs_by_date[gran] = pl.DataFrame().lazy()

        self.plot_metrics = {}
        self.plot_metrics["g1"] = self.config["stats_by_date"]["group1_columns"]
        self.plot_metrics["g2"] = self.config["stats_by_date"]["group2_columns"]
        self.plot_metrics["g3"] = self.config["stats_by_date"]["group3_columns"]

    def _load_bet_data(self) -> None:
        if self.lf_bet.collect_schema().len() > 0 and self.sessions:
            return

        file_paths = [f for f in self.config["stats_by_bet"]["files"] if f]
        if not file_paths:
            self.lf_bet = pl.DataFrame().lazy()
            self.sessions = []
            self.bet_metrics = []
            return

        lf_bet_raw = read_files(file_paths, lazy_load=True, return_as_list=False, parallel_mode="thread")
        if not isinstance(lf_bet_raw, pl.LazyFrame) or lf_bet_raw.collect_schema().len() == 0:
            self.lf_bet = pl.DataFrame().lazy()
            self.sessions = []
            self.bet_metrics = []
            return

        self.lf_bet = lf_bet_raw.with_columns(pl.col("bet_index").cast(pl.Float64)).filter(
            pl.col("bet_index").is_not_nan()
        )
        schema_names = self.lf_bet.collect_schema().names()
        exclude_bet_cols = {"session_start_date", "session_group", "bet_index"}
        self.bet_metrics = [c for c in schema_names if c not in exclude_bet_cols]
        sessions_df = self.lf_bet.select(pl.col("session_start_date").unique().cast(pl.Utf8)).collect()
        self.sessions = sorted(sessions_df.to_series().to_list()) if not sessions_df.is_empty() else []

    def _load_weekly_report_data(self) -> None:
        self._weekly_report_df = None
        wr = self.config.get("weekly_report")
        if not wr:
            return
        paths = [f for f in wr.get("files", []) if f]
        if not paths:
            logger.warning("weekly_report is present but weekly_report.files is empty; no data will load.")
            return
        try:
            raw = read_files(paths, lazy_load=False, expand_s3_prefixes=True)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                self._weekly_report_df = transform_source_data(raw)
        except Exception as e:
            logger.warning("Failed to load weekly report data: %s", e)
            self._weekly_report_df = None

    def _ensure_plot_metrics_loaded(self) -> None:
        """Ensure plot_metrics has g1/g2/g3 keys; repopulate from config if missing."""
        if "g1" not in self.plot_metrics and "stats_by_date" in self.config:
            sd = self.config["stats_by_date"]
            self.plot_metrics["g1"] = sd.get("group1_columns", [])
            self.plot_metrics["g2"] = sd.get("group2_columns", [])
            self.plot_metrics["g3"] = sd.get("group3_columns", [])

    def _reload_config(self, config_file: str) -> None:
        self.config = load_config(config_file)
        self._load_date_data()
        self._load_weekly_report_data()
        # self._load_bet_data()

    def _compute_group_date_ranges(self) -> Tuple[
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Any],
    ]:
        """
        For Stats by Group: Compute for 3 sequential two-week ranges (6 weeks total, most recent).
        Returns: min_date, max_date, g1_start, g1_end, g2_start, g2_end, g3_start, g3_end
        """
        # tab_group always uses the "day" dataset.
        granularity = "day"
        lf_date = self.lfs_by_date.get(granularity, pl.DataFrame().lazy())
        if lf_date.collect_schema().len() == 0 or self.date_col not in lf_date.collect_schema().names():
            return (None, None, None, None, None, None, None, None)
        agg = lf_date.select(
            pl.col(self.date_col).min().alias("min_d"),
            pl.col(self.date_col).max().alias("max_d"),
        ).collect()
        min_date = agg.item(0, "min_d")
        max_date = agg.item(0, "max_d")
        if min_date is None or max_date is None:
            return (None, None, None, None, None, None, None, None)

        # Calculate three consecutive 2-week ranges ending at max_date
        g3_end = max_date
        g3_start = max(min_date, g3_end - timedelta(days=13))
        g2_end = g3_start - timedelta(days=1)
        g2_start = max(min_date, g2_end - timedelta(days=13)) if g2_end >= min_date else None
        g1_end = g2_start - timedelta(days=1) if g2_start is not None else None
        g1_start = max(min_date, g1_end - timedelta(days=13)) if g1_end is not None and g1_end >= min_date else None
        return min_date, max_date, g1_start, g1_end, g2_start, g2_end, g3_start, g3_end

    @property
    def date_range(self) -> Tuple[Any, Any]:
        granularity = list(self.date_files_config.keys())[0]
        lf_date = self.lfs_by_date.get(granularity, pl.DataFrame().lazy())
        if lf_date.collect_schema().len() == 0 or self.date_col not in lf_date.collect_schema().names():
            return None, None
        agg = lf_date.select(
            pl.col(self.date_col).min().alias("min_d"),
            pl.col(self.date_col).max().alias("max_d"),
        ).collect()
        return agg.item(0, "min_d"), agg.item(0, "max_d")

    @property
    def date_files_config(self) -> Dict[str, Union[str, List[str]]]:
        raw = self.config["stats_by_date"].get("files", {})
        if isinstance(raw, list):
            return {"day": cast(List[str], raw)}
        return cast(Dict[str, Union[str, List[str]]], raw)

    @staticmethod
    def to_rgba(color: str, alpha: float = 0.2) -> str:
        try:
            return "rgba" + str(tuple(int(c * 255) for c in mcolors.to_rgb(color)) + (alpha,))
        except Exception:
            return f"rgba(0,0,0,{alpha})"

    @staticmethod
    def _parse_date(date_value: Optional[Any]) -> Optional[datetime]:
        """
        Convert date value from Dash date picker (string) to Python datetime.

        Args:
            date_value: String date from Dash picker (e.g., "2024-12-02" or "2024-12-02T00:00:00")
                       or datetime object, or None

        Returns:
            datetime object or None
        """
        if date_value is None:
            return None
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, str):
            date_str = date_value.strip()
            if not date_str:
                return None
            try:
                # Try ISO format first (handles "2024-12-02T00:00:00" or "2024-12-02T00:00:00Z")
                # Replace Z with +00:00 for fromisoformat
                if "Z" in date_str:
                    date_str = date_str.replace("Z", "+00:00")
                return datetime.fromisoformat(date_str)
            except ValueError:
                try:
                    # Fallback to date-only format "2024-12-02"
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError as e:
                    logger.warning(f"Could not parse date value '{date_value}': {e}")
                    return None
        # Try to convert other types (e.g., pandas Timestamp, numpy datetime64)
        try:
            return datetime.fromisoformat(str(date_value))
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse date value '{date_value}' (type: {type(date_value)})")
            return None

    @staticmethod
    def hybrid_transform(arr: Union[np.ndarray, pl.Series], thresh: float) -> np.ndarray:
        """Hybrid scale: linear below thresh, log above. Input array-like, returns numpy."""
        a = np.asarray(arr, dtype=float)
        mask = np.isfinite(a)
        out = np.full_like(a, np.nan)
        out[mask] = np.where(
            a[mask] <= thresh,
            a[mask],
            thresh * (1.0 + np.log(np.maximum(a[mask] / thresh, 1.0))),
        )
        return out

    @staticmethod
    def get_ticks(all_y: Union[List[float], np.ndarray], thresh: float, n_ticks: int = 8) -> np.ndarray:
        """Generate tick values for hybrid log axis from flat list of values."""
        arr = np.asarray(all_y, dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.array([0, thresh])
        lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
        if lo >= hi:
            return np.array([lo])
        below = np.linspace(lo, min(hi, thresh), max(2, n_ticks // 2))
        above = valid[valid > thresh]
        if len(above) > 0:
            log_hi = np.log(np.nanmax(above) / thresh + 1e-12)
            if log_hi > 0:
                t_above = thresh * (1.0 + np.linspace(0, log_hi, max(2, n_ticks // 2)))
                ticks = np.unique(np.r_[below, thresh, t_above])
            else:
                ticks = np.unique(below)
        else:
            ticks = np.unique(below)
        return ticks

    def compute_axis_range(
        self,
        df_groups: List[pl.DataFrame],
        metrics: List[str],
        log_scale: bool,
        log_thresh: float,
    ) -> Optional[Tuple[float, float]]:
        """Compute shared y-axis range from list of Polars DataFrames and metrics."""
        vals: List[float] = []
        for df in df_groups:
            if df.is_empty():
                continue
            for m in metrics:
                if m not in df.columns:
                    continue
                s = df.get_column(m).cast(pl.Float64).fill_null(float("nan"))
                vals.extend(s.to_numpy().tolist())
        valid = [v for v in vals if np.isfinite(v)]
        if not valid:
            return None
        arr = np.array(valid)
        if log_scale:
            arr = self.hybrid_transform(arr, log_thresh)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        return (lo, hi * 1.05 if hi > lo else hi + 1.0)

    def update_log_ticks(
        self,
        fig: go.Figure,
        df_groups: List[pl.DataFrame],
        left_metrics: List[str],
        right_metrics: List[str],
        log_thresh: float,
    ) -> None:
        """Set y-axis tick values for hybrid log scale."""
        all_left: List[float] = []
        all_right: List[float] = []
        for df in df_groups:
            for m in left_metrics:
                if m in df.columns:
                    all_left.extend(df.get_column(m).cast(pl.Float64).to_numpy().tolist())
            for m in right_metrics:
                if m in df.columns:
                    all_right.extend(df.get_column(m).cast(pl.Float64).to_numpy().tolist())
        if all_left:
            yticks = self.get_ticks(all_left, log_thresh)
            fig.update_yaxes(
                tickvals=self.hybrid_transform(yticks, log_thresh).tolist(),
                ticktext=[f"{v:.0f}" for v in yticks],
                secondary_y=False,
            )
        if all_right:
            yticks_r = self.get_ticks(all_right, log_thresh)
            fig.update_yaxes(
                tickvals=self.hybrid_transform(yticks_r, log_thresh).tolist(),
                ticktext=[f"{v:.0f}" for v in yticks_r],
                secondary_y=True,
            )

    # ------------------------------------------------------------------
    # Layout Builders
    # ------------------------------------------------------------------

    def _build_main_layout(self) -> None:
        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="config-dropdown",
                                    options=self.config_files,
                                    value=self.config_files[0]["value"],
                                    clearable=False,
                                    style={
                                        "width": "330px",
                                        "marginLeft": "8px",
                                        "marginRight": "12px",
                                    },
                                ),
                                html.Button(
                                    "Save Config",
                                    id="save-config-btn",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "auto",
                                        "marginRight": "8px",
                                        "padding": "6px 12px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#377EB8",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Button(
                                    "Load Config",
                                    id="load-config-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 12px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#4DAF4A",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "fontSize": "13px",
                                    },
                                ),
                                html.Div(
                                    id="save-config-modal",
                                    style={
                                        "display": "none",
                                        "position": "fixed",
                                        "top": "0",
                                        "left": "0",
                                        "width": "100%",
                                        "height": "100%",
                                        "backgroundColor": "rgba(0,0,0,0.5)",
                                        "zIndex": 1000,
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                        "flexDirection": "row",
                                    },
                                    children=[
                                        html.Div(
                                            [
                                                html.H4("Save dashboard config", style={"marginTop": "0"}),
                                                html.Label("Config name:", style=Styles.CONTROL_LABEL),
                                                dcc.Input(
                                                    id="save-config-filename",
                                                    type="text",
                                                    placeholder="my_config.json",
                                                    style={**Styles.INPUT_FULLWIDTH, "marginBottom": "12px"},
                                                ),
                                                html.Label("Or overwrite existing:", style=Styles.CONTROL_LABEL),
                                                dcc.Dropdown(
                                                    id="save-config-overwrite",
                                                    options=[],
                                                    placeholder="Select to overwrite...",
                                                    clearable=True,
                                                    style={"marginBottom": "12px"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Save",
                                                            id="save-config-confirm",
                                                            n_clicks=0,
                                                            style={
                                                                "marginRight": "8px",
                                                                "padding": "8px 16px",
                                                                "cursor": "pointer",
                                                                "backgroundColor": "#377EB8",
                                                                "color": "white",
                                                                "border": "none",
                                                                "borderRadius": "4px",
                                                            },
                                                        ),
                                                        html.Button(
                                                            "Cancel",
                                                            id="save-config-cancel",
                                                            n_clicks=0,
                                                            style={
                                                                "padding": "8px 16px",
                                                                "cursor": "pointer",
                                                                "backgroundColor": "#999",
                                                                "color": "white",
                                                                "border": "none",
                                                                "borderRadius": "4px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"display": "flex", "marginTop": "12px"},
                                                ),
                                            ],
                                            style={
                                                "backgroundColor": "white",
                                                "padding": "24px",
                                                "borderRadius": "8px",
                                                "minWidth": "350px",
                                                "boxShadow": "0 4px 20px rgba(0,0,0,0.15)",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="load-config-dropdown-container",
                                    style={"display": "none", "marginLeft": "8px", "minWidth": "200px"},
                                    children=[
                                        html.Label(
                                            "Select config to load:", style={"fontSize": "12px", "marginBottom": "4px"}
                                        ),
                                        dcc.Dropdown(
                                            id="load-config-dropdown",
                                            options=[],
                                            placeholder="Loading...",
                                            clearable=True,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "AI Assistant",
                                    id="chat-toggle-btn",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "12px",
                                        "padding": "6px 14px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#FF7F00",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "fontSize": "13px",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "center",
                                "width": "100%",
                                "padding": "8px 0",
                                "flexWrap": "wrap",
                                "rowGap": "6px",
                            },
                        ),
                        html.Div(
                            [
                                dcc.Tabs(
                                    id="navigator-tabs",
                                    value="tab-date",
                                    children=[
                                        dcc.Tab(
                                            label="Stats by Date",
                                            value="tab-date",
                                            style=Styles.NAV_TAB,
                                            selected_style=Styles.NAV_TAB_SELECTED,
                                        ),
                                        dcc.Tab(
                                            label="Stats by Group",
                                            value="tab-group",
                                            style=Styles.NAV_TAB,
                                            selected_style=Styles.NAV_TAB_SELECTED,
                                        ),
                                        dcc.Tab(
                                            label="Stats Deepdive",
                                            value="tab-viz",
                                            style=Styles.NAV_TAB,
                                            selected_style=Styles.NAV_TAB_SELECTED,
                                        ),
                                        dcc.Tab(
                                            label="Weekly Report",
                                            value="tab-weekly",
                                            style=Styles.NAV_TAB,
                                            selected_style=Styles.NAV_TAB_SELECTED,
                                        ),
                                        dcc.Tab(
                                            label="Stats by Bet",
                                            value="tab-bet",
                                            style=Styles.NAV_TAB,
                                            selected_style=Styles.NAV_TAB_SELECTED,
                                        ),
                                        dcc.Tab(
                                            label="Report",
                                            value="tab-report",
                                            style=Styles.NAV_TAB,
                                            selected_style=Styles.NAV_TAB_SELECTED,
                                        ),
                                    ],
                                    style={"width": "1280px"},
                                ),
                            ],
                            style={
                                "width": "100%",
                                "padding": "0 8px",
                            },
                        ),
                    ],
                    style={
                        "width": "100%",
                        "minWidth": "330px",
                        "display": "flex",
                        "flexDirection": "column",
                        "borderBottom": "1px solid #eee",
                        "marginBottom": "20px",
                    },
                ),
                dcc.Store(id="tab-date-g1-state", data=None),
                dcc.Store(id="tab-date-g2-state", data=None),
                dcc.Store(id="tab-date-g3-state", data=None),
                dcc.Store(id="tab-group-g1-state", data=None),
                dcc.Store(id="tab-group-g2-state", data=None),
                dcc.Store(id="tab-group-g3-state", data=None),
                dcc.Store(id="tab-viz-p1-state", data=None),
                dcc.Store(id="tab-viz-p2-state", data=None),
                dcc.Store(id="tab-bet-state", data=None),
                dcc.Store(id="dashboard-load-trigger", data=None),
                dcc.Store(id="chat-history", data=[]),
                dcc.Store(id="chat-pending-request", data=None),
                html.Div(
                    self._layout_stats_by_date(),
                    id="tab-date-content",
                    style={**Styles.PAGE_CONTENT, "display": "block"},
                ),
                html.Div(
                    self._layout_stats_by_group(),
                    id="tab-group-content",
                    style={**Styles.PAGE_CONTENT, "display": "none"},
                ),
                html.Div(
                    self._layout_stats_visualization(),
                    id="tab-viz-content",
                    style={**Styles.PAGE_CONTENT, "display": "none"},
                ),
                html.Div(
                    self._layout_weekly_report(),
                    id="tab-weekly-content",
                    style={**Styles.PAGE_CONTENT, "display": "none"},
                ),
                html.Div(
                    self._layout_stats_by_bet(),
                    id="tab-bet-content",
                    style={**Styles.PAGE_CONTENT, "display": "none"},
                ),
                html.Div(
                    self._layout_report_tab(),
                    id="tab-report-content",
                    style={**Styles.PAGE_CONTENT, "display": "none"},
                ),
                self._layout_chat_dialog(),
            ]
        )

    def _layout_report_tab(self) -> html.Div:
        """Report tab: collects figures, generates descriptions/summary, exports to Confluence."""
        return html.Div(
            [
                dcc.Store(id="report-figures", data=[]),
                dcc.Store(id="report-summary", data=""),
                # Top bar: Confluence URL + language + export button
                html.Div(
                    [
                        dcc.Input(
                            id="report-doc-link",
                            type="url",
                            placeholder="Confluence doc URL",
                            debounce=True,
                            persistence=True,
                            persistence_type="session",
                            style={
                                "flex": "1",
                                "minWidth": "320px",
                                "marginRight": "8px",
                                "padding": "6px 10px",
                                "border": "1px solid #ccc",
                                "borderRadius": "4px",
                                "fontSize": "13px",
                                "height": "32px",
                                "boxSizing": "border-box",
                            },
                        ),
                        dcc.Dropdown(
                            id="report-language",
                            options=_dropdown_option_rows(
                                [
                                    ("English", "en"),
                                    ("Chinese (Simplified)", "zh-Hans"),
                                    ("Chinese (Traditional)", "zh-Hant"),
                                ]
                            ),
                            value="en",
                            clearable=False,
                            persistence=True,
                            persistence_type="session",
                            style={"width": "200px", "marginRight": "8px"},
                        ),
                        html.Button(
                            "Export to Doc",
                            id="report-export-btn",
                            n_clicks=0,
                            style={
                                "padding": "6px 14px",
                                "cursor": "pointer",
                                "backgroundColor": "#984EA3",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "4px",
                                "fontSize": "13px",
                                "fontWeight": "bold",
                            },
                        ),
                        html.Div(
                            id="report-export-status",
                            style={"marginLeft": "12px", "fontSize": "12px", "color": "#666"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginBottom": "16px",
                    },
                ),
                # Summary panel
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Summary", style={**Styles.PANEL_HEADER, "display": "inline-block"}),
                                html.Button(
                                    "Generate summary",
                                    id="report-generate-summary-btn",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "12px",
                                        "padding": "4px 10px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#377EB8",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "fontSize": "12px",
                                    },
                                ),
                                html.Span(
                                    id="report-summary-status",
                                    style={"marginLeft": "10px", "fontSize": "11px", "color": "#666"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        dcc.Textarea(
                            id="report-summary-display",
                            placeholder=(
                                "(No summary yet — click 'Generate summary'. You can edit "
                                "the result, or add lines like  /prompt: tighten the second "
                                "paragraph  to instruct the next regenerate.)"
                            ),
                            style={
                                "width": "100%",
                                "marginTop": "10px",
                                "padding": "8px",
                                "fontSize": "13px",
                                "lineHeight": "1.6",
                                "color": "#333",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                                "minHeight": "120px",
                                "fontFamily": "inherit",
                                "resize": "vertical",
                                "boxSizing": "border-box",
                            },
                        ),
                    ],
                    style=Styles.SECTION,
                ),
                # Per-figure panels container, rendered from report-figures store
                html.Div(id="report-figures-container"),
            ],
            style={"padding": "8px"},
        )

    def _layout_inline_group_column_dropdown(
        self,
        label: str,
        elem_id: str,
        options: list[dcc.Dropdown.Options],
        value: Optional[str],
        *,
        clearable: bool,
        margin_left: str = "8px",
        extra_style: Optional[Dict[str, Any]] = None,
    ) -> html.Div:
        """Label and group-column dropdown on one row (shared by tab_date header and tab_group header)."""
        style: Dict[str, Any] = {
            **Styles.FLEX_ROW_CENTER,
            "alignItems": "center",
            "marginLeft": margin_left,
            "marginRight": "0",
            "flexShrink": 0,
        }
        if extra_style:
            style.update(extra_style)
        return html.Div(
            [
                html.Label(label, style={**Styles.CONTROL_LABEL, "marginRight": "6px", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id=elem_id,
                    options=options,
                    value=value,
                    clearable=clearable,
                    style=Styles.DROPDOWN_NARROW,
                ),
            ],
            style=style,
        )

    def _layout_add_to_report_panel(self, graph_id: str) -> html.Div:
        """'Add to report' button + status row, sitting just under one figure."""
        return html.Div(
            [
                html.Button(
                    "Add to report",
                    id={"type": "add-to-report-btn", "graph_id": graph_id},
                    n_clicks=0,
                    style={
                        "marginRight": "8px",
                        "padding": "5px 12px",
                        "cursor": "pointer",
                        "backgroundColor": "#984EA3",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(
                    id={"type": "add-to-report-status", "graph_id": graph_id},
                    style={"fontSize": "11px", "color": "#666"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "padding": "6px 8px",
                "borderTop": "1px solid #eee",
                "backgroundColor": "#fafafa",
            },
        )

    def _layout_show_group_checklist_sections(
        self,
        checklist_id_prefix: str,
        ug1_opts: list[dcc.Dropdown.Options],
        ug1_defaults: list[str],
        ug2_opts: list[dcc.Dropdown.Options],
        ug2_defaults: list[str],
        col2_hide_style: Dict[str, str],
    ) -> List[Component]:
        """Show Group 1 / Show Group 2 blocks (same structure in tab_date and tab_group plot panels)."""
        return [
            html.Div(
                [
                    html.Label("Show Group 1:", style=Styles.CONTROL_LABEL),
                    dcc.Checklist(
                        id=f"{checklist_id_prefix}-show-ug-1",
                        options=ug1_opts,
                        value=ug1_defaults,
                        labelStyle=Styles.CHECKLIST_BLOCK,
                        style={"columnCount": 3, "columnGap": "8px"},
                    ),
                ],
                style=Styles.CONTROL_GROUP,
            ),
            html.Div(
                [
                    html.Hr(style=Styles.HR),
                    html.Div(
                        [
                            html.Label("Show Group 2:", style=Styles.CONTROL_LABEL),
                            dcc.Checklist(
                                id=f"{checklist_id_prefix}-show-ug-2",
                                options=ug2_opts,
                                value=ug2_defaults,
                                labelStyle=Styles.CHECKLIST_BLOCK,
                                style={"columnCount": 3, "columnGap": "8px"},
                            ),
                        ],
                        style=Styles.CONTROL_GROUP,
                    ),
                ],
                style=col2_hide_style,
            ),
        ]

    def _layout_date_tab_show_ug_and_log_controls(
        self,
        checklist_id_prefix: str,
        panel_id: str,
        ug1_opts: list[dcc.Dropdown.Options],
        ug1_defaults: list[str],
        ug2_opts: list[dcc.Dropdown.Options],
        ug2_defaults: list[str],
        col2_hide_style: Dict[str, str],
    ) -> List[Component]:
        """Tail of tab_date sidebar: same rhythm as tab_group (HR → Show Group 1/2 → HR → bottom controls)."""
        return [
            html.Hr(style=Styles.HR),
            *self._layout_show_group_checklist_sections(
                checklist_id_prefix,
                ug1_opts,
                ug1_defaults,
                ug2_opts,
                ug2_defaults,
                col2_hide_style,
            ),
            html.Hr(style=Styles.HR),
            html.Div(
                [
                    html.Label("Logarithmic Scaling:", style=Styles.CONTROL_LABEL),
                    html.Div(
                        [
                            dcc.Checklist(
                                id=f"date-{panel_id}-log",
                                options=_dropdown_option_rows([(" Hybrid Log", "ON")]),
                                value=[],
                                style=Styles.CHECKLIST_INLINE,
                            ),
                            html.Span("Thresh: ", style={"fontSize": "0.9em"}),
                            dcc.Input(
                                id=f"date-{panel_id}-thresh",
                                type="number",
                                value=10,
                                style=Styles.INPUT_SMALL,
                            ),
                        ],
                        style=Styles.FLEX_ROW_CENTER,
                    ),
                ],
                style=Styles.CONTROL_GROUP,
            ),
        ]

    def _layout_stats_by_date(self) -> html.Div:
        min_date, max_date = self.date_range

        col1 = self.df_user_group_cols[0] if self.df_user_group_cols else None
        col2 = self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None
        has_col2 = col2 is not None

        ug1_opts, ug1_defaults = self._col_opts_and_default(col1)
        ug2_opts, ug2_defaults = self._col_opts_and_default(col2)
        _col2_hide = {"display": "none"} if not has_col2 else {}

        # Top bar only: matches tab_group (date ranges + column selectors on one row; no per-panel controls here).
        date_header_bar = html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Filter Date Range:",
                            style={**Styles.CONTROL_LABEL, "marginRight": "10px", "display": "inline-block"},
                        ),
                        dcc.DatePickerRange(
                            id="date-picker-range",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            initial_visible_month=min_date,
                            start_date=min_date,
                            end_date=max_date,
                            display_format="YYYY-MM-DD",
                            style={"verticalAlign": "middle", "marginRight": "30px"},
                        ),
                        html.Label(
                            "Granularity:",
                            style={
                                **Styles.CONTROL_LABEL,
                                "marginRight": "10px",
                                "display": "inline-block",
                                "marginLeft": "20px",
                            },
                        ),
                        dcc.Dropdown(
                            id="date-granularity",
                            options=_dropdown_options_from_strings(list(self.date_files_config.keys())),
                            value=list(self.date_files_config.keys())[0],
                            clearable=False,
                            style=Styles.DROPDOWN_NARROW,
                        ),
                        self._layout_inline_group_column_dropdown(
                            "Group Column 1:",
                            "date-ug-col-1",
                            self.df_user_group_col_options,
                            col1,
                            clearable=False,
                            margin_left="20px",
                        ),
                        self._layout_inline_group_column_dropdown(
                            "Group Column 2:",
                            "date-ug-col-2",
                            self.df_user_group_col_options,
                            col2,
                            clearable=True,
                            margin_left="8px",
                            extra_style=_col2_hide,
                        ),
                    ],
                    style={**Styles.FLEX_ROW_CENTER, "alignItems": "flex-start", "flexWrap": "wrap"},
                ),
            ],
            style=Styles.SECTION_GROUP,
        )

        download_config = dict(
            toImageButtonOptions=dict(format="png", height=600, width=1200, scale=3), displaylogo=False
        )

        def create_group_panel(group_id: str, label: str) -> html.Div:
            download_config["toImageButtonOptions"]["filename"] = f"stats_by_date{label}"  # type: ignore
            return html.Div(
                [
                    html.H4(label, style=Styles.PANEL_HEADER),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Left Axis Metrics:", style=Styles.CONTROL_LABEL),
                                            dcc.Dropdown(
                                                id=f"date-{group_id}-left-metrics",
                                                options=_dropdown_options_from_strings(self.plot_metrics[group_id]),
                                                value=(
                                                    [self.plot_metrics[group_id][0]]
                                                    if self.plot_metrics[group_id] and group_id == "g1"
                                                    else []
                                                ),
                                                multi=True,
                                                style=Styles.DROPDOWN_WIDE,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Right Axis Metrics:", style=Styles.CONTROL_LABEL),
                                            dcc.Dropdown(
                                                id=f"date-{group_id}-right-metrics",
                                                options=_dropdown_options_from_strings(self.plot_metrics[group_id]),
                                                value=[],
                                                multi=True,
                                                style=Styles.DROPDOWN_WIDE,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    *self._layout_date_tab_show_ug_and_log_controls(
                                        f"date-{group_id}",
                                        group_id,
                                        ug1_opts,
                                        ug1_defaults,
                                        ug2_opts,
                                        ug2_defaults,
                                        _col2_hide,
                                    ),
                                ],
                                style=Styles.CONTROL_PANEL_CONTAINER,
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id=f"date-{group_id}-plot",
                                        style={"height": "430px"},
                                        config=cast(Any, download_config),
                                    ),
                                    self._layout_add_to_report_panel(f"date-{group_id}-plot"),
                                ],
                                style=Styles.GRAPH_CONTAINER,
                            ),
                        ],
                        style=Styles.FLEX_ROW,
                    ),
                ],
                style=Styles.BOX,
            )

        sd_cfg = self.config["stats_by_date"]
        return html.Div(
            [
                date_header_bar,
                create_group_panel("g1", f"DateMetrics: {sd_cfg['group1_label']}"),
                create_group_panel("g2", f"DateMetrics: {sd_cfg['group2_label']}"),
                create_group_panel("g3", f"DateMetrics: {sd_cfg['group3_label']}"),
            ]
        )

    def _layout_stats_by_group(self) -> html.Div:
        # Use ranges from a loaded config if available; otherwise compute defaults
        ranges = self._loaded_group_date_ranges or self._compute_group_date_ranges()

        col1 = self.df_user_group_cols[0] if self.df_user_group_cols else None
        col2 = self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None
        has_col2 = col2 is not None

        ug1_opts, ug1_defaults = self._col_opts_and_default(col1)
        ug2_opts, ug2_defaults = self._col_opts_and_default(col2)

        # Both dropdowns always show all available columns; auto-swap handles conflicts at runtime
        col1_init_opts = self.df_user_group_col_options
        col2_init_opts = self.df_user_group_col_options

        _col2_hide = {"display": "none"} if not has_col2 else {}

        # Helper to create range picker blocks
        def range_block(label, idx, start, end, init_month):
            return html.Div(
                [
                    html.Label(label, style=Styles.CONTROL_LABEL),
                    dcc.DatePickerRange(
                        id=f"date-picker-range{idx}",
                        min_date_allowed=ranges[0],
                        max_date_allowed=ranges[1],
                        initial_visible_month=init_month or ranges[0],
                        start_date=start or ranges[0],
                        end_date=end or start,
                        display_format="YYYY-MM-DD",
                        style={"width": "100%"},
                    ),
                ],
                style={"marginRight": "20px"},
            )

        date_group_picker = html.Div(
            [
                html.Div(
                    [
                        range_block("Range 1 (Oldest)", 1, ranges[2], ranges[3], ranges[2]),
                        range_block("Range 2", 2, ranges[4], ranges[5], ranges[4]),
                        range_block("Range 3 (Newest)", 3, ranges[6], ranges[7], ranges[6]),
                        self._layout_inline_group_column_dropdown(
                            "Group Column 1:",
                            "group-ug-col-1",
                            col1_init_opts,
                            col1,
                            clearable=False,
                            margin_left="8px",
                        ),
                        self._layout_inline_group_column_dropdown(
                            "Group Column 2:",
                            "group-ug-col-2",
                            col2_init_opts,
                            col2,
                            clearable=True,
                            margin_left="8px",
                            extra_style=_col2_hide,
                        ),
                    ],
                    style={**Styles.FLEX_ROW_CENTER, "alignItems": "flex-start", "flexWrap": "wrap"},
                ),
            ],
            style=Styles.SECTION_GROUP,
        )

        download_config = dict(
            toImageButtonOptions=dict(format="png", height=600, width=1200, scale=3), displaylogo=False
        )

        def create_group_panel(group_id: str, label: str) -> html.Div:
            dl_id = f"group-{group_id}-plot"
            return html.Div(
                [
                    html.H4(label, style=Styles.PANEL_HEADER),
                    html.Div(
                        [
                            # --- CONTROL PANEL ---
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Metric:", style=Styles.CONTROL_LABEL),
                                            dcc.Dropdown(
                                                id=f"group-{group_id}-metrics",
                                                options=_dropdown_options_from_strings(self.plot_metrics[group_id]),
                                                value=(
                                                    self.plot_metrics[group_id][0]
                                                    if self.plot_metrics[group_id] and group_id == "g1"
                                                    else None
                                                ),
                                                multi=False,
                                                style=Styles.DROPDOWN_WIDE,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Display Mode:", style=Styles.CONTROL_LABEL),
                                            dcc.RadioItems(
                                                id=f"group-{group_id}-display",
                                                options=_dropdown_option_rows(
                                                    [("Box Plot", "box"), ("Bar (Mean)", "bar")]
                                                ),
                                                value="box",
                                                labelStyle={"display": "inline-block", "marginRight": "15px"},
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Hr(style=Styles.HR),
                                    html.Div(
                                        [
                                            html.Label("Show Date Range(s):", style=Styles.CONTROL_LABEL),
                                            dcc.Checklist(
                                                id=f"group-{group_id}-show-r1",
                                                options=_dropdown_option_rows([("Range 1", "ON")]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                            dcc.Checklist(
                                                id=f"group-{group_id}-show-r2",
                                                options=_dropdown_option_rows([("Range 2", "ON")]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                            dcc.Checklist(
                                                id=f"group-{group_id}-show-r3",
                                                options=_dropdown_option_rows([("Range 3", "ON")]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Hr(style=Styles.HR),
                                    *self._layout_show_group_checklist_sections(
                                        f"group-{group_id}",
                                        ug1_opts,
                                        ug1_defaults,
                                        ug2_opts,
                                        ug2_defaults,
                                        _col2_hide,
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Clip Data:", style=Styles.CONTROL_LABEL),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id=f"group-{group_id}-clip-enable",
                                                        options=_dropdown_option_rows([("Enable", "ON")]),
                                                        value=[],
                                                        style=Styles.CHECKLIST_INLINE,
                                                    ),
                                                    html.Label("Min:", style={"marginLeft": "5px"}),
                                                    dcc.Input(
                                                        id=f"group-{group_id}-clip-min",
                                                        type="number",
                                                        placeholder="min",
                                                        style=Styles.INPUT_SMALL,
                                                    ),
                                                    html.Label("Max:", style={"marginLeft": "5px"}),
                                                    dcc.Input(
                                                        id=f"group-{group_id}-clip-max",
                                                        type="number",
                                                        placeholder="max",
                                                        style=Styles.INPUT_SMALL,
                                                    ),
                                                ],
                                                style=Styles.FLEX_ROW_CENTER,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                ],
                                style=Styles.CONTROL_PANEL_CONTAINER,
                            ),
                            # --- GRAPH PANEL ---
                            html.Div(
                                [
                                    dcc.Graph(id=dl_id, style={"height": "430px"}, config=cast(Any, download_config)),
                                    self._layout_add_to_report_panel(dl_id),
                                ],
                                style=Styles.GRAPH_CONTAINER,
                            ),
                        ],
                        style=Styles.FLEX_ROW,
                    ),
                ],
                style=Styles.BOX,
            )

        return html.Div(
            [
                date_group_picker,
                create_group_panel("g1", f"Group Metrics: {self.config['stats_by_date']['group1_label']}"),
                create_group_panel("g2", f"Group Metrics: {self.config['stats_by_date']['group2_label']}"),
                create_group_panel("g3", f"Group Metrics: {self.config['stats_by_date']['group3_label']}"),
            ]
        )

    def _viz_derived_metric_options(self) -> list[dcc.Dropdown.Options]:
        """Derived metric options, filtered to those whose user-column dependencies are present.

        ``DataMetrics.METRICS`` is the union across game families — slot-only
        columns like ``user_num_bets_fg`` aren't in fishhunter's ETL output, and
        offering metrics that depend on them produces ``ColumnNotFoundError``
        when the user picks one. ``DataMetrics.metric_user_col_deps`` does the
        introspection so this list is always a subset of what's computable on
        the currently-loaded data.
        """
        gran = next(iter(self.date_files_config.keys()), None)
        lf = self.lfs_by_date.get(gran) if gran else None
        if lf is None or lf.collect_schema().len() == 0:
            return _dropdown_options_from_strings(sorted(DataMetrics.METRICS))
        schema_cols = set(lf.collect_schema().names())
        valid = sorted(m for m in DataMetrics.METRICS if DataMetrics.metric_user_col_deps(m).issubset(schema_cols))
        return _dropdown_options_from_strings(valid)

    def _viz_user_metric_options(self) -> list[dcc.Dropdown.Options]:
        """User-level raw metric options — every ``user_*`` column actually present in the loaded parquet.

        Derived from the dataframe schema (not a hardcoded list) so any ``user_*`` column produced
        by the per-game ETL — e.g. ``user_avg_delta_t_seconds``, ``user_fg_ratio``,
        ``user_mathtable_change``, ``user_hit_rate`` — automatically appears here without a code
        change.  Two types of columns are excluded:

        - Columns computed by ``DataMetrics`` (``user_rtp_median``, ``user_rtp_ultilization_ratio``)
          — they belong in the Derived panel even if the ETL pre-materialises them.
        - Columns listed in ``stats_by_date.user_group_cols`` (e.g. ``ai_group``, ``user_group``,
          ``user_group2``) — they are group-by keys, not metrics to plot.

        Falls back to ``ENRICH_USER_ROW_INPUT_COLUMNS`` (with the same exclusions) if no data is
        loaded yet.
        """
        excluded_group_cols = set(self.df_user_group_cols or [])
        gran = next(iter(self.date_files_config.keys()), None)
        lf = self.lfs_by_date.get(gran) if gran else None
        if lf is not None and lf.collect_schema().len() > 0:
            names = lf.collect_schema().names()
            cols = sorted(
                c
                for c in names
                if c.startswith("user_")
                and c != "user_id"
                and c not in DataMetrics.METRICS
                and c not in excluded_group_cols
            )
            if cols:
                return _dropdown_options_from_strings(cols)
        fallback = ENRICH_USER_ROW_INPUT_COLUMNS - {"user_id"} - DataMetrics.METRICS - excluded_group_cols
        return _dropdown_options_from_strings(sorted(fallback))

    _VIZ_DISPLAY_MODES: List[Tuple[str, str]] = [
        ("Histogram", "histogram"),
        ("Heatmap", "heatmap"),
        ("Scatter Plot", "scatter"),
    ]
    # Metric-selector rows: MAX_ROWS slots are pre-rendered with fixed IDs.  The first
    # ``DEFAULT_FILLED`` rows are pre-populated with default metrics; the row immediately
    # after the last filled row is always shown empty (so the user can add one more).
    # Visibility is data-driven — no explicit "Add metrics" button needed.
    _VIZ_MS_MAX_ROWS: int = 16
    _VIZ_HM_DEFAULT_FILLED: int = 3
    _VIZ_SM_DEFAULT_FILLED: int = 2

    def _layout_stats_visualization(self) -> html.Div:
        """Layout for the 'Stats Deepdive' tab.

        Header mirrors ``stats_by_group`` (3 date pickers + 2 group-column selectors).
        Two plotting panels — DataMetrics-derived and user-level raw — share the same
        controls (metrics multi-select, display mode, range/group toggles, clipping).
        """
        ranges = self._loaded_group_date_ranges or self._compute_group_date_ranges()

        col1 = self.df_user_group_cols[0] if self.df_user_group_cols else None
        col2 = self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None
        has_col2 = col2 is not None

        ug1_opts, ug1_defaults = self._col_opts_and_default(col1)
        ug2_opts, ug2_defaults = self._col_opts_and_default(col2)

        col1_init_opts = self.df_user_group_col_options
        col2_init_opts = self.df_user_group_col_options

        _col2_hide = {"display": "none"} if not has_col2 else {}

        def range_block(label: str, idx: int, start: Any, end: Any, init_month: Any) -> html.Div:
            return html.Div(
                [
                    html.Label(label, style=Styles.CONTROL_LABEL),
                    dcc.DatePickerRange(
                        id=f"viz-date-picker-range{idx}",
                        min_date_allowed=ranges[0],
                        max_date_allowed=ranges[1],
                        initial_visible_month=init_month or ranges[0],
                        start_date=start or ranges[0],
                        end_date=end or start,
                        display_format="YYYY-MM-DD",
                        style={"width": "100%"},
                    ),
                ],
                style={"marginRight": "20px"},
            )

        date_group_picker = html.Div(
            [
                html.Div(
                    [
                        range_block("Range 1 (Oldest)", 1, ranges[2], ranges[3], ranges[2]),
                        range_block("Range 2", 2, ranges[4], ranges[5], ranges[4]),
                        range_block("Range 3 (Newest)", 3, ranges[6], ranges[7], ranges[6]),
                        self._layout_inline_group_column_dropdown(
                            "Group Column 1:",
                            "viz-ug-col-1",
                            col1_init_opts,
                            col1,
                            clearable=False,
                            margin_left="8px",
                        ),
                        self._layout_inline_group_column_dropdown(
                            "Group Column 2:",
                            "viz-ug-col-2",
                            col2_init_opts,
                            col2,
                            clearable=True,
                            margin_left="8px",
                            extra_style=_col2_hide,
                        ),
                    ],
                    style={**Styles.FLEX_ROW_CENTER, "alignItems": "flex-start", "flexWrap": "wrap"},
                ),
            ],
            style=Styles.SECTION_GROUP,
        )

        download_config: Dict[str, Any] = dict(
            toImageButtonOptions=dict(format="png", height=700, width=1200, scale=3),
            displaylogo=False,
            # Plotly installs a ResizeObserver when responsive is True; without this the
            # plot measures its container only at creation (0px while display:none) and
            # keeps that stale width, then stretches on the first real resize event.
            responsive=True,
        )

        def create_panel(panel_id: str, label: str, metric_options: list[dcc.Dropdown.Options]) -> html.Div:
            dl_id = f"viz-{panel_id}-plot"

            def build_metric_rows(prefix: str, default_filled: int) -> List[Component]:
                """Build MAX_ROWS per-metric rows for heatmap (prefix='hm') / scatter (prefix='sm').

                The first ``default_filled`` rows are pre-populated with the first alphabetical
                metrics (so the panel is useful immediately) plus one trailing empty row so the
                user can "add more" simply by picking a value.  Extra rows are rendered but
                hidden — a visibility callback driven by current row values shows/hides them
                dynamically: ``show rows 0..max_filled+1``.
                """
                default_metrics: List[Optional[str]] = [
                    (str(opt["value"]) if i < len(metric_options) else None)
                    for i, opt in enumerate(metric_options[: self._VIZ_MS_MAX_ROWS])
                ]
                while len(default_metrics) < self._VIZ_MS_MAX_ROWS:
                    default_metrics.append(None)
                rows: List[Component] = []
                # ``default_filled`` rows filled + 1 trailing empty = ``default_filled + 1`` visible initially.
                initial_visible_until = min(default_filled, self._VIZ_MS_MAX_ROWS - 1)
                for i in range(self._VIZ_MS_MAX_ROWS):
                    default_val = default_metrics[i] if i < default_filled else None
                    row_visible = i <= initial_visible_until
                    rows.append(
                        html.Div(
                            [
                                html.Label(
                                    f"Metric {i + 1}:",
                                    style={
                                        **Styles.CONTROL_LABEL,
                                        "marginBottom": "0",
                                        "marginRight": "6px",
                                        "minWidth": "72px",
                                        "whiteSpace": "nowrap",
                                    },
                                ),
                                dcc.Dropdown(
                                    id=f"viz-{panel_id}-{prefix}-metric-{i}",
                                    options=metric_options,
                                    value=default_val,
                                    clearable=True,
                                    style={"flex": "1", "minWidth": "120px"},
                                ),
                                dcc.Checklist(
                                    id=f"viz-{panel_id}-{prefix}-log-{i}",
                                    options=_dropdown_option_rows([(" Log", "ON")]),
                                    value=[],
                                    style={
                                        **Styles.CHECKLIST_INLINE,
                                        "marginBottom": "0",
                                        "marginLeft": "6px",
                                        "whiteSpace": "nowrap",
                                    },
                                ),
                            ],
                            id=f"viz-{panel_id}-{prefix}-row-{i}",
                            style={
                                **Styles.FLEX_ROW_CENTER,
                                "marginBottom": "4px",
                                "display": "flex" if row_visible else "none",
                            },
                        )
                    )
                return rows

            hm_rows = build_metric_rows("hm", self._VIZ_HM_DEFAULT_FILLED)
            sm_rows = build_metric_rows("sm", self._VIZ_SM_DEFAULT_FILLED)
            return html.Div(
                [
                    html.H4(label, style=Styles.PANEL_HEADER),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Display Mode:", style=Styles.CONTROL_LABEL),
                                            dcc.RadioItems(
                                                id=f"viz-{panel_id}-display",
                                                options=_dropdown_option_rows(self._VIZ_DISPLAY_MODES),
                                                value="histogram",
                                                labelStyle={"display": "inline-block", "marginRight": "15px"},
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    # General Metrics multi-select — visible for histogram / scatter.
                                    html.Div(
                                        [
                                            html.Label("Metrics:", style=Styles.CONTROL_LABEL),
                                            dcc.Dropdown(
                                                id=f"viz-{panel_id}-metrics",
                                                options=metric_options,
                                                value=[],
                                                multi=True,
                                                style=Styles.DROPDOWN_WIDE,
                                            ),
                                        ],
                                        id=f"viz-{panel_id}-metrics-container",
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    # Heatmap metrics — visible only for heatmap mode.  Each row is a
                                    # single-column dropdown plus an optional "Log" (Yeo-Johnson) flag.
                                    # Rows auto-add: always shows one trailing empty row; filling it reveals
                                    # the next; clearing the last filled row collapses the trailing empty.
                                    html.Div(
                                        hm_rows,
                                        id=f"viz-{panel_id}-hm-container",
                                        style={**Styles.CONTROL_GROUP, "display": "none"},
                                    ),
                                    # Scatter metrics — same dynamic-row logic as heatmap, default 2 filled.
                                    html.Div(
                                        sm_rows,
                                        id=f"viz-{panel_id}-sm-container",
                                        style={**Styles.CONTROL_GROUP, "display": "none"},
                                    ),
                                    # Histogram-only controls — live inside the fixed-width sidebar so
                                    # showing/hiding them never changes the graph container's size.
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Bins:",
                                                        style={
                                                            **Styles.CONTROL_LABEL,
                                                            "marginBottom": "0",
                                                            "marginRight": "6px",
                                                            "whiteSpace": "nowrap",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id=f"viz-{panel_id}-nbins",
                                                        options=_dropdown_option_rows(
                                                            [
                                                                ("50", "50"),
                                                                ("100", "100"),
                                                                ("200", "200"),
                                                                ("500", "500"),
                                                            ]
                                                        ),
                                                        value="50",
                                                        clearable=False,
                                                        style={"width": "80px", "marginRight": "10px"},
                                                    ),
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-log-y",
                                                        options=_dropdown_option_rows([(" Log Y", "ON")]),
                                                        value=[],
                                                        style={**Styles.CHECKLIST_INLINE, "marginBottom": "0"},
                                                    ),
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-normalize",
                                                        options=_dropdown_option_rows([(" Normalize", "ON")]),
                                                        value=[],
                                                        style={**Styles.CHECKLIST_INLINE, "marginBottom": "0"},
                                                    ),
                                                ],
                                                style={
                                                    **Styles.FLEX_ROW_CENTER,
                                                    "flexWrap": "wrap",
                                                    "gap": "4px",
                                                },
                                            ),
                                        ],
                                        id=f"viz-{panel_id}-hist-controls",
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    # Scatter-only controls — shown only when Display Mode == "scatter".
                                    # Lives inside the fixed-width sidebar so toggling does not reflow the graph.
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-scatter-log-x",
                                                        options=_dropdown_option_rows([(" Log X", "ON")]),
                                                        value=[],
                                                        style={**Styles.CHECKLIST_INLINE, "marginBottom": "0"},
                                                    ),
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-scatter-symlog-x",
                                                        options=_dropdown_option_rows([(" Sym Log X", "ON")]),
                                                        value=[],
                                                        style={**Styles.CHECKLIST_INLINE, "marginBottom": "0"},
                                                    ),
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-scatter-log-y",
                                                        options=_dropdown_option_rows([(" Log Y", "ON")]),
                                                        value=[],
                                                        style={**Styles.CHECKLIST_INLINE, "marginBottom": "0"},
                                                    ),
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-scatter-symlog-y",
                                                        options=_dropdown_option_rows([(" Sym Log Y", "ON")]),
                                                        value=[],
                                                        style={**Styles.CHECKLIST_INLINE, "marginBottom": "0"},
                                                    ),
                                                ],
                                                style={**Styles.FLEX_ROW_CENTER, "flexWrap": "wrap", "gap": "4px"},
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-outliers-enable",
                                                        options=_dropdown_option_rows([(" Remove outliers", "ON")]),
                                                        value=[],
                                                        style={**Styles.CHECKLIST_INLINE, "marginBottom": "0"},
                                                    ),
                                                    html.Label(
                                                        "σ:",
                                                        style={
                                                            **Styles.CONTROL_LABEL,
                                                            "marginBottom": "0",
                                                            "marginLeft": "6px",
                                                            "marginRight": "4px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id=f"viz-{panel_id}-outliers-std",
                                                        options=_dropdown_option_rows(
                                                            [
                                                                ("2", "2"),
                                                                ("2.5", "2.5"),
                                                                ("3", "3"),
                                                                ("3.5", "3.5"),
                                                                ("4", "4"),
                                                            ]
                                                        ),
                                                        value="3",
                                                        clearable=False,
                                                        style={"width": "70px"},
                                                    ),
                                                ],
                                                style={
                                                    **Styles.FLEX_ROW_CENTER,
                                                    "flexWrap": "wrap",
                                                    "gap": "4px",
                                                    "marginTop": "6px",
                                                },
                                            ),
                                        ],
                                        id=f"viz-{panel_id}-scatter-controls",
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Hr(style=Styles.HR),
                                    html.Div(
                                        [
                                            html.Label("Show Date Range(s):", style=Styles.CONTROL_LABEL),
                                            dcc.Checklist(
                                                id=f"viz-{panel_id}-show-r1",
                                                options=_dropdown_option_rows([("Range 1", "ON")]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                            dcc.Checklist(
                                                id=f"viz-{panel_id}-show-r2",
                                                options=_dropdown_option_rows([("Range 2", "ON")]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                            dcc.Checklist(
                                                id=f"viz-{panel_id}-show-r3",
                                                options=_dropdown_option_rows([("Range 3", "ON")]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Hr(style=Styles.HR),
                                    *self._layout_show_group_checklist_sections(
                                        f"viz-{panel_id}",
                                        ug1_opts,
                                        ug1_defaults,
                                        ug2_opts,
                                        ug2_defaults,
                                        _col2_hide,
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Clip Data:", style=Styles.CONTROL_LABEL),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id=f"viz-{panel_id}-clip-enable",
                                                        options=_dropdown_option_rows([("Enable", "ON")]),
                                                        value=[],
                                                        style=Styles.CHECKLIST_INLINE,
                                                    ),
                                                    html.Label("Min:", style={"marginLeft": "5px"}),
                                                    dcc.Input(
                                                        id=f"viz-{panel_id}-clip-min",
                                                        type="number",
                                                        placeholder="min",
                                                        style=Styles.INPUT_SMALL,
                                                    ),
                                                    html.Label("Max:", style={"marginLeft": "5px"}),
                                                    dcc.Input(
                                                        id=f"viz-{panel_id}-clip-max",
                                                        type="number",
                                                        placeholder="max",
                                                        style=Styles.INPUT_SMALL,
                                                    ),
                                                ],
                                                style=Styles.FLEX_ROW_CENTER,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                ],
                                style=Styles.CONTROL_PANEL_CONTAINER,
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id=dl_id, style={"height": "900px"}, config=cast(Any, download_config)),
                                    self._layout_add_to_report_panel(dl_id),
                                ],
                                style=Styles.GRAPH_CONTAINER,
                            ),
                        ],
                        style=Styles.FLEX_ROW,
                    ),
                ],
                style=Styles.BOX,
            )

        return html.Div(
            [
                date_group_picker,
                create_panel("p1", "Derived Metrics (DataMetrics)", self._viz_derived_metric_options()),
                create_panel("p2", "User-Level Metrics", self._viz_user_metric_options()),
            ]
        )

    def _layout_weekly_report(self) -> html.Div:
        wr = self.config.get("weekly_report") or {}
        game_id = wr.get("game_id")
        if not game_id:
            return html.Div(
                html.P(
                    [
                        "Weekly report is not configured for this dashboard. Add a weekly_report section "
                        "(game_id, files, currency_type) to the YAML. Also make sure the weekly-report "
                        "parquet data has been uploaded to S3 at ",
                        html.Code(
                            "s3://bituslabs-team-ai/etl-results/jobs/output_weekly_report_all_games/ops_daily_report"
                        ),
                        " (produced by ",
                        html.Code("jobs/operation_daily_report/etl_weekly_report_all_games.py"),
                        ").",
                    ],
                    style={"color": "#64748b"},
                ),
                style=Styles.PAGE_CONTENT,
            )

        df = self._weekly_report_df
        currency = (wr.get("currency_type") or "CNY").upper()

        if df is None or df.empty:
            return html.Div(
                html.P(
                    "No weekly report data loaded. Run jobs/operation_daily_report/etl_weekly_report_all_games.py "
                    "and set weekly_report.files to the S3 path (see dashboard_config-*.yaml).",
                    style={"color": "#64748b"},
                ),
                style=Styles.PAGE_CONTENT,
            )

        if "game_id" not in df.columns or "currency_type" not in df.columns:
            return html.Div(
                html.P(
                    "Weekly report data must include game_id and currency_type columns.",
                    style={"color": "#b91c1c"},
                ),
                style=Styles.PAGE_CONTENT,
            )

        _mask = (df["game_id"] == game_id) & (df["currency_type"] == currency)
        game_df = df.loc[_mask].sort_values(by="bj_date_key")
        if game_df.empty:
            return html.Div(
                html.P(
                    f"No rows for game_id={game_id!r} and currency_type={currency!r}. "
                    "Check ETL output and weekly_report.game_id in this config.",
                    style={"color": "#64748b"},
                ),
                style=Styles.PAGE_CONTENT,
            )

        store_json = game_df.to_json(date_format="iso", orient="split")
        _, _, default_end = weekly_report_end_date_bounds(game_df)
        sliced_for_draft = slice_weekly_report_by_end_date(game_df, default_end) if default_end else game_df
        weekly_draft_children: List[Component] = build_draft_preview_single(sliced_for_draft, game_id)
        subject_line = format_weekly_report_subject(default_end) if default_end else "运营周报"

        return html.Div(
            style=Styles.PAGE_CONTENT,
            children=[
                html.H2("Weekly Report", style={**Styles.PANEL_HEADER, "marginBottom": "12px"}),
                html.P(
                    f"Configured game: {game_id} · Currency: {currency}",
                    style={"color": "#64748b", "marginBottom": "16px"},
                ),
                dcc.Store(id="weekly-report-store", data=store_json),
                html.Div(
                    style={"marginBottom": "16px"},
                    children=[
                        html.Label("Suggested subject line", style=Styles.CONTROL_LABEL),
                        html.Div(
                            id="weekly-report-subject-line",
                            children=subject_line,
                            style={"fontWeight": "600", "color": "#0f172a"},
                        ),
                    ],
                ),
                layout_weekly_report_panel(game_df, game_id),
                html.H3("Weekly summary (figures & table)", style={**Styles.PANEL_HEADER, "fontSize": "18px"}),
                html.Div(id="weekly-report-draft", children=weekly_draft_children),
            ],
        )

    def _layout_stats_by_bet(self) -> html.Div:
        return html.Div(
            [
                # --- LEFT SIDEBAR ---
                html.Div(
                    [
                        # Card 1: Selectors
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Select Session:", style=Styles.CONTROL_LABEL),
                                        dcc.Dropdown(
                                            id="session-dropdown",
                                            options=_dropdown_options_from_strings(self.sessions),
                                            value=self.sessions[0] if self.sessions else None,
                                            clearable=False,
                                            style=Styles.DROPDOWN_WIDE,
                                        ),
                                    ],
                                    style=Styles.CONTROL_GROUP,
                                ),
                                html.Hr(style=Styles.HR),
                                html.Div(
                                    [
                                        html.Label("Compare Strategies:", style=Styles.CONTROL_LABEL),
                                        dcc.Checklist(
                                            id="strategy-checklist",
                                            options=_dropdown_options_from_strings(self.df_bet_groups),
                                            value=self.df_bet_groups[: self.DEFAULT_VISIBLE_GROUPS],
                                            labelStyle=Styles.CHECKLIST_BLOCK,
                                        ),
                                    ],
                                    style=Styles.CONTROL_GROUP,
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                        # Card 2: Metrics
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Metrics (Left Axis):", style=Styles.CONTROL_LABEL),
                                        dcc.Dropdown(
                                            id="metric-checklist",
                                            options=_dropdown_options_from_strings(self.bet_metrics),
                                            value=[self.bet_metrics[0]] if self.bet_metrics else [],
                                            multi=True,
                                            style=Styles.DROPDOWN_WIDE,
                                        ),
                                    ],
                                    style=Styles.CONTROL_GROUP,
                                ),
                                html.Div(
                                    [
                                        html.Label("Metrics (Right Axis):", style=Styles.CONTROL_LABEL),
                                        dcc.Dropdown(
                                            id="right-axis-checklist",
                                            options=_dropdown_options_from_strings(self.bet_metrics),
                                            value=[],
                                            multi=True,
                                            style=Styles.DROPDOWN_WIDE,
                                        ),
                                    ],
                                    style=Styles.CONTROL_GROUP,
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                        # Card 3: Display Options
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="share-left-yscale-check",
                                            options=_dropdown_option_rows([(" Share Left Scale", "ON")]),
                                            value=[],
                                            style=Styles.CHECKLIST_BLOCK,
                                        ),
                                        dcc.Checklist(
                                            id="share-right-yscale-check",
                                            options=_dropdown_option_rows([(" Share Right Scale", "ON")]),
                                            value=[],
                                            style=Styles.CHECKLIST_BLOCK,
                                        ),
                                    ],
                                    style=Styles.CONTROL_GROUP,
                                ),
                                html.Hr(style=Styles.HR),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="log-check",
                                            options=_dropdown_option_rows([(" Hybrid Log Scale", "ON")]),
                                            value=[],
                                            style=Styles.CHECKLIST_BLOCK,
                                        ),
                                        html.Label(
                                            "Linear Thresh:", style={**Styles.CONTROL_LABEL, "marginTop": "8px"}
                                        ),
                                        dcc.Input(
                                            id="linear-thresh", type="number", value=10, style=Styles.INPUT_FULLWIDTH
                                        ),
                                    ],
                                    style=Styles.CONTROL_GROUP,
                                ),
                                html.Hr(style=Styles.HR),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="filter-check",
                                            options=_dropdown_option_rows([("Max Number of Bets", "ON")]),
                                            value=["ON"],
                                            style=Styles.CHECKLIST_BLOCK,
                                        ),
                                        dcc.Input(
                                            id="filter-thresh", type="number", value=5000, style=Styles.INPUT_FULLWIDTH
                                        ),
                                    ],
                                    style=Styles.CONTROL_GROUP,
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                    ],
                    style=Styles.CONTROL_PANEL_CONTAINER,
                ),
                # --- MAIN PLOT ---
                html.Div(
                    [
                        dcc.Graph(id="combined-plot", style={"height": "950px"}),
                        self._layout_add_to_report_panel("combined-plot"),
                    ],
                    style=Styles.GRAPH_CONTAINER,
                ),
            ],
            style=Styles.FLEX_ROW,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        @self.app.callback(
            Output("tab-date-content", "children"),
            Output("tab-group-content", "children"),
            Output("tab-viz-content", "children"),
            Output("tab-weekly-content", "children"),
            Output("tab-bet-content", "children"),
            # Reset every persistent Store on a config switch so
            # ``restore_dashboard_state`` (which fires on tab-content change)
            # short-circuits at its trigger guard and can't re-apply the
            # previous config's metric/group selections to the freshly built
            # dropdowns. Without this, a user who has loaded any saved S3
            # config in the session leaks ss02 metric picks (e.g.
            # ``active_user_no_fg_ratio``) into a fishhunter layout and
            # crashes the viz callback on a missing column.
            Output("dashboard-load-trigger", "data", allow_duplicate=True),
            Output("tab-date-g1-state", "data", allow_duplicate=True),
            Output("tab-date-g2-state", "data", allow_duplicate=True),
            Output("tab-date-g3-state", "data", allow_duplicate=True),
            Output("tab-group-g1-state", "data", allow_duplicate=True),
            Output("tab-group-g2-state", "data", allow_duplicate=True),
            Output("tab-group-g3-state", "data", allow_duplicate=True),
            Output("tab-viz-p1-state", "data", allow_duplicate=True),
            Output("tab-viz-p2-state", "data", allow_duplicate=True),
            Output("tab-bet-state", "data", allow_duplicate=True),
            Input("config-dropdown", "value"),
            prevent_initial_call=True,
        )
        def update_config(selected_config: str) -> Any:
            self._reset_state()
            self._reload_config(selected_config)
            self._ensure_tab_data_loaded()
            return (
                self._layout_stats_by_date(),
                self._layout_stats_by_group(),
                self._layout_stats_visualization(),
                self._layout_weekly_report(),
                self._layout_stats_by_bet(),
                None,  # dashboard-load-trigger
                None,  # tab-date-g1-state
                None,  # tab-date-g2-state
                None,  # tab-date-g3-state
                None,  # tab-group-g1-state
                None,  # tab-group-g2-state
                None,  # tab-group-g3-state
                None,  # tab-viz-p1-state
                None,  # tab-viz-p2-state
                None,  # tab-bet-state
            )

        @self.app.callback(
            Output("tab-date-content", "style"),
            Output("tab-group-content", "style"),
            Output("tab-viz-content", "style"),
            Output("tab-weekly-content", "style"),
            Output("tab-bet-content", "style"),
            Output("tab-report-content", "style"),
            Input("navigator-tabs", "value"),
            prevent_initial_call=False,
        )
        def render_content(tab: str) -> Any:
            date_style = {**Styles.PAGE_CONTENT, "display": "block" if tab == "tab-date" else "none"}
            group_style = {**Styles.PAGE_CONTENT, "display": "block" if tab == "tab-group" else "none"}
            viz_style = {**Styles.PAGE_CONTENT, "display": "block" if tab == "tab-viz" else "none"}
            weekly_style = {**Styles.PAGE_CONTENT, "display": "block" if tab == "tab-weekly" else "none"}
            bet_style = {**Styles.PAGE_CONTENT, "display": "block" if tab == "tab-bet" else "none"}
            report_style = {**Styles.PAGE_CONTENT, "display": "block" if tab == "tab-report" else "none"}
            return date_style, group_style, viz_style, weekly_style, bet_style, report_style

        @self.app.callback(
            Output("weekly-report-chart", "figure"),
            Output("weekly-report-summary-card", "children"),
            Input("weekly-report-store", "data"),
            Input("weekly-report-metric-select", "value"),
            Input("weekly-report-end-date", "date"),
        )
        def update_weekly_report_chart(
            store_json: Optional[str],
            metric_key: Optional[str],
            end_date_iso: Optional[str],
        ):
            if not store_json or not metric_key or not end_date_iso:
                return empty_figure("Select a metric or load data."), html.Div(
                    "暂无数据",
                    style={"color": "#475569", "fontWeight": "bold"},
                )
            game_df = dataframe_from_split_json(store_json)
            game_df = slice_weekly_report_by_end_date(game_df, end_date_iso)
            if game_df.empty:
                return empty_figure("No data."), html.Div(
                    "暂无数据",
                    style={"color": "#475569", "fontWeight": "bold"},
                )
            gid = str(game_df["game_id"].iloc[0])
            window_days = get_fixed_window_days(game_df)
            metric = next((m for m in METRICS_BY_GAME.get(gid, []) if m.key == metric_key), None)
            if metric is None:
                return empty_figure("Unknown metric."), html.Div()
            metric_summary = compare_metric(game_df, metric, window_days)
            return build_chart(game_df, metric, window_days), build_metric_summary_card(metric_summary, window_days)

        @self.app.callback(
            Output("weekly-report-subject-line", "children"),
            Output("weekly-report-draft", "children"),
            Input("weekly-report-store", "data"),
            Input("weekly-report-end-date", "date"),
            prevent_initial_call=True,
        )
        def update_weekly_report_subject_draft(
            store_json: Optional[str],
            end_date_iso: Optional[str],
        ):
            if not store_json or not end_date_iso:
                return no_update, no_update
            game_df = dataframe_from_split_json(store_json)
            sliced = slice_weekly_report_by_end_date(game_df, end_date_iso)
            gid = str(sliced["game_id"].iloc[0]) if not sliced.empty and "game_id" in sliced.columns else ""
            subject = format_weekly_report_subject(end_date_iso)
            draft_children: List[Component] = build_draft_preview_single(sliced, gid)
            return subject, draft_children

        # For "Stats by Date": update picker to full range on granularity switch
        @self.app.callback(
            Output("date-picker-range", "min_date_allowed"),
            Output("date-picker-range", "max_date_allowed"),
            Output("date-picker-range", "start_date", allow_duplicate=True),
            Output("date-picker-range", "end_date", allow_duplicate=True),
            Output("date-picker-range", "initial_visible_month"),
            Input("date-granularity", "value"),
            State("dashboard-load-trigger", "data"),
            prevent_initial_call=True,
        )
        def update_date_picker_on_granularity(granularity: str, load_trigger: Any):
            if load_trigger is not None:
                return (no_update,) * 5
            gran_lf = self.lfs_by_date.get(granularity, pl.DataFrame().lazy())
            if gran_lf.collect_schema().len() == 0 or self.date_col not in gran_lf.collect_schema().names():
                return [None] * 5
            agg = gran_lf.select(
                pl.col(self.date_col).min().alias("min_d"),
                pl.col(self.date_col).max().alias("max_d"),
            ).collect()
            min_date, max_date = agg.item(0, "min_d"), agg.item(0, "max_d")
            return (min_date, max_date, min_date, max_date, min_date)

        # For "Stats by Group" (tab_group): set three date pickers (range1/2/3).
        # Rules:
        # - Initial load: compute 3 sequential 2-week default windows.
        # - Config load: if config provides tab_group.date_ranges, apply them to all 3 pickers.
        # - Edge case: if config date_ranges is empty/missing, fall back to defaults.
        @self.app.callback(
            Output("date-picker-range1", "min_date_allowed"),
            Output("date-picker-range1", "max_date_allowed"),
            Output("date-picker-range1", "initial_visible_month"),
            Output("date-picker-range2", "min_date_allowed"),
            Output("date-picker-range2", "max_date_allowed"),
            Output("date-picker-range2", "initial_visible_month"),
            Output("date-picker-range3", "min_date_allowed"),
            Output("date-picker-range3", "max_date_allowed"),
            Output("date-picker-range3", "initial_visible_month"),
            Input("dashboard-load-trigger", "data"),
            prevent_initial_call=True,
        )
        def update_tab_group_date_pickers(load_trigger: Any):
            # Let `_layout_stats_by_group()` be the single source of truth for:
            # - min_date_allowed / max_date_allowed
            # - initial_visible_month
            # Avoid updating bounds here because it can clamp restored start/end values
            # back to default windows during config load.
            return [no_update] * 9

        # Asymmetric auto-swap to avoid a Dash dependency cycle:
        # col1 change → may auto-update col2 (no cycle, since col2-change callback never outputs col1)
        # col2 change → only updates its own checklists; if col2==col1 the plot handles it gracefully
        # ── Why these callbacks update OPTIONS only, never VALUE ──────────────────────────────
        #
        # Dash executes callbacks in topological (dependency) order: if callback A outputs
        # component X, any callback B that has X as an Input fires *after* A completes.
        #
        # The config-restore path is:
        #   1. load_config_select()       → writes g1/g2/g3 stores + sets dashboard-load-trigger
        #   2. restore_dashboard_state()  → reads stores, outputs to ALL UI components at once,
        #                                   including group-ug-col-1.value AND
        #                                   group-g{1,2,3}-show-ug-1.value (saved selections).
        #   3. on_ug_col_1_change()       → fires BECAUSE step 2 changed group-ug-col-1.value.
        #                                   If this callback also outputs show-ug-1.value,
        #                                   it runs AFTER step 2 and OVERWRITES the saved
        #                                   selections with a hardcoded default (e.g. ["all"]).
        #
        # Result without this fix: saved checklist selections are silently discarded every time
        # a config is loaded, because the column-change callback always fires last and resets
        # the value to the first option.
        #
        # Fix: column-change callbacks output OPTIONS only.
        #   • OPTIONS tell the checklist what is *available* to select (needs refreshing when
        #     the column changes, because different columns have different unique values).
        #   • VALUE (what is currently selected) is owned by two actors only:
        #       – the layout (sets the initial default at page render), and
        #       – restore_dashboard_state (sets saved selections on config load).
        #     Interactive column switches by the user leave the value untouched; if the old
        #     selections don't exist in the new column's options, Dash silently drops them and
        #     the checklist shows an empty selection — expected UX prompting the user to
        #     re-choose from the new column's values.
        @self.app.callback(
            [Output("group-ug-col-2", "value")] + [Output(f"group-g{i}-show-ug-1", "options") for i in range(1, 4)],
            Input("group-ug-col-1", "value"),
            State("group-ug-col-2", "value"),
            prevent_initial_call=True,
        )
        def on_ug_col_1_change(col1_val: Any, current_col2: Any):
            # Auto-swap col2 if it would clash with the new col1; refresh show-ug-1 options only.
            new_col2 = current_col2 if current_col2 != col1_val else self._next_col(col1_val)
            opts = self._make_ug_opts(col1_val)
            return [new_col2] + [opts] * 3

        @self.app.callback(
            [Output(f"group-g{i}-show-ug-2", "options") for i in range(1, 4)],
            Input("group-ug-col-2", "value"),
            prevent_initial_call=True,
        )
        def on_ug_col_2_change(col2_val: Any):
            # Refresh show-ug-2 options only; see note above for why value is not touched.
            opts = self._make_ug_opts(col2_val)
            return [opts] * 3

        # ── Date-tab column selectors: same asymmetric pattern as tab_group ────────────────────────
        # on_date_ug_col_1_change: auto-swap col2 to avoid conflict, refresh show-ug-1 checklist options.
        # on_date_ug_col_2_change: only refresh show-ug-2 options (no output to col1 → no DAG cycle).
        # Neither callback touches checklist VALUES — owned by layout defaults and restore_dashboard_state.
        @self.app.callback(
            [
                Output("date-ug-col-2", "value"),
                Output(f"date-g1-show-ug-1", "options"),
                Output(f"date-g2-show-ug-1", "options"),
                Output(f"date-g3-show-ug-1", "options"),
            ],
            Input("date-ug-col-1", "value"),
            State("date-ug-col-2", "value"),
            prevent_initial_call=True,
        )
        def on_date_ug_col_1_change(col1_val: Any, current_col2: Any):
            new_col2 = current_col2 if current_col2 != col1_val else self._next_col(col1_val)
            opts = self._make_ug_opts(col1_val)
            return [new_col2] + [opts] * 3

        @self.app.callback(
            [
                Output(f"date-g1-show-ug-2", "options"),
                Output(f"date-g2-show-ug-2", "options"),
                Output(f"date-g3-show-ug-2", "options"),
            ],
            Input("date-ug-col-2", "value"),
            prevent_initial_call=True,
        )
        def on_date_ug_col_2_change(col2_val: Any):
            opts = self._make_ug_opts(col2_val)
            return [opts] * 3

        def _wrap_bet_plot(session, groups, left_m, right_m, share_l, share_r, log_v, log_t, filt_c, filt_t):
            fig = self.update_bet_plot(session, groups, left_m, right_m, share_l, share_r, log_v, log_t, filt_c, filt_t)
            state = {
                "session": session,
                "strategy": groups,
                "left_metrics": left_m,
                "right_metrics": right_m,
                "share_left": share_l,
                "share_right": share_r,
                "log": log_v,
                "log_thresh": log_t,
                "filter_check": filt_c,
                "filter_thresh": filt_t,
            }
            return fig, state

        self.app.callback(
            Output("combined-plot", "figure"),
            Output("tab-bet-state", "data"),
            [
                Input("session-dropdown", "value"),
                Input("strategy-checklist", "value"),
                Input("metric-checklist", "value"),
                Input("right-axis-checklist", "value"),
                Input("share-left-yscale-check", "value"),
                Input("share-right-yscale-check", "value"),
                Input("log-check", "value"),
                Input("linear-thresh", "value"),
                Input("filter-check", "value"),
                Input("filter-thresh", "value"),
            ],
        )(_wrap_bet_plot)

        def _date_plot_with_state_factory(gi: str):
            def _inner(
                left_m: Any,
                right_m: Any,
                log_v: Any,
                thresh: Any,
                gran: Any,
                start_d: Any,
                end_d: Any,
                date_ug_col_1: Any,
                date_ug_col_2: Any,
                show_ug_1: Any,
                show_ug_2: Any,
            ):
                fig = self.update_date_plot(
                    left_m,
                    right_m,
                    log_v,
                    thresh,
                    gran,
                    start_d,
                    end_d,
                    date_ug_col_1,
                    date_ug_col_2,
                    show_ug_1,
                    show_ug_2,
                )
                state = {
                    "left_metrics": left_m,
                    "right_metrics": right_m,
                    "log": log_v,
                    "thresh": thresh,
                    "date_granularity": gran,
                    "start_date": start_d,
                    "end_date": end_d,
                    "date_ug_col_1": date_ug_col_1,
                    "date_ug_col_2": date_ug_col_2,
                    "show_ug_1": show_ug_1,
                    "show_ug_2": show_ug_2,
                }
                return fig, state

            return _inner

        for i in range(1, 4):
            self.app.callback(
                Output(f"date-g{i}-plot", "figure"),
                Output(f"tab-date-g{i}-state", "data"),
                [
                    Input(f"date-g{i}-left-metrics", "value"),
                    Input(f"date-g{i}-right-metrics", "value"),
                    Input(f"date-g{i}-log", "value"),
                    Input(f"date-g{i}-thresh", "value"),
                    Input("date-granularity", "value"),
                    Input("date-picker-range", "start_date"),
                    Input("date-picker-range", "end_date"),
                    Input("date-ug-col-1", "value"),
                    Input("date-ug-col-2", "value"),
                    Input(f"date-g{i}-show-ug-1", "value"),
                    Input(f"date-g{i}-show-ug-2", "value"),
                ],
            )(_date_plot_with_state_factory(f"g{i}"))

        def _group_plot_with_state_factory(gi: str):
            orig = self.update_group_plot_combined_factory(gi)

            def _inner(
                metrics: Any,
                display: Any,
                r1_s: Any,
                r1_e: Any,
                r2_s: Any,
                r2_e: Any,
                r3_s: Any,
                r3_e: Any,
                show_r1: Any,
                show_r2: Any,
                show_r3: Any,
                clip_en: Any,
                clip_min: Any,
                clip_max: Any,
                ug_col_1: Any,
                ug_col_2: Any,
                show_ug_1: Any,
                show_ug_2: Any,
            ):
                fig = orig(
                    metrics,
                    display,
                    r1_s,
                    r1_e,
                    r2_s,
                    r2_e,
                    r3_s,
                    r3_e,
                    show_r1,
                    show_r2,
                    show_r3,
                    clip_en,
                    clip_min,
                    clip_max,
                    ug_col_1,
                    ug_col_2,
                    show_ug_1,
                    show_ug_2,
                )
                state = {
                    "metrics": metrics,
                    "display": display,
                    "date_range1_start": r1_s,
                    "date_range1_end": r1_e,
                    "date_range2_start": r2_s,
                    "date_range2_end": r2_e,
                    "date_range3_start": r3_s,
                    "date_range3_end": r3_e,
                    "show_r1": show_r1,
                    "show_r2": show_r2,
                    "show_r3": show_r3,
                    "clip_enable": clip_en,
                    "clip_min": clip_min,
                    "clip_max": clip_max,
                    "ug_col_1": ug_col_1,
                    "ug_col_2": ug_col_2,
                    "show_ug_1": show_ug_1,
                    "show_ug_2": show_ug_2,
                }
                return fig, state

            return _inner

        for i in range(1, 4):
            self.app.callback(
                Output(f"group-g{i}-plot", "figure"),
                Output(f"tab-group-g{i}-state", "data"),
                [
                    Input(f"group-g{i}-metrics", "value"),
                    Input(f"group-g{i}-display", "value"),
                    Input("date-picker-range1", "start_date"),
                    Input("date-picker-range1", "end_date"),
                    Input("date-picker-range2", "start_date"),
                    Input("date-picker-range2", "end_date"),
                    Input("date-picker-range3", "start_date"),
                    Input("date-picker-range3", "end_date"),
                    Input(f"group-g{i}-show-r1", "value"),
                    Input(f"group-g{i}-show-r2", "value"),
                    Input(f"group-g{i}-show-r3", "value"),
                    Input(f"group-g{i}-clip-enable", "value"),
                    Input(f"group-g{i}-clip-min", "value"),
                    Input(f"group-g{i}-clip-max", "value"),
                    Input("group-ug-col-1", "value"),
                    Input("group-ug-col-2", "value"),
                    Input(f"group-g{i}-show-ug-1", "value"),
                    Input(f"group-g{i}-show-ug-2", "value"),
                ],
            )(_group_plot_with_state_factory(f"g{i}"))

        # ── Stats Deepdive tab ─────────────────────────────────────────
        @self.app.callback(
            [Output("viz-ug-col-2", "value")] + [Output(f"viz-{pid}-show-ug-1", "options") for pid in ("p1", "p2")],
            Input("viz-ug-col-1", "value"),
            State("viz-ug-col-2", "value"),
            prevent_initial_call=True,
        )
        def on_viz_ug_col_1_change(col1_val: Any, current_col2: Any):
            new_col2 = current_col2 if current_col2 != col1_val else self._next_col(col1_val)
            opts = self._make_ug_opts(col1_val)
            return [new_col2] + [opts] * 2

        @self.app.callback(
            [Output(f"viz-{pid}-show-ug-2", "options") for pid in ("p1", "p2")],
            Input("viz-ug-col-2", "value"),
            prevent_initial_call=True,
        )
        def on_viz_ug_col_2_change(col2_val: Any):
            opts = self._make_ug_opts(col2_val)
            return [opts] * 2

        # Show/hide the per-mode controls (histogram-only vs scatter-only) per panel based
        # on Display Mode.  The controls live inside the fixed-width sidebar, so toggling
        # their display does NOT reflow the graph container — the plot keeps its stable
        # width across mode switches.
        def _make_toggle_mode_controls(pid: str, mode_key: str) -> Callable[..., Any]:
            visible_style: Dict[str, Any] = dict(Styles.CONTROL_GROUP)
            hidden_style: Dict[str, Any] = {"display": "none"}

            def _inner(mode: Any) -> Any:
                return visible_style if str(mode or "") == mode_key else hidden_style

            _inner.__name__ = f"toggle_viz_{mode_key}_controls_{pid}"
            return _inner

        for _pid in ("p1", "p2"):
            self.app.callback(
                Output(f"viz-{_pid}-hist-controls", "style"),
                Input(f"viz-{_pid}-display", "value"),
            )(_make_toggle_mode_controls(_pid, "histogram"))
            self.app.callback(
                Output(f"viz-{_pid}-scatter-controls", "style"),
                Input(f"viz-{_pid}-display", "value"),
            )(_make_toggle_mode_controls(_pid, "scatter"))

        # General Metrics multi-select is shown ONLY for histogram (heatmap and scatter both
        # use the per-row metric selector container instead).
        def _make_toggle_metrics_general(pid: str) -> Callable[..., Any]:
            visible: Dict[str, Any] = dict(Styles.CONTROL_GROUP)
            hidden: Dict[str, Any] = {"display": "none"}

            def _inner(mode: Any) -> Any:
                return visible if str(mode or "") == "histogram" else hidden

            _inner.__name__ = f"toggle_viz_metrics_general_{pid}"
            return _inner

        for _pid in ("p1", "p2"):
            self.app.callback(
                Output(f"viz-{_pid}-metrics-container", "style"),
                Input(f"viz-{_pid}-display", "value"),
            )(_make_toggle_metrics_general(_pid))
            self.app.callback(
                Output(f"viz-{_pid}-hm-container", "style"),
                Input(f"viz-{_pid}-display", "value"),
            )(_make_toggle_mode_controls(_pid, "heatmap"))
            self.app.callback(
                Output(f"viz-{_pid}-sm-container", "style"),
                Input(f"viz-{_pid}-display", "value"),
            )(_make_toggle_mode_controls(_pid, "scatter"))

        # Metric-row visibility is purely data-driven: show rows 0..max_filled+1 (always one
        # trailing empty slot so the user can add a new metric simply by picking a value; clearing
        # the last filled row collapses the empty slot, effectively removing that metric).
        def _make_ms_rows_visibility(pid: str, prefix: str) -> Callable[..., Any]:
            visible_style: Dict[str, Any] = {**Styles.FLEX_ROW_CENTER, "marginBottom": "4px", "display": "flex"}
            hidden_style: Dict[str, Any] = {**Styles.FLEX_ROW_CENTER, "marginBottom": "4px", "display": "none"}
            max_rows = self._VIZ_MS_MAX_ROWS

            def _inner(*values: Any) -> Any:
                filled_idx = [i for i, v in enumerate(values) if v not in (None, "", [])]
                max_filled = max(filled_idx) if filled_idx else -1
                visible_until = min(max_rows - 1, max_filled + 1)
                return [visible_style if i <= visible_until else hidden_style for i in range(max_rows)]

            _inner.__name__ = f"viz_{prefix}_rows_visibility_{pid}"
            return _inner

        for _pid in ("p1", "p2"):
            for _prefix in ("hm", "sm"):
                self.app.callback(
                    [Output(f"viz-{_pid}-{_prefix}-row-{i}", "style") for i in range(self._VIZ_MS_MAX_ROWS)],
                    [Input(f"viz-{_pid}-{_prefix}-metric-{i}", "value") for i in range(self._VIZ_MS_MAX_ROWS)],
                )(_make_ms_rows_visibility(_pid, _prefix))

        def _viz_plot_with_state_factory(pid: str) -> Callable[..., Any]:
            base = self.update_viz_plot_factory(pid)

            def _inner(
                metrics: Any,
                display: Any,
                r1_s: Any,
                r1_e: Any,
                r2_s: Any,
                r2_e: Any,
                r3_s: Any,
                r3_e: Any,
                show_r1: Any,
                show_r2: Any,
                show_r3: Any,
                clip_en: Any,
                clip_min: Any,
                clip_max: Any,
                ug_col_1: Any,
                ug_col_2: Any,
                show_ug_1: Any,
                show_ug_2: Any,
                nbins: Any,
                log_y: Any,
                normalize: Any,
                scatter_log_x: Any,
                scatter_log_y: Any,
                scatter_symlog_x: Any,
                scatter_symlog_y: Any,
                outliers_enable: Any,
                outliers_std: Any,
                *ms_rows: Any,
            ):
                # ms_rows flattened: [hm_m0..hm_m7, hm_l0..hm_l7, sm_m0..sm_m7, sm_l0..sm_l7]
                max_rows = self._VIZ_MS_MAX_ROWS
                hm_metrics = list(ms_rows[:max_rows])
                hm_logs = list(ms_rows[max_rows : 2 * max_rows])
                sm_metrics = list(ms_rows[2 * max_rows : 3 * max_rows])
                sm_logs = list(ms_rows[3 * max_rows : 4 * max_rows])
                fig = base(
                    metrics,
                    display,
                    r1_s,
                    r1_e,
                    r2_s,
                    r2_e,
                    r3_s,
                    r3_e,
                    show_r1,
                    show_r2,
                    show_r3,
                    clip_en,
                    clip_min,
                    clip_max,
                    ug_col_1,
                    ug_col_2,
                    show_ug_1,
                    show_ug_2,
                    nbins,
                    log_y,
                    normalize,
                    scatter_log_x,
                    scatter_log_y,
                    scatter_symlog_x,
                    scatter_symlog_y,
                    outliers_enable,
                    outliers_std,
                    hm_metrics,
                    hm_logs,
                    sm_metrics,
                    sm_logs,
                )
                state = {
                    "metrics": metrics,
                    "display": display,
                    "date_range1_start": r1_s,
                    "date_range1_end": r1_e,
                    "date_range2_start": r2_s,
                    "date_range2_end": r2_e,
                    "date_range3_start": r3_s,
                    "date_range3_end": r3_e,
                    "show_r1": show_r1,
                    "show_r2": show_r2,
                    "show_r3": show_r3,
                    "clip_enable": clip_en,
                    "clip_min": clip_min,
                    "clip_max": clip_max,
                    "ug_col_1": ug_col_1,
                    "ug_col_2": ug_col_2,
                    "show_ug_1": show_ug_1,
                    "show_ug_2": show_ug_2,
                    "nbins": nbins,
                    "log_y": log_y,
                    "normalize": normalize,
                    "scatter_log_x": scatter_log_x,
                    "scatter_log_y": scatter_log_y,
                    "scatter_symlog_x": scatter_symlog_x,
                    "scatter_symlog_y": scatter_symlog_y,
                    "outliers_enable": outliers_enable,
                    "outliers_std": outliers_std,
                    "hm_metrics": hm_metrics,
                    "hm_logs": hm_logs,
                    "sm_metrics": sm_metrics,
                    "sm_logs": sm_logs,
                }
                return fig, state

            return _inner

        for pid in ("p1", "p2"):
            self.app.callback(
                Output(f"viz-{pid}-plot", "figure"),
                Output(f"tab-viz-{pid}-state", "data"),
                [
                    Input(f"viz-{pid}-metrics", "value"),
                    Input(f"viz-{pid}-display", "value"),
                    Input("viz-date-picker-range1", "start_date"),
                    Input("viz-date-picker-range1", "end_date"),
                    Input("viz-date-picker-range2", "start_date"),
                    Input("viz-date-picker-range2", "end_date"),
                    Input("viz-date-picker-range3", "start_date"),
                    Input("viz-date-picker-range3", "end_date"),
                    Input(f"viz-{pid}-show-r1", "value"),
                    Input(f"viz-{pid}-show-r2", "value"),
                    Input(f"viz-{pid}-show-r3", "value"),
                    Input(f"viz-{pid}-clip-enable", "value"),
                    Input(f"viz-{pid}-clip-min", "value"),
                    Input(f"viz-{pid}-clip-max", "value"),
                    Input("viz-ug-col-1", "value"),
                    Input("viz-ug-col-2", "value"),
                    Input(f"viz-{pid}-show-ug-1", "value"),
                    Input(f"viz-{pid}-show-ug-2", "value"),
                    Input(f"viz-{pid}-nbins", "value"),
                    Input(f"viz-{pid}-log-y", "value"),
                    Input(f"viz-{pid}-normalize", "value"),
                    Input(f"viz-{pid}-scatter-log-x", "value"),
                    Input(f"viz-{pid}-scatter-log-y", "value"),
                    Input(f"viz-{pid}-scatter-symlog-x", "value"),
                    Input(f"viz-{pid}-scatter-symlog-y", "value"),
                    Input(f"viz-{pid}-outliers-enable", "value"),
                    Input(f"viz-{pid}-outliers-std", "value"),
                    *[Input(f"viz-{pid}-hm-metric-{i}", "value") for i in range(self._VIZ_MS_MAX_ROWS)],
                    *[Input(f"viz-{pid}-hm-log-{i}", "value") for i in range(self._VIZ_MS_MAX_ROWS)],
                    *[Input(f"viz-{pid}-sm-metric-{i}", "value") for i in range(self._VIZ_MS_MAX_ROWS)],
                    *[Input(f"viz-{pid}-sm-log-{i}", "value") for i in range(self._VIZ_MS_MAX_ROWS)],
                ],
            )(_viz_plot_with_state_factory(pid))

        # Save config: open modal and fetch existing configs
        # NOTE: Output("save-config-overwrite", "value") is included so the dropdown is
        # always reset to None when the modal opens.  Without this, a value selected in a
        # previous session (including a cancelled save) would persist and cause
        # save_config_confirm to treat it as an overwrite target even when the user types a
        # brand-new filename — silently writing to the old path and never creating the new file.
        @self.app.callback(
            Output("save-config-modal", "style"),
            Output("save-config-overwrite", "options"),
            Output("save-config-overwrite", "value"),
            Output("save-config-filename", "value"),
            Input("save-config-btn", "n_clicks"),
            Input("save-config-cancel", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_save_modal(save_clicks: int, cancel_clicks: int):
            triggered = callback_context.triggered_id
            if triggered == "save-config-btn":
                try:
                    bucket, prefix = parse_s3_path(DASHBOARD_CONFIG_S3_PATH)
                    prefix_ = prefix.rstrip("/") + "/" if prefix else ""
                    files = list_s3_files(bucket, prefix_, pattern=r"\.json$")
                    opts = [{"label": Path(f).name, "value": f} for f in files]
                except Exception as e:
                    logger.warning(f"Failed to list configs: {e}")
                    opts = []
                # Clear both fields every time the modal opens so stale state can't
                # accidentally route a "save as new" into an old overwrite path.
                return (
                    {
                        "display": "flex",
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                        "height": "100%",
                        "backgroundColor": "rgba(0,0,0,0.5)",
                        "zIndex": 1000,
                        "justifyContent": "center",
                        "alignItems": "center",
                    },
                    opts,
                    None,
                    "",
                )
            return (
                {
                    "display": "none",
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100%",
                    "height": "100%",
                    "backgroundColor": "rgba(0,0,0,0.5)",
                    "zIndex": 1000,
                },
                [],
                None,
                no_update,
            )

        # Save config: perform save to S3 (full state)
        modal_hidden = {
            "display": "none",
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0,0,0,0.5)",
            "zIndex": 1000,
        }

        @self.app.callback(
            Output("save-config-filename", "value", allow_duplicate=True),
            Output("save-config-overwrite", "value", allow_duplicate=True),
            Output("save-config-modal", "style", allow_duplicate=True),
            Input("save-config-confirm", "n_clicks"),
            State("save-config-filename", "value"),
            State("save-config-overwrite", "value"),
            State("config-dropdown", "value"),
            State("navigator-tabs", "value"),
            State("tab-date-g1-state", "data"),
            State("tab-date-g2-state", "data"),
            State("tab-date-g3-state", "data"),
            State("tab-group-g1-state", "data"),
            State("tab-group-g2-state", "data"),
            State("tab-group-g3-state", "data"),
            State("tab-viz-p1-state", "data"),
            State("tab-viz-p2-state", "data"),
            State("viz-ug-col-1", "value"),
            State("viz-ug-col-2", "value"),
            State("date-granularity", "value"),
            State("date-picker-range", "start_date"),
            State("date-picker-range", "end_date"),
            State("date-ug-col-1", "value"),
            State("date-ug-col-2", "value"),
            State("group-ug-col-1", "value"),
            State("group-ug-col-2", "value"),
            State("date-g1-left-metrics", "value"),
            State("date-g1-right-metrics", "value"),
            State("date-g1-log", "value"),
            State("date-g1-thresh", "value"),
            State("date-g2-left-metrics", "value"),
            State("date-g2-right-metrics", "value"),
            State("date-g2-log", "value"),
            State("date-g2-thresh", "value"),
            State("date-g3-left-metrics", "value"),
            State("date-g3-right-metrics", "value"),
            State("date-g3-log", "value"),
            State("date-g3-thresh", "value"),
            State("tab-bet-state", "data"),
            prevent_initial_call=True,
        )
        def save_config_confirm(
            n_clicks: int,
            filename: Optional[str],
            overwrite: Optional[str],
            config_file: str,
            current_tab: str,
            date_tab_g1_state: Optional[Dict],
            date_tab_g2_state: Optional[Dict],
            date_tab_g3_state: Optional[Dict],
            group_tab_g1_state: Optional[Dict],
            group_tab_g2_state: Optional[Dict],
            group_tab_g3_state: Optional[Dict],
            viz_tab_p1_state: Optional[Dict],
            viz_tab_p2_state: Optional[Dict],
            viz_ug_col_1: Any,
            viz_ug_col_2: Any,
            date_granularity: Optional[str],
            date_picker_start_date: Any,
            date_picker_end_date: Any,
            date_ug_col_1: Any,
            date_ug_col_2: Any,
            group_ug_col_1: Any,
            group_ug_col_2: Any,
            date_g1_left_metrics: Any,
            date_g1_right_metrics: Any,
            date_g1_log: Any,
            date_g1_thresh: Any,
            date_g2_left_metrics: Any,
            date_g2_right_metrics: Any,
            date_g2_log: Any,
            date_g2_thresh: Any,
            date_g3_left_metrics: Any,
            date_g3_right_metrics: Any,
            date_g3_log: Any,
            date_g3_thresh: Any,
            bet: Optional[Dict],
        ):
            if not n_clicks:
                return no_update, no_update, no_update
            bucket, prefix = parse_s3_path(DASHBOARD_CONFIG_S3_PATH)
            prefix_ = prefix.rstrip("/") + "/"
            if overwrite:
                s3_path = overwrite
            else:
                name = (filename or "config").strip()
                if not name.endswith(".json"):
                    name = name + ".json"
                s3_path = f"s3://{bucket}/{prefix_}{name}"
            # tab_date: shared date_range for the Date tab, plus per-panel (g1/g2/g3) without start/end
            date_keys_date_tab = ("start_date", "end_date")
            date_range_tab_date: Dict[str, Any] = {}
            # Prefer current picker values to avoid stale tab-date store values.
            if date_picker_start_date is not None and date_picker_end_date is not None:
                date_range_tab_date = {
                    "start_date": date_picker_start_date,
                    "end_date": date_picker_end_date,
                }
            elif date_tab_g1_state:
                date_range_tab_date = {
                    k: date_tab_g1_state.get(k) for k in date_keys_date_tab if date_tab_g1_state.get(k) is not None
                }

            def _date_tab_panel_for_config(panel_state: Optional[Dict]) -> Optional[Dict]:
                if not panel_state:
                    return panel_state
                # Strip shared date range and tab-level column keys (show_ug_* stay per-panel).
                strip_date = set(date_keys_date_tab) | {"date_ug_col_1", "date_ug_col_2", "groups"}
                return {k: v for k, v in panel_state.items() if k not in strip_date}

            def _merge_date_tab_panel_for_save(store: Optional[Dict], ui_fields: Dict[str, Any]) -> Optional[Dict]:
                merged = {**(store or {}), **ui_fields}
                return _date_tab_panel_for_config(merged)

            # tab_group: shared date_ranges for the Group tab (range1/2/3), plus per-panel (g1/g2/g3) without those keys
            date_range_keys = (
                "date_range1_start",
                "date_range1_end",
                "date_range2_start",
                "date_range2_end",
                "date_range3_start",
                "date_range3_end",
            )
            date_range_tab_group: Dict[str, Any] = {}
            if group_tab_g1_state:
                date_range_tab_group = {
                    k: group_tab_g1_state.get(k) for k in date_range_keys if group_tab_g1_state.get(k) is not None
                }

            def _group_tab_panel_for_config(panel_state: Optional[Dict]) -> Optional[Dict]:
                if not panel_state:
                    return panel_state
                # Keep only per-panel fields; strip shared range1/2/3 and tab-level col selectors.
                strip_group = set(date_range_keys) | {"ug_col_1", "ug_col_2"}
                return {k: v for k, v in panel_state.items() if k not in strip_group}

            date_tab_panel_g1 = _merge_date_tab_panel_for_save(
                date_tab_g1_state,
                {
                    "left_metrics": date_g1_left_metrics,
                    "right_metrics": date_g1_right_metrics,
                    "log": date_g1_log,
                    "thresh": date_g1_thresh,
                    "date_granularity": date_granularity,
                },
            )
            date_tab_panel_g2 = _merge_date_tab_panel_for_save(
                date_tab_g2_state,
                {
                    "left_metrics": date_g2_left_metrics,
                    "right_metrics": date_g2_right_metrics,
                    "log": date_g2_log,
                    "thresh": date_g2_thresh,
                    "date_granularity": date_granularity,
                },
            )
            date_tab_panel_g3 = _merge_date_tab_panel_for_save(
                date_tab_g3_state,
                {
                    "left_metrics": date_g3_left_metrics,
                    "right_metrics": date_g3_right_metrics,
                    "log": date_g3_log,
                    "thresh": date_g3_thresh,
                    "date_granularity": date_granularity,
                },
            )

            payload: Dict[str, Any] = {
                "config_file": config_file,
                "current_tab": current_tab,
                "tab_date": {
                    "date_range": date_range_tab_date,
                    "date_ug_col_1": date_ug_col_1,
                    "date_ug_col_2": date_ug_col_2,
                    "g1": date_tab_panel_g1,
                    "g2": date_tab_panel_g2,
                    "g3": date_tab_panel_g3,
                },
                "tab_group": {
                    "date_ranges": date_range_tab_group,
                    "ug_col_1": group_ug_col_1,
                    "ug_col_2": group_ug_col_2,
                    "g1": _group_tab_panel_for_config(group_tab_g1_state),
                    "g2": _group_tab_panel_for_config(group_tab_g2_state),
                    "g3": _group_tab_panel_for_config(group_tab_g3_state),
                },
                "tab_viz": {
                    "date_ranges": (
                        {k: viz_tab_p1_state.get(k) for k in date_range_keys if viz_tab_p1_state.get(k) is not None}
                        if viz_tab_p1_state
                        else {}
                    ),
                    "ug_col_1": viz_ug_col_1,
                    "ug_col_2": viz_ug_col_2,
                    "p1": _group_tab_panel_for_config(viz_tab_p1_state),
                    "p2": _group_tab_panel_for_config(viz_tab_p2_state),
                },
                "tab_bet": bet,
            }
            normalized = _normalize_dates_in_state(payload)
            payload = normalized if normalized is not None else payload
            payload = _json_sanitize(payload)
            try:
                write_json_to_s3(payload, s3_path)
            except Exception as e:
                logger.error("Failed to save config to S3: %s", e, exc_info=True)
            return "", None, modal_hidden

        # Load config: show dropdown when Load Config is clicked
        @self.app.callback(
            Output("load-config-dropdown-container", "style"),
            Output("load-config-dropdown", "options"),
            Output("load-config-dropdown", "value"),
            Input("load-config-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def show_load_dropdown(n_clicks: int):
            if not n_clicks:
                return no_update, no_update, no_update
            try:
                bucket, prefix = parse_s3_path(DASHBOARD_CONFIG_S3_PATH)
                prefix_ = prefix.rstrip("/") + "/"
                files = list_s3_files(bucket, prefix_, pattern=r"\.json$")
                opts = [{"label": Path(f).name, "value": f} for f in files]
            except Exception as e:
                logger.warning(f"Failed to list configs: {e}")
                opts = []
            return {"display": "block", "marginLeft": "8px", "minWidth": "200px"}, opts, None

        # Load config: apply when user selects a file (populate stores + trigger restore)
        @self.app.callback(
            Output("config-dropdown", "value", allow_duplicate=True),
            Output("navigator-tabs", "value", allow_duplicate=True),
            Output("load-config-dropdown", "value", allow_duplicate=True),
            Output("tab-date-g1-state", "data", allow_duplicate=True),
            Output("tab-date-g2-state", "data", allow_duplicate=True),
            Output("tab-date-g3-state", "data", allow_duplicate=True),
            Output("tab-group-g1-state", "data", allow_duplicate=True),
            Output("tab-group-g2-state", "data", allow_duplicate=True),
            Output("tab-group-g3-state", "data", allow_duplicate=True),
            Output("tab-viz-p1-state", "data", allow_duplicate=True),
            Output("tab-viz-p2-state", "data", allow_duplicate=True),
            Output("tab-bet-state", "data", allow_duplicate=True),
            Output("dashboard-load-trigger", "data", allow_duplicate=True),
            Input("load-config-dropdown", "value"),
            prevent_initial_call=True,
        )
        def load_config_select(s3_uri: Optional[str]):
            nothing = (no_update,) * 13
            if not s3_uri:
                return nothing
            try:
                data = read_json_from_s3(s3_uri)
                config_file = data.get("config_file")
                current_tab = data.get("current_tab", "tab-date")
                tab_date = data.get("tab_date", {})
                tab_group = data.get("tab_group", {})
                tab_viz = data.get("tab_viz", {}) or {}
                tab_bet = data.get("tab_bet")
                if not config_file:
                    return nothing
                # Ensure config_file matches an option (path may differ by machine)
                saved_basename = Path(config_file).name
                if not any(str(opt["value"]) == config_file for opt in self.config_files):
                    match = next(
                        (opt["value"] for opt in self.config_files if Path(str(opt["value"])).name == saved_basename),
                        None,
                    )
                    if match:
                        config_file = match
                # Merge shared date_range and group-selector values into each panel for restore
                date_range = tab_date.get("date_range") or {}
                _date_ug_col_1 = tab_date.get(
                    "date_ug_col_1", self.df_user_group_cols[0] if self.df_user_group_cols else None
                )
                _date_ug_col_2 = tab_date.get(
                    "date_ug_col_2", self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None
                )
                _legacy_date_show_ug_1 = tab_date.get("date_show_ug_1")
                _legacy_date_show_ug_2 = tab_date.get("date_show_ug_2")

                def _d_with_range(gi: str) -> Optional[Dict]:
                    g = tab_date.get(gi)
                    if not g:
                        return g
                    merged: Dict[str, Any] = {**date_range, **g}
                    merged["date_ug_col_1"] = _date_ug_col_1
                    merged["date_ug_col_2"] = _date_ug_col_2
                    if _legacy_date_show_ug_1 is not None and merged.get("show_ug_1") is None:
                        merged["show_ug_1"] = _legacy_date_show_ug_1
                    if _legacy_date_show_ug_2 is not None and merged.get("show_ug_2") is None:
                        merged["show_ug_2"] = _legacy_date_show_ug_2
                    return merged

                d1_out = _d_with_range("g1")
                d2_out = _d_with_range("g2")
                d3_out = _d_with_range("g3")

                # Ensure tab_group has date_ranges; if missing/empty, fall back to default recent ranges
                if not tab_group.get("date_ranges"):
                    (
                        _min_d,
                        _max_d,
                        g1_start,
                        g1_end,
                        g2_start,
                        g2_end,
                        g3_start,
                        g3_end,
                    ) = self._compute_group_date_ranges()
                    tab_group["date_ranges"] = {
                        "date_range1_start": _normalize_date_value(g1_start),
                        "date_range1_end": _normalize_date_value(g1_end),
                        "date_range2_start": _normalize_date_value(g2_start),
                        "date_range2_end": _normalize_date_value(g2_end),
                        "date_range3_start": _normalize_date_value(g3_start),
                        "date_range3_end": _normalize_date_value(g3_end),
                    }

                # Cache loaded date ranges so the group layout can reuse them instead of recomputing defaults
                dr = tab_group.get("date_ranges") or {}
                if dr:
                    s1, e1 = dr.get("date_range1_start"), dr.get("date_range1_end")
                    s2, e2 = dr.get("date_range2_start"), dr.get("date_range2_end")
                    s3, e3 = dr.get("date_range3_start"), dr.get("date_range3_end")
                    # Compute overall min/max using string comparison (safe for YYYY-MM-DD)
                    starts = [d for d in [s1, s2, s3] if d is not None]
                    ends = [d for d in [e1, e2, e3] if d is not None]
                    min_d = min(starts) if starts else None
                    max_d = max(ends) if ends else None
                    self._loaded_group_date_ranges = (min_d, max_d, s1, e1, s2, e2, s3, e3)
                else:
                    self._loaded_group_date_ranges = None
                # Merge shared date_ranges into each g for restore
                date_ranges = dr
                ug_col_1 = tab_group.get("ug_col_1", self.df_user_group_cols[0] if self.df_user_group_cols else None)
                ug_col_2 = tab_group.get(
                    "ug_col_2", self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None
                )

                def _g_with_dates(gi: str) -> Optional[Dict]:
                    g = tab_group.get(gi)
                    if not g:
                        return g
                    merged_g: Dict[str, Any] = {**date_ranges, **g}
                    merged_g["ug_col_1"] = ug_col_1
                    merged_g["ug_col_2"] = ug_col_2
                    return merged_g

                g1_out = _g_with_dates("g1")
                g2_out = _g_with_dates("g2")
                g3_out = _g_with_dates("g3")

                # tab_viz: mirror tab_group's shape — date_ranges + ug_col_1/2 + per-panel state
                viz_date_ranges = tab_viz.get("date_ranges") or {}
                viz_ug_col_1 = tab_viz.get("ug_col_1", self.df_user_group_cols[0] if self.df_user_group_cols else None)
                viz_ug_col_2 = tab_viz.get(
                    "ug_col_2", self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None
                )

                def _p_with_dates(pi: str) -> Optional[Dict]:
                    p = tab_viz.get(pi)
                    if not p:
                        return p
                    merged_p: Dict[str, Any] = {**viz_date_ranges, **p}
                    merged_p["ug_col_1"] = viz_ug_col_1
                    merged_p["ug_col_2"] = viz_ug_col_2
                    return merged_p

                p1_out = _p_with_dates("p1")
                p2_out = _p_with_dates("p2")

                return (
                    config_file,
                    current_tab,
                    None,
                    d1_out,
                    d2_out,
                    d3_out,
                    g1_out,
                    g2_out,
                    g3_out,
                    p1_out,
                    p2_out,
                    tab_bet,
                    datetime.now().isoformat(),
                )
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
            return (no_update,) * 13

        # Restore: apply loaded state to all UI components when load-trigger fires
        @self.app.callback(
            Output("dashboard-load-trigger", "data", allow_duplicate=True),
            Output("date-granularity", "value", allow_duplicate=True),
            Output("date-picker-range", "start_date", allow_duplicate=True),
            Output("date-picker-range", "end_date", allow_duplicate=True),
            Output("date-ug-col-1", "value", allow_duplicate=True),
            Output("date-ug-col-2", "value", allow_duplicate=True),
            Output("date-g1-left-metrics", "value", allow_duplicate=True),
            Output("date-g1-right-metrics", "value", allow_duplicate=True),
            Output("date-g1-log", "value", allow_duplicate=True),
            Output("date-g1-thresh", "value", allow_duplicate=True),
            Output("date-g1-show-ug-1", "value", allow_duplicate=True),
            Output("date-g1-show-ug-2", "value", allow_duplicate=True),
            Output("date-g2-left-metrics", "value", allow_duplicate=True),
            Output("date-g2-right-metrics", "value", allow_duplicate=True),
            Output("date-g2-log", "value", allow_duplicate=True),
            Output("date-g2-thresh", "value", allow_duplicate=True),
            Output("date-g2-show-ug-1", "value", allow_duplicate=True),
            Output("date-g2-show-ug-2", "value", allow_duplicate=True),
            Output("date-g3-left-metrics", "value", allow_duplicate=True),
            Output("date-g3-right-metrics", "value", allow_duplicate=True),
            Output("date-g3-log", "value", allow_duplicate=True),
            Output("date-g3-thresh", "value", allow_duplicate=True),
            Output("date-g3-show-ug-1", "value", allow_duplicate=True),
            Output("date-g3-show-ug-2", "value", allow_duplicate=True),
            Output("date-picker-range1", "start_date", allow_duplicate=True),
            Output("date-picker-range1", "end_date", allow_duplicate=True),
            Output("date-picker-range2", "start_date", allow_duplicate=True),
            Output("date-picker-range2", "end_date", allow_duplicate=True),
            Output("date-picker-range3", "start_date", allow_duplicate=True),
            Output("date-picker-range3", "end_date", allow_duplicate=True),
            Output("group-ug-col-1", "value", allow_duplicate=True),
            Output("group-ug-col-2", "value", allow_duplicate=True),
            Output("group-g1-metrics", "value", allow_duplicate=True),
            Output("group-g1-display", "value", allow_duplicate=True),
            Output("group-g1-show-r1", "value", allow_duplicate=True),
            Output("group-g1-show-r2", "value", allow_duplicate=True),
            Output("group-g1-show-r3", "value", allow_duplicate=True),
            Output("group-g1-clip-enable", "value", allow_duplicate=True),
            Output("group-g1-clip-min", "value", allow_duplicate=True),
            Output("group-g1-clip-max", "value", allow_duplicate=True),
            Output("group-g1-show-ug-1", "value", allow_duplicate=True),
            Output("group-g1-show-ug-2", "value", allow_duplicate=True),
            Output("group-g2-metrics", "value", allow_duplicate=True),
            Output("group-g2-display", "value", allow_duplicate=True),
            Output("group-g2-show-r1", "value", allow_duplicate=True),
            Output("group-g2-show-r2", "value", allow_duplicate=True),
            Output("group-g2-show-r3", "value", allow_duplicate=True),
            Output("group-g2-clip-enable", "value", allow_duplicate=True),
            Output("group-g2-clip-min", "value", allow_duplicate=True),
            Output("group-g2-clip-max", "value", allow_duplicate=True),
            Output("group-g2-show-ug-1", "value", allow_duplicate=True),
            Output("group-g2-show-ug-2", "value", allow_duplicate=True),
            Output("group-g3-metrics", "value", allow_duplicate=True),
            Output("group-g3-display", "value", allow_duplicate=True),
            Output("group-g3-show-r1", "value", allow_duplicate=True),
            Output("group-g3-show-r2", "value", allow_duplicate=True),
            Output("group-g3-show-r3", "value", allow_duplicate=True),
            Output("group-g3-clip-enable", "value", allow_duplicate=True),
            Output("group-g3-clip-min", "value", allow_duplicate=True),
            Output("group-g3-clip-max", "value", allow_duplicate=True),
            Output("group-g3-show-ug-1", "value", allow_duplicate=True),
            Output("group-g3-show-ug-2", "value", allow_duplicate=True),
            Output("viz-date-picker-range1", "start_date", allow_duplicate=True),
            Output("viz-date-picker-range1", "end_date", allow_duplicate=True),
            Output("viz-date-picker-range2", "start_date", allow_duplicate=True),
            Output("viz-date-picker-range2", "end_date", allow_duplicate=True),
            Output("viz-date-picker-range3", "start_date", allow_duplicate=True),
            Output("viz-date-picker-range3", "end_date", allow_duplicate=True),
            Output("viz-ug-col-1", "value", allow_duplicate=True),
            Output("viz-ug-col-2", "value", allow_duplicate=True),
            Output("viz-p1-metrics", "value", allow_duplicate=True),
            Output("viz-p1-display", "value", allow_duplicate=True),
            Output("viz-p1-show-r1", "value", allow_duplicate=True),
            Output("viz-p1-show-r2", "value", allow_duplicate=True),
            Output("viz-p1-show-r3", "value", allow_duplicate=True),
            Output("viz-p1-clip-enable", "value", allow_duplicate=True),
            Output("viz-p1-clip-min", "value", allow_duplicate=True),
            Output("viz-p1-clip-max", "value", allow_duplicate=True),
            Output("viz-p1-show-ug-1", "value", allow_duplicate=True),
            Output("viz-p1-show-ug-2", "value", allow_duplicate=True),
            Output("viz-p1-nbins", "value", allow_duplicate=True),
            Output("viz-p1-log-y", "value", allow_duplicate=True),
            Output("viz-p1-normalize", "value", allow_duplicate=True),
            Output("viz-p1-scatter-log-x", "value", allow_duplicate=True),
            Output("viz-p1-scatter-log-y", "value", allow_duplicate=True),
            Output("viz-p1-scatter-symlog-x", "value", allow_duplicate=True),
            Output("viz-p1-scatter-symlog-y", "value", allow_duplicate=True),
            Output("viz-p1-outliers-enable", "value", allow_duplicate=True),
            Output("viz-p1-outliers-std", "value", allow_duplicate=True),
            *[Output(f"viz-p1-hm-metric-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            *[Output(f"viz-p1-hm-log-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            *[Output(f"viz-p1-sm-metric-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            *[Output(f"viz-p1-sm-log-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            Output("viz-p2-metrics", "value", allow_duplicate=True),
            Output("viz-p2-display", "value", allow_duplicate=True),
            Output("viz-p2-show-r1", "value", allow_duplicate=True),
            Output("viz-p2-show-r2", "value", allow_duplicate=True),
            Output("viz-p2-show-r3", "value", allow_duplicate=True),
            Output("viz-p2-clip-enable", "value", allow_duplicate=True),
            Output("viz-p2-clip-min", "value", allow_duplicate=True),
            Output("viz-p2-clip-max", "value", allow_duplicate=True),
            Output("viz-p2-show-ug-1", "value", allow_duplicate=True),
            Output("viz-p2-show-ug-2", "value", allow_duplicate=True),
            Output("viz-p2-nbins", "value", allow_duplicate=True),
            Output("viz-p2-log-y", "value", allow_duplicate=True),
            Output("viz-p2-normalize", "value", allow_duplicate=True),
            Output("viz-p2-scatter-log-x", "value", allow_duplicate=True),
            Output("viz-p2-scatter-log-y", "value", allow_duplicate=True),
            Output("viz-p2-scatter-symlog-x", "value", allow_duplicate=True),
            Output("viz-p2-scatter-symlog-y", "value", allow_duplicate=True),
            Output("viz-p2-outliers-enable", "value", allow_duplicate=True),
            Output("viz-p2-outliers-std", "value", allow_duplicate=True),
            *[Output(f"viz-p2-hm-metric-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            *[Output(f"viz-p2-hm-log-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            *[Output(f"viz-p2-sm-metric-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            *[Output(f"viz-p2-sm-log-{i}", "value", allow_duplicate=True) for i in range(self._VIZ_MS_MAX_ROWS)],
            Output("session-dropdown", "value", allow_duplicate=True),
            Output("strategy-checklist", "value", allow_duplicate=True),
            Output("metric-checklist", "value", allow_duplicate=True),
            Output("right-axis-checklist", "value", allow_duplicate=True),
            Output("share-left-yscale-check", "value", allow_duplicate=True),
            Output("share-right-yscale-check", "value", allow_duplicate=True),
            Output("log-check", "value", allow_duplicate=True),
            Output("linear-thresh", "value", allow_duplicate=True),
            Output("filter-check", "value", allow_duplicate=True),
            Output("filter-thresh", "value", allow_duplicate=True),
            Input("tab-date-content", "children"),
            State("dashboard-load-trigger", "data"),
            State("tab-date-g1-state", "data"),
            State("tab-date-g2-state", "data"),
            State("tab-date-g3-state", "data"),
            State("tab-group-g1-state", "data"),
            State("tab-group-g2-state", "data"),
            State("tab-group-g3-state", "data"),
            State("tab-viz-p1-state", "data"),
            State("tab-viz-p2-state", "data"),
            State("tab-bet-state", "data"),
            prevent_initial_call=True,
        )
        def restore_dashboard_state(
            _tab_children: Any,
            trigger: Any,
            d1: Optional[Dict],
            d2: Optional[Dict],
            d3: Optional[Dict],
            g1: Optional[Dict],
            g2: Optional[Dict],
            g3: Optional[Dict],
            v1: Optional[Dict],
            v2: Optional[Dict],
            bet: Optional[Dict],
        ):
            if trigger is None:
                return (no_update,) * 246
            # Keep the load trigger token so default-picker callbacks don't recompute
            # and clamp restored DatePickerRange values.
            out: List[Any] = [trigger]
            sd = d1 or {}
            out.extend(
                [
                    sd.get("date_granularity"),
                    _normalize_date_value(sd.get("start_date")),
                    _normalize_date_value(sd.get("end_date")),
                    sd.get("date_ug_col_1", self.df_user_group_cols[0] if self.df_user_group_cols else None),
                    sd.get("date_ug_col_2", self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None),
                ]
            )
            for s in [d1, d2, d3]:
                sd = s or {}
                out.extend(
                    [
                        sd.get("left_metrics"),
                        sd.get("right_metrics"),
                        sd.get("log"),
                        sd.get("thresh"),
                        sd.get("show_ug_1", no_update),
                        sd.get("show_ug_2", no_update),
                    ]
                )
            sg = g1 or {}
            out.extend(
                [
                    _normalize_date_value(sg.get("date_range1_start")),
                    _normalize_date_value(sg.get("date_range1_end")),
                    _normalize_date_value(sg.get("date_range2_start")),
                    _normalize_date_value(sg.get("date_range2_end")),
                    _normalize_date_value(sg.get("date_range3_start")),
                    _normalize_date_value(sg.get("date_range3_end")),
                    sg.get("ug_col_1", self.df_user_group_cols[0] if self.df_user_group_cols else None),
                    sg.get("ug_col_2", self.df_user_group_cols[1] if len(self.df_user_group_cols) > 1 else None),
                ]
            )
            for s in [g1, g2, g3]:
                sg = s or {}
                out.extend(
                    [
                        sg.get("metrics"),
                        sg.get("display"),
                        sg.get("show_r1"),
                        sg.get("show_r2"),
                        sg.get("show_r3"),
                        sg.get("clip_enable"),
                        sg.get("clip_min"),
                        sg.get("clip_max"),
                        sg.get("show_ug_1", no_update),
                        sg.get("show_ug_2", no_update),
                    ]
                )

            # Viz tab: shared date ranges + ug cols (from v1), then per-panel fields.
            # Use `no_update` as the fallback so loading a legacy config (no tab_viz section)
            # does not overwrite the viz DatePickerRange / dropdown / checklist values with
            # nulls — which otherwise surfaces as "Cannot read properties of null
            # (reading 'indexOf')" from the date picker JS when start_date/end_date is None.
            def _viz_or_skip(d: Optional[Dict], key: str) -> Any:
                if not d or key not in d or d.get(key) is None:
                    return no_update
                return d[key]

            def _viz_date_or_skip(d: Optional[Dict], key: str) -> Any:
                val = _viz_or_skip(d, key)
                return _normalize_date_value(val) if val is not no_update else no_update

            def _viz_hm_list_or_skip(d: Optional[Dict], key: str, n: int) -> List[Any]:
                """Expand a saved list-valued heatmap field (hm_metrics / hm_logs) to n slots.

                Missing or short lists get ``no_update`` for the uncovered slots so untouched
                rows aren't overwritten on config load.
                """
                if not d or key not in d or d.get(key) is None:
                    return [no_update] * n
                raw = d[key]
                if not isinstance(raw, list):
                    return [no_update] * n
                out_list: List[Any] = []
                for i in range(n):
                    out_list.append(raw[i] if i < len(raw) else no_update)
                return out_list

            out.extend(
                [
                    _viz_date_or_skip(v1, "date_range1_start"),
                    _viz_date_or_skip(v1, "date_range1_end"),
                    _viz_date_or_skip(v1, "date_range2_start"),
                    _viz_date_or_skip(v1, "date_range2_end"),
                    _viz_date_or_skip(v1, "date_range3_start"),
                    _viz_date_or_skip(v1, "date_range3_end"),
                    _viz_or_skip(v1, "ug_col_1"),
                    _viz_or_skip(v1, "ug_col_2"),
                ]
            )
            for s in [v1, v2]:
                out.extend(
                    [
                        _viz_or_skip(s, "metrics"),
                        _viz_or_skip(s, "display"),
                        _viz_or_skip(s, "show_r1"),
                        _viz_or_skip(s, "show_r2"),
                        _viz_or_skip(s, "show_r3"),
                        _viz_or_skip(s, "clip_enable"),
                        _viz_or_skip(s, "clip_min"),
                        _viz_or_skip(s, "clip_max"),
                        _viz_or_skip(s, "show_ug_1"),
                        _viz_or_skip(s, "show_ug_2"),
                        _viz_or_skip(s, "nbins"),
                        _viz_or_skip(s, "log_y"),
                        _viz_or_skip(s, "normalize"),
                        _viz_or_skip(s, "scatter_log_x"),
                        _viz_or_skip(s, "scatter_log_y"),
                        _viz_or_skip(s, "scatter_symlog_x"),
                        _viz_or_skip(s, "scatter_symlog_y"),
                        _viz_or_skip(s, "outliers_enable"),
                        _viz_or_skip(s, "outliers_std"),
                        *_viz_hm_list_or_skip(s, "hm_metrics", self._VIZ_MS_MAX_ROWS),
                        *_viz_hm_list_or_skip(s, "hm_logs", self._VIZ_MS_MAX_ROWS),
                        *_viz_hm_list_or_skip(s, "sm_metrics", self._VIZ_MS_MAX_ROWS),
                        *_viz_hm_list_or_skip(s, "sm_logs", self._VIZ_MS_MAX_ROWS),
                    ]
                )
            sb = bet or {}
            out.extend(
                [
                    sb.get("session"),
                    sb.get("strategy"),
                    sb.get("left_metrics"),
                    sb.get("right_metrics"),
                    sb.get("share_left"),
                    sb.get("share_right"),
                    sb.get("log"),
                    sb.get("log_thresh"),
                    sb.get("filter_check"),
                    sb.get("filter_thresh"),
                ]
            )
            return tuple(out)

    def _get_plot_groups(self, lf: pl.LazyFrame, group_col: str) -> list:
        """
        Extract unique non-None groups from a LazyFrame by group column.
        Returns groups sorted alphabetically (by string representation).
        """
        if lf.collect_schema().len() > 0 and group_col in lf.collect_schema().names():
            uniq = lf.select(pl.col(group_col).unique()).collect()
            groups = [x for x in uniq.to_series().to_list() if x is not None]
            return sorted(groups, key=str)
        return []

    def _ensure_user_group_filter(self) -> None:
        """Set df_user_group_col(s) from YAML.

        Accepts either the new ``user_group_cols`` list or the legacy ``user_group_col`` string.
        ``df_user_group_col`` is the first configured column (used as default filter column).
        ``df_user_group_col_options`` drives both the date-tab and group-tab column picker dropdowns.
        ``df_user_group_dropdown_options`` is kept for backward-compat but no longer used in the UI.
        """
        sd = self.config.get("stats_by_date", {})
        # Accept list (new) or scalar string (legacy).
        raw = sd.get("user_group_cols") or sd.get("user_group_col")
        if isinstance(raw, str):
            cols = [raw.strip()] if raw.strip() else []
        elif isinstance(raw, list):
            cols = [str(c).strip() for c in raw if str(c).strip()]
        else:
            cols = []

        self.df_user_group_cols = cols
        self.df_user_group_col = cols[0] if cols else ""
        self.df_user_group_col_options = [{"label": c, "value": c} for c in cols]

        if not self.df_user_group_col:
            self.df_user_group_dropdown_options = _dropdown_options_from_strings(["all"])
            return
        default_gran = list(self.date_files_config.keys())[0]
        lf = self.lfs_by_date.get(default_gran, pl.DataFrame().lazy())
        col = self.df_user_group_col
        if lf.collect_schema().len() == 0 or col not in lf.collect_schema().names():
            self.df_user_group_dropdown_options = _dropdown_options_from_strings(["all"])
            return
        uniq = self._get_plot_groups(lf, col)
        self.df_user_group_dropdown_options = _dropdown_options_from_strings(["all"]) + _dropdown_options_from_strings(
            [str(u) for u in uniq]
        )

    def _apply_user_group_filter(
        self, df: pl.DataFrame, user_group_val: Any, col: Optional[str] = None
    ) -> pl.DataFrame:
        col = col or self.df_user_group_col
        if not col or col not in df.columns:
            return df
        v = user_group_val if user_group_val is not None else "all"
        if str(v) == "all":
            return df
        return df.filter(pl.col(col) == v)

    def _apply_ug_filter_list(self, df: pl.DataFrame, selected_vals: Any, col: Optional[str] = None) -> pl.DataFrame:
        """Filter df to rows where col is in selected_vals (union). 'all' means no filter."""
        col = col or self.df_user_group_col
        if not col or col not in df.columns:
            return df
        vals_raw = selected_vals if isinstance(selected_vals, list) else ([selected_vals] if selected_vals else [])
        clean = [v for v in vals_raw if str(v) != "all"]
        if not clean:
            return df
        return df.filter(pl.col(col).is_in(clean))

    @staticmethod
    def _effective_ug_col2(ug_col_1: Any, ug_col_2: Any) -> Optional[str]:
        """Return second UG column only when configured and distinct from the first."""
        return ug_col_2 if (ug_col_2 and ug_col_2 != ug_col_1) else None

    @staticmethod
    def _normalize_ug_selection(selected_vals: Any) -> List[Any]:
        """Normalize checklist values; strips 'all' so empty means no explicit filter."""
        vals_raw = selected_vals if isinstance(selected_vals, list) else ([selected_vals] if selected_vals else [])
        return [v for v in vals_raw if str(v) != "all"]

    def _filter_df_for_ug_selection(
        self,
        df: pl.DataFrame,
        ug_col_1: Any,
        ug_col_2: Any,
        show_ug_1: Any,
        show_ug_2: Any,
    ) -> Tuple[pl.DataFrame, Optional[str], List[Any], List[Any]]:
        """
        Apply the two user-group checklist filters and return:
        (filtered_df, effective_col2, clean_vals1, clean_vals2).
        """
        clean_1 = self._normalize_ug_selection(show_ug_1)
        effective_col2 = self._effective_ug_col2(ug_col_1, ug_col_2)
        clean_2 = self._normalize_ug_selection(show_ug_2) if effective_col2 else []
        out = self._apply_ug_filter_list(df, clean_1, col=ug_col_1 or None)
        if effective_col2:
            out = self._apply_ug_filter_list(out, clean_2, col=effective_col2)
        return out, effective_col2, clean_1, clean_2

    def _make_ug_opts(self, col: Any) -> list[dcc.Dropdown.Options]:
        """Return dropdown/checklist options for a user-group column (includes 'all' first)."""
        if not col:
            return []
        gran = list(self.date_files_config.keys())[0]
        lf = self.lfs_by_date.get(gran, pl.DataFrame().lazy())
        schema = lf.collect_schema()
        if schema.len() == 0 or col not in schema.names():
            return _dropdown_options_from_strings(["all"])
        uniq = self._get_plot_groups(lf, col)
        return _dropdown_options_from_strings(["all"]) + _dropdown_options_from_strings([str(u) for u in uniq])

    def _col_opts_and_default(self, col: Optional[str]) -> tuple[list[dcc.Dropdown.Options], list[str]]:
        """Return (options, [first_value]) for a user-group column checklist/dropdown."""
        if not col:
            return [], []
        opts = self._make_ug_opts(col)
        if not opts:
            return [], []
        return opts, [str(opts[0]["value"])]

    def _next_col(self, chosen: Any) -> Optional[str]:
        """Return the first configured user-group column that is not ``chosen``."""
        for c in self.df_user_group_cols:
            if c != chosen:
                return c
        return None

    def _aggregate_stats_from_user_rows_enabled(self) -> bool:
        return bool(self.config.get("stats_by_date", {}).get("aggregate_stats_from_user_rows", False))

    def _stats_by_date_all_chart_columns(self) -> List[str]:
        """Union of group1/2/3 column lists (order preserved, first occurrence wins)."""
        sd = self.config.get("stats_by_date", {})
        seen: List[str] = []
        for k in ("group1_columns", "group2_columns", "group3_columns"):
            for c in sd.get(k) or []:
                if c not in seen:
                    seen.append(c)
        return seen

    def _parquet_required_columns_for_enrich(self) -> set[str]:
        """Columns that must exist in the loaded frame before enrich: registry inputs, plotted Parquet metrics, keys.

        ``user_group_col`` is intentionally excluded: DataMetrics handles its absence gracefully
        via ``_has_user_group`` and falls back to ``[date_col, group_col]`` keys.
        """
        chart_non_enrich = set(self._stats_by_date_all_chart_columns()) - ENRICH_PRODUCED_COLUMNS
        return (
            set(ENRICH_USER_ROW_INPUT_COLUMNS)
            | chart_non_enrich
            | {
                self.date_col,
                self.df_date_group_col,
            }
        )

    def _make_data_metrics(
        self,
        df: pl.DataFrame,
        start_dt: datetime,
        end_dt: datetime,
        granularity: str,
        include_group_col: bool = True,
    ) -> DataMetrics:
        """Return a ``DataMetrics`` instance configured for this dashboard.

        ``df`` must already be filtered to the desired user_group *before* this
        call.  user_group is therefore not included in ``key_cols``; DataMetrics
        computes all metrics for the cohort as supplied.
        """
        key_cols = [self.date_col]
        if include_group_col and self.df_date_group_col:
            key_cols.append(self.df_date_group_col)
        return DataMetrics(
            df,
            start_dt=start_dt,
            end_dt=end_dt,
            key_cols=key_cols,
            granularity=granularity or "day",
        )

    def _ensure_date_plot_groups(self) -> None:
        """Set ``df_date_group_col`` from YAML. Plot lines use distinct values in that column after user-group filters."""
        if self.config.get("stats_by_date", {}).get("group_col"):
            self.df_date_group_col = self.config["stats_by_date"]["group_col"]
        self._ensure_user_group_filter()

    def _ensure_bet_groups(self) -> None:
        """Populate df_bet_group_col and df_bet_groups from config and bet data."""
        self._load_bet_data()
        if self.config.get("stats_by_bet", {}).get("group_col"):
            self.df_bet_group_col = self.config["stats_by_bet"]["group_col"]
            self.df_bet_groups = self._get_plot_groups(self.lf_bet, self.df_bet_group_col)

    def _ensure_tab_data_loaded(self) -> None:
        """Ensure data for all tabs is loaded (for rendering all tab contents)."""
        self._ensure_plot_metrics_loaded()
        self._ensure_date_plot_groups()
        self._ensure_bet_groups()

    def _render_tab_content(self, current_tab: str) -> Any:
        if current_tab == "tab-date":
            self._ensure_tab_data_loaded()
            return self._layout_stats_by_date()
        elif current_tab == "tab-group":
            self._ensure_tab_data_loaded()
            return self._layout_stats_by_group()
        elif current_tab == "tab-viz":
            self._ensure_tab_data_loaded()
            return self._layout_stats_visualization()
        elif current_tab == "tab-weekly":
            return self._layout_weekly_report()
        elif current_tab == "tab-bet":
            self._ensure_tab_data_loaded()
            return self._layout_stats_by_bet()
        else:
            return html.Div("404 Error")

    def _render_all_tab_contents(self, current_tab: str) -> html.Div:
        """Render all three tab contents with visibility toggled - keeps all components in DOM for restore."""
        self._ensure_tab_data_loaded()
        date_visible = "block" if current_tab == "tab-date" else "none"
        group_visible = "block" if current_tab == "tab-group" else "none"
        viz_visible = "block" if current_tab == "tab-viz" else "none"
        weekly_visible = "block" if current_tab == "tab-weekly" else "none"
        bet_visible = "block" if current_tab == "tab-bet" else "none"
        return html.Div(
            [
                html.Div(self._layout_stats_by_date(), id="tab-date-content", style={"display": date_visible}),
                html.Div(self._layout_stats_by_group(), id="tab-group-content", style={"display": group_visible}),
                html.Div(self._layout_stats_visualization(), id="tab-viz-content", style={"display": viz_visible}),
                html.Div(self._layout_weekly_report(), id="tab-weekly-content", style={"display": weekly_visible}),
                html.Div(self._layout_stats_by_bet(), id="tab-bet-content", style={"display": bet_visible}),
            ]
        )

    # ------------------------------------------------------------------
    # Plot Logic
    # ------------------------------------------------------------------

    @staticmethod
    def _user_group_label(user_group_filter: Any) -> str:
        """Return `` (new)`` style suffix when a specific user-group filter is active."""
        val = str(user_group_filter).strip() if user_group_filter is not None else ""
        return f" ({val})" if val and val.lower() not in ("all", "") else ""

    def update_date_plot(
        self,
        left_metrics: Any,
        right_metrics: Any,
        log_val: Any,
        log_thresh: Any,
        date_granularity: Any,
        start_date: Any,
        end_date: Any,
        date_ug_col_1: Any = None,
        date_ug_col_2: Any = None,
        show_ug_1: Any = None,
        show_ug_2: Any = None,
    ) -> go.Figure:
        if not left_metrics and not right_metrics:
            return go.Figure()

        log_scale = "ON" in (log_val or [])
        thresh = float(log_thresh) if log_thresh else 10.0
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        lf = self.lfs_by_date.get(date_granularity, pl.DataFrame().lazy())
        if lf.collect_schema().len() == 0:
            return fig

        # Convert string dates from Dash picker to datetime objects
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)
        if start_dt is None or end_dt is None:
            return fig

        load_end = (
            end_dt + timedelta(days=RETENTION_LOAD_EXTRA_DAYS)
            if self._aggregate_stats_from_user_rows_enabled()
            else end_dt
        )
        df_raw = lf.filter((pl.col(self.date_col) >= start_dt) & (pl.col(self.date_col) <= load_end)).collect()
        if df_raw.is_empty():
            return fig

        def _vals_with_all(sel: Any) -> List[Any]:
            vals_raw = sel if isinstance(sel, list) else ([sel] if sel else [])
            if not vals_raw:
                return ["all"]
            vals = []
            seen = set()
            for v in vals_raw:
                key = str(v)
                if key in seen:
                    continue
                seen.add(key)
                vals.append(v)
            return vals

        effective_col2 = self._effective_ug_col2(date_ug_col_1, date_ug_col_2)
        ug1_vals = _vals_with_all(show_ug_1)
        ug2_vals = _vals_with_all(show_ug_2) if effective_col2 else ["all"]
        clean_ug1 = self._normalize_ug_selection(show_ug_1)
        clean_ug2 = self._normalize_ug_selection(show_ug_2) if effective_col2 else []

        def _cohort_label(v1: Any, v2: Any) -> str:
            s1, s2 = str(v1), str(v2)
            if not effective_col2:
                return "all" if s1 == "all" else s1
            if s1 == "all" and s2 == "all":
                return "all"
            if s1 == "all":
                return s2
            if s2 == "all":
                return s1
            return f"{s1}|{s2}"

        # Build one combined (all strategy groups) DataMetrics per selected UG cohort.
        cohort_entries: List[Tuple[str, DataMetrics, pl.DataFrame]] = []
        seen_labels: set[str] = set()
        for v1 in ug1_vals:
            for v2 in ug2_vals:
                df_c = self._apply_ug_filter_list(
                    df_raw,
                    [] if str(v1) == "all" else [v1],
                    col=date_ug_col_1 or None,
                )
                if effective_col2:
                    df_c = self._apply_ug_filter_list(
                        df_c,
                        [] if str(v2) == "all" else [v2],
                        col=effective_col2,
                    )
                if df_c.is_empty():
                    continue
                df_c_date = df_c.filter((pl.col(self.date_col) >= start_dt) & (pl.col(self.date_col) <= end_dt))
                if df_c_date.is_empty():
                    continue
                label = _cohort_label(v1, v2)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                dm_c = self._make_data_metrics(
                    df_c,
                    start_dt,
                    end_dt,
                    str(date_granularity or "day"),
                    include_group_col=False,
                )
                cohort_entries.append((label, dm_c, df_c_date))

        if not cohort_entries:
            return fig

        color_map = {
            f"{label}:{m}": Styles.COLORS[j % len(Styles.COLORS)]
            for i, m in enumerate(left_metrics + right_metrics)
            for j, (label, _, _) in enumerate(cohort_entries)
        }
        line_style_map = {
            f"{label}:{m}": Styles.LINE_SHAPE[i % len(Styles.LINE_SHAPE)]
            for i, m in enumerate(left_metrics + right_metrics)
            for j, (label, _, _) in enumerate(cohort_entries)
        }

        # Accumulate y values per axis for log-scale tick calculation.
        all_y_left: List[float] = []
        all_y_right: List[float] = []

        for strat, dm, df_group in cohort_entries:

            def add_scatter_plot(metrics: Any, secondary_y: bool) -> None:
                for m in metrics:
                    x, y_raw, y_lower, y_upper = dm.plot_values(m, df_group)
                    if len(x) == 0:
                        continue

                    y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw
                    y_lower_plot = self.hybrid_transform(y_lower, thresh) if log_scale else y_lower
                    y_upper_plot = self.hybrid_transform(y_upper, thresh) if log_scale else y_upper

                    if secondary_y:
                        all_y_right.extend(y_raw[~np.isnan(y_raw)].tolist())
                    else:
                        all_y_left.extend(y_raw[~np.isnan(y_raw)].tolist())

                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y_plot.tolist() if hasattr(y_plot, "tolist") else list(y_plot),
                            name=f"{strat}:{m}",
                            mode="lines+markers",
                            line=dict(
                                color=color_map.get(f"{strat}:{m}", "black"),
                                dash=line_style_map.get(f"{strat}:{m}", "solid"),
                            ),
                            legendgroup=strat,
                        ),
                        secondary_y=secondary_y,
                    )

                    lower_arr = np.asarray(y_lower_plot)
                    upper_arr = np.asarray(y_upper_plot)
                    if not np.allclose(lower_arr, upper_arr, equal_nan=True):
                        base_color = color_map.get(f"{strat}:{m}", "black")
                        x_list = x if isinstance(x, list) else list(x)
                        fig.add_trace(
                            go.Scatter(
                                x=x_list + x_list[::-1],
                                y=(upper_arr.tolist() if hasattr(upper_arr, "tolist") else list(upper_arr))
                                + (lower_arr.tolist() if hasattr(lower_arr, "tolist") else list(lower_arr))[::-1],
                                fill="toself",
                                fillcolor=self.to_rgba(base_color, 0.1),
                                line=dict(color="rgba(255,255,255,0)"),
                                hoverinfo="skip",
                                showlegend=False,
                                legendgroup=strat,
                                name=f"{strat}:{m} 95% CI",
                            ),
                            secondary_y=secondary_y,
                        )

            add_scatter_plot(left_metrics or [], False)
            add_scatter_plot(right_metrics or [], True)

        metrics_label = ", ".join(m for m in [*(left_metrics or []), *(right_metrics or [])] if m)

        # Build a compact filter label from the selected checklist values
        def _label_from_vals(vals: Any) -> str:
            clean = [v for v in (vals or []) if str(v) != "all"]
            return f"({', '.join(str(v) for v in clean)})" if clean else ""

        ug1_lbl = _label_from_vals(clean_ug1)
        ug2_lbl = _label_from_vals(clean_ug2) if effective_col2 else ""
        ug_label = " " + " ".join(p for p in [ug1_lbl, ug2_lbl] if p) if (ug1_lbl or ug2_lbl) else ""
        # ug_label = (ug_label + " [all groups combined]").strip()
        fig.update_layout(
            title=f"{metrics_label}{ug_label}" if metrics_label else None,
            height=400,
            margin=dict(l=50, r=10, t=40, b=15),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.90),
            hovermode="x unified",
            xaxis_title="Date (Beijing)",
            font=dict(size=16),
            xaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis2=dict(title_font=dict(size=18), tickfont=dict(size=15)),
        )

        if log_scale:
            if all_y_left:
                yticks = self.get_ticks(all_y_left, thresh)
                fig.update_yaxes(
                    tickvals=self.hybrid_transform(yticks, thresh).tolist(),
                    ticktext=[f"{v:.0f}" for v in yticks],
                    secondary_y=False,
                )
            if all_y_right:
                yticks_r = self.get_ticks(all_y_right, thresh)
                fig.update_yaxes(
                    tickvals=self.hybrid_transform(yticks_r, thresh).tolist(),
                    ticktext=[f"{v:.0f}" for v in yticks_r],
                    secondary_y=True,
                )
        return fig

    def update_bet_plot(
        self,
        session,
        groups,
        left_metrics,
        right_metrics,
        share_left,
        share_right,
        log_val,
        log_thresh,
        filter_check,
        filter_thresh,
    ):
        self._load_bet_data()
        if self.lf_bet.collect_schema().len() == 0 or not self.sessions:
            return go.Figure()
        share_left_y = "ON" in (share_left or [])
        share_right_y = "ON" in (share_right or [])
        log_scale = "ON" in (log_val or [])
        do_filter = "ON" in (filter_check or [])
        log_thresh = max(float(log_thresh), 1.0) if log_thresh else 10.0
        lf_sess = self.lf_bet.filter(pl.col("session_start_date").cast(pl.Utf8) == str(session)).sort("bet_index")
        if do_filter and filter_thresh:
            lf_sess = lf_sess.filter(pl.col("bet_index") <= float(filter_thresh))
        df_sess = lf_sess.collect()
        if df_sess.is_empty():
            return go.Figure()
        group_col = self.config["stats_by_bet"].get("group_col", "session_group")
        df_groups = [df_sess.filter(pl.col(group_col) == s) if s else pl.DataFrame() for s in groups]
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Metrics by Strategies: {session}"],
            specs=[[{"secondary_y": True}]],
            vertical_spacing=0.05,
            x_title="Bet Index",
        )
        left_range = self.compute_axis_range(df_groups, left_metrics, log_scale, log_thresh) if share_left_y else None
        right_range = (
            self.compute_axis_range(df_groups, right_metrics, log_scale, log_thresh) if share_right_y else None
        )
        if not left_metrics and right_metrics:
            fig.add_trace(
                go.Scatter(
                    x=np.array([0]),
                    y=np.array([0]),
                    name="",
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=None,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        for i, df_g in enumerate(df_groups):
            if df_g.is_empty():
                continue
            metric_colors = {
                f"{s}:{m}": Styles.COLORS[i + j * len(groups) % len(Styles.COLORS)]
                for i, s in enumerate(groups)
                for j, m in enumerate(left_metrics)
            }
            self._add_traces_to_fig(
                fig,
                df_g,
                groups[i],
                left_metrics,
                metric_colors,
                log_scale,
                log_thresh,
                is_right=False,
                y_range=left_range,
            )
            metric_colors = {
                f"{s}:{m}": Styles.COLORS[i + j * len(groups) % len(Styles.COLORS)]
                for i, s in enumerate(groups)
                for j, m in enumerate(right_metrics)
            }
            self._add_traces_to_fig(
                fig,
                df_g,
                groups[i],
                right_metrics,
                metric_colors,
                log_scale,
                log_thresh,
                is_right=True,
                y_range=right_range,
            )
        fig.update_layout(
            legend=dict(
                orientation="h",
                traceorder="normal",
                yanchor="top",
                y=1.10,
                xanchor="right",
                x=0.95,
                borderwidth=0,
                font=dict(size=20),
            ),
            font=dict(size=16),
            xaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis2=dict(title_font=dict(size=18), tickfont=dict(size=15)),
        )
        if log_scale:
            self.update_log_ticks(fig, df_groups, left_metrics, right_metrics, log_thresh)
        fig.update_layout(height=950, template="plotly_white", margin=dict(l=60, r=60, t=50, b=50))
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=1, col=1)
        return fig

    def _add_traces_to_fig(self, fig, df: pl.DataFrame, strat, metrics, colors, log_scale, thresh, is_right, y_range):
        if not metrics:
            return
        y_max_local = -inf
        for m in metrics:
            if m not in df.columns:
                continue
            y = df.get_column(m).cast(pl.Float64).fill_null(float("nan")).forward_fill().to_numpy()
            y_plot = self.hybrid_transform(y, thresh) if log_scale else y
            curr_max = float(np.nanmax(y_plot)) if len(y_plot) > 0 else 0.0
            y_max_local = max(y_max_local, curr_max)
            fig.add_trace(
                go.Scatter(
                    x=df.get_column("bet_index").to_list(),
                    y=y_plot.tolist(),
                    name=f"{strat}:{m}{' (R)' if is_right else ''}",
                    mode="lines",
                    line=dict(width=2.5, color=colors.get(f"{strat}:{m}", "#333"), dash="dash" if is_right else None),
                    legendgroup=m,
                    showlegend=True,
                ),
                row=1,
                col=1,
                secondary_y=is_right,
            )
        y_range = y_range if y_range else (0, y_max_local * 1.05)
        fig.update_yaxes(range=y_range, row=1, col=1, secondary_y=is_right)

    def update_group_plot_combined_factory(self, group_id: str) -> Callable[..., Any]:
        """
        Callback factory: Plots date-group stats for a metric, for three date ranges, in a single figure.
        Now supports optional data clipping, range toggle, and no redundant legend.
        """

        def callback(
            metric: Any,
            display_mode: Any,
            r1_start: Any,
            r1_end: Any,
            r2_start: Any,
            r2_end: Any,
            r3_start: Any,
            r3_end: Any,
            show_r1: Any,
            show_r2: Any,
            show_r3: Any,
            clip_enable: Any,
            clip_min: Any,
            clip_max: Any,
            ug_col_1: Any,
            ug_col_2: Any,
            show_ug_1: Any,
            show_ug_2: Any,
        ) -> go.Figure:
            # Normalize metric: multi=False dropdown should give a string, but guard
            # against legacy saved configs that stored it as a single-item list.
            if isinstance(metric, list):
                metric = metric[0] if metric else None
            if not metric or r1_start is None or r1_end is None:
                return go.Figure()
            granularity = list(self.date_files_config.keys())[0]
            lf = self.lfs_by_date.get(granularity, pl.DataFrame().lazy())
            if lf.collect_schema().len() == 0:
                return go.Figure()
            dates_used = [
                self._parse_date(r1_start),
                self._parse_date(r1_end),
                self._parse_date(r2_start),
                self._parse_date(r2_end),
                self._parse_date(r3_start),
                self._parse_date(r3_end),
            ]
            valid_dates = [d for d in dates_used if d is not None]
            if not valid_dates:
                return go.Figure()
            min_d, max_d = min(valid_dates), max(valid_dates)
            load_end = (
                max_d + timedelta(days=RETENTION_LOAD_EXTRA_DAYS)
                if self._aggregate_stats_from_user_rows_enabled()
                else max_d
            )
            file_gran = list(self.date_files_config.keys())[0]

            df_loaded = lf.filter((pl.col(self.date_col) >= min_d) & (pl.col(self.date_col) <= load_end)).collect()
            if df_loaded.is_empty() or self.df_date_group_col not in df_loaded.columns:
                return go.Figure()

            selected_g1 = self._normalize_ug_selection(show_ug_1)
            effective_col2 = self._effective_ug_col2(ug_col_1, ug_col_2)
            selected_g2 = self._normalize_ug_selection(show_ug_2) if effective_col2 else []
            g1_vals = sorted(selected_g1, key=str) if selected_g1 else ["all"]
            g2_vals = (sorted(selected_g2, key=str) if selected_g2 else ["all"]) if effective_col2 else ["all"]

            # Build one DataMetrics per (g1, g2) combination, filtering data before metrics.
            # combo_entries: list of (g1_val, g2_val, metric_df)
            combo_entries: List[Tuple[str, str, pl.DataFrame]] = []
            for g1 in g1_vals:
                for g2 in g2_vals:
                    g1_sel = [g1] if g1 != "all" else []
                    g2_sel = [g2] if (effective_col2 and g2 != "all") else []
                    df_c, _, _, _ = self._filter_df_for_ug_selection(
                        df_loaded,
                        ug_col_1,
                        effective_col2,
                        g1_sel,
                        g2_sel,
                    )
                    if df_c.is_empty():
                        continue
                    dm_c = self._make_data_metrics(df_c, min_d, max_d, file_gran)
                    if metric not in dm_c:
                        logger.warning(
                            "plot metric %r not found for (%r=%r, %r=%r)",
                            metric,
                            ug_col_1,
                            g1,
                            ug_col_2,
                            g2,
                        )
                        continue
                    mdf = dm_c[str(metric)]
                    if mdf is None:
                        continue
                    combo_entries.append((g1, g2, mdf))

            if not combo_entries:
                return go.Figure()

            base_colors = Styles.COLORS
            fig = go.Figure()

            def _format_date_range(start_dt: datetime, end_dt: datetime) -> str:
                return f"{start_dt.strftime('%m/%d/%Y')}-{end_dt.strftime('%m/%d/%Y')}"

            def _get_vals(metric_df: pl.DataFrame, start_dt: datetime, end_dt: datetime) -> np.ndarray:
                return (
                    metric_df.filter((pl.col(self.date_col) >= start_dt) & (pl.col(self.date_col) <= end_dt))
                    .get_column(str(metric))
                    .drop_nulls()
                    .to_numpy()
                    .astype(float)
                )

            data_ranges = [
                {"start": self._parse_date(r1_start), "end": self._parse_date(r1_end), "show": "ON" in (show_r1 or [])},
                {"start": self._parse_date(r2_start), "end": self._parse_date(r2_end), "show": "ON" in (show_r2 or [])},
                {"start": self._parse_date(r3_start), "end": self._parse_date(r3_end), "show": "ON" in (show_r3 or [])},
            ]
            data_ranges = [r for r in data_ranges if r["start"] is not None and r["end"] is not None]
            data_by_range = [r for r in data_ranges if r["show"]]
            enable_clip = clip_enable is not None and "ON" in (clip_enable or [])
            cmin = clip_min if enable_clip and clip_min is not None else None
            cmax = clip_max if enable_clip and clip_max is not None else None

            show_g1_in_label = len(g1_vals) > 1 or (g1_vals and g1_vals[0] != "all")
            show_g2_in_label = effective_col2 and (len(g2_vals) > 1 or (g2_vals and g2_vals[0] != "all"))

            def _build_label(g1: str, g2: str, date_range_str: str) -> str:
                """Label: g1 first (primary grouping), g2 second, date range last."""
                parts = []
                if show_g1_in_label:
                    parts.append(g1)
                if show_g2_in_label:
                    parts.append(f"({g2})")
                parts.append(date_range_str)
                return "<br>".join(parts)

            def _iter_plot_data():
                """Yield (label, color_base, range_idx, vals) for every non-empty (combo, range)."""
                for ci, (g1, g2, metric_df) in enumerate(combo_entries):
                    color = base_colors[ci % len(base_colors)]
                    for ri, range_info in enumerate(data_by_range):
                        start_dt = range_info["start"]
                        end_dt = range_info["end"]
                        if start_dt is None or end_dt is None:
                            continue
                        vals = _get_vals(metric_df, cast(datetime, start_dt), cast(datetime, end_dt))
                        if enable_clip and (cmin is not None or cmax is not None):
                            vals = np.clip(
                                vals,
                                cmin if cmin is not None else -np.inf,
                                cmax if cmax is not None else np.inf,
                            )
                        if len(vals) == 0:
                            continue
                        label = _build_label(
                            g1,
                            g2,
                            _format_date_range(cast(datetime, start_dt), cast(datetime, end_dt)),
                        )
                        yield label, color, ri, vals

            if display_mode == "box":
                for label, color_base, ri, vals in _iter_plot_data():
                    fig.add_trace(
                        go.Box(
                            y=vals,
                            name=label,
                            boxmean="sd",
                            marker=dict(
                                color=color_base,
                                opacity=1.0 if ri == 0 else 0.45,
                                line=dict(color="#333", width=1),
                            ),
                            showlegend=False,
                        )
                    )
            else:  # bar, mean + 500 bootstrap CI + text stats
                N_BOOTSTRAP = 500
                x_labels = []
                ydata = []
                err_upper = []
                err_lower = []
                text_stats = []
                mcolors_box = []
                for label, color_base, ri, vals in _iter_plot_data():
                    x_labels.append(label)
                    mean_val = float(np.mean(vals))
                    ydata.append(mean_val)
                    lower_ci, upper_ci = bootstrap_worker(vals, n_boot=N_BOOTSTRAP)
                    err_upper.append(upper_ci - mean_val if np.isfinite(upper_ci) else 0.0)
                    err_lower.append(mean_val - lower_ci if np.isfinite(lower_ci) else 0.0)
                    med_val = float(np.median(vals))
                    min_val = float(np.min(vals))
                    max_val = float(np.max(vals))
                    n_samples = len(vals)
                    text_stats.append(
                        f"n={n_samples}<br>μ={mean_val:.2f}<br>med={med_val:.2f}"
                        f"<br>max={max_val:.2f}<br>min={min_val:.2f}"
                    )
                    mcolors_box.append(
                        dict(color=color_base, opacity=1.0, line=dict(color="#333", width=1))
                        if ri == 0
                        else dict(
                            color=color_base, opacity=0.5, line=dict(color="#333", width=1), pattern=dict(shape="/")
                        )
                    )
                if x_labels:
                    fig.add_trace(
                        go.Bar(
                            x=x_labels,
                            y=ydata,
                            error_y=dict(
                                type="data",
                                array=err_upper,
                                arrayminus=err_lower,
                                symmetric=False,
                            ),
                            text=text_stats,
                            textposition="none",
                            marker={"color": [c["color"] for c in mcolors_box], "opacity": None},
                            customdata=[c for c in mcolors_box],
                            hovertemplate="%{x}<br>mean=%{y:.2f}<br>95%% CI (500 boot)<br>%{text}",
                            showlegend=False,
                        )
                    )
                    annotations = []
                    for i in range(len(x_labels)):
                        annotations.append(
                            dict(
                                x=i - 0.25,
                                y=ydata[i],
                                xref="x",
                                yref="y",
                                text=text_stats[i],
                                xanchor="right",
                                yanchor="bottom",
                                align="left",
                                showarrow=False,
                                font=dict(size=14),
                            )
                        )
                    fig.update_layout(annotations=annotations, margin=dict(l=220))
                    fig_data = cast(Any, fig.data)
                    for i in range(len(fig_data)):
                        trace = fig_data[i]
                        if getattr(trace, "customdata", None) is None:
                            continue
                        mcolors_box = trace.customdata
                        marker = cast(Any, trace.marker)
                        marker.color = [m["color"] for m in mcolors_box]
                        if any("pattern" in m for m in mcolors_box):
                            if not hasattr(marker, "pattern"):
                                marker.pattern = dict(shape=[""] * len(trace.x))
                            cast(Any, marker.pattern).shape = [
                                m.get("pattern", {}).get("shape", "") for m in mcolors_box
                            ]
                        marker.opacity = [m.get("opacity", 1.0) for m in mcolors_box]

            mode_title = "Box Plot" if display_mode == "box" else "Bar (Mean ± 95% CI, 500 bootstrap)"
            subtitle = ""
            if enable_clip:
                subtitle = " (Clipped"
                if cmin is not None:
                    subtitle += f" min={cmin}"
                if cmax is not None:
                    subtitle += f" max={cmax}"
                subtitle += ")"
            fig.update_layout(
                title=f"{mode_title}: {metric}{subtitle}",
                yaxis_title=metric,
                template="plotly_white",
                hovermode="closest",
                font=dict(size=16),
            )
            return fig

        return callback

    # ------------------------------------------------------------------
    # Stats Deepdive (histogram / heatmap / scatter)
    # ------------------------------------------------------------------

    def _viz_collect_panel_data(
        self,
        panel_id: str,
        metrics: List[str],
        r1_start: Any,
        r1_end: Any,
        r2_start: Any,
        r2_end: Any,
        r3_start: Any,
        r3_end: Any,
        show_r1: Any,
        show_r2: Any,
        show_r3: Any,
        clip_enable: Any,
        clip_min: Any,
        clip_max: Any,
        ug_col_1: Any,
        ug_col_2: Any,
        show_ug_1: Any,
        show_ug_2: Any,
    ) -> List[Tuple[str, pl.DataFrame]]:
        """Collect one DataFrame per (range × g1 × g2) combo with the selected metric columns.

        For ``panel_id == "p1"`` (derived metrics) rows are at (date, date_group_col) granularity,
        joined from ``DataMetrics[metric]`` for each selected metric.  For ``panel_id == "p2"``
        (user-level metrics) rows are kept at the per-user granularity.
        """
        if not metrics:
            return []
        granularity = list(self.date_files_config.keys())[0]
        lf = self.lfs_by_date.get(granularity, pl.DataFrame().lazy())
        if lf.collect_schema().len() == 0:
            return []

        data_ranges = [
            {
                "label": "R1",
                "start": self._parse_date(r1_start),
                "end": self._parse_date(r1_end),
                "show": "ON" in (show_r1 or []),
            },
            {
                "label": "R2",
                "start": self._parse_date(r2_start),
                "end": self._parse_date(r2_end),
                "show": "ON" in (show_r2 or []),
            },
            {
                "label": "R3",
                "start": self._parse_date(r3_start),
                "end": self._parse_date(r3_end),
                "show": "ON" in (show_r3 or []),
            },
        ]
        active_ranges = [r for r in data_ranges if r["show"] and r["start"] is not None and r["end"] is not None]
        if not active_ranges:
            return []

        min_d = min(cast(datetime, r["start"]) for r in active_ranges)
        max_d = max(cast(datetime, r["end"]) for r in active_ranges)
        load_end = (
            max_d + timedelta(days=RETENTION_LOAD_EXTRA_DAYS)
            if self._aggregate_stats_from_user_rows_enabled()
            else max_d
        )
        df_loaded = lf.filter((pl.col(self.date_col) >= min_d) & (pl.col(self.date_col) <= load_end)).collect()
        if df_loaded.is_empty():
            return []

        selected_g1 = self._normalize_ug_selection(show_ug_1)
        effective_col2 = self._effective_ug_col2(ug_col_1, ug_col_2)
        selected_g2 = self._normalize_ug_selection(show_ug_2) if effective_col2 else []
        g1_vals: List[Any] = sorted(selected_g1, key=str) if selected_g1 else ["all"]
        g2_vals: List[Any] = (sorted(selected_g2, key=str) if selected_g2 else ["all"]) if effective_col2 else ["all"]

        enable_clip = clip_enable is not None and "ON" in (clip_enable or [])
        cmin = float(clip_min) if enable_clip and clip_min is not None else None
        cmax = float(clip_max) if enable_clip and clip_max is not None else None

        out: List[Tuple[str, pl.DataFrame]] = []
        for g1 in g1_vals:
            for g2 in g2_vals:
                g1_sel: List[Any] = [g1] if str(g1) != "all" else []
                g2_sel: List[Any] = [g2] if (effective_col2 and str(g2) != "all") else []
                df_ug, _, _, _ = self._filter_df_for_ug_selection(
                    df_loaded,
                    ug_col_1,
                    effective_col2,
                    g1_sel,
                    g2_sel,
                )
                if df_ug.is_empty():
                    continue

                if panel_id == "p1":
                    dm = self._make_data_metrics(df_ug, min_d, max_d, granularity)
                    combined: Optional[pl.DataFrame] = None
                    combined_keys: List[str] = []
                    for m in metrics:
                        if m not in dm:
                            continue
                        mdf = dm[str(m)]
                        if mdf is None or mdf.is_empty():
                            continue
                        if combined is None:
                            combined = mdf
                            combined_keys = [c for c in mdf.columns if c != m]
                        else:
                            join_on = [c for c in combined_keys if c in mdf.columns]
                            if not join_on:
                                continue
                            combined = combined.join(mdf, on=join_on, how="inner")
                    if combined is None or combined.is_empty():
                        continue
                    source_df = combined
                    date_col_name = combined_keys[0] if combined_keys else self.date_col
                else:
                    keep_cols = [c for c in metrics if c in df_ug.columns]
                    if not keep_cols:
                        continue
                    source_df = df_ug.select([self.date_col] + keep_cols)
                    date_col_name = self.date_col

                for r in active_ranges:
                    r_start = cast(datetime, r["start"])
                    r_end = cast(datetime, r["end"])
                    rdf = source_df.filter((pl.col(date_col_name) >= r_start) & (pl.col(date_col_name) <= r_end))
                    if rdf.is_empty():
                        continue
                    rdf = rdf.drop(date_col_name) if date_col_name in rdf.columns else rdf
                    keep_metric_cols = [c for c in metrics if c in rdf.columns]
                    if not keep_metric_cols:
                        continue
                    rdf = rdf.select(keep_metric_cols)
                    if enable_clip and (cmin is not None or cmax is not None):
                        lo = cmin if cmin is not None else -float("inf")
                        hi = cmax if cmax is not None else float("inf")
                        rdf = rdf.with_columns(
                            [
                                pl.col(c).cast(pl.Float64).clip(lower_bound=lo, upper_bound=hi).alias(c)
                                for c in keep_metric_cols
                            ]
                        )
                    parts: List[str] = []
                    if str(g1) != "all":
                        parts.append(str(g1))
                    if effective_col2 and str(g2) != "all":
                        parts.append(str(g2))
                    parts.append(f"{cast(str, r['label'])}: {r_start.strftime('%m/%d')}-{r_end.strftime('%m/%d')}")
                    label = " | ".join(parts) if parts else cast(str, r["label"])
                    out.append((label, rdf))

        return out

    def update_viz_plot_factory(self, panel_id: str) -> Callable[..., Any]:
        """Callback factory for the Stats Deepdive tab.

        Dispatches to histogram / heatmap / pairplot based on ``display_mode`` and the number
        of selected metrics.
        """

        def callback(
            metrics: Any,
            display_mode: Any,
            r1_start: Any,
            r1_end: Any,
            r2_start: Any,
            r2_end: Any,
            r3_start: Any,
            r3_end: Any,
            show_r1: Any,
            show_r2: Any,
            show_r3: Any,
            clip_enable: Any,
            clip_min: Any,
            clip_max: Any,
            ug_col_1: Any,
            ug_col_2: Any,
            show_ug_1: Any,
            show_ug_2: Any,
            nbins: Any = 50,
            log_y: Any = None,
            normalize: Any = None,
            scatter_log_x: Any = None,
            scatter_log_y: Any = None,
            scatter_symlog_x: Any = None,
            scatter_symlog_y: Any = None,
            outliers_enable: Any = None,
            outliers_std: Any = "3",
            hm_metrics: Optional[List[Any]] = None,
            hm_logs: Optional[List[Any]] = None,
            sm_metrics: Optional[List[Any]] = None,
            sm_logs: Optional[List[Any]] = None,
        ) -> go.Figure:
            mode = str(display_mode or "histogram")

            # Heatmap / scatter both use per-row metric selectors with an optional Yeo-Johnson
            # log flag.  Histogram uses the shared multi-select "Metrics".
            def _collect_rows(
                raw_metrics: Optional[List[Any]], raw_logs: Optional[List[Any]]
            ) -> List[Tuple[str, bool]]:
                pairs: List[Tuple[str, bool]] = []
                metrics_raw = raw_metrics or []
                logs_raw = raw_logs or []
                seen: set[str] = set()
                for i in range(self._VIZ_MS_MAX_ROWS):
                    m = metrics_raw[i] if i < len(metrics_raw) else None
                    if not m:
                        continue
                    m_str = str(m)
                    if m_str in seen:
                        continue
                    seen.add(m_str)
                    log_flag = "ON" in (logs_raw[i] if i < len(logs_raw) and logs_raw[i] else [])
                    pairs.append((m_str, log_flag))
                return pairs

            mode_pairs: List[Tuple[str, bool]] = []
            if mode == "heatmap":
                mode_pairs = _collect_rows(hm_metrics, hm_logs)
                metrics_list: List[str] = [m for m, _ in mode_pairs]
            elif mode == "scatter":
                mode_pairs = _collect_rows(sm_metrics, sm_logs)
                metrics_list = [m for m, _ in mode_pairs]
            else:
                metrics_list = [str(m) for m in (metrics or [])]
            if not metrics_list:
                return go.Figure()
            if mode == "heatmap" and len(metrics_list) < 2:
                fig = go.Figure()
                fig.add_annotation(
                    text="Select at least two metrics to show a correlation heatmap.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    font=dict(size=14),
                )
                return fig

            combos = self._viz_collect_panel_data(
                panel_id,
                metrics_list,
                r1_start,
                r1_end,
                r2_start,
                r2_end,
                r3_start,
                r3_end,
                show_r1,
                show_r2,
                show_r3,
                clip_enable,
                clip_min,
                clip_max,
                ug_col_1,
                ug_col_2,
                show_ug_1,
                show_ug_2,
            )
            if not combos:
                return go.Figure()

            try:
                nbins_int = int(nbins) if nbins is not None else 50
            except (TypeError, ValueError):
                nbins_int = 50
            log_y_on = "ON" in (log_y or [])
            normalize_on = "ON" in (normalize or [])
            scatter_log_x_on = "ON" in (scatter_log_x or [])
            scatter_log_y_on = "ON" in (scatter_log_y or [])
            scatter_symlog_x_on = "ON" in (scatter_symlog_x or [])
            scatter_symlog_y_on = "ON" in (scatter_symlog_y or [])
            # Symlog wins if both are on (it's strictly more general and handles negatives).
            if scatter_symlog_x_on:
                scatter_log_x_on = False
            if scatter_symlog_y_on:
                scatter_log_y_on = False
            outliers_on = "ON" in (outliers_enable or [])
            try:
                outliers_threshold = float(outliers_std) if outliers_std is not None else 3.0
            except (TypeError, ValueError):
                outliers_threshold = 3.0

            active_outliers_threshold: Optional[float] = outliers_threshold if outliers_on else None

            if mode == "histogram":
                return self._viz_build_histogram(
                    metrics_list, combos, nbins=nbins_int, log_y=log_y_on, normalize=normalize_on
                )
            if mode == "heatmap":
                log_flags = {m: flag for m, flag in mode_pairs}
                return self._viz_build_heatmap(metrics_list, combos, log_flags=log_flags)
            if mode == "scatter":
                scatter_log_flags = {m: flag for m, flag in mode_pairs}
                if len(metrics_list) == 1:
                    return self._viz_build_histogram(
                        metrics_list, combos, nbins=nbins_int, log_y=log_y_on, normalize=normalize_on
                    )
                if len(metrics_list) == 2:
                    return self._viz_build_scatter_pair(
                        metrics_list,
                        combos,
                        log_x=scatter_log_x_on,
                        log_y=scatter_log_y_on,
                        symlog_x=scatter_symlog_x_on,
                        symlog_y=scatter_symlog_y_on,
                        outliers_threshold=active_outliers_threshold,
                        log_flags=scatter_log_flags,
                    )
                return self._viz_build_scatter_grid(
                    metrics_list,
                    combos,
                    log_x=scatter_log_x_on,
                    log_y=scatter_log_y_on,
                    symlog_x=scatter_symlog_x_on,
                    symlog_y=scatter_symlog_y_on,
                    outliers_threshold=active_outliers_threshold,
                    log_flags=scatter_log_flags,
                )
            return go.Figure()

        return callback

    @staticmethod
    def _viz_values(df: pl.DataFrame, metric: str) -> np.ndarray:
        if metric not in df.columns:
            return np.array([])
        return df.get_column(metric).cast(pl.Float64).drop_nulls().to_numpy()

    @staticmethod
    def _viz_symlog(arr: np.ndarray) -> np.ndarray:
        """Symmetric log transform: ``sign(x) * log10(|x| + 1)``.

        Preserves sign, passes through 0 unchanged, and stays roughly linear near 0 while
        compressing large magnitudes — so negative and positive values can share the axis.
        """
        a = np.asarray(arr, dtype=float)
        return np.sign(a) * np.log10(np.abs(a) + 1.0)

    @staticmethod
    def _viz_symlog_ticks(vmin: float, vmax: float) -> Tuple[List[float], List[str]]:
        """Return (tickvals, ticktext) for a symlog-transformed axis spanning [vmin, vmax]
        in original (untransformed) units.  Places ticks at 0 and ±10^k out to the data extent.
        """
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return [0.0], ["0"]
        lo = min(vmin, 0.0)
        hi = max(vmax, 0.0)
        max_mag = max(abs(lo), abs(hi), 1.0)
        max_exp = int(np.ceil(np.log10(max_mag)))
        originals: List[float] = [0.0]
        for exp in range(0, max_exp + 1):
            v = 10.0**exp
            if v <= hi:
                originals.append(v)
            if -v >= lo:
                originals.append(-v)
        originals = sorted(set(originals))
        tickvals = [float(np.sign(t) * np.log10(abs(t) + 1.0)) for t in originals]
        ticktext = [f"{t:g}" for t in originals]
        return tickvals, ticktext

    @staticmethod
    def _viz_drop_outliers(pdf: pd.DataFrame, threshold: Optional[float]) -> pd.DataFrame:
        """Drop rows where any column's absolute z-score exceeds ``threshold``.

        Operates per-column: a row is kept only when every selected metric stays within
        ``threshold`` standard deviations of that metric's mean.  A column with zero / NaN
        std is ignored (no rows removed on its account).  ``threshold is None`` returns ``pdf``
        unchanged.
        """
        if threshold is None or pdf.empty:
            return pdf
        mask = pd.Series(True, index=pdf.index)
        for col in pdf.columns:
            s = pdf[col]
            std = s.std()
            if std is None or not np.isfinite(std) or std == 0:
                continue
            mask &= (s - s.mean()).abs() / std <= threshold
        return pdf.loc[mask]

    def _viz_build_histogram(
        self,
        metrics: List[str],
        combos: List[Tuple[str, pl.DataFrame]],
        nbins: int = 50,
        log_y: bool = False,
        normalize: bool = False,
    ) -> go.Figure:
        n = len(metrics)
        cols = min(2, n)
        rows = int(np.ceil(n / cols))
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics)
        base_colors = Styles.COLORS
        shown_labels: set[str] = set()
        # When normalize is on, each trace's bars show the relative frequency (fraction of
        # the combo's samples per bin).  Using "probability" here normalises per-trace, so
        # overlayed histograms with different sample sizes remain visually comparable.
        histnorm = "probability" if normalize else ""
        for i, metric in enumerate(metrics):
            r = i // cols + 1
            c = i % cols + 1
            for ci, (label, df) in enumerate(combos):
                vals = self._viz_values(df, metric)
                if len(vals) == 0:
                    continue
                color = base_colors[ci % len(base_colors)]
                fig.add_trace(
                    go.Histogram(
                        x=vals,
                        name=label,
                        marker=dict(color=color),
                        opacity=0.55,
                        legendgroup=label,
                        showlegend=label not in shown_labels,
                        nbinsx=int(nbins),
                        histnorm=histnorm,
                    ),
                    row=r,
                    col=c,
                )
                shown_labels.add(label)
        fig.update_layout(
            barmode="overlay",
            template="plotly_white",
            autosize=True,
            margin=dict(l=70, r=20, t=70, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=16)),
            font=dict(size=18),
        )
        y_title = "Frequency" if normalize else "Count"
        fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=16))
        fig.update_yaxes(title_text=y_title, title_font=dict(size=18), tickfont=dict(size=16))
        for ann in fig.layout.annotations or []:
            ann.font = dict(size=20)
        if log_y:
            fig.update_yaxes(type="log")
        return fig

    def _viz_build_heatmap(
        self,
        metrics: List[str],
        combos: List[Tuple[str, pl.DataFrame]],
        log_flags: Optional[Dict[str, bool]] = None,
    ) -> go.Figure:
        log_flags = log_flags or {}
        n_combos = len(combos)
        cols = min(2, n_combos) if n_combos > 0 else 1
        rows = int(np.ceil(n_combos / cols)) if n_combos > 0 else 1
        titles = [label for label, _ in combos] or [""]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
        # Lazy import sklearn so the rest of the dashboard keeps working if it's not installed.
        power_transformer = None
        if any(log_flags.values()):
            try:
                from sklearn.preprocessing import PowerTransformer  # type: ignore[import-not-found]

                power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            except ImportError:
                logger.warning("Heatmap log-transform requested but sklearn is not installed; skipping Yeo-Johnson.")
                power_transformer = None
        axis_labels = [f"{m}*" if log_flags.get(m) and power_transformer is not None else m for m in metrics]
        for idx, (_label, df) in enumerate(combos):
            r = idx // cols + 1
            c = idx % cols + 1
            available = [m for m in metrics if m in df.columns]
            if len(available) < 2:
                continue
            pdf = cast(pd.DataFrame, df.select(available).to_pandas().apply(pd.to_numeric, errors="coerce"))
            # Apply Yeo-Johnson to each flagged column independently (per-combo fit).
            if power_transformer is not None:
                for m in available:
                    if not log_flags.get(m):
                        continue
                    col = np.asarray(pdf[m].to_numpy(), dtype=float).reshape(-1, 1)
                    try:
                        transformed = np.asarray(power_transformer.fit_transform(col)).ravel()
                        pdf[m] = transformed
                    except Exception as exc:
                        logger.warning("Yeo-Johnson failed for %r: %s", m, exc)
            corr_values, text_matrix = self._viz_corr_with_significance(pdf)
            # Re-label axes to show which columns were transformed.
            x_labels = [f"{m}†" if log_flags.get(m) and power_transformer is not None else m for m in available]
            fig.add_trace(
                go.Heatmap(
                    z=corr_values,
                    x=x_labels,
                    y=x_labels,
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="corr"),
                    text=text_matrix,
                    texttemplate="%{text}",
                    hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
                ),
                row=r,
                col=c,
            )
        fig.update_layout(
            template="plotly_white",
            autosize=True,
            # Top margin a little larger to leave room for the significance legend above the
            # subplot titles; bottom margin stays minimal now that the legend moved up.
            margin=dict(l=80, r=40, t=90, b=60),
            font=dict(size=14),
        )
        # Legend for * / ** significance stars and (if any) the Yeo-Johnson † axis marker.
        # Placed at the top of the figure (above subplot titles) so it never overlaps with
        # x-tick labels at the bottom.
        legend_text = "* p < 0.05    ** p < 0.01"
        if any(log_flags.get(m) and power_transformer is not None for m in metrics):
            legend_text += "    † Yeo-Johnson transformed before correlation"
        fig.add_annotation(
            text=legend_text,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=14, color="#64748b"),
        )
        # Silence the unused-variable warning about axis_labels if not referenced.
        _ = axis_labels
        return fig

    @staticmethod
    def _viz_corr_with_significance(pdf: pd.DataFrame, decimals: int = 2) -> Tuple[np.ndarray, List[List[str]]]:
        """Pearson correlation matrix + a parallel text matrix with significance stars.

        Returns ``(r_matrix, text_matrix)`` where ``text_matrix[i][j]`` is formatted as
        ``f"{r:.2f}"`` with a trailing ``*`` when p < 0.05 and ``**`` when p < 0.01.  P-values
        are computed via the standard Pearson t-statistic ``t = r * sqrt((n-2)/(1-r²))`` and
        scipy's Student-t survival function.  If scipy isn't installed, the stars are
        silently omitted (and a warning is logged once per call).

        Edge cases:
          - Diagonal (r=1, i==j) is never starred.
          - Cells where fewer than 3 non-null pairs contributed get no stars (test is invalid).
          - NaN correlations render as ``"nan"`` with no stars.
        """
        corr_values = np.asarray(pdf.corr().values, dtype=float)
        K = corr_values.shape[0]
        text: List[List[str]] = [[""] * K for _ in range(K)]
        # Pairwise non-null sample sizes: (K x K) matrix where entry (i,j) = # rows where
        # both column i and column j are non-null.
        notna = pdf.notna().astype(int).to_numpy()  # shape (N, K)
        n_matrix = notna.T @ notna

        p_matrix: Optional[np.ndarray] = None
        try:
            from scipy.stats import t as scipy_t  # type: ignore[import-not-found]

            with np.errstate(divide="ignore", invalid="ignore"):
                dof = np.maximum(n_matrix - 2, 0)
                denom = 1.0 - corr_values**2
                denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
                t_stat = corr_values * np.sqrt(dof / denom)
                p_matrix = 2.0 * scipy_t.sf(np.abs(t_stat), df=dof)
        except ImportError:
            logger.warning(
                "Heatmap significance stars requested but scipy is not installed; skipping p-value annotations."
            )
        for i in range(K):
            for j in range(K):
                r = corr_values[i, j]
                if not np.isfinite(r):
                    text[i][j] = "nan"
                    continue
                base = f"{r:.{decimals}f}"
                if i == j or p_matrix is None or n_matrix[i, j] < 3:
                    text[i][j] = base
                    continue
                p = p_matrix[i, j]
                if not np.isfinite(p):
                    text[i][j] = base
                elif p < 0.01:
                    text[i][j] = f"{base}**"
                elif p < 0.05:
                    text[i][j] = f"{base}*"
                else:
                    text[i][j] = base
        return corr_values, text

    @staticmethod
    def _viz_yeo_johnson_columns(pdf: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply Yeo-Johnson to each named column in ``pdf`` (in-place on a copy).

        Lazy-imports sklearn.  If sklearn is not installed, logs a warning and returns
        ``pdf`` unchanged so the rest of the plot still renders.
        """
        if not cols:
            return pdf
        try:
            from sklearn.preprocessing import PowerTransformer  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("Per-metric log-transform requested but sklearn is not installed; skipping Yeo-Johnson.")
            return pdf
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        out = pdf.copy()
        for col in cols:
            if col not in out.columns:
                continue
            arr = np.asarray(out[col].to_numpy(), dtype=float).reshape(-1, 1)
            try:
                out[col] = np.asarray(pt.fit_transform(arr)).ravel()
            except Exception as exc:
                logger.warning("Yeo-Johnson failed for %r: %s", col, exc)
        return out

    def _viz_build_scatter_pair(
        self,
        metrics: List[str],
        combos: List[Tuple[str, pl.DataFrame]],
        log_x: bool = False,
        log_y: bool = False,
        symlog_x: bool = False,
        symlog_y: bool = False,
        outliers_threshold: Optional[float] = None,
        log_flags: Optional[Dict[str, bool]] = None,
    ) -> go.Figure:
        log_flags = log_flags or {}
        cols_to_log = [m for m in metrics if log_flags.get(m)]
        mx, my = metrics[0], metrics[1]
        mx_label = f"{mx}*" if log_flags.get(mx) else mx
        my_label = f"{my}*" if log_flags.get(my) else my
        fig = go.Figure()
        base_colors = Styles.COLORS
        all_x_raw: List[float] = []
        all_y_raw: List[float] = []
        for ci, (label, df) in enumerate(combos):
            if mx not in df.columns or my not in df.columns:
                continue
            pdf = cast(pd.DataFrame, df.select([mx, my]).to_pandas().apply(pd.to_numeric, errors="coerce")).dropna()
            pdf = self._viz_drop_outliers(pdf, outliers_threshold)
            if pdf.empty:
                continue
            pdf = self._viz_yeo_johnson_columns(pdf, cols_to_log)
            xs = np.asarray(pdf[mx].values, dtype=float)
            ys = np.asarray(pdf[my].values, dtype=float)
            if symlog_x:
                all_x_raw.extend(xs.tolist())
                xs = self._viz_symlog(xs)
            if symlog_y:
                all_y_raw.extend(ys.tolist())
                ys = self._viz_symlog(ys)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=label,
                    marker=dict(color=base_colors[ci % len(base_colors)], size=6, opacity=0.6),
                )
            )
        fig.update_layout(
            template="plotly_white",
            xaxis_title=mx_label + (" (symlog)" if symlog_x else ""),
            yaxis_title=my_label + (" (symlog)" if symlog_y else ""),
            autosize=True,
            margin=dict(l=70, r=20, t=40, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            font=dict(size=14),
        )
        if symlog_x and all_x_raw:
            tv, tt = self._viz_symlog_ticks(min(all_x_raw), max(all_x_raw))
            fig.update_xaxes(tickvals=tv, ticktext=tt)
        elif log_x:
            fig.update_xaxes(type="log")
        if symlog_y and all_y_raw:
            tv, tt = self._viz_symlog_ticks(min(all_y_raw), max(all_y_raw))
            fig.update_yaxes(tickvals=tv, ticktext=tt)
        elif log_y:
            fig.update_yaxes(type="log")
        return fig

    def _viz_build_scatter_grid(
        self,
        metrics: List[str],
        combos: List[Tuple[str, pl.DataFrame]],
        log_x: bool = False,
        log_y: bool = False,
        symlog_x: bool = False,
        symlog_y: bool = False,
        outliers_threshold: Optional[float] = None,
        log_flags: Optional[Dict[str, bool]] = None,
    ) -> go.Figure:
        log_flags = log_flags or {}
        cols_to_log = [m for m in metrics if log_flags.get(m)]
        metric_labels = [f"{m}*" if log_flags.get(m) else m for m in metrics]
        n = len(metrics)
        fig = make_subplots(
            rows=n, cols=n, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.03, vertical_spacing=0.03
        )
        base_colors = Styles.COLORS
        shown_labels: set[str] = set()
        # Pre-filter each combo to drop outliers across *all* selected metrics once, so the
        # same row mask is applied on the diagonal (histogram) and off-diagonal (scatter).
        filtered_combos: List[Tuple[str, pd.DataFrame]] = []
        for label, df in combos:
            available = [m for m in metrics if m in df.columns]
            if not available:
                continue
            pdf = cast(pd.DataFrame, df.select(available).to_pandas().apply(pd.to_numeric, errors="coerce")).dropna()
            pdf = self._viz_drop_outliers(pdf, outliers_threshold)
            # Apply Yeo-Johnson to flagged columns *before* computing symlog ranges so ticks
            # reflect the transformed distribution.
            pdf = self._viz_yeo_johnson_columns(pdf, cols_to_log)
            filtered_combos.append((label, pdf))
        # Per-metric raw-value ranges (untransformed) — needed for symlog tick generation.
        metric_ranges: Dict[str, Tuple[float, float]] = {}
        for m in metrics:
            vals: List[float] = []
            for _, pdf in filtered_combos:
                if m in pdf.columns:
                    vals.extend(pdf[m].dropna().tolist())
            if vals:
                metric_ranges[m] = (float(min(vals)), float(max(vals)))
        for i, m_row in enumerate(metrics):
            for j, m_col in enumerate(metrics):
                r, c = i + 1, j + 1
                for ci, (label, pdf) in enumerate(filtered_combos):
                    color = base_colors[ci % len(base_colors)]
                    show_legend = (i == 0 and j == 0) and (label not in shown_labels)
                    if i == j:
                        # Diagonal: histogram of raw values (no log/symlog transform applied —
                        # histogram Y axis is counts, and transforming X would distort the bins).
                        if m_row not in pdf.columns:
                            continue
                        h_vals = np.asarray(pdf[m_row].values)
                        if len(h_vals) == 0:
                            continue
                        fig.add_trace(
                            go.Histogram(
                                x=h_vals,
                                name=label,
                                marker=dict(color=color),
                                opacity=0.55,
                                legendgroup=label,
                                showlegend=show_legend,
                                nbinsx=30,
                            ),
                            row=r,
                            col=c,
                        )
                    else:
                        if m_col not in pdf.columns or m_row not in pdf.columns:
                            continue
                        if pdf.empty:
                            continue
                        xs = np.asarray(pdf[m_col].values, dtype=float)
                        ys = np.asarray(pdf[m_row].values, dtype=float)
                        if symlog_x:
                            xs = self._viz_symlog(xs)
                        if symlog_y:
                            ys = self._viz_symlog(ys)
                        fig.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode="markers",
                                name=label,
                                marker=dict(color=color, size=4, opacity=0.55),
                                legendgroup=label,
                                showlegend=show_legend,
                            ),
                            row=r,
                            col=c,
                        )
                    shown_labels.add(label)
                if j == 0:
                    fig.update_yaxes(title_text=metric_labels[i], row=r, col=c)
                if i == n - 1:
                    fig.update_xaxes(title_text=metric_labels[j], row=r, col=c)
                # Apply log / symlog scales only to non-diagonal cells.
                if i != j:
                    if symlog_x and m_col in metric_ranges:
                        tv, tt = self._viz_symlog_ticks(*metric_ranges[m_col])
                        fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=c)
                    elif log_x:
                        fig.update_xaxes(type="log", row=r, col=c)
                    if symlog_y and m_row in metric_ranges:
                        tv, tt = self._viz_symlog_ticks(*metric_ranges[m_row])
                        fig.update_yaxes(tickvals=tv, ticktext=tt, row=r, col=c)
                    elif log_y:
                        fig.update_yaxes(type="log", row=r, col=c)
        fig.update_layout(
            barmode="overlay",
            template="plotly_white",
            autosize=True,
            margin=dict(l=70, r=20, t=40, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            font=dict(size=12),
        )
        return fig

    # ------------------------------------------------------------------
    # AI Chat — floating dialog
    # ------------------------------------------------------------------

    def _layout_chat_dialog(self) -> html.Div:
        """Build the floating, draggable, resizable AI chat dialog."""
        return html.Div(
            id="chat-dialog",
            children=[
                # ── Title bar (drag handle) ──
                html.Div(
                    [
                        html.Span(
                            "AI Assistant",
                            style={"fontWeight": "bold", "fontSize": "14px", "color": "white"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "\u2014",
                                    id="chat-minimize-btn",
                                    n_clicks=0,
                                    title="Minimize",
                                    style={
                                        "background": "none",
                                        "border": "none",
                                        "color": "white",
                                        "fontSize": "16px",
                                        "cursor": "pointer",
                                        "padding": "0 6px",
                                        "lineHeight": "1",
                                    },
                                ),
                                html.Button(
                                    "\u2716",
                                    id="chat-close-btn",
                                    n_clicks=0,
                                    title="Close",
                                    style={
                                        "background": "none",
                                        "border": "none",
                                        "color": "white",
                                        "fontSize": "14px",
                                        "cursor": "pointer",
                                        "padding": "0 6px",
                                        "lineHeight": "1",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                    ],
                    id="chat-titlebar",
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "padding": "8px 14px",
                        "backgroundColor": "#FF7F00",
                        "borderRadius": "10px 10px 0 0",
                        "cursor": "move",
                        "userSelect": "none",
                    },
                ),
                # ── Body (collapsible) ──
                html.Div(
                    id="chat-body",
                    children=[
                        # Provider selector + hints
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Model: ",
                                            style={"fontWeight": "bold", "fontSize": "12px", "marginRight": "4px"},
                                        ),
                                        dcc.Dropdown(
                                            id="chat-provider-select",
                                            options=_CHAT_PROVIDER_DROPDOWN_OPTIONS,
                                            value=os.environ.get("CHAT_MODEL_KEY", "gemini:gemini-2.5-flash"),
                                            clearable=False,
                                            style={"width": "200px", "fontSize": "12px"},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center"},
                                ),
                                html.Span(
                                    "Try: column meaning \u2022 group definitions \u2022 rtp vs user_rtp",
                                    style={"fontSize": "11px", "color": "#888", "marginLeft": "12px"},
                                ),
                            ],
                            style={
                                "padding": "6px 14px",
                                "backgroundColor": "#fdf6ec",
                                "borderBottom": "1px solid #eee",
                                "display": "flex",
                                "alignItems": "center",
                                "flexWrap": "wrap",
                                "gap": "4px",
                            },
                        ),
                        # Messages area
                        html.Div(
                            id="chat-messages-container",
                            children=[],
                            style={
                                "flex": "1",
                                "overflowY": "auto",
                                "padding": "12px 14px",
                                "backgroundColor": "#ffffff",
                            },
                        ),
                        # Status line
                        html.Div(
                            id="chat-status",
                            style={"fontSize": "11px", "color": "#999", "padding": "0 14px 4px 14px"},
                        ),
                        # Input row
                        html.Div(
                            [
                                dcc.Input(
                                    id="chat-input",
                                    type="text",
                                    placeholder="Ask about a column, metric, or group...",
                                    style={
                                        "flex": "1",
                                        "padding": "8px 12px",
                                        "fontSize": "13px",
                                        "border": "1px solid #d0d0d0",
                                        "borderRadius": "6px",
                                        "marginRight": "6px",
                                        "outline": "none",
                                    },
                                    debounce=True,
                                    n_submit=0,
                                ),
                                html.Button(
                                    "Send",
                                    id="chat-send-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "8px 16px",
                                        "backgroundColor": "#FF7F00",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                        "fontSize": "13px",
                                        "fontWeight": "bold",
                                    },
                                ),
                                html.Button(
                                    "Clear",
                                    id="chat-clear-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "8px 10px",
                                        "backgroundColor": "#bbb",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                        "fontSize": "13px",
                                        "marginLeft": "4px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "padding": "8px 14px 10px 14px",
                                "borderTop": "1px solid #eee",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "overflow": "hidden",
                    },
                ),
            ],
            style={
                "display": "none",
                "position": "fixed",
                "bottom": "80px",
                "right": "40px",
                "width": "480px",
                "height": "550px",
                "minWidth": "340px",
                "minHeight": "250px",
                "backgroundColor": "#ffffff",
                "border": "1px solid #d0d0d0",
                "borderRadius": "10px",
                "boxShadow": "0 8px 32px rgba(0,0,0,0.18)",
                "zIndex": "9999",
                "flexDirection": "column",
                "resize": "both",
                "overflow": "hidden",
            },
        )

    def _register_chat_callbacks(self) -> None:
        """Register AI chat callbacks (only if dependencies are installed)."""

        # Toggle / close / minimize work regardless of chat agent availability
        @self.app.callback(
            Output("chat-dialog", "style"),
            Output("chat-body", "style"),
            Input("chat-toggle-btn", "n_clicks"),
            Input("chat-close-btn", "n_clicks"),
            Input("chat-minimize-btn", "n_clicks"),
            State("chat-dialog", "style"),
            State("chat-body", "style"),
            prevent_initial_call=True,
        )
        def toggle_chat_dialog(
            toggle_clicks: int,
            close_clicks: int,
            minimize_clicks: int,
            dialog_style: dict,
            body_style: dict,
        ) -> Any:
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update

            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
            dialog_style = dict(dialog_style or {})
            body_style = dict(body_style or {})

            if triggered_id == "chat-close-btn":
                dialog_style["display"] = "none"
                return dialog_style, no_update

            if triggered_id == "chat-minimize-btn":
                is_minimized = body_style.get("display") == "none"
                if is_minimized:
                    body_style["display"] = "flex"
                    dialog_style["height"] = dialog_style.get("_prevHeight", "550px")
                else:
                    dialog_style["_prevHeight"] = dialog_style.get("height", "550px")
                    body_style["display"] = "none"
                    dialog_style["height"] = "auto"
                return dialog_style, body_style

            if triggered_id == "chat-toggle-btn":
                is_visible = dialog_style.get("display") != "none"
                dialog_style["display"] = "none" if is_visible else "flex"
                if not is_visible:
                    body_style["display"] = "flex"
                return dialog_style, body_style

            return no_update, no_update

        # Drag support via clientside callback
        self.app.clientside_callback(
            """
            function() {
                var titlebar = document.getElementById('chat-titlebar');
                var dialog = document.getElementById('chat-dialog');
                if (!titlebar || !dialog || titlebar._dragInit) return;
                titlebar._dragInit = true;

                var offsetX = 0, offsetY = 0, isDragging = false;

                titlebar.addEventListener('mousedown', function(e) {
                    if (e.target.tagName === 'BUTTON') return;
                    isDragging = true;
                    var rect = dialog.getBoundingClientRect();
                    offsetX = e.clientX - rect.left;
                    offsetY = e.clientY - rect.top;
                    document.body.style.userSelect = 'none';
                });

                document.addEventListener('mousemove', function(e) {
                    if (!isDragging) return;
                    var newLeft = e.clientX - offsetX;
                    var newTop = e.clientY - offsetY;
                    newLeft = Math.max(0, Math.min(newLeft, window.innerWidth - 100));
                    newTop = Math.max(0, Math.min(newTop, window.innerHeight - 50));
                    dialog.style.left = newLeft + 'px';
                    dialog.style.top = newTop + 'px';
                    dialog.style.right = 'auto';
                    dialog.style.bottom = 'auto';
                });

                document.addEventListener('mouseup', function() {
                    isDragging = false;
                    document.body.style.userSelect = '';
                });

                return window.dash_clientside.no_update;
            }
            """,
            Output("chat-titlebar", "data-drag"),
            Input("chat-dialog", "style"),
        )

        # Step 1: Show user message immediately + "Thinking..." indicator
        @self.app.callback(
            Output("chat-messages-container", "children", allow_duplicate=True),
            Output("chat-history", "data", allow_duplicate=True),
            Output("chat-input", "value", allow_duplicate=True),
            Output("chat-status", "children", allow_duplicate=True),
            Output("chat-pending-request", "data"),
            Input("chat-send-btn", "n_clicks"),
            Input("chat-input", "n_submit"),
            Input("chat-clear-btn", "n_clicks"),
            State("chat-input", "value"),
            State("chat-history", "data"),
            State("config-dropdown", "value"),
            State("chat-provider-select", "value"),
            prevent_initial_call=True,
        )
        def handle_chat_submit(
            send_clicks: int,
            n_submit: int,
            clear_clicks: int,
            user_input: Optional[str],
            history: list,
            config_path: str,
            provider: str,
        ) -> Any:
            ctx = callback_context
            if not ctx.triggered:
                return no_update, no_update, no_update, no_update, no_update

            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if triggered_id == "chat-clear-btn":
                return [], [], "", "Chat cleared.", None

            if not user_input or not user_input.strip():
                return no_update, no_update, no_update, no_update, no_update

            history = list(history or [])
            history.append({"role": "user", "content": user_input.strip()})

            thinking_history = history + [{"role": "assistant", "content": "Thinking..."}]
            messages_ui = self._render_chat_messages(thinking_history)

            model_label = provider.split(":")[-1] if ":" in provider else provider
            pending = {
                "message": user_input.strip(),
                "history": [{"role": m["role"], "content": m["content"]} for m in history[:-1]],
                "dashboard_config": config_path or "",
                "model": provider,
            }
            return messages_ui, history, "", f"Waiting for response ({model_label})...", pending

        # Step 2: Call the AI agent API and display the response
        @self.app.callback(
            Output("chat-messages-container", "children"),
            Output("chat-history", "data"),
            Output("chat-status", "children"),
            Input("chat-pending-request", "data"),
            prevent_initial_call=True,
        )
        def handle_chat_response(pending: Optional[dict]) -> Any:
            import json as _json
            import urllib.error
            import urllib.request

            if not pending:
                return no_update, no_update, no_update

            model_key = pending.get("model", "")
            model_label = model_key.split(":")[-1] if ":" in model_key else model_key

            try:
                payload = _json.dumps(pending).encode()
                req = urllib.request.Request(
                    f"{_CHAT_API_URL}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = _json.loads(resp.read().decode())
                response = body.get("response", "No response received.")
                elapsed = body.get("elapsed_ms", "")
                status = f"({model_label}) {elapsed}ms" if elapsed else f"({model_label})"
            except urllib.error.URLError as exc:
                logger.exception("Chat agent unreachable at %s", _CHAT_API_URL)
                response = (
                    f"Cannot reach the AI Agent at `{_CHAT_API_URL}`.\n\n"
                    "**Local dev:** start the agent first:\n"
                    "```\nPYTHONPATH=src uvicorn dashboards.chat_api:app --port 8051\n```\n\n"
                    "**ECS:** ensure the ai-chat-agent service is running and "
                    "`CHAT_API_URL` is set to its ALB URL."
                )
                status = f"Agent unreachable at {_CHAT_API_URL}"
            except Exception as exc:
                logger.exception("Chat agent error")
                response = f"Error: {exc}"
                status = "An error occurred."

            # Retrieve current history from the pending request context
            history = pending.get("history", [])
            history.append({"role": "user", "content": pending["message"]})
            history.append({"role": "assistant", "content": response})

            messages_ui = self._render_chat_messages(history)
            return messages_ui, history, status

    def _register_report_tab_callbacks(self) -> None:
        """Wire 'Add to report' buttons + Report tab interactions (description, summary, export)."""
        figure_states = [
            State("date-g1-plot", "figure"),
            State("date-g2-plot", "figure"),
            State("date-g3-plot", "figure"),
            State("group-g1-plot", "figure"),
            State("group-g2-plot", "figure"),
            State("group-g3-plot", "figure"),
            State("viz-p1-plot", "figure"),
            State("viz-p2-plot", "figure"),
            State("combined-plot", "figure"),
        ]
        figure_state_graph_ids = [
            "date-g1-plot",
            "date-g2-plot",
            "date-g3-plot",
            "group-g1-plot",
            "group-g2-plot",
            "group-g3-plot",
            "viz-p1-plot",
            "viz-p2-plot",
            "combined-plot",
        ]

        @self.app.callback(
            Output("report-figures", "data", allow_duplicate=True),
            Output({"type": "add-to-report-status", "graph_id": ALL}, "children"),
            Input({"type": "add-to-report-btn", "graph_id": ALL}, "n_clicks"),
            State({"type": "add-to-report-btn", "graph_id": ALL}, "id"),
            State("report-figures", "data"),
            State("config-dropdown", "value"),
            State("navigator-tabs", "value"),
            *figure_states,
            prevent_initial_call=True,
        )
        def add_to_report(
            click_counts: List[Optional[int]],
            button_ids: List[Dict[str, str]],
            current_figures: Optional[List[Dict[str, Any]]],
            config_name: Optional[str],
            tab_value: Optional[str],
            *figures: Optional[Dict[str, Any]],
        ) -> Any:
            n = len(button_ids)
            triggered = callback_context.triggered_id
            if not isinstance(triggered, dict) or triggered.get("type") != "add-to-report-btn":
                return no_update, [no_update] * n

            target_graph_id = triggered.get("graph_id")
            if not isinstance(target_graph_id, str):
                return no_update, [no_update] * n

            try:
                btn_idx = next(i for i, bid in enumerate(button_ids) if bid.get("graph_id") == target_graph_id)
            except StopIteration:
                return no_update, [no_update] * n

            statuses: List[Any] = [no_update] * n
            if not (click_counts[btn_idx] or 0):
                return no_update, statuses

            try:
                fig_idx = figure_state_graph_ids.index(target_graph_id)
            except ValueError:
                statuses[btn_idx] = html.Span("Unknown figure id.", style={"color": "#c00"})
                return no_update, statuses

            figure_dict = figures[fig_idx] if fig_idx < len(figures) else None
            if not figure_dict or not figure_dict.get("data"):
                statuses[btn_idx] = html.Span("Figure is empty — render it first.", style={"color": "#c00"})
                return no_update, statuses

            try:
                from dashboards.report_agent.data_summary import extract_data_summary
            except ImportError as exc:
                statuses[btn_idx] = html.Span(f"Missing dep: {exc}", style={"color": "#c00"})
                return no_update, statuses

            width, height = _FIGURE_EXPORT_DIMENSIONS.get(target_graph_id, (1200, 600))
            entry = {
                "id": uuid.uuid4().hex,
                "graph_id": target_graph_id,
                "tab": tab_value,
                "config_name": config_name,
                "fig_dict": figure_dict,
                "data_summary": extract_data_summary(figure_dict),
                "description": "",
                "export_width": width,
                "export_height": height,
            }
            new_figures = list(current_figures or []) + [entry]
            statuses[btn_idx] = html.Span(
                f"Added to report (now {len(new_figures)}).",
                style={"color": "#2a7a2a"},
            )
            return new_figures, statuses

        @self.app.callback(
            Output("report-figures-container", "children"),
            Input("report-figures", "data"),
        )
        def render_report_panels(figures: Optional[List[Dict[str, Any]]]) -> Any:
            figures = figures or []
            if not figures:
                return html.Div(
                    "No figures yet — click 'Add to report' under any chart to start.",
                    style={"color": "#888", "fontStyle": "italic", "padding": "16px"},
                )
            panels: List[Component] = []
            for idx, fig in enumerate(figures, start=1):
                fig_id = fig.get("id", "")
                description = fig.get("description") or ""
                graph_id = fig.get("graph_id", "")
                panels.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4(
                                        f"Figure {idx}",
                                        style={**Styles.PANEL_HEADER, "display": "inline-block"},
                                    ),
                                    html.Span(
                                        f"  ({graph_id})",
                                        style={"color": "#888", "fontSize": "12px", "marginLeft": "8px"},
                                    ),
                                    html.Button(
                                        "Generate description",
                                        id={"type": "report-generate-desc-btn", "fig_id": fig_id},
                                        n_clicks=0,
                                        style={
                                            "marginLeft": "12px",
                                            "padding": "4px 10px",
                                            "cursor": "pointer",
                                            "backgroundColor": "#377EB8",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "4px",
                                            "fontSize": "12px",
                                        },
                                    ),
                                    html.Button(
                                        "Remove figure",
                                        id={"type": "report-remove-btn", "fig_id": fig_id},
                                        n_clicks=0,
                                        style={
                                            "marginLeft": "8px",
                                            "padding": "4px 10px",
                                            "cursor": "pointer",
                                            "backgroundColor": "#E41A1C",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "4px",
                                            "fontSize": "12px",
                                        },
                                    ),
                                    html.Span(
                                        id={"type": "report-desc-status", "fig_id": fig_id},
                                        style={"marginLeft": "10px", "fontSize": "11px", "color": "#666"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                            dcc.Graph(
                                figure=fig.get("fig_dict") or {},
                                config={"displayModeBar": False},
                                style={"marginTop": "10px"},
                            ),
                            dcc.Textarea(
                                id={"type": "report-figure-desc-text", "fig_id": fig_id},
                                value=description,
                                placeholder=(
                                    "Click 'Generate description' to draft. You can edit "
                                    "the response, or add lines like  /prompt: focus on "
                                    "the spike on Apr 15  to instruct the next regenerate."
                                ),
                                style={
                                    "width": "100%",
                                    "marginTop": "10px",
                                    "padding": "8px",
                                    "fontSize": "13px",
                                    "lineHeight": "1.6",
                                    "color": "#333",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "4px",
                                    "minHeight": "120px",
                                    "fontFamily": "inherit",
                                    "resize": "vertical",
                                    "boxSizing": "border-box",
                                },
                            ),
                        ],
                        style=Styles.SECTION,
                    )
                )
            return panels

        @self.app.callback(
            Output("report-figures", "data", allow_duplicate=True),
            Output({"type": "report-desc-status", "fig_id": ALL}, "children"),
            Input({"type": "report-generate-desc-btn", "fig_id": ALL}, "n_clicks"),
            State({"type": "report-generate-desc-btn", "fig_id": ALL}, "id"),
            # Textarea state, so unblurred edits and any /prompt: lines the
            # user just typed reach the LLM even if the blur-time persist
            # callback runs out of order with this one.
            State({"type": "report-figure-desc-text", "fig_id": ALL}, "id"),
            State({"type": "report-figure-desc-text", "fig_id": ALL}, "value"),
            State("report-figures", "data"),
            State("report-language", "value"),
            prevent_initial_call=True,
        )
        def generate_description_callback(
            click_counts: List[Optional[int]],
            button_ids: List[Dict[str, str]],
            textarea_ids: List[Dict[str, str]],
            textarea_values: List[Optional[str]],
            current_figures: Optional[List[Dict[str, Any]]],
            language: Optional[str],
        ) -> Any:
            n = len(button_ids)
            triggered = callback_context.triggered_id
            if not isinstance(triggered, dict) or triggered.get("type") != "report-generate-desc-btn":
                return no_update, [no_update] * n

            target_fig_id = triggered.get("fig_id")
            try:
                btn_idx = next(i for i, bid in enumerate(button_ids) if bid.get("fig_id") == target_fig_id)
            except StopIteration:
                return no_update, [no_update] * n

            statuses: List[Any] = [no_update] * n
            if not (click_counts[btn_idx] or 0):
                return no_update, statuses

            figures = list(current_figures or [])
            try:
                fig_idx = next(i for i, f in enumerate(figures) if f.get("id") == target_fig_id)
            except StopIteration:
                statuses[btn_idx] = html.Span("Figure not found.", style={"color": "#c00"})
                return no_update, statuses

            # Prefer the live textarea value over the Store (the user may
            # have typed without blurring).
            latest_description: Optional[str] = figures[fig_idx].get("description") or None
            try:
                tx_idx = next(i for i, tid in enumerate(textarea_ids) if tid.get("fig_id") == target_fig_id)
                latest_description = textarea_values[tx_idx] or latest_description
            except StopIteration:
                pass

            try:
                from dashboards.report_agent.description import generate_description
            except ImportError as exc:
                statuses[btn_idx] = html.Span(f"Missing dep: {exc}", style={"color": "#c00"})
                return no_update, statuses

            try:
                text = generate_description(
                    data_summary=figures[fig_idx].get("data_summary") or {},
                    language=language or "en",
                    existing_description=latest_description,
                )
            except Exception as exc:
                logger.exception("Generate description failed for fig_id=%s", target_fig_id)
                statuses[btn_idx] = html.Span(f"Failed: {exc}", style={"color": "#c00"})
                return no_update, statuses

            updated = dict(figures[fig_idx])
            updated["description"] = text
            figures[fig_idx] = updated
            statuses[btn_idx] = html.Span("Updated.", style={"color": "#2a7a2a"})
            return figures, statuses

        @self.app.callback(
            Output("report-figures", "data", allow_duplicate=True),
            Input({"type": "report-remove-btn", "fig_id": ALL}, "n_clicks"),
            State({"type": "report-remove-btn", "fig_id": ALL}, "id"),
            State("report-figures", "data"),
            prevent_initial_call=True,
        )
        def remove_figure_callback(
            click_counts: List[Optional[int]],
            button_ids: List[Dict[str, str]],
            current_figures: Optional[List[Dict[str, Any]]],
        ) -> Any:
            triggered = callback_context.triggered_id
            if not isinstance(triggered, dict) or triggered.get("type") != "report-remove-btn":
                return no_update
            target_fig_id = triggered.get("fig_id")
            try:
                btn_idx = next(i for i, bid in enumerate(button_ids) if bid.get("fig_id") == target_fig_id)
            except StopIteration:
                return no_update
            if not (click_counts[btn_idx] or 0):
                return no_update
            figures = [f for f in (current_figures or []) if f.get("id") != target_fig_id]
            return figures

        @self.app.callback(
            Output("report-summary", "data"),
            Output("report-summary-status", "children"),
            Input("report-generate-summary-btn", "n_clicks"),
            State("report-figures", "data"),
            State("report-language", "value"),
            # Live summary textarea value too, for the same unblurred-edits
            # reason as generate_description_callback.
            State("report-summary-display", "value"),
            prevent_initial_call=True,
        )
        def generate_summary_callback(
            n_clicks: int,
            figures: Optional[List[Dict[str, Any]]],
            language: Optional[str],
            current_summary_text: Optional[str],
        ) -> Any:
            if not (n_clicks or 0):
                return no_update, no_update
            figures = figures or []
            if not figures:
                return no_update, html.Span("Add figures first.", style={"color": "#c00"})
            try:
                from dashboards.report_agent.description import generate_summary
            except ImportError as exc:
                return no_update, html.Span(f"Missing dep: {exc}", style={"color": "#c00"})
            try:
                text = generate_summary(
                    figures=figures,
                    language=language or "en",
                    existing_summary=current_summary_text or None,
                )
            except Exception as exc:
                logger.exception("Generate summary failed")
                return no_update, html.Span(f"Failed: {exc}", style={"color": "#c00"})
            return text, html.Span("Updated.", style={"color": "#2a7a2a"})

        @self.app.callback(
            Output("report-summary-display", "value"),
            Input("report-summary", "data"),
        )
        def render_summary(text: Optional[str]) -> Any:
            return text or ""

        # Persist textarea edits (description + summary) into the underlying
        # Stores on blur. Without this, render_report_panels would resync the
        # textareas from the Store on the next re-render and the user's
        # edits would be lost. Equality short-circuit avoids spurious writes.
        @self.app.callback(
            Output("report-figures", "data", allow_duplicate=True),
            Input({"type": "report-figure-desc-text", "fig_id": ALL}, "n_blur"),
            State({"type": "report-figure-desc-text", "fig_id": ALL}, "id"),
            State({"type": "report-figure-desc-text", "fig_id": ALL}, "value"),
            State("report-figures", "data"),
            prevent_initial_call=True,
        )
        def persist_description_edits(
            blurs: List[Optional[int]],
            ids: List[Dict[str, str]],
            values: List[Optional[str]],
            current_figures: Optional[List[Dict[str, Any]]],
        ) -> Any:
            triggered = callback_context.triggered_id
            if not isinstance(triggered, dict) or triggered.get("type") != "report-figure-desc-text":
                return no_update
            target_fig_id = triggered.get("fig_id")
            try:
                tx_idx = next(i for i, tid in enumerate(ids) if tid.get("fig_id") == target_fig_id)
            except StopIteration:
                return no_update
            new_value = values[tx_idx] or ""
            figures = list(current_figures or [])
            try:
                f_idx = next(i for i, f in enumerate(figures) if f.get("id") == target_fig_id)
            except StopIteration:
                return no_update
            if (figures[f_idx].get("description") or "") == new_value:
                return no_update
            updated = dict(figures[f_idx])
            updated["description"] = new_value
            figures[f_idx] = updated
            return figures

        @self.app.callback(
            Output("report-summary", "data", allow_duplicate=True),
            Input("report-summary-display", "n_blur"),
            State("report-summary-display", "value"),
            State("report-summary", "data"),
            prevent_initial_call=True,
        )
        def persist_summary_edits(
            n_blur: Optional[int],
            new_value: Optional[str],
            current_value: Optional[str],
        ) -> Any:
            cleaned = new_value or ""
            if cleaned == (current_value or ""):
                return no_update
            return cleaned

        @self.app.callback(
            Output("report-export-status", "children"),
            Input("report-export-btn", "n_clicks"),
            State("report-doc-link", "value"),
            State("report-figures", "data"),
            State("report-summary", "data"),
            prevent_initial_call=True,
        )
        def export_report_callback(
            n_clicks: int,
            doc_url: Optional[str],
            figures: Optional[List[Dict[str, Any]]],
            summary: Optional[str],
        ) -> Any:
            if not (n_clicks or 0):
                return no_update
            if not doc_url:
                return html.Span("Set the Confluence doc URL first.", style={"color": "#c00"})
            figures = figures or []
            if not figures:
                return html.Span("No figures to export.", style={"color": "#c00"})
            try:
                from dashboards.report_agent.exporter import export_report
            except ImportError as exc:
                return html.Span(f"Missing dep: {exc}", style={"color": "#c00"})
            try:
                result = export_report(page_url=doc_url, summary=summary or "", figures=figures)
            except Exception as exc:
                logger.exception("Export report failed")
                return html.Span(f"Failed: {exc}", style={"color": "#c00"})
            return html.Span(
                f"Exported {result.figures_uploaded} figure(s) to page {result.page_id} (v{result.page_version}).",
                style={"color": "#2a7a2a"},
            )

    @staticmethod
    def _render_chat_messages(history: list) -> List[Component]:
        """Convert chat history to Dash HTML components."""
        components: List[Component] = []
        for msg in history:
            is_user = msg.get("role") == "user"
            components.append(
                html.Div(
                    [
                        html.Div(
                            "You" if is_user else "AI",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "12px",
                                "color": "#377EB8" if is_user else "#4DAF4A",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            dcc.Markdown(
                                msg.get("content", ""),
                                style={"fontSize": "13px", "lineHeight": "1.5"},
                            ),
                        ),
                    ],
                    style={
                        "padding": "8px 12px",
                        "marginBottom": "6px",
                        "borderRadius": "8px",
                        "backgroundColor": "#e8f4fd" if is_user else "#f0f9f0",
                        "borderLeft": f"3px solid {'#377EB8' if is_user else '#4DAF4A'}",
                    },
                )
            )
        return components

    def run(self, debug: bool = True) -> None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.host_ip = s.getsockname()[0]
            s.close()
        except Exception:
            self.host_ip = "127.0.0.1"
        # Use DASHBOARD_PUBLIC_URL when deployed (e.g. http://bituslabs.ds.dashboard.com)
        public_url = os.environ.get("DASHBOARD_PUBLIC_URL")
        display_url = public_url if public_url else f"http://{self.host_ip}:8050"
        print(f"Dashboard running on {display_url}")
        self.app.run(debug=debug, host="0.0.0.0", port=8050)


# ── WSGI entry-point (used by both Gunicorn and direct `python` execution) ──────────────
#
# Gunicorn needs a module-level WSGI callable.  We initialise the dashboard once here so
# that `gunicorn dashboards.game_stats_monitor:server` works out of the box.
#
# Why Gunicorn instead of Flask's dev server (app.run()):
#   • Flask's Werkzeug dev server is single-threaded: one long callback (e.g. DataMetrics
#     bootstrap CI on a large dataset) blocks every other request → 503s under real load.
#   • Gunicorn with --worker-class gthread spins up N threads per worker, so concurrent
#     callbacks from different browser tabs are handled in parallel without the memory cost
#     of multiple full-process workers (each of which would independently reload S3 data).
#   • --timeout 300 gives heavy computations 5 min before the worker is killed and replaced,
#     preventing indefinite hangs without killing short requests unnecessarily.
setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")
_config_dir = os.environ.get("DASHBOARD_CONFIG_DIR", str(Path(__file__).resolve().parent))
_dashboard = GameStatsDashboard(_config_dir)

# `server` is the Flask WSGI app Gunicorn binds to:
#   gunicorn --workers 1 --worker-class gthread --threads 8 --timeout 300 \
#            --bind 0.0.0.0:8050 dashboards.game_stats_monitor:server
server = _dashboard.app.server

# REST API (/api/chat, /api/metadata/*) is served by the standalone
# AI Chat Agent service (FastAPI + Uvicorn) at CHAT_API_URL.
# No Flask blueprint needed here — the dashboard calls the agent over HTTP.

if __name__ == "__main__":
    debug = os.environ.get("DASHBOARD_DEBUG", "true").lower() in ("1", "true", "yes")
    _dashboard.run(debug=debug)
