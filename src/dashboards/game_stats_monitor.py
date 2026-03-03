import logging
import os
import re
import socket
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from math import inf
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.dashboard_utils import bootstrap_worker, load_config
from bituslabs_ds.s3_utils import read_files

# ==========================================
# Styling & Constants
# ==========================================

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


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

    config_files: List[Dict[str, str]]
    config: dict
    lfs_by_date: Dict[str, pl.LazyFrame]
    lf_bet: pl.LazyFrame
    df_plot_groups: List[str]
    df_bet_groups: List[str]
    df_date_group_col: str
    df_bet_group_col: str
    sessions: List[str]
    bet_metrics: List[str]
    plot_metrics: Dict[str, List[str]]
    host_ip: str
    config_dir: str
    app: Dash

    def __init__(self, config_dir: str, host_ip: str = "127.0.0.1"):
        self.host_ip: str = host_ip
        self.config_dir: str = config_dir

        self.config_files = self._find_config_files()
        self._reset_state()
        self.config = load_config(self.config_files[0]["value"])

        self.app: Dash = Dash(__name__, suppress_callback_exceptions=True)
        self._build_main_layout()
        self._register_callbacks()
        self._load_date_data()

    @property
    def date_col(self) -> str:
        return self.config["stats_by_date"]["date_col"]

    def _reset_state(self) -> None:
        self.config = {}
        self.lfs_by_date = {}
        self.lf_bet = pl.DataFrame().lazy()
        self.df_plot_groups = []
        self.df_bet_groups = []
        self.df_date_group_col = ""
        self.df_bet_group_col = ""
        self.bet_metrics = []
        self.plot_metrics = {}

    def _find_config_files(self) -> List[Dict[str, str]]:
        config_files: List[Dict[str, str]] = []
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
        return sorted(config_files, key=lambda x: x["label"])

    def _load_data(self) -> None:
        self._load_bet_data()
        self._load_date_data()

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

    def _ensure_plot_metrics_loaded(self) -> None:
        """Ensure plot_metrics has g1/g2/g3 keys; repopulate from config if missing."""
        if "g1" not in self.plot_metrics and "stats_by_date" in self.config:
            sd = self.config["stats_by_date"]
            self.plot_metrics["g1"] = sd.get("group1_columns", [])
            self.plot_metrics["g2"] = sd.get("group2_columns", [])
            self.plot_metrics["g3"] = sd.get("group3_columns", [])

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

    def _reload_config(self, config_file: str) -> None:
        self.config = load_config(config_file)
        self._load_date_data()

    def _compute_group_date_ranges(self, granularity: Optional[str] = None) -> Tuple[
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
        if granularity is None:
            granularity = list(self.date_files_config.keys())[0]
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
        files_config = self.config["stats_by_date"].get("files", {})
        if isinstance(files_config, list):
            files_config = {"day": files_config}
        return files_config

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
                        dcc.Dropdown(
                            id="config-dropdown",
                            options=cast(Any, self.config_files),
                            value=self.config_files[0]["value"],
                            clearable=False,
                            style=dict(
                                {
                                    "width": "250px",
                                    "marginRight": "50px",
                                    "marginLeft": "8px",
                                    "marginTop": "2px",
                                    "marginBottom": "0px",
                                }
                            ),
                        ),
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
                                    label="Stats by Bet",
                                    value="tab-bet",
                                    style=Styles.NAV_TAB,
                                    selected_style=Styles.NAV_TAB_SELECTED,
                                ),
                            ],
                            style={"width": "1050px"},
                        ),
                    ],
                    style=Styles.NAV_CONTAINER,
                ),
                html.Div(
                    id="page-content",
                    style=Styles.PAGE_CONTENT,
                ),
            ]
        )

    def _layout_stats_by_date(self) -> html.Div:
        granularity: str = list(self.date_files_config.keys())[0]
        min_date, max_date = self.date_range

        date_picker = html.Div(
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
                            style={**Styles.CONTROL_LABEL, "marginRight": "10px", "display": "inline-block"},
                        ),
                        dcc.Dropdown(
                            id="date-granularity",
                            options=cast(
                                Any, [{"label": gran, "value": gran} for gran in list(self.date_files_config.keys())]
                            ),
                            value=list(self.date_files_config.keys())[0],
                            clearable=False,
                            style=Styles.DROPDOWN_NARROW,
                        ),
                    ],
                    style=Styles.FLEX_ROW_CENTER,
                ),
            ],
            style=Styles.SECTION,
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
                            # --- LEFT CONTROL PANEL ---
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Left Axis Metrics:", style=Styles.CONTROL_LABEL),
                                            dcc.Dropdown(
                                                id=f"date-{group_id}-left-metrics",
                                                options=cast(Any, self.plot_metrics[group_id]),
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
                                                options=cast(Any, self.plot_metrics[group_id]),
                                                value=[],
                                                multi=True,
                                                style=Styles.DROPDOWN_WIDE,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Hr(style=Styles.HR),
                                    html.Div(
                                        [
                                            html.Label("Logarithmic Scaling:", style=Styles.CONTROL_LABEL),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id=f"date-{group_id}-log",
                                                        options=cast(Any, [{"label": " Hybrid Log", "value": "ON"}]),
                                                        value=[],
                                                        style=Styles.CHECKLIST_INLINE,
                                                    ),
                                                    html.Span("Thresh: ", style={"fontSize": "0.9em"}),
                                                    dcc.Input(
                                                        id=f"date-{group_id}-thresh",
                                                        type="number",
                                                        value=10,
                                                        style=Styles.INPUT_SMALL,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Hr(style=Styles.HR),
                                    html.Div(
                                        [
                                            html.Label("Groups to Show:", style=Styles.CONTROL_LABEL),
                                            dcc.Checklist(
                                                id=f"date-{group_id}-group",
                                                options=cast(Any, self.df_plot_groups),
                                                value=self.df_plot_groups[: self.DEFAULT_VISIBLE_GROUPS],
                                                labelStyle=Styles.CHECKLIST_BLOCK,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                ],
                                style=Styles.CONTROL_PANEL_CONTAINER,
                            ),
                            # --- RIGHT GRAPH PANEL ---
                            html.Div(
                                [
                                    dcc.Graph(id=f"date-{group_id}-plot", style={"height": "400px"}, config=download_config)  # type: ignore
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
                date_picker,
                create_group_panel("g1", f"Date Metrics: {self.config['stats_by_date']['group1_label']}"),
                create_group_panel("g2", f"Date Metrics: {self.config['stats_by_date']['group2_label']}"),
                create_group_panel("g3", f"Date Metrics: {self.config['stats_by_date']['group3_label']}"),
            ]
        )

    def _layout_stats_by_group(self) -> html.Div:
        ranges = self._compute_group_date_ranges()

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
                    ],
                    style=Styles.FLEX_ROW_CENTER,
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
                                                options=cast(Any, self.plot_metrics[group_id]),
                                                value=(
                                                    [self.plot_metrics[group_id][0]]
                                                    if self.plot_metrics[group_id] and group_id == "g1"
                                                    else []
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
                                                options=cast(
                                                    Any,
                                                    [
                                                        {"label": "Box Plot", "value": "box"},
                                                        {"label": "Bar (Mean)", "value": "bar"},
                                                    ],
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
                                            html.Label("Group(s) to Show:", style=Styles.CONTROL_LABEL),
                                            dcc.Checklist(
                                                id=f"group-{group_id}-group",
                                                options=cast(Any, self.df_plot_groups),
                                                value=self.df_plot_groups[: self.DEFAULT_VISIBLE_GROUPS],
                                                labelStyle=Styles.CHECKLIST_BLOCK,
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
                                                options=cast(Any, [{"label": "Range 1", "value": "ON"}]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                            dcc.Checklist(
                                                id=f"group-{group_id}-show-r2",
                                                options=cast(Any, [{"label": "Range 2", "value": "ON"}]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                            dcc.Checklist(
                                                id=f"group-{group_id}-show-r3",
                                                options=cast(Any, [{"label": "Range 3", "value": "ON"}]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                        ],
                                        style=Styles.CONTROL_GROUP,
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Clip Data:", style=Styles.CONTROL_LABEL),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id=f"group-{group_id}-clip-enable",
                                                        options=cast(Any, [{"label": "Enable", "value": "ON"}]),
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
                                    dcc.Graph(id=dl_id, style={"height": "430px"}, config=download_config)  # type: ignore
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
                                            options=cast(Any, [{"label": s, "value": s} for s in self.sessions]),
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
                                            options=cast(Any, [{"label": s, "value": s} for s in self.df_bet_groups]),
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
                                            options=cast(Any, [{"label": m, "value": m} for m in self.bet_metrics]),
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
                                            options=cast(Any, [{"label": m, "value": m} for m in self.bet_metrics]),
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
                                            options=cast(Any, [{"label": " Share Left Scale", "value": "ON"}]),
                                            value=[],
                                            style=Styles.CHECKLIST_BLOCK,
                                        ),
                                        dcc.Checklist(
                                            id="share-right-yscale-check",
                                            options=cast(Any, [{"label": " Share Right Scale", "value": "ON"}]),
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
                                            options=cast(Any, [{"label": " Hybrid Log Scale", "value": "ON"}]),
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
                                            options=cast(Any, [{"label": "Max Number of Bets", "value": "ON"}]),
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
                html.Div([dcc.Graph(id="combined-plot", style={"height": "950px"})], style=Styles.GRAPH_CONTAINER),
            ],
            style=Styles.FLEX_ROW,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        @self.app.callback(
            Output("page-content", "children", allow_duplicate=True),
            Input("config-dropdown", "value"),
            State("navigator-tabs", "value"),
            prevent_initial_call=True,
        )
        def update_config(selected_config: str, current_tab: str) -> Any:
            self._reset_state()
            self._reload_config(selected_config)
            return self._render_tab_content(current_tab)

        @self.app.callback(
            Output("page-content", "children"),
            Input("navigator-tabs", "value"),
            prevent_initial_call=False,
        )
        def render_content(tab: str) -> Any:
            return self._render_tab_content(tab)

        # For "Stats by Date": update picker to full range on granularity switch
        @self.app.callback(
            Output("date-picker-range", "min_date_allowed"),
            Output("date-picker-range", "max_date_allowed"),
            Output("date-picker-range", "start_date"),
            Output("date-picker-range", "end_date"),
            Output("date-picker-range", "initial_visible_month"),
            Input("date-granularity", "value"),
        )
        def update_date_picker_on_granularity(granularity: str):
            gran_lf = self.lfs_by_date.get(granularity, pl.DataFrame().lazy())
            if gran_lf.collect_schema().len() == 0 or self.date_col not in gran_lf.collect_schema().names():
                return [None] * 5
            agg = gran_lf.select(
                pl.col(self.date_col).min().alias("min_d"),
                pl.col(self.date_col).max().alias("max_d"),
            ).collect()
            min_date, max_date = agg.item(0, "min_d"), agg.item(0, "max_d")
            return (min_date, max_date, min_date, max_date, min_date)

        # For "Stats by Group": set three pickers for three most recent 2-week ranges
        @self.app.callback(
            Output("date-picker-range1", "min_date_allowed"),
            Output("date-picker-range1", "max_date_allowed"),
            Output("date-picker-range1", "start_date"),
            Output("date-picker-range1", "end_date"),
            Output("date-picker-range1", "initial_visible_month"),
            Output("date-picker-range2", "min_date_allowed"),
            Output("date-picker-range2", "max_date_allowed"),
            Output("date-picker-range2", "start_date"),
            Output("date-picker-range2", "end_date"),
            Output("date-picker-range2", "initial_visible_month"),
            Output("date-picker-range3", "min_date_allowed"),
            Output("date-picker-range3", "max_date_allowed"),
            Output("date-picker-range3", "start_date"),
            Output("date-picker-range3", "end_date"),
            Output("date-picker-range3", "initial_visible_month"),
            Input("date-granularity", "value"),
        )
        def update_group_date_pickers_on_granularity(granularity: str):
            (
                min_date,
                max_date,
                group1_start,
                group1_end,
                group2_start,
                group2_end,
                group3_start,
                group3_end,
            ) = self._compute_group_date_ranges(granularity)
            return (
                min_date,
                max_date,
                group1_start if group1_start is not None else min_date,
                group1_end if group1_end is not None else group1_start,
                group1_start if group1_start is not None else min_date,
                min_date,
                max_date,
                group2_start if group2_start is not None else min_date,
                group2_end if group2_end is not None else group2_start,
                group2_start if group2_start is not None else min_date,
                min_date,
                max_date,
                group3_start if group3_start is not None else min_date,
                group3_end if group3_end is not None else max_date,
                group3_start if group3_start is not None else min_date,
            )

        # Add group-plot callbacks for 3 date ranges, with show/hide toggles
        for i in range(1, 4):
            self.app.callback(
                Output(f"group-g{i}-plot", "figure"),
                [
                    Input(f"group-g{i}-metrics", "value"),
                    Input(f"group-g{i}-display", "value"),
                    Input(f"group-g{i}-group", "value"),
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
                ],
            )(self.update_group_plot_combined_factory(f"g{i}"))

        self.app.callback(
            Output("combined-plot", "figure"),
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
        )(self.update_bet_plot)

        for i in range(1, 4):
            self.app.callback(
                Output(f"date-g{i}-plot", "figure"),
                [
                    Input(f"date-g{i}-left-metrics", "value"),
                    Input(f"date-g{i}-right-metrics", "value"),
                    Input(f"date-g{i}-log", "value"),
                    Input(f"date-g{i}-thresh", "value"),
                    Input(f"date-g{i}-group", "value"),
                    Input("date-granularity", "value"),
                    Input("date-picker-range", "start_date"),
                    Input("date-picker-range", "end_date"),
                ],
            )(self.update_date_plot)

    def _get_plot_groups(self, lf: pl.LazyFrame, group_col: str) -> list:
        """
        Helper function to extract unique non-None groups from given LazyFrame and group column.
        """
        if lf.collect_schema().len() > 0 and group_col in lf.collect_schema().names():
            uniq = lf.select(pl.col(group_col).unique()).collect()
            return [x for x in uniq.to_series().to_list() if x is not None]
        return []

    def _render_tab_content(self, current_tab: str) -> Any:
        if current_tab == "tab-date":
            self._ensure_plot_metrics_loaded()
            default_gran = list(self.date_files_config.keys())[0]
            if self.config["stats_by_date"]["group_col"]:
                self.df_date_group_col = self.config["stats_by_date"]["group_col"]
                lf = self.lfs_by_date.get(default_gran, pl.DataFrame().lazy())
                self.df_plot_groups = self._get_plot_groups(lf, self.df_date_group_col)
            return self._layout_stats_by_date()
        elif current_tab == "tab-group":
            self._ensure_plot_metrics_loaded()
            default_gran = list(self.date_files_config.keys())[0]
            if self.config["stats_by_date"]["group_col"]:
                self.df_date_group_col = self.config["stats_by_date"]["group_col"]
                lf = self.lfs_by_date.get(default_gran, pl.DataFrame().lazy())
                self.df_plot_groups = self._get_plot_groups(lf, self.df_date_group_col)
            return self._layout_stats_by_group()
        elif current_tab == "tab-bet":
            self._load_bet_data()
            if self.config["stats_by_bet"]["group_col"]:
                self.df_bet_group_col = self.config["stats_by_bet"]["group_col"]
                lf_bet = self.lf_bet
                self.df_bet_groups = self._get_plot_groups(lf_bet, self.df_bet_group_col)
            return self._layout_stats_by_bet()
        else:
            return html.Div("404 Error")

    # ------------------------------------------------------------------
    # Plot Logic
    # ------------------------------------------------------------------

    def update_date_plot(
        self,
        left_metrics: Any,
        right_metrics: Any,
        log_val: Any,
        log_thresh: Any,
        groups: Any,
        date_granularity: Any,
        start_date: Any,
        end_date: Any,
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

        df_date = lf.filter((pl.col(self.date_col) >= start_dt) & (pl.col(self.date_col) <= end_dt)).collect()
        if df_date.is_empty():
            return fig

        if self.df_date_group_col == "" or self.df_date_group_col not in df_date.columns:
            logger.warning(f"Group column {self.df_date_group_col} not found in dataframe")
            logger.warning(f"Using default group column 'group'")
            group_col = "group"
            df_date = df_date.with_columns(pl.lit("total").alias("group"))
            groups = ["total"]
            color_map = {
                f"{s}:{m}": Styles.COLORS[i % len(Styles.COLORS)]
                for i, m in enumerate(left_metrics + right_metrics)
                for j, s in enumerate(groups)
            }
            line_style_map = {
                f"{s}:{m}": Styles.LINE_SHAPE[0]
                for i, m in enumerate(left_metrics + right_metrics)
                for j, s in enumerate(groups)
            }
        else:
            group_col = self.df_date_group_col
            color_map = {
                f"{s}:{m}": Styles.COLORS[j % len(Styles.COLORS)]
                for i, m in enumerate(left_metrics + right_metrics)
                for j, s in enumerate(groups)
            }
            line_style_map = {
                f"{s}:{m}": Styles.LINE_SHAPE[i % len(Styles.LINE_SHAPE)]
                for i, m in enumerate(left_metrics + right_metrics)
                for j, s in enumerate(groups)
            }

        with ProcessPoolExecutor() as executor:
            for strat in groups:
                print(f"Processing group: {strat}")
                df_group = df_date.filter(pl.col(group_col) == strat).sort(self.date_col)
                if df_group.is_empty():
                    continue

                def add_scatter_plot(metrics: Any, secondary_y: bool) -> None:
                    date_vals = df_group.get_column(self.date_col)
                    for m in metrics:
                        if m not in df_group.columns:
                            continue

                        if self.date_col in df_group.columns:
                            agg_df = (
                                df_group.group_by(self.date_col)
                                .agg(pl.col(m).mean().alias("_mean"))
                                .sort(self.date_col)
                            )
                            dates = agg_df.get_column(self.date_col)
                            mean_vals = agg_df.get_column("_mean")
                            arrays_to_bootstrap = [
                                df_group.filter(pl.col(self.date_col) == d).get_column(m).drop_nulls().to_numpy()
                                for d in dates
                            ]
                            ci_results = list(executor.map(bootstrap_worker, arrays_to_bootstrap))
                            lowers, uppers = zip(*ci_results)
                            y_raw = mean_vals.to_numpy()
                            y_lower = np.array([l if l == l else y_raw[i] for i, l in enumerate(lowers)])
                            y_upper = np.array([u if u == u else y_raw[i] for i, u in enumerate(uppers)])
                            x = dates.to_list()
                            y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw
                            y_lower_plot = self.hybrid_transform(y_lower, thresh) if log_scale else y_lower
                            y_upper_plot = self.hybrid_transform(y_upper, thresh) if log_scale else y_upper
                        else:
                            y_raw = df_group.get_column(m).to_numpy()
                            y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw
                            x = date_vals.to_list()
                            y_lower_plot = y_plot
                            y_upper_plot = y_plot

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

                        lower_arr = y_lower_plot if isinstance(y_lower_plot, np.ndarray) else np.asarray(y_lower_plot)
                        upper_arr = y_upper_plot if isinstance(y_upper_plot, np.ndarray) else np.asarray(y_upper_plot)
                        if not np.allclose(lower_arr, upper_arr):
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

        fig.update_layout(
            height=400,
            margin=dict(l=50, r=10, t=5, b=15),
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
            all_y: List[float] = []
            for c in left_metrics or []:
                if c in df_date.columns:
                    all_y.extend(df_date.get_column(c).cast(pl.Float64).to_numpy().tolist())
            if all_y:
                yticks = self.get_ticks(all_y, thresh)
                fig.update_yaxes(
                    tickvals=self.hybrid_transform(yticks, thresh).tolist(),
                    ticktext=[f"{v:.0f}" for v in yticks],
                    secondary_y=False,
                )
            all_y_r: List[float] = []
            for c in right_metrics or []:
                if c in df_date.columns:
                    all_y_r.extend(df_date.get_column(c).cast(pl.Float64).to_numpy().tolist())
            if all_y_r:
                yticks_r = self.get_ticks(all_y_r, thresh)
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
            groups: Any,
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
        ) -> go.Figure:
            if not metric or not groups or r1_start is None or r1_end is None:
                return go.Figure()
            granularity = list(self.date_files_config.keys())[0]
            lf = self.lfs_by_date.get(granularity, pl.DataFrame().lazy())
            if lf.collect_schema().len() == 0:
                return go.Figure()
            # Collect only the date range needed for the three ranges to limit memory
            # Convert string dates from Dash pickers to datetime objects
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
            df_date = lf.filter((pl.col(self.date_col) >= min_d) & (pl.col(self.date_col) <= max_d)).collect()
            if df_date.is_empty() or self.df_date_group_col not in df_date.columns or metric not in df_date.columns:
                return go.Figure()
            base_colors = Styles.COLORS
            fig = go.Figure()

            def _label(group, l):
                return f"{group} [Range{l}]"

            # Parse date strings to datetime objects
            data_ranges = [
                {
                    "start": self._parse_date(r1_start),
                    "end": self._parse_date(r1_end),
                    "label": "1",
                    "show": "ON" in (show_r1 or []),
                },
                {
                    "start": self._parse_date(r2_start),
                    "end": self._parse_date(r2_end),
                    "label": "2",
                    "show": "ON" in (show_r2 or []),
                },
                {
                    "start": self._parse_date(r3_start),
                    "end": self._parse_date(r3_end),
                    "label": "3",
                    "show": "ON" in (show_r3 or []),
                },
            ]
            # Filter out ranges with None dates
            data_ranges = [r for r in data_ranges if r["start"] is not None and r["end"] is not None]
            # Only keep enabled
            data_by_range = [r for r in data_ranges if r["show"]]
            enable_clip = clip_enable is not None and "ON" in (clip_enable or [])
            cmin = clip_min if enable_clip and clip_min is not None else None
            cmax = clip_max if enable_clip and clip_max is not None else None

            if display_mode == "box":
                for gi, group in enumerate(groups):
                    color_base = base_colors[gi % len(base_colors)]
                    for ri, range_info in enumerate(data_by_range):
                        start_dt = range_info["start"]
                        end_dt = range_info["end"]
                        if start_dt is None or end_dt is None:
                            continue
                        df_g = df_date.filter(
                            (pl.col(self.date_col) >= start_dt)
                            & (pl.col(self.date_col) <= end_dt)
                            & (pl.col(self.df_date_group_col) == group)
                        )
                        vals = df_g.get_column(metric).drop_nulls().to_numpy()
                        if enable_clip and (cmin is not None or cmax is not None):
                            vals = np.clip(
                                vals, cmin if cmin is not None else -np.inf, cmax if cmax is not None else np.inf
                            )
                        if len(vals) == 0:
                            continue
                        box_label = _label(group, range_info["label"])
                        fig.add_trace(
                            go.Box(
                                y=vals,
                                name=box_label,
                                boxmean="sd",
                                marker=dict(
                                    color=color_base, opacity=1.0 if ri == 0 else 0.45, line=dict(color="#333", width=1)
                                ),
                                showlegend=False,
                            )
                        )
            else:  # bar, mean+std
                x_labels = []
                ydata = []
                edata = []
                mcolors_box = []
                for gi, group in enumerate(groups):
                    color_base = base_colors[gi % len(base_colors)]
                    for ri, range_info in enumerate(data_by_range):
                        start_dt = range_info["start"]
                        end_dt = range_info["end"]
                        if start_dt is None or end_dt is None:
                            continue
                        df_g = df_date.filter(
                            (pl.col(self.date_col) >= start_dt)
                            & (pl.col(self.date_col) <= end_dt)
                            & (pl.col(self.df_date_group_col) == group)
                        )
                        vals = df_g.get_column(metric).drop_nulls().to_numpy()
                        if enable_clip and (cmin is not None or cmax is not None):
                            vals = np.clip(
                                vals, cmin if cmin is not None else -np.inf, cmax if cmax is not None else np.inf
                            )
                        if len(vals) == 0:
                            continue
                        bar_label = _label(group, range_info["label"])
                        x_labels.append(bar_label)
                        ydata.append(float(np.mean(vals)))
                        edata.append(float(np.std(vals)))
                        if ri == 0:
                            mcolors_box.append(dict(color=color_base, opacity=1.0, line=dict(color="#333", width=1)))
                        else:
                            mcolors_box.append(
                                dict(
                                    color=color_base,
                                    opacity=0.5,
                                    line=dict(color="#333", width=1),
                                    pattern=dict(shape="/"),
                                )
                            )
                if x_labels:
                    fig.add_trace(
                        go.Bar(
                            x=x_labels,
                            y=ydata,
                            error_y=dict(type="data", array=edata),
                            marker={"color": [c["color"] for c in mcolors_box], "opacity": None},
                            customdata=[c for c in mcolors_box],
                            hovertemplate="%{x}: %{y:.2f} ± %{error_y.array:.2f}",
                            showlegend=False,
                        )
                    )
                    for i, bar in enumerate(fig.data):
                        if hasattr(bar, "customdata") and bar.customdata is not None:
                            mcolors_box = bar.customdata
                            fig.data[i].marker.color = [m["color"] for m in mcolors_box]
                            if any("pattern" in m for m in mcolors_box):
                                if not hasattr(fig.data[i].marker, "pattern"):
                                    fig.data[i].marker.pattern = dict(shape=[""] * len(bar.x))
                                fig.data[i].marker.pattern.shape = [
                                    m.get("pattern", {}).get("shape", "") for m in mcolors_box
                                ]
                            fig.data[i].marker.opacity = [m.get("opacity", 1.0) for m in mcolors_box]

            mode_title = "Box Plot" if display_mode == "box" else "Bar (Mean ± Std)"
            subtitle = ""
            if enable_clip:
                subtitle = f" (Clipped"
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
                # legend removed (redundant with explicit labels)
            )
            return fig

        return callback

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


if __name__ == "__main__":
    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")
    config_dir = os.environ.get("DASHBOARD_CONFIG_DIR", str(Path(__file__).resolve().parent))
    debug = os.environ.get("DASHBOARD_DEBUG", "false").lower() in ("1", "true", "yes")
    dashboard = GameStatsDashboard(config_dir)
    dashboard.run(debug=debug)
