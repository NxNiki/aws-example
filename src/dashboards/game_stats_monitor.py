import logging
import os
import re
import socket
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from math import inf
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.s3_utils import read_local_cache
from bituslabs_ds.utils import bootstrap_worker, load_config

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
        "padding": "20px",
        "boxShadow": "0 2px 10px 0 rgba(0,0,0,0.05)",
        "marginBottom": "20px",
    }
    PANEL_HEADER: Dict[str, Any] = {
        "marginTop": "0",
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
        "marginBottom": "26px",
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
    CONTROL_PANEL_CONTAINER: Dict[str, Any] = {
        "width": "20%",
        "display": "inline-block",
        "verticalAlign": "top",
        "paddingRight": "22px",
        "paddingLeft": "2px",
        "boxSizing": "border-box",
    }

    GRAPH_CONTAINER: Dict[str, Any] = {
        "width": "100%",
        "display": "inline-block",
        "verticalAlign": "top",
        "padding": "28px",
        "marginRight": "10px",
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
        "marginLeft": "0px",
    }
    PAGE_CONTENT: Dict[str, Any] = {
        "padding": "0 20px 20px 20px",
        "backgroundColor": "#f4f7f6",
        "minHeight": "100vh",
    }

    # Inputs and controls
    LABEL: Dict[str, Any] = {
        "fontWeight": "bold",
        "marginRight": "10px",
    }
    CONTROL_LABEL: Dict[str, Any] = {
        "fontWeight": "bold",
        "marginBottom": "5px",
        "display": "block",
    }
    GROUP_CHECKLIST_LABEL: Dict[str, Any] = {
        "display": "block",
        "marginBottom": "5px",
    }
    GROUP_CHECKLIST_STYLE: Dict[str, Any] = {
        "marginBottom": "10px",
    }
    MARGIN_BOTTOM14: Dict[str, Any] = {
        "marginBottom": "14px",
        "marginTop": "8px",
    }
    HR: Dict[str, Any] = {
        "marginTop": "2px",
        "marginBottom": "2px",
    }

    DROPDOWN_NARROW: Dict[str, Any] = {"width": "150px"}
    DROPDOWN_WIDE: Dict[str, Any] = {"minWidth": "200px"}
    INPUT_SMALL: Dict[str, Any] = {
        "width": "60px",
        "marginLeft": "2px",
    }
    INPUT_FULLWIDTH: Dict[str, Any] = {
        "width": "100%",
    }
    CHECKLIST_INLINE: Dict[str, Any] = {
        "display": "inline-block",
        "marginRight": "10px",
    }


# ==========================================
# Dashboard Logic
# ==========================================


class GameStatsDashboard:

    DEFAULT_VISIBLE_GROUPS: int = 2

    config_files: List[Dict[str, str]]
    config: dict
    dfs_by_date: Dict[str, pd.DataFrame]
    df_bet: pd.DataFrame
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
        self.dfs_by_date = {}
        self.df_bet = pd.DataFrame()
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
        if not self.df_bet.empty:
            return

        file_paths = [f for f in self.config["stats_by_bet"]["files"] if f]

        bet_data: List[pd.DataFrame] = []
        with ThreadPoolExecutor() as executor:
            bet_data = list(executor.map(read_local_cache, file_paths))

        if bet_data:
            self.df_bet = pd.concat(bet_data)
            self.df_bet["bet_index"] = pd.to_numeric(self.df_bet["bet_index"], errors="coerce")
            self.df_bet = self.df_bet.dropna(subset=["bet_index"])
            self.sessions = sorted(self.df_bet["session_start_date"].astype(str).unique())
            exclude_bet_cols = ["session_start_date", "session_group", "bet_index"]
            self.bet_metrics = [c for c in self.df_bet.columns if c not in exclude_bet_cols]
        else:
            self.df_bet = pd.DataFrame()
            self.sessions = []
            self.bet_metrics = []

    def _load_date_data(self) -> None:
        with ThreadPoolExecutor() as executor:
            for gran, file_paths in self.date_files_config.items():
                daily_data: List[pd.DataFrame] = []
                if isinstance(file_paths, str):
                    file_paths = [file_paths]

                valid_paths = [f for f in file_paths if f]
                if valid_paths:
                    daily_data = list(executor.map(read_local_cache, valid_paths))

                if daily_data:
                    self.dfs_by_date[gran] = pd.concat(daily_data)
                    if self.date_col in self.dfs_by_date[gran].columns:
                        self.dfs_by_date[gran][self.date_col] = pd.to_datetime(self.dfs_by_date[gran][self.date_col])
                else:
                    self.dfs_by_date[gran] = pd.DataFrame()

        self.plot_metrics = {}
        self.plot_metrics["g1"] = self.config["stats_by_date"]["group1_columns"]
        self.plot_metrics["g2"] = self.config["stats_by_date"]["group2_columns"]
        self.plot_metrics["g3"] = self.config["stats_by_date"]["group3_columns"]

    def _reload_config(self, config_file: str) -> None:
        self.config = load_config(config_file)
        self._load_date_data()

    def _compute_group_date_ranges(self, granularity: Optional[str] = None) -> Tuple[
        Optional[pd.Timestamp],
        Optional[pd.Timestamp],
        Optional[pd.Timestamp],
        Optional[pd.Timestamp],
        Optional[pd.Timestamp],
        Optional[pd.Timestamp],
        Optional[pd.Timestamp],
        Optional[pd.Timestamp],
    ]:
        """
        For Stats by Group: Compute for 3 sequential two-week ranges (6 weeks total, most recent).
        Returns: min_date, max_date, g1_start, g1_end, g2_start, g2_end, g3_start, g3_end
        """
        if granularity is None:
            granularity = list(self.date_files_config.keys())[0]
        df_date = self.dfs_by_date.get(granularity, pd.DataFrame())
        if df_date.empty or self.date_col not in df_date.columns:
            return (None, None, None, None, None, None, None, None)
        min_date = pd.to_datetime(df_date[self.date_col].min())
        max_date = pd.to_datetime(df_date[self.date_col].max())

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
        df_date = self.dfs_by_date.get(granularity, pd.DataFrame())
        if df_date.empty or self.date_col not in df_date.columns:
            return None, None
        min_d = df_date[self.date_col].min()
        max_d = df_date[self.date_col].max()
        return min_d, max_d

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
        df_date: pd.DataFrame = self.dfs_by_date.get(granularity, pd.DataFrame())
        min_date, max_date = None, None
        if not df_date.empty and self.date_col in df_date.columns:
            min_date, max_date = df_date[self.date_col].min(), df_date[self.date_col].max()

        date_picker = html.Div(
            [
                html.Div(
                    [
                        html.Label("Filter Date Range:", style=Styles.LABEL),
                        dcc.DatePickerRange(
                            id="date-picker-range",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            initial_visible_month=min_date,
                            start_date=min_date,
                            end_date=max_date,
                            display_format="YYYY-MM-DD",
                            style={"verticalAlign": "middle", "marginRight": "20px"},
                        ),
                        html.Div(style={"width": "30px"}),
                        html.Label("Select Date Granularity:", style=Styles.LABEL),
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

        download_config = dict[str, dict[str, str | int] | bool](
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
                                    html.Label("Left Axis Metrics:", style=Styles.CONTROL_LABEL),
                                    dcc.Dropdown(
                                        id=f"date-{group_id}-left-metrics",
                                        options=cast(Any, self.plot_metrics[group_id]),
                                        value=[self.plot_metrics[group_id][0]] if self.plot_metrics[group_id] else [],
                                        multi=True,
                                        style=Styles.DROPDOWN_WIDE,
                                    ),
                                    html.Br(),
                                    html.Label(
                                        "Right Axis Metrics:", style=dict(Styles.CONTROL_LABEL, marginTop="2px")
                                    ),
                                    dcc.Dropdown(
                                        id=f"date-{group_id}-right-metrics",
                                        options=cast(Any, self.plot_metrics[group_id]),
                                        value=[],
                                        multi=True,
                                        style=Styles.DROPDOWN_WIDE,
                                    ),
                                    html.Br(),
                                    html.Hr(style=Styles.HR),
                                    html.Label("Logarithmic Scaling:", style=Styles.CONTROL_LABEL),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id=f"date-{group_id}-log",
                                                options=cast(Any, [{"label": " Hybrid Log", "value": "ON"}]),
                                                value=[],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                            html.Label("Thresh: ", style={"fontSize": "0.9em"}),
                                            dcc.Input(
                                                id=f"date-{group_id}-thresh",
                                                type="number",
                                                value=10,
                                                style=Styles.INPUT_SMALL,
                                            ),
                                        ],
                                        style={"marginTop": "10px"},
                                    ),
                                    html.Br(),
                                    html.Hr(style=Styles.HR),
                                    html.Label("Groups to Show:", style=Styles.CONTROL_LABEL),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id=f"date-{group_id}-group",
                                                options=cast(Any, self.df_plot_groups),
                                                value=self.df_plot_groups[: self.DEFAULT_VISIBLE_GROUPS],
                                                labelStyle=Styles.GROUP_CHECKLIST_LABEL,
                                                style=Styles.GROUP_CHECKLIST_STYLE,
                                            )
                                        ],
                                        style={"marginTop": "10px"},
                                    ),
                                ],
                                style=Styles.CONTROL_PANEL_CONTAINER,
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id=f"date-{group_id}-plot", style={"height": "400px"}, config=download_config  # type: ignore
                                    )
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
        (
            min_date,
            max_date,
            group1_start,
            group1_end,
            group2_start,
            group2_end,
            group3_start,
            group3_end,
        ) = self._compute_group_date_ranges()

        date_group_picker = html.Div(
            [
                html.Div(
                    [
                        html.Label("Date Range 1 (Oldest):", style=Styles.LABEL),
                        dcc.DatePickerRange(
                            id="date-picker-range1",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            initial_visible_month=group1_start if group1_start is not None else min_date,
                            start_date=group1_start if group1_start is not None else min_date,
                            end_date=group1_end if group1_end is not None else group1_start,
                            display_format="YYYY-MM-DD",
                            style={"verticalAlign": "middle", "marginRight": "36px"},
                        ),
                        html.Label("Date Range 2:", style=Styles.LABEL),
                        dcc.DatePickerRange(
                            id="date-picker-range2",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            initial_visible_month=group2_start if group2_start is not None else min_date,
                            start_date=group2_start if group2_start is not None else min_date,
                            end_date=group2_end if group2_end is not None else group2_start,
                            display_format="YYYY-MM-DD",
                            style={"verticalAlign": "middle", "marginRight": "36px"},
                        ),
                        html.Label("Date Range 3 (Most Recent):", style=Styles.LABEL),
                        dcc.DatePickerRange(
                            id="date-picker-range3",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            initial_visible_month=group3_start if group3_start is not None else min_date,
                            start_date=group3_start if group3_start is not None else min_date,
                            end_date=group3_end if group3_end is not None else max_date,
                            display_format="YYYY-MM-DD",
                            style={"verticalAlign": "middle", "marginRight": "18px"},
                        ),
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
            metric_dropdown_id = f"group-{group_id}-metrics"
            display_toggle_id = f"group-{group_id}-display"
            group_selector_id = f"group-{group_id}-group"
            range_cb_id1 = f"group-{group_id}-show-r1"
            range_cb_id2 = f"group-{group_id}-show-r2"
            range_cb_id3 = f"group-{group_id}-show-r3"
            # --- CLIP controls:
            clip_check_id = f"group-{group_id}-clip-enable"
            clip_min_id = f"group-{group_id}-clip-min"
            clip_max_id = f"group-{group_id}-clip-max"

            return html.Div(
                [
                    html.H4(label, style=Styles.PANEL_HEADER),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Metric:", style=Styles.CONTROL_LABEL),
                                    dcc.Dropdown(
                                        id=metric_dropdown_id,
                                        options=cast(Any, self.plot_metrics[group_id]),
                                        value=[self.plot_metrics[group_id][0]] if self.plot_metrics[group_id] else [],
                                        multi=False,
                                        style={"minWidth": "240px"},
                                    ),
                                    html.Br(),
                                    html.Hr(style=Styles.HR),
                                    html.Label("Display Mode:", style=Styles.CONTROL_LABEL),
                                    dcc.RadioItems(
                                        id=display_toggle_id,
                                        options=cast(
                                            Any,
                                            [
                                                {"label": "Box Plot", "value": "box"},
                                                {"label": "Bar (Mean)", "value": "bar"},
                                            ],
                                        ),
                                        value="box",
                                        labelStyle={"display": "inline-block", "marginRight": "12px"},
                                        style=Styles.MARGIN_BOTTOM14,
                                    ),
                                    html.Br(),
                                    html.Hr(style=Styles.HR),
                                    html.Label("Group(s) to Show:", style=Styles.CONTROL_LABEL),
                                    dcc.Checklist(
                                        id=group_selector_id,
                                        options=cast(Any, self.df_plot_groups),
                                        value=self.df_plot_groups[: self.DEFAULT_VISIBLE_GROUPS],
                                        labelStyle=Styles.GROUP_CHECKLIST_LABEL,
                                        style={"marginBottom": "14px", "marginTop": "8px"},
                                    ),
                                    html.Br(),
                                    html.Hr(style=Styles.HR),
                                    html.Label("Show Date Range(s):", style=Styles.CONTROL_LABEL),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id=range_cb_id1,
                                                options=cast(Any, [{"label": "Show Range 1", "value": "ON"}]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE | {"marginRight": "18px"},
                                            ),
                                            dcc.Checklist(
                                                id=range_cb_id2,
                                                options=cast(Any, [{"label": "Show Range 2", "value": "ON"}]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE | {"marginRight": "18px"},
                                            ),
                                            dcc.Checklist(
                                                id=range_cb_id3,
                                                options=cast(Any, [{"label": "Show Range 3", "value": "ON"}]),
                                                value=["ON"],
                                                style=Styles.CHECKLIST_INLINE,
                                            ),
                                        ],
                                        style={"marginTop": "8px", "marginBottom": "4px"},
                                    ),
                                    html.Hr(style=Styles.HR),
                                    html.Label(
                                        "Clip Data (Box/Bar):", style={**Styles.CONTROL_LABEL, "display": "block"}
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id=clip_check_id,
                                                options=cast(Any, [{"label": " Enable Clip", "value": "ON"}]),
                                                value=[],
                                                style={"display": "inline-block", "marginRight": "12px"},
                                            ),
                                            html.Label("Min:", style={"marginLeft": "10px"}),
                                            dcc.Input(
                                                id=clip_min_id,
                                                type="number",
                                                placeholder="min",
                                                style={"width": "70px"},
                                                value=None,
                                            ),
                                            html.Label("Max:", style={"marginLeft": "10px"}),
                                            dcc.Input(
                                                id=clip_max_id,
                                                type="number",
                                                placeholder="max",
                                                style={"width": "70px"},
                                                value=None,
                                            ),
                                        ],
                                        style={"marginTop": "8px", "marginBottom": "4px"},
                                    ),
                                ],
                                style=Styles.CONTROL_PANEL_CONTAINER,
                            ),
                            html.Div(
                                [dcc.Graph(id=dl_id, style={"height": "430px"}, config=download_config)],  # type: ignore
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
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Select Date:", style={**Styles.CONTROL_LABEL, "marginBottom": "10px"}),
                                dcc.Dropdown(
                                    id="session-dropdown",
                                    options=cast(Any, [{"label": s, "value": s} for s in self.sessions]),
                                    value=self.sessions[0] if self.sessions else None,
                                    clearable=False,
                                ),
                                html.Hr(style=Styles.HR),
                                html.Label(
                                    "Compare Strategies:", style={**Styles.CONTROL_LABEL, "marginBottom": "10px"}
                                ),
                                dcc.Checklist(
                                    id="strategy-checklist",
                                    options=cast(Any, [{"label": s, "value": s} for s in self.df_bet_groups]),
                                    value=self.df_bet_groups[: self.DEFAULT_VISIBLE_GROUPS],
                                    labelStyle=Styles.GROUP_CHECKLIST_LABEL,
                                    style=Styles.GROUP_CHECKLIST_STYLE,
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                        html.Div(
                            [
                                html.Label("Metrics (Left Axis):", style=Styles.CONTROL_LABEL),
                                dcc.Dropdown(
                                    id="metric-checklist",
                                    options=cast(Any, [{"label": m, "value": m} for m in self.bet_metrics]),
                                    value=[self.bet_metrics[0]] if self.bet_metrics else [],
                                    multi=True,
                                    style=Styles.MARGIN_BOTTOM14,
                                ),
                                html.Label("Metrics (Right Axis):", style=Styles.CONTROL_LABEL),
                                dcc.Dropdown(
                                    id="right-axis-checklist",
                                    options=cast(Any, [{"label": m, "value": m} for m in self.bet_metrics]),
                                    value=[],
                                    multi=True,
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="share-left-yscale-check",
                                    options=cast(Any, [{"label": " Share Left Scale", "value": "ON"}]),
                                    value=[],
                                ),
                                dcc.Checklist(
                                    id="share-right-yscale-check",
                                    options=cast(Any, [{"label": " Share Right Scale", "value": "ON"}]),
                                    value=[],
                                ),
                                html.Hr(style=Styles.HR),
                                dcc.Checklist(
                                    id="log-check",
                                    options=cast(Any, [{"label": " Hybrid Log Scale", "value": "ON"}]),
                                    value=[],
                                ),
                                html.Div(
                                    [
                                        html.Label("Linear Thresh:"),
                                        dcc.Input(
                                            id="linear-thresh", type="number", value=10, style=Styles.INPUT_FULLWIDTH
                                        ),
                                    ],
                                    style={"marginTop": "5px"},
                                ),
                                html.Hr(style=Styles.HR),
                                dcc.Checklist(
                                    id="filter-check",
                                    options=cast(Any, [{"label": "Max Number of Bets", "value": "ON"}]),
                                    value=["ON"],
                                ),
                                dcc.Input(
                                    id="filter-thresh",
                                    type="number",
                                    value=5000,
                                    style={**Styles.INPUT_FULLWIDTH, "marginTop": "5px"},
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                    ],
                    style=Styles.CONTROL_PANEL_CONTAINER,
                ),
                html.Div([dcc.Graph(id="combined-plot", style={"height": "950px"})], style=Styles.GRAPH_CONTAINER),
            ],
            style={"display": "flex", "width": "100%"},
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
            gran = granularity
            gran_df = self.dfs_by_date.get(gran, pd.DataFrame())
            if gran_df.empty or self.date_col not in gran_df.columns:
                return [None] * 5
            min_date = gran_df[self.date_col].min()
            max_date = gran_df[self.date_col].max()
            return (
                min_date,
                max_date,
                min_date,
                max_date,
                min_date,
            )

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

    def _render_tab_content(self, current_tab: str) -> Any:
        if current_tab == "tab-date":
            default_gran = list(self.date_files_config.keys())[0]
            if self.config["stats_by_date"]["group_col"]:
                self.df_date_group_col = self.config["stats_by_date"]["group_col"]
                self.df_plot_groups = self.dfs_by_date[default_gran][self.df_date_group_col].unique().tolist()
            return self._layout_stats_by_date()
        elif current_tab == "tab-group":
            default_gran = list(self.date_files_config.keys())[0]
            if self.config["stats_by_date"]["group_col"]:
                self.df_date_group_col = self.config["stats_by_date"]["group_col"]
                self.df_plot_groups = self.dfs_by_date[default_gran][self.df_date_group_col].unique().tolist()
            return self._layout_stats_by_group()
        elif current_tab == "tab-bet":
            self._load_bet_data()
            if self.config["stats_by_bet"]["group_col"]:
                self.df_bet_group_col = self.config["stats_by_bet"]["group_col"]
                self.df_bet_groups = self.df_bet[self.df_bet_group_col].unique().tolist()
            return self._layout_stats_by_bet()
        else:
            return html.Div("404 Error")

    # ------------------------------------------------------------------
    # Plot Logic
    # ------------------------------------------------------------------

    def update_date_plot(
        self, left_metrics, right_metrics, log_val, log_thresh, groups, date_granularity, start_date, end_date
    ):
        if not left_metrics and not right_metrics:
            return go.Figure()

        log_scale = "ON" in (log_val or [])
        thresh = float(log_thresh) if log_thresh else 10.0
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        df_date = self.dfs_by_date[date_granularity]
        df_date = df_date[(df_date[self.date_col] >= start_date) & (df_date[self.date_col] <= end_date)].copy()

        if self.df_date_group_col == "" or self.df_date_group_col not in df_date.columns:
            group_col = "group"
            df_date["group"] = "total"
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

        # Initialize ProcessPoolExecutor for parallel bootstrap calculations
        # We start it here because creating it inside the inner loop is inefficient
        with ProcessPoolExecutor() as executor:
            for strat in groups:
                df_strat = df_date[df_date[group_col] == strat].sort_values(self.date_col)
                if df_strat.empty:
                    continue

                # Helper to add trace (requires calculating CI first)
                def add_scatter_plot(metrics, secondary_y):
                    x_vals = df_strat[self.date_col]
                    for m in metrics:
                        if m not in df_strat.columns:
                            continue

                        if self.date_col in df_strat.columns:
                            grouped = df_strat.groupby(self.date_col)[m]
                            mean_vals = grouped.mean()

                            # Prepare data for parallel execution
                            # We collect all arrays that need bootstrapping for this metric
                            dates = mean_vals.index.tolist()
                            arrays_to_bootstrap = [grouped.get_group(d).dropna().values for d in dates]

                            # Execute bootstrap in parallel
                            ci_results = list(executor.map(bootstrap_worker, arrays_to_bootstrap))

                            # Unpack results
                            lowers, uppers = zip(*ci_results)

                            y_raw = mean_vals
                            y_lower = pd.Series(lowers, index=mean_vals.index).fillna(y_raw)
                            y_upper = pd.Series(uppers, index=mean_vals.index).fillna(y_raw)

                            x = mean_vals.index
                            y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw
                            y_lower_plot = self.hybrid_transform(y_lower, thresh) if log_scale else y_lower
                            y_upper_plot = self.hybrid_transform(y_upper, thresh) if log_scale else y_upper
                        else:
                            # Fallback if no date column grouping (unlikely given logic above)
                            y_raw = df_strat[m]
                            y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw
                            x = x_vals
                            y_lower_plot = y_plot
                            y_upper_plot = y_plot

                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=y_plot,
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

                        # Add CI Band if meaningful
                        if any(y_lower_plot != y_upper_plot):
                            base_color = color_map.get(f"{strat}:{m}", "black")
                            fig.add_trace(
                                go.Scatter(
                                    x=list(x) + list(x[::-1]),
                                    y=list(y_upper_plot) + list(y_lower_plot[::-1]),
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

                add_scatter_plot(left_metrics, False)
                add_scatter_plot(right_metrics, True)

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
            all_y = []
            if left_metrics:
                all_y.extend(df_date[left_metrics].values.flatten())
            if all_y:
                yticks = self.get_ticks(all_y, thresh)
                fig.update_yaxes(
                    tickvals=self.hybrid_transform(yticks, thresh),
                    ticktext=[f"{v:.0f}" for v in yticks],
                    secondary_y=False,
                )
            all_y_r = []
            if right_metrics:
                all_y_r.extend(df_date[right_metrics].values.flatten())
            if all_y_r:
                yticks_r = self.get_ticks(all_y_r, thresh)
                fig.update_yaxes(
                    tickvals=self.hybrid_transform(yticks_r, thresh),
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
        if self.df_bet.empty:
            return go.Figure()
        share_left_y = "ON" in (share_left or [])
        share_right_y = "ON" in (share_right or [])
        log_scale = "ON" in (log_val or [])
        do_filter = "ON" in (filter_check or [])
        log_thresh = max(float(log_thresh), 1.0) if log_thresh else 10.0
        df_sess = self.df_bet[self.df_bet["session_start_date"] == session].sort_values("bet_index")
        if do_filter and filter_thresh:
            df_sess = df_sess[df_sess["bet_index"] <= float(filter_thresh)]
        df_groups = [df_sess[df_sess["session_group"] == s] if s else pd.DataFrame() for s in groups]
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
            if df_g.empty:
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

    def _add_traces_to_fig(self, fig, df, strat, metrics, colors, log_scale, thresh, is_right, y_range):
        if not metrics:
            return
        y_max_local = -inf
        for m in metrics:
            if m not in df.columns:
                continue
            y = pd.to_numeric(df[m], errors="coerce").ffill()
            y_plot = self.hybrid_transform(y, thresh) if log_scale else y
            curr_max = np.nanmax(y_plot) if len(y_plot) > 0 else 0
            y_max_local = max(y_max_local, curr_max)
            fig.add_trace(
                go.Scatter(
                    x=df["bet_index"],
                    y=y_plot,
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
            metric,
            display_mode,
            groups,
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
        ):
            if not metric or not groups or r1_start is None or r1_end is None:
                return go.Figure()
            granularity = list(self.date_files_config.keys())[0]
            df_date = self.dfs_by_date[granularity]
            if df_date.empty or self.df_date_group_col not in df_date.columns or metric not in df_date.columns:
                return go.Figure()
            base_colors = Styles.COLORS
            fig = go.Figure()

            def _label(group, l):
                return f"{group} [Range{l}]"

            data_ranges = [
                {"start": r1_start, "end": r1_end, "label": "1", "show": "ON" in (show_r1 or [])},
                {"start": r2_start, "end": r2_end, "label": "2", "show": "ON" in (show_r2 or [])},
                {"start": r3_start, "end": r3_end, "label": "3", "show": "ON" in (show_r3 or [])},
            ]
            # Only keep enabled
            data_by_range = [r for r in data_ranges if r["show"]]
            enable_clip = clip_enable is not None and "ON" in (clip_enable or [])
            cmin = clip_min if enable_clip and clip_min is not None else None
            cmax = clip_max if enable_clip and clip_max is not None else None

            if display_mode == "box":
                for gi, group in enumerate(groups):
                    color_base = base_colors[gi % len(base_colors)]
                    for ri, range_info in enumerate(data_by_range):
                        mask = (df_date[self.date_col] >= range_info["start"]) & (
                            df_date[self.date_col] <= range_info["end"]
                        )
                        df_g = df_date.loc[mask]
                        df_g = df_g[df_g[self.df_date_group_col] == group]
                        vals = df_g[metric].dropna().values
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
                        mask = (df_date[self.date_col] >= range_info["start"]) & (
                            df_date[self.date_col] <= range_info["end"]
                        )
                        df_g = df_date.loc[mask]
                        df_g = df_g[df_g[self.df_date_group_col] == group]
                        vals = df_g[metric].dropna().values
                        if enable_clip and (cmin is not None or cmax is not None):
                            vals = np.clip(
                                vals, cmin if cmin is not None else -np.inf, cmax if cmax is not None else np.inf
                            )
                        if len(vals) == 0:
                            continue
                        bar_label = _label(group, range_info["label"])
                        x_labels.append(bar_label)
                        ydata.append(np.mean(vals))
                        edata.append(np.std(vals))
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
                        for xi in range(len(bar.x)):
                            this_marker = bar.customdata[xi]
                            fig.data[i].marker.color = [m["color"] for m in mcolors_box]
                            if "pattern" in this_marker:
                                if not hasattr(fig.data[i].marker, "pattern"):
                                    fig.data[i].marker.pattern = dict(shape=[""] * len(bar.x))
                                fig.data[i].marker.pattern.shape = [
                                    m.get("pattern", {}).get("shape", "") for m in mcolors_box
                                ]
                            if "opacity" in this_marker:
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
        except:
            self.host_ip = "127.0.0.1"
        print(f"Dashboard running on http://{self.host_ip}:8050")
        self.app.run(debug=debug, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")
    config_dir = "/Users/niuxin/Documents/aws-example/src/dashboards"
    dashboard = GameStatsDashboard(config_dir)
    dashboard.run()
