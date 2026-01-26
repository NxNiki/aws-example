import logging
import os
import re
import socket
from math import inf
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.s3_utils import read_local_cache
from bituslabs_ds.utils import load_config

# ==========================================
# 1. Styling & Constants
# ==========================================

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


class Styles:
    COLORS = [
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#FF7F00",
        "#984EA3",
        "#A65628",
        "#F781BF",
        "#999999",
    ]

    LINE_SHAPE = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

    NAV_TAB_SELECTED = {
        "borderTop": "3px solid #377EB8",
        "borderBottom": "1px solid white",
        "backgroundColor": "white",
        "color": "#377EB8",
        "fontWeight": "bold",
        "padding": "12px",
    }

    NAV_TAB = {"padding": "12px", "backgroundColor": "#f9f9f9", "color": "#555", "border": "1px solid #d6d6d6"}

    BOX = {
        "border": "1px solid #e0e0e0",
        "borderRadius": "8px",
        "backgroundColor": "#ffffff",
        "padding": "20px",
        "boxShadow": "0 2px 10px 0 rgba(0,0,0,0.05)",
        "marginBottom": "20px",
    }

    CONTROL_PANEL_CONTAINER = {
        "width": "20%",
        "display": "inline-block",
        "verticalAlign": "top",
        "paddingRight": "20px",
        "boxSizing": "border-box",
    }

    GRAPH_CONTAINER = {
        "width": "100%",
        "display": "inline-block",
        "verticalAlign": "top",
        "padding": "20px",
        "marginRight": "10px",
    }

    FLEX_ROW = {
        "display": "flex",
        "flexDirection": "row",
        "justifyContent": "space-between",
        "alignItems": "flex-start",
    }


# ==========================================
# 2. Dashboard Logic
# ==========================================


class GameStatsDashboard:

    def __init__(self, config_dir: str, host_ip: str = "127.0.0.1"):
        self.host_ip = host_ip
        self.config_dir = config_dir

        self.config_files = self._find_config_files()
        self._reset_state()
        self.config = load_config(self.config_files[0]["value"])

        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self._build_main_layout()
        self._register_callbacks()
        self._load_date_data()

    @property
    def date_col(self):
        return self.config["stats_by_date"]["date_col"]

    def _reset_state(self):

        self.config = {}
        self.df_date = {}
        self.df_bet = pd.DataFrame()
        self.df_date_groups = []
        self.df_bet_groups = []
        self.df_date_group_col = ""
        self.df_bet_group_col = ""
        self.bet_metrics = []
        self.date_metrics = []

    def _find_config_files(self) -> List[Dict[str, str]]:
        """Find all dashboard_config*.yaml files in the config directory."""
        config_files = []
        config_dir_path = Path(self.config_dir)

        if not config_dir_path.exists() or not config_dir_path.is_dir():
            # Fallback to current file's directory
            config_dir_path = Path(__file__).parent

        pattern = re.compile(r"^dashboard_config.*\.yaml$")
        for file_path in config_dir_path.glob("*.yaml"):
            if pattern.match(file_path.name):
                # Load config to get title
                try:
                    temp_config = load_config(str(file_path))
                    title = temp_config.get("title", file_path.stem)
                    config_files.append({"label": title, "value": str(file_path.absolute())})  # Use absolute path
                except Exception as e:
                    logger.warning(f"Could not load config {file_path}: {e}")
                    config_files.append(
                        {"label": file_path.stem, "value": str(file_path.absolute())}  # Use absolute path
                    )

        return sorted(config_files, key=lambda x: x["label"])

    def _load_data(self):
        """Load all data based on current config."""

        # --- 1. Process Bet Data (Granular) ---
        self._load_bet_data()

        # --- 2. Process Date Data (Aggregated) ---
        self._load_date_data()

    def _load_bet_data(self):
        """Load bet-level data from config."""
        if not self.df_bet.empty:
            return

        bet_data = []
        for f in self.config["stats_by_bet"]["files"]:
            if f:  # Skip empty file paths
                bet_data.append(read_local_cache(f))

        if bet_data:
            self.df_bet = pd.concat(bet_data)
            self.df_bet["bet_index"] = pd.to_numeric(self.df_bet["bet_index"], errors="coerce")
            self.df_bet = self.df_bet.dropna(subset=["bet_index"])
            self.sessions: List[str] = sorted(self.df_bet["session_start_date"].astype(str).unique())
            exclude_bet_cols = ["session_start_date", "session_group", "bet_index"]
            self.bet_metrics: List[str] = [c for c in self.df_bet.columns if c not in exclude_bet_cols]
        else:
            self.df_bet = pd.DataFrame()
            self.sessions = []
            self.bet_metrics = []

    def _load_date_data(self):
        """Load date-level data from config."""
        for gran, file_paths in self.date_files_config.items():
            daily_data = []
            if isinstance(file_paths, str):
                file_paths = [file_paths]

            for f in file_paths:
                if f:  # Skip empty file paths
                    daily_data.append(read_local_cache(f))

            if daily_data:
                self.df_date[gran] = pd.concat(daily_data)
                if self.date_col in self.df_date[gran].columns:
                    self.df_date[gran][self.date_col] = pd.to_datetime(self.df_date[gran][self.date_col])
            else:
                self.df_date[gran] = pd.DataFrame()

        # Exclude grouping columns for date stats
        self.date_metrics = {}
        self.date_metrics["g1"] = self.config["stats_by_date"]["group1_columns"]
        self.date_metrics["g2"] = self.config["stats_by_date"]["group2_columns"]
        self.date_metrics["g3"] = self.config["stats_by_date"]["group3_columns"]

    def _reload_config(self, config_file: str):
        """Reload config and data from a new config file."""
        self.config = load_config(config_file)
        self._load_date_data()

    @property
    def date_range(self) -> Tuple[Any, Any]:
        """
        Returns (min_date, max_date) from the current df_date.
        Returns (None, None) if dataframe is empty.
        """
        # Ensure we are returning Python date objects or strings, not Timestamps
        # This helps Dash serialize the data correctly.
        min_date = None
        max_date = None
        for _, df_date in self.df_date.items():
            if df_date.empty or self.date_col not in df_date.columns:
                continue
            cur_min = df_date[self.date_col].min()
            cur_max = df_date[self.date_col].max()
            if min_date is None or (cur_min is not None and cur_min < min_date):
                min_date = cur_min
            if max_date is None or (cur_max is not None and cur_max > max_date):
                max_date = cur_max

        return min_date, max_date

    @property
    def date_files_config(self):
        files_config = self.config["stats_by_date"].get("files", {})

        if isinstance(files_config, list):
            files_config = {"day": files_config}

        return files_config

    @staticmethod
    def to_rgba(color, alpha=0.2):
        try:
            return "rgba" + str(tuple(int(c * 255) for c in mcolors.to_rgb(color)) + (alpha,))
        except Exception:
            return f"rgba(0,0,0,{alpha})"

    @staticmethod
    def bootstrap_ci(arr, n_boot=1000, ci_level=0.95):
        arr = arr.dropna().values
        if len(arr) == 0:
            return np.nan, np.nan

        if min(arr) == max(arr):
            return arr[0], arr[0]

        boot_means = []
        for _ in range(n_boot):
            samples = np.random.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(samples)))
        lower = np.percentile(boot_means, (1 - ci_level) / 2 * 100)
        upper = np.percentile(boot_means, (1 + ci_level) / 2 * 100)
        return lower, upper

    # ------------------------------------------------------------------
    # Layout Builders
    # ------------------------------------------------------------------

    def _build_main_layout(self):
        self.app.layout = html.Div(
            [
                # --- Header / Navigator ---
                html.Div(
                    [
                        dcc.Dropdown(
                            id="config-dropdown",
                            options=self.config_files,
                            value=self.config_files[0]["value"],
                            clearable=False,
                            style={
                                "width": "250px",
                                "marginRight": "50px",
                                "marginLeft": "8px",
                                "marginTop": "2px",
                                "marginBottom": "0px",
                            },
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
                                    label="Stats by Bet",
                                    value="tab-bet",
                                    style=Styles.NAV_TAB,
                                    selected_style=Styles.NAV_TAB_SELECTED,
                                ),
                            ],
                            style={"width": "1050px"},
                        ),
                    ],
                    style={
                        "width": "100%",
                        "minWidth": "330px",
                        "display": "flex",
                        "flexDirection": "row",
                        "justifyContent": "flex-start",
                        "alignItems": "flex-start",
                        "borderBottom": "1px solid #eee",
                        "marginBottom": "20px",
                        "marginLeft": "0px",
                    },
                ),
                # --- Main Content Area ---
                html.Div(
                    id="page-content",
                    style={"padding": "0 20px 20px 20px", "backgroundColor": "#f4f7f6", "minHeight": "100vh"},
                ),
            ]
        )

    def _layout_stats_by_date(self):
        """Layout for the date view with 3 group plots."""

        min_date, max_date = self.date_range
        date_picker = html.Div(
            [
                html.Div(
                    [
                        html.Label("Filter Date Range:", style={"fontWeight": "bold", "marginRight": "10px"}),
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
                        # Add spacing between components
                        html.Div(style={"width": "30px"}),
                        html.Label("Select Date Granularity:", style={"fontWeight": "bold", "marginRight": "10px"}),
                        dcc.Dropdown(
                            id="date-granularity",
                            options=list(self.date_files_config.keys()),
                            value=list(self.date_files_config.keys())[0],
                            clearable=False,
                            # Ensure the dropdown has a defined width so it doesn't collapse
                            style={"width": "150px"},
                        ),
                    ],
                    # Use Flexbox to align all children horizontally and vertically ***
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "alignItems": "center",  # Vertically aligns items in the center
                        "justifyContent": "flex-start",  # Aligns items to the left
                    },
                ),
            ],
            # Outer box styling remains the same
            style={
                "marginBottom": "20px",
                "padding": "15px",
                "backgroundColor": "white",
                "borderRadius": "8px",
                "border": "1px solid #e0e0e0",
            },
        )

        # Define the download configuration once to reuse it
        download_config = {
            "toImageButtonOptions": {
                "format": "png",  # one of png, svg, jpeg, webp
                "height": 600,  # Set the height of the download
                "width": 1200,  # Set the width of the download
                "scale": 3,  # Multiply title/legend/axis fonts by 3 (High Res)
            },
            "displaylogo": False,  # Optional: Hide the "Plotly" logo
        }

        def create_group_panel(group_id, label):

            download_config["toImageButtonOptions"]["filename"] = f"fh_stats_by_date{label}"

            return html.Div(
                [
                    html.H4(label, style={"marginTop": "0", "color": "#377EB8"}),
                    html.Div(
                        [
                            # Controls
                            html.Div(
                                [
                                    html.Label("Left Axis Metrics:"),
                                    dcc.Dropdown(
                                        id=f"date-{group_id}-left-metrics",
                                        options=[{"label": m, "value": m} for m in self.date_metrics[group_id]],
                                        value=[self.date_metrics[group_id][0]] if self.date_metrics[group_id] else [],
                                        multi=True,
                                        style={"minWidth": "200px"},
                                    ),
                                    html.Label("Right Axis Metrics:", style={"marginTop": "10px"}),
                                    dcc.Dropdown(
                                        id=f"date-{group_id}-right-metrics",
                                        options=[{"label": m, "value": m} for m in self.date_metrics[group_id]],
                                        value=[],
                                        multi=True,
                                        style={"minWidth": "200px"},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id=f"date-{group_id}-log",
                                                options=[{"label": " Hybrid Log", "value": "ON"}],
                                                value=[],
                                                style={"display": "inline-block", "marginRight": "10px"},
                                            ),
                                            html.Label("Thresh: ", style={"fontSize": "0.9em"}),
                                            dcc.Input(
                                                id=f"date-{group_id}-thresh",
                                                type="number",
                                                value=10,
                                                style={"width": "60px"},
                                            ),
                                        ],
                                        style={"marginTop": "10px"},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id=f"date-{group_id}-group",
                                                options=[{"label": s, "value": s} for s in self.df_date_groups],
                                                value=self.df_date_groups[
                                                    :4
                                                ],  # Select up to the first 4 strategies by default
                                                labelStyle={"display": "block", "marginBottom": "5px"},
                                                style={"marginBottom": "10px"},
                                            )
                                        ],
                                        style={"marginTop": "10px"},
                                    ),
                                ],
                                style=Styles.CONTROL_PANEL_CONTAINER,
                            ),
                            # Graph
                            html.Div(
                                [
                                    dcc.Graph(
                                        id=f"date-{group_id}-plot", style={"height": "400px"}, config=download_config
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

    def _layout_stats_by_bet(self):
        """Layout logic for the 'Bet' tab."""
        # Add a row container with display:flex so the left and right panels
        # widths (from CONTROL_PANEL_CONTAINER and GRAPH_CONTAINER) will actually matter
        return html.Div(
            [
                # --- Left Control Panel ---
                html.Div(
                    [
                        # 1. Date & Strategy Selectors
                        html.Div(
                            [
                                html.Label("Select Date:", style={"marginBottom": "10px"}),
                                dcc.Dropdown(
                                    id="session-dropdown",
                                    options=[{"label": s, "value": s} for s in self.sessions],
                                    value=self.sessions[0] if self.sessions else None,
                                    clearable=False,
                                ),
                                html.Hr(),
                                html.Label("Compare Strategies:", style={"marginBottom": "10px"}),
                                dcc.Checklist(
                                    id="strategy-checklist",
                                    options=[{"label": s, "value": s} for s in self.df_bet_groups],
                                    value=self.df_bet_groups[:4],  # Select up to the first 4 strategies by default
                                    labelStyle={"display": "block", "marginBottom": "5px"},
                                    style={"marginBottom": "10px"},
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                        # 2. Metric Selectors
                        html.Div(
                            [
                                html.Label("Metrics (Left Axis):"),
                                dcc.Dropdown(
                                    id="metric-checklist",
                                    options=self.bet_metrics,
                                    value=[self.bet_metrics[0]] if self.bet_metrics else [],
                                    multi=True,
                                    style={"marginBottom": "10px"},
                                ),
                                html.Label("Metrics (Right Axis):"),
                                dcc.Dropdown(id="right-axis-checklist", options=self.bet_metrics, value=[], multi=True),
                            ],
                            style=Styles.BOX,
                        ),
                        # 3. Settings & Filters
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="share-left-yscale-check",
                                    options=[{"label": " Share Left Scale", "value": "ON"}],
                                    value=[],
                                ),
                                dcc.Checklist(
                                    id="share-right-yscale-check",
                                    options=[{"label": " Share Right Scale", "value": "ON"}],
                                    value=[],
                                ),
                                html.Hr(),
                                dcc.Checklist(
                                    id="log-check", options=[{"label": " Hybrid Log Scale", "value": "ON"}], value=[]
                                ),
                                html.Div(
                                    [
                                        html.Label("Linear Thresh:"),
                                        dcc.Input(id="linear-thresh", type="number", value=10, style={"width": "100%"}),
                                    ],
                                    style={"marginTop": "5px"},
                                ),
                                html.Hr(),
                                dcc.Checklist(
                                    id="filter-check",
                                    options=[{"label": "Max Number of Bets", "value": "ON"}],
                                    value=["ON"],
                                ),
                                dcc.Input(
                                    id="filter-thresh",
                                    type="number",
                                    value=5000,
                                    style={"width": "100%", "marginTop": "5px"},
                                ),
                            ],
                            style=Styles.BOX,
                        ),
                    ],
                    style=Styles.CONTROL_PANEL_CONTAINER,
                ),
                # --- Right Graph Panel ---
                html.Div([dcc.Graph(id="combined-plot", style={"height": "950px"})], style=Styles.GRAPH_CONTAINER),
            ],
            style={"display": "flex", "width": "100%"},
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self):
        # 0. Config Dropdown - Reload data when config changes
        @self.app.callback(
            Output("page-content", "children", allow_duplicate=True),
            Input("config-dropdown", "value"),
            State("navigator-tabs", "value"),
            prevent_initial_call=True,
        )
        def update_config(selected_config, current_tab):
            """Reload data when config changes."""
            self._reset_state()
            self._reload_config(selected_config)
            return self._render_tab_content(current_tab)

        # 1. Tab Switcher - only updates page content
        @self.app.callback(
            Output("page-content", "children"),
            Input("navigator-tabs", "value"),
            prevent_initial_call=False,
        )
        def render_content(tab):
            return self._render_tab_content(tab)

        # 2. Stats by Bet - Main Plot
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

        # 3. Stats by Date - Group Plots (always g1, g2, g3)
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
            )(self.update_date_group_plot)

    def _render_tab_content(self, current_tab):

        if current_tab == "tab-date":
            default_gran = list(self.date_files_config.keys())[0]
            if self.config["stats_by_date"]["group_col"]:
                self.df_date_group_col = self.config["stats_by_date"]["group_col"]
                self.df_date_groups: List[str] = self.df_date[default_gran][self.df_date_group_col].unique().tolist()
            return self._layout_stats_by_date()

        elif current_tab == "tab-bet":
            self._load_bet_data()
            if self.config["stats_by_bet"]["group_col"]:
                self.df_bet_group_col = self.config["stats_by_bet"]["group_col"]
                self.df_bet_groups: List[str] = self.df_bet[self.df_bet_group_col].unique().tolist()
            return self._layout_stats_by_bet()
        else:
            return html.Div("404 Error")

    # ------------------------------------------------------------------
    # Plot Logic
    # ------------------------------------------------------------------

    def update_date_group_plot(
        self, left_metrics, right_metrics, log_val, log_thresh, groups, date_granularity, start_date, end_date
    ):
        """
        Generates a plot using self.df_date.
        X-axis = date
        Series = Grouped by the specified group column
        """
        if not left_metrics and not right_metrics:
            return go.Figure()

        log_scale = "ON" in (log_val or [])
        thresh = float(log_thresh) if log_thresh else 10.0

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        df_date = self.df_date[date_granularity]
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

        def add_scatter_plot(metrics, strat, df_strat, secondary_y):
            x_vals = df_strat[self.date_col]

            for m in metrics:
                if m not in df_strat.columns:
                    continue

                # Compute mean and confidence interval for each date
                if self.date_col in df_strat.columns:
                    grouped = df_strat.groupby(self.date_col)[m]
                    mean_vals = grouped.mean()

                    # Bootstrapped 95% confidence interval
                    y_raw = mean_vals
                    y_lower = y_raw.copy()
                    y_upper = y_raw.copy()
                    for date_idx in mean_vals.index:
                        arr = grouped.get_group(date_idx)
                        lower, upper = self.bootstrap_ci(arr)
                        y_lower.loc[date_idx] = lower
                        y_upper.loc[date_idx] = upper
                    y_lower = y_lower.fillna(y_raw)
                    y_upper = y_upper.fillna(y_raw)

                    x = mean_vals.index
                    y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw
                    y_lower_plot = self.hybrid_transform(y_lower, thresh) if log_scale else y_lower
                    y_upper_plot = self.hybrid_transform(y_upper, thresh) if log_scale else y_upper
                else:
                    # Fallback: original handling if for some reason grouping isn't possible
                    y_raw = df_strat[m]
                    y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw
                    x = x_vals
                    y_lower_plot = y_plot
                    y_upper_plot = y_plot

                # Plot main line (mean)
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

                # Plot confidence interval as a filled area
                if any((y_lower_plot != y_upper_plot)):
                    base_color = color_map.get(f"{strat}:{m}", "black")
                    fig.add_trace(
                        go.Scatter(
                            x=list(x) + list(x[::-1]),
                            y=list(y_upper_plot) + list(y_lower_plot[::-1]),
                            fill="toself",
                            fillcolor=self.to_rgba(base_color, 0.1),
                            line=dict(color="rgba(255,255,255,0)"),  # No border
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=strat,
                            name=f"{strat}:{m} 95% CI",
                        ),
                        secondary_y=secondary_y,
                    )

        for strat in groups:

            df_strat = df_date[df_date[group_col] == strat].sort_values(self.date_col)
            if df_strat.empty:
                continue

            add_scatter_plot(left_metrics, strat, df_strat, False)
            add_scatter_plot(right_metrics, strat, df_strat, True)

        # Formatting
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=10, t=5, b=15),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.90),
            hovermode="x unified",
            xaxis_title="Date (Beijing)",
            font=dict(size=16),  # Increase global font size for the plot
            xaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis2=dict(title_font=dict(size=18), tickfont=dict(size=15)),  # In case of secondary axis
        )

        if log_scale:
            # Smart Ticks for Date view
            all_y = []
            if left_metrics:
                all_y.extend(self.df_date[left_metrics].values.flatten())
            if all_y:
                yticks = self.get_ticks(all_y, thresh)
                fig.update_yaxes(
                    tickvals=self.hybrid_transform(yticks, thresh),
                    ticktext=[f"{v:.0f}" for v in yticks],
                    secondary_y=False,
                )

            all_y_r = []
            if right_metrics:
                all_y_r.extend(self.df_date[right_metrics].values.flatten())
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
        strategies,
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

        # -- 1. Setup Flags --
        share_left_y = "ON" in (share_left or [])
        share_right_y = "ON" in (share_right or [])
        log_scale = "ON" in (log_val or [])
        do_filter = "ON" in (filter_check or [])
        log_thresh = max(float(log_thresh), 1.0) if log_thresh else 10.0

        # -- 2. Filter Data --
        df_sess = self.df_bet[self.df_bet["session_start_date"] == session].sort_values("bet_index")
        if do_filter and filter_thresh:
            df_sess = df_sess[df_sess["bet_index"] <= float(filter_thresh)]

        df_groups = [df_sess[df_sess["session_group"] == s] if s else pd.DataFrame() for s in strategies]

        # -- 3. Init Figure --
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Metrics by Strategies: {session}"],
            specs=[[{"secondary_y": True}]],
            vertical_spacing=0.05,
            x_title="Bet Index",
        )

        # -- 4. Y Ranges --
        left_range = self.compute_axis_range(df_groups, left_metrics, log_scale, log_thresh) if share_left_y else None
        right_range = (
            self.compute_axis_range(df_groups, right_metrics, log_scale, log_thresh) if share_right_y else None
        )

        # -- 5. Add dummy trace for left axis if only right metrics are selected
        if not left_metrics and right_metrics:
            # Add a single invisible dummy trace to initialize the left y-axis
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

        # -- 6. Add Traces --
        for i, df_g in enumerate(df_groups):
            if df_g.empty:
                continue

            # Left Axis
            metric_colors = {
                f"{s}:{m}": Styles.COLORS[i + j * len(strategies) % len(Styles.COLORS)]
                for i, s in enumerate(strategies)
                for j, m in enumerate(left_metrics)
            }
            self._add_traces_to_fig(
                fig,
                df_g,
                strategies[i],
                left_metrics,
                metric_colors,
                log_scale,
                log_thresh,
                is_right=False,
                y_range=left_range,
            )
            # Right Axis
            metric_colors = {
                f"{s}:{m}": Styles.COLORS[i + j * len(strategies) % len(Styles.COLORS)]
                for i, s in enumerate(strategies)
                for j, m in enumerate(right_metrics)
            }
            self._add_traces_to_fig(
                fig,
                df_g,
                strategies[i],
                right_metrics,
                metric_colors,
                log_scale,
                log_thresh,
                is_right=True,
                y_range=right_range,
            )

        # Move legend closer by updating the layout just after trace is added
        fig.update_layout(
            legend=dict(
                orientation="h",
                traceorder="normal",
                yanchor="top",
                y=1.10,  # push legend further above plot area
                xanchor="right",
                x=0.95,  # center legend over plot
                borderwidth=0,
                font=dict(size=20),
            ),
            font=dict(size=16),
            xaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis=dict(title_font=dict(size=18), tickfont=dict(size=15)),
            yaxis2=dict(title_font=dict(size=18), tickfont=dict(size=15)),
        )

        # -- 6. Ticks & Layout --
        if log_scale:
            self.update_log_ticks(fig, df_groups, left_metrics, right_metrics, log_thresh)

        fig.update_layout(height=950, template="plotly_white", margin=dict(l=60, r=60, t=50, b=50))
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=1, col=1)

        return fig

    def _add_traces_to_fig(self, fig, df, strat, metrics, colors, log_scale, thresh, is_right, y_range):
        if not metrics:
            # Don't add dummy traces here - handle it at the plot level
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

    # ------------------------------------------------------------------
    # Math Helpers
    # ------------------------------------------------------------------

    def compute_axis_range(self, dfs, metrics, log_scale, thresh):
        if not metrics:
            return None
        vals = []
        for df in dfs:
            for m in metrics:
                if m in df.columns:
                    y = pd.to_numeric(df[m], errors="coerce").dropna()
                    vals.append(y)
        if not vals:
            return None
        all_v = pd.concat(vals)
        if log_scale:
            all_v = self.hybrid_transform(all_v, thresh)
        return (0, all_v.max())

    def update_log_ticks(self, fig, dfs, left_ms, right_ms, thresh):
        all_left = []
        all_right = []
        for df in dfs:
            if left_ms:
                all_left.extend(df[left_ms].values.flatten())
            if right_ms:
                all_right.extend(df[right_ms].values.flatten())

        def set_ticks(data, is_right):
            if not len(data):
                return
            ticks = self.get_ticks(data, thresh)
            tick_vals = self.hybrid_transform(ticks, thresh)
            tick_text = [f"{int(t)}" if t >= 1 else f"{t:.2f}" for t in ticks]
            fig.update_yaxes(tickvals=tick_vals, ticktext=tick_text, secondary_y=is_right, row=1, col=1)

        set_ticks(all_left, False)
        set_ticks(all_right, True)

    @staticmethod
    def hybrid_transform(y, thresh):
        y = np.array(y, dtype=float)
        eps = 1e-9
        return np.where(y <= thresh, y, thresh + np.log10(y + eps))

    @staticmethod
    def get_ticks(y_values, thresh, nticks=10):
        y_values = np.array(y_values, dtype=float)
        y_values = y_values[~np.isnan(y_values)]
        if len(y_values) == 0:
            return np.array([0, thresh])
        y_max = y_values.max()
        lin_ticks = np.linspace(0, thresh, num=4)
        if y_max > thresh:
            start_exp = np.ceil(np.log10(thresh))
            end_exp = np.ceil(np.log10(y_max))
            log_ticks = np.logspace(start_exp, end_exp, num=int(end_exp - start_exp) + 1)
            combined = np.unique(np.concatenate([lin_ticks, log_ticks]))
            return combined[combined <= y_max * 1.1]
        return lin_ticks

    def run(self, debug=True):
        try:
            # specific trick to get the reliable IP address
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
