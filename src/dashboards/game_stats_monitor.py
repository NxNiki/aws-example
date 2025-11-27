import logging
import os
import socket
from math import inf
from typing import Any, Dict, List, Optional, Tuple, Union

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
    def __init__(self, config_file: str, host_ip: str = "127.0.0.1"):
        self.host_ip = host_ip
        self.config = load_config(config_file)
        self.groups: List[str] = self.config["groups"]

        # --- 1. Process Bet Data (Granular) ---
        bet_data = []
        for f in self.config["stats_by_bet"]["files"]:
            bet_data.append(read_local_cache(f))

        if bet_data:
            self.df_bet = pd.concat(bet_data)
            self.df_bet["bet_index"] = pd.to_numeric(self.df_bet["bet_index"], errors="coerce")
            self.df_bet = self.df_bet.dropna(subset=["bet_index"])
            self.sessions: List[str] = sorted(self.df_bet["session_start_date"].astype(str).unique())
            exclude_bet_cols = ["session_start_date", "session_group", "bet_index"]
            self.bet_metrics: List[str] = [c for c in self.df_bet.columns if c not in exclude_bet_cols]

        # --- 2. Process Date Data (Aggregated) ---
        daily_data = []
        for f in self.config["stats_by_date"]["files"]:
            daily_data.append(read_local_cache(f))

        if daily_data:
            self.df_date = pd.concat(daily_data)

            self.df_date["bj_date"] = pd.to_datetime(self.df_date["bj_date"])

        # Exclude grouping columns for date stats
        self.date_metrics = {}
        self.date_metrics["g1"] = self.config["stats_by_date"]["group1_columns"]
        self.date_metrics["g2"] = self.config["stats_by_date"]["group2_columns"]
        self.date_metrics["g3"] = self.config["stats_by_date"]["group3_columns"]

        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self._build_main_layout()
        self._register_callbacks()

    # ------------------------------------------------------------------
    # Layout Builders
    # ------------------------------------------------------------------

    def _build_main_layout(self):
        self.app.layout = html.Div(
            [
                # --- Header / Navigator ---
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2(
                                    "Fish Hunter Analytics",
                                    style={"margin": "0", "paddingBottom": "10px", "color": "#333", "fontSize": "24px"},
                                ),
                                # Adjusted Navigator: Width 40%, Min Width 400px for portability
                                html.Div(
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
                                    ),
                                    style={"width": "40%", "minWidth": "400px"},
                                ),
                            ],
                            style={"padding": "20px 20px 0 20px"},
                        )
                    ],
                    style={"backgroundColor": "white", "borderBottom": "1px solid #eee", "marginBottom": "20px"},
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
                                    ),
                                    html.Label("Right Axis Metrics:", style={"marginTop": "10px"}),
                                    dcc.Dropdown(
                                        id=f"date-{group_id}-right-metrics",
                                        options=[{"label": m, "value": m} for m in self.date_metrics[group_id]],
                                        value=[],
                                        multi=True,
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
                create_group_panel("g1", "Date Metrics: DAU and Retention"),
                create_group_panel("g2", "Date Metrics: RTP and Profit"),
                create_group_panel("g3", "Date Metrics: Bullets and Fish"),
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
                                    options=[{"label": s, "value": s} for s in self.groups],
                                    value=self.groups[:4],  # Select up to the first 4 strategies by default
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
        # 1. Tab Switcher
        @self.app.callback(Output("page-content", "children"), Input("navigator-tabs", "value"))
        def render_content(tab):
            if tab == "tab-date":
                return self._layout_stats_by_date()
            elif tab == "tab-bet":
                return self._layout_stats_by_bet()
            return html.Div("404 Error")

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

        # 3. Stats by Date - Group Plots
        for i in range(1, 4):
            self.app.callback(
                Output(f"date-g{i}-plot", "figure"),
                [
                    Input(f"date-g{i}-left-metrics", "value"),
                    Input(f"date-g{i}-right-metrics", "value"),
                    Input(f"date-g{i}-log", "value"),
                    Input(f"date-g{i}-thresh", "value"),
                ],
            )(self.update_date_group_plot)

    # ------------------------------------------------------------------
    # Plot Logic
    # ------------------------------------------------------------------

    def update_date_group_plot(self, left_metrics, right_metrics, log_val, log_thresh):
        """
        Generates a plot using self.df_date.
        X-axis = bj_date
        Series = Grouped by the specified group column
        """
        if not left_metrics and not right_metrics:
            return go.Figure()

        log_scale = "ON" in (log_val or [])
        thresh = float(log_thresh) if log_thresh else 10.0

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        group_col = self.config["stats_by_date"]["group_col"]

        df_date = self.df_date.copy()
        if group_col == "":
            df_date["group"] = "All"
            group_col = "group"

        strategies = df_date[group_col].unique()
        color_map = {
            f"{s}:{m}": Styles.COLORS[j % len(Styles.COLORS)]
            for i, m in enumerate(left_metrics + right_metrics)
            for j, s in enumerate(strategies)
        }

        line_style_map = {
            f"{s}:{m}": Styles.LINE_SHAPE[i % len(Styles.LINE_SHAPE)]
            for i, m in enumerate(left_metrics + right_metrics)
            for j, s in enumerate(strategies)
        }

        for strat in strategies:
            # Filter by daily_group
            df_strat = df_date[df_date[group_col] == strat].sort_values("bj_date")

            if df_strat.empty:
                continue

            # UPDATED: Use 'bj_date' for X-axis
            x_vals = df_strat["bj_date"]

            # Plot Left Axis Metrics
            if left_metrics:
                for m in left_metrics:
                    if m not in df_strat.columns:
                        continue
                    y_raw = df_strat[m]
                    y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_plot,
                            name=f"{strat}:{m}",
                            mode="lines+markers",
                            line=dict(
                                color=color_map.get(f"{strat}:{m}", "black"),
                                dash=line_style_map.get(f"{strat}:{m}", "solid"),
                            ),
                            legendgroup=strat,
                        ),
                        secondary_y=False,
                    )

            # Plot Right Axis Metrics
            if right_metrics:
                for m in right_metrics:
                    if m not in df_strat.columns:
                        continue
                    y_raw = df_strat[m]
                    y_plot = self.hybrid_transform(y_raw, thresh) if log_scale else y_raw

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_plot,
                            name=f"{strat}:{m}(R)",
                            mode="lines+markers",
                            line=dict(
                                color=color_map.get(f"{strat}:{m}", "black"),
                                dash=line_style_map.get(f"{strat}:{m}", "solid"),
                            ),
                            legendgroup=strat,
                        ),
                        secondary_y=True,
                    )

        # Formatting
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=10, t=5, b=15),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.90),
            hovermode="x unified",
            xaxis_title="Date (bj_date)",
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

        # -- 5. Add Traces --
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
            if not is_right:
                # Fix for Plotly bug: When only the right y-axis has metrics and the left does not, Plotly errors with
                # "TypeError: Cannot read properties of undefined (reading 'rangemode')". Adding an invisible dummy trace
                # ensures the left y-axis is instantiated so the right axis renders without error.
                fig.add_trace(
                    go.Scatter(
                        x=np.array([0]),
                        y=np.array([0]),
                        name="",
                        mode="lines",
                        line=dict(width=0.5),
                        legendgroup=None,
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                pass
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

    # config_file = "/Users/niuxin/Documents/aws-example/src/dashboards/dashboard_config-fishhunter.yaml"
    config_file = "/Users/niuxin/Documents/aws-example/src/dashboards/dashboard_config-ss01.yaml"
    dashboard = GameStatsDashboard(config_file)
    dashboard.run()
