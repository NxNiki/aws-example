from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from bituslabs_ds.s3_utils import read_local_cache

box_style: Dict[str, Any] = {
    "border": "2px solid #377EB8",
    "borderRadius": "15px",
    "backgroundColor": "#F9F9F9",
    "padding": "18px 18px 15px 18px",
    "boxShadow": "0 4px 24px 0 rgba(55,126,184,0.12)",
    "marginBottom": "26px",
    "marginTop": "6px",
}

check_box_style: Dict[str, Any] = {
    "flex": "1 1 0",
    "marginRight": "3%",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "flex-start",
    "minWidth": "0",
}


class FishHunterDashboard:
    def __init__(self, df: pd.DataFrame, host_ip: str = "192.168.2.115"):
        self.df = df.copy()
        # Ensure bet_index is numeric
        self.df["bet_index"] = pd.to_numeric(self.df["bet_index"], errors="coerce")
        self.df = self.df.dropna(subset=["bet_index"])
        self.sessions: List[Any] = sorted(self.df["session_start_date"].unique())
        self.strategies: List[str] = ["BOOST", "DYNA_RTP", "DEFAULT", "PA", "Null"]
        self.metrics: List[str] = [
            c for c in self.df.columns if c not in ["session_start_date", "session_group", "bet_index"]
        ]
        self.colors_hex: List[str] = [
            "#E41A1C",
            "#377EB8",
            "#4DAF4A",
            "#FF7F00",
            "#984EA3",
            "#A65628",
            "#F781BF",
            "#999999",
        ]
        self.host_ip = host_ip
        self.app = Dash(__name__)
        self._build_layout()
        self._register_callbacks()

    def _build_layout(self):
        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        self._date_strategy_panel(),
                        self._metric_right_panel(),
                        self._log_filter_panel(),
                    ],
                    style={
                        "width": "15%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingRight": "10px",
                        "paddingLeft": "20px",
                        "paddingTop": "20px",
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="combined-plot",
                            style={
                                "height": "950px",
                                "marginBottom": "10px",
                                "width": "100%",
                            },
                        )
                    ],
                    style={
                        "width": "80%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "marginLeft": "0",
                    },
                ),
            ]
        )

    def _date_strategy_panel(self):
        # Separated out for readability/OOP
        s = self.strategies
        return html.Div(
            [
                html.Label("Date:"),
                dcc.Dropdown(
                    id="session-dropdown",
                    options=[{"label": s, "value": s} for s in self.sessions],
                    value=self.sessions[0],
                    clearable=False,
                    style={"marginTop": "10px", "marginBottom": "15px"},
                ),
                html.Label("Strategy A:", style={"marginTop": "15px"}),
                dcc.Dropdown(
                    id="strategy-a-dropdown",
                    options=[{"label": a, "value": a} for a in s],
                    value=s[0],
                    clearable=False,
                    style={"marginTop": "10px", "marginBottom": "15px"},
                ),
                html.Label("Strategy B:", style={"marginTop": "15px"}),
                dcc.Dropdown(
                    id="strategy-b-dropdown",
                    options=[{"label": b, "value": b} for b in s],
                    value=s[1] if len(s) > 1 else s[0],
                    clearable=False,
                    style={"marginTop": "10px", "marginBottom": "15px"},
                ),
                html.Label("Strategy C:", style={"marginTop": "15px"}),
                dcc.Dropdown(
                    id="strategy-c-dropdown",
                    options=[{"label": c, "value": c} for c in s],
                    value=s[2] if len(s) > 2 else s[0],
                    clearable=False,
                    style={"marginTop": "10px", "marginBottom": "15px"},
                ),
                html.Label("Strategy D:", style={"marginTop": "15px"}),
                dcc.Dropdown(
                    id="strategy-d-dropdown",
                    options=[{"label": d, "value": d} for d in s],
                    value=s[3] if len(s) > 3 else s[0],
                    clearable=False,
                    style={"marginTop": "10px", "marginBottom": "15px"},
                ),
            ],
            style=box_style,
        )

    def _metric_right_panel(self):
        metrics = self.metrics
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Metrics:", style={"marginTop": "15px"}),
                                html.Div(
                                    dcc.Checklist(
                                        id="metric-checklist",
                                        options=[{"label": m, "value": m} for m in metrics],
                                        value=[metrics[0]] if metrics else [],
                                        style={
                                            "marginTop": "10px",
                                            "marginBottom": "15px",
                                            "columnCount": 1,
                                        },
                                    ),
                                ),
                            ],
                            style=check_box_style,
                        ),
                        html.Div(
                            [
                                html.Label("Plot on Right Axis", style={"marginTop": "15px", "display": "block"}),
                                html.Div(
                                    dcc.Checklist(
                                        id="right-axis-checklist",
                                        options=[{"label": "", "value": m} for m in metrics],
                                        value=[],
                                        style={
                                            "marginTop": "10px",
                                            "marginBottom": "15px",
                                            "columnCount": 1,
                                        },
                                    ),
                                ),
                            ],
                            style=check_box_style,
                        ),
                    ],
                    style={
                        "width": "100%",
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "flex-start",  # << key to align tops
                        "maxHeight": "40vh",
                        "overflowY": "auto",
                        "marginBottom": "10px",
                        "border": "1px solid #eee",
                        "paddingRight": "10px",  # To make room for scrollbar
                    },
                ),
            ],
            style=box_style,
        )

    def _log_filter_panel(self):
        # Split left/right for log and session filtering
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Checklist(
                            id="share-left-yscale-check",
                            options=[{"label": "Share left y scale", "value": "ON"}],
                            value=[],
                            style={"marginTop": "10px", "marginBottom": "0px", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="share-right-yscale-check",
                            options=[{"label": "Share right y scale", "value": "ON"}],
                            value=[],
                            style={"marginTop": "10px", "marginBottom": "0px", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="log-check",
                            options=[{"label": "Hybrid log", "value": "ON"}],
                            value=[],
                            style={"marginTop": "10px", "marginBottom": "0px"},
                        ),
                        html.Label("Linear threshold:", style={"marginTop": "10px", "marginRight": "5px"}),
                        dcc.Input(
                            id="linear-thresh",
                            type="number",
                            value=10,
                            min=0,
                            style={
                                "width": "70px",
                                "marginTop": "3px",
                                "marginBottom": "5px",
                                "display": "inline-block",
                            },
                        ),
                    ],
                    style={
                        **box_style,
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "width": "47%",
                        "marginRight": "2%",
                    },
                ),
                html.Div(
                    [
                        dcc.Checklist(
                            id="filter-check",
                            options=[{"label": "Max sessions", "value": "ON"}],
                            value=["ON"],
                            style={"marginTop": "10px", "marginBottom": "5px"},
                        ),
                        html.Label("Num Sessions:", style={"marginTop": "15px", "marginRight": "5px"}),
                        dcc.Input(
                            id="filter-thresh",
                            type="number",
                            value=5000,
                            min=0,
                            style={
                                "width": "70px",
                                "marginTop": "10px",
                                "marginBottom": "15px",
                                "display": "inline-block",
                            },
                        ),
                    ],
                    style={**box_style, "display": "inline-block", "verticalAlign": "top", "width": "47%"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "space-between",
                "width": "100%",
            },
        )

    def _register_callbacks(self):
        # This callback enforces that a value can only exist in ONE list at a time.
        self.app.callback(
            [Output("metric-checklist", "value"), Output("right-axis-checklist", "value")],
            [Input("metric-checklist", "value"), Input("right-axis-checklist", "value")],
            prevent_initial_call=True,
        )(self.enforce_mutual_exclusivity)

        self.app.callback(
            Output("combined-plot", "figure"),
            Input("session-dropdown", "value"),
            Input("strategy-a-dropdown", "value"),
            Input("strategy-b-dropdown", "value"),
            Input("strategy-c-dropdown", "value"),
            Input("strategy-d-dropdown", "value"),
            Input("metric-checklist", "value"),
            Input("right-axis-checklist", "value"),
            Input("share-left-yscale-check", "value"),
            Input("share-right-yscale-check", "value"),
            Input("log-check", "value"),
            Input("linear-thresh", "value"),
            Input("filter-check", "value"),
            Input("filter-thresh", "value"),
        )(self.update_plot)

    # This callback enforces that a value can only exist in ONE list at a time.
    def enforce_mutual_exclusivity(
        self, left_values: Optional[List[str]], right_values: Optional[List[str]]
    ) -> Tuple[Any, Any]:
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        left_values = left_values or []
        right_values = right_values or []

        if trigger_id == "metric-checklist":
            new_right = [item for item in right_values if item not in left_values]
            return no_update, new_right
        elif trigger_id == "right-axis-checklist":
            new_left = [item for item in left_values if item not in right_values]
            return new_left, no_update
        return no_update, no_update

    def update_plot(
        self,
        session: Any,
        strat_a: Optional[str],
        strat_b: Optional[str],
        strat_c: Optional[str],
        strat_d: Optional[str],
        left_metrics: List[str],
        right_metrics: List[str],
        share_left_y_check: str,
        share_right_y_check: str,
        log_scale_check: str,
        log_thresh: Union[float, int],
        filter_bets_check: str,
        filter_thresh: Union[float, int],
    ) -> go.Figure:

        share_left_y = "ON" in share_left_y_check
        share_right_y = "ON" in share_right_y_check
        log_scale = "ON" in log_scale_check
        filter_bets = "ON" in filter_bets_check

        df_sess: pd.DataFrame = self.df[self.df["session_start_date"] == session].sort_values("bet_index")
        if filter_bets:
            df_sess = df_sess[(df_sess["bet_index"] <= filter_thresh)]

        # Grab the dataframes for the four strategies
        strategies = [strat_a, strat_b, strat_c, strat_d]
        df_session: List[pd.DataFrame] = [
            df_sess[df_sess["session_group"] == strat] if strat else pd.DataFrame() for strat in strategies
        ]

        log_thresh = max(float(log_thresh), 1.0)

        fig: go.Figure = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            subplot_titles=tuple(strategies),
            specs=[[{"secondary_y": True}] for _ in range(4)],
        )

        metrics_all = (left_metrics or []) + (right_metrics or [])
        metric_colors: Dict[str, str] = {
            metric: self.colors_hex[i % len(self.colors_hex)] for i, metric in enumerate(metrics_all)
        }

        # Compute axis y-range globally for all four rows/strategies, so they share y range
        if share_right_y:
            right_axis_range = self.compute_axis_range(df_session, right_metrics, log_scale, log_thresh)
        else:
            right_axis_range = None

        if share_left_y:
            left_axis_range = self.compute_axis_range(df_session, left_metrics, log_scale, log_thresh)
        else:
            left_axis_range = None

        for row, df_group in enumerate(df_session, 1):
            if not left_metrics and not df_group.empty:
                # Add dummy invisible trace if left metrics is empty but right is plotted
                self.add_dummy_primary_yaxis_trace(fig, df_group, row)
            else:
                self.add_metric_traces(
                    fig,
                    df_group,
                    row,
                    left_metrics,
                    metric_colors,
                    log_scale,
                    log_thresh,
                    right_axis=False,
                    y_range=left_axis_range,
                )
                fig.update_yaxes(title="", showticklabels=True, showgrid=True, secondary_y=False, row=row, col=1)

            # Add right axis metrics (with shared range)
            if right_metrics:
                self.add_metric_traces(
                    fig,
                    df_group,
                    row,
                    right_metrics,
                    metric_colors,
                    log_scale,
                    log_thresh,
                    right_axis=True,
                    y_range=right_axis_range,
                )
                fig.update_yaxes(title="", showticklabels=True, showgrid=True, secondary_y=True, row=row, col=1)

        # Apply unified log ticks if log scaling is enabled
        if log_scale:
            self.update_log_ticks(fig, df_session, left_metrics, right_metrics, log_thresh)

        fig.add_annotation(
            text=f"Session Date: {session}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.08,
            showarrow=False,
            font=dict(size=20, color="black"),
            xanchor="center",
            yanchor="top",
        )

        fig.update_layout(
            height=950,
            template="plotly_white",
            margin=dict(l=70, r=50, t=80, b=50),
        )

        range_slider_row: int = max(i + 1 for i in range(len(df_session)) if not df_session[i].empty)
        fig.update_xaxes(
            title_text="bet_index",
            rangeslider=dict(visible=True, thickness=0.05, bgcolor="white"),
            rangeslider_visible=True,
            row=range_slider_row,
            col=1,
        )
        for r in range(1, range_slider_row):
            fig.update_xaxes(title_text="bet_index", rangeslider=dict(visible=False), row=r, col=1)

        return fig

    def compute_axis_range(
        self, dfs: List[pd.DataFrame], metrics: List[str], log_check: bool, log_thresh: float
    ) -> Optional[Tuple[float, float]]:
        """Compute shared min and max across all metrics."""
        ys = []
        for df in dfs:
            if not df.empty and metrics:
                for metric in metrics:
                    if metric in df.columns:
                        y = pd.to_numeric(df[metric], errors="coerce").ffill()
                        y_plot = self.hybrid_transform(y, log_thresh) if log_check else y
                        vals = y_plot.dropna() if hasattr(y_plot, "dropna") else y_plot[~np.isnan(y_plot)]
                        if len(vals) > 0:
                            ys.append(np.asarray(vals))
        if ys:
            all_y = np.concatenate(ys)
            if len(all_y) > 0:
                y_min = float(np.nanmin(all_y))
                y_max = float(np.nanmax(all_y))
                if log_check:
                    y_min = max(y_min, 1e-9)
                return y_min, y_max
        return None

    def add_metric_traces(
        self,
        fig: go.Figure,
        df_group: pd.DataFrame,
        row: int,
        metrics: List[str],
        metric_colors: Dict[str, str],
        log_check: bool,
        log_thresh: float,
        right_axis: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        if df_group.empty or not metrics:
            return

        y_max = 0
        for metric in metrics:
            if metric not in df_group.columns:
                continue
            y = pd.to_numeric(df_group[metric], errors="coerce").ffill()
            y_plot = self.hybrid_transform(y, log_thresh) if log_check else y
            y_max = max(y_max, max(y_plot))
            fig.add_trace(
                go.Scatter(
                    x=df_group["bet_index"],
                    y=y_plot,
                    name=f"{metric} (right)" if right_axis else metric,
                    mode="lines",
                    line=dict(
                        width=2,
                        color=metric_colors.get(metric, "#000000"),
                        dash="dash" if right_axis else None,
                    ),
                    opacity=0.6,
                    legendgroup=metric,
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
                secondary_y=right_axis,
            )
        # Set shared axis range if y_range provided
        if y_range is not None:
            y_range = (0, y_range[1])
        else:
            y_range = (0, y_max)
        fig.update_yaxes(range=y_range, row=row, col=1, secondary_y=right_axis)

    def add_dummy_primary_yaxis_trace(self, fig, df_group, row):
        """Add invisible dummy trace to primary y-axis when no metrics are plotted."""
        fig.add_trace(
            go.Scatter(
                x=[df_group["bet_index"].iloc[0]],
                y=[0],
                mode="lines",
                marker=dict(opacity=0),  # Invisible
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(title="", showticklabels=False, showgrid=False, secondary_y=False, row=row, col=1)

    def update_log_ticks(
        self,
        fig: go.Figure,
        df_session: List[pd.DataFrame],
        sel_metrics: List[str],
        right_axis_metrics: List[str],
        log_thresh: float,
    ):
        """Set tick values (and labels) for log axis on both left/right axis for all four rows."""
        for i, df_sess_row in enumerate(df_session):
            if not df_sess_row.empty:
                # Left axis ticks
                if sel_metrics:
                    yticks = self.get_ticks(df_sess_row[sel_metrics].values.flatten(), log_thresh)
                    fig.update_yaxes(
                        tickvals=self.hybrid_transform(yticks, log_thresh),
                        ticktext=[f"{v:.0f}" for v in yticks],
                        row=i + 1,
                        col=1,
                        secondary_y=False,
                    )
                # Right axis ticks (if any)
                if right_axis_metrics:
                    yticks = self.get_ticks(df_sess_row[right_axis_metrics].values.flatten(), log_thresh)
                    fig.update_yaxes(
                        tickvals=self.hybrid_transform(yticks, log_thresh),
                        ticktext=[f"{v:.0f}" for v in yticks],
                        row=i + 1,
                        col=1,
                        secondary_y=True,
                    )

    @staticmethod
    def hybrid_transform(y: Union[np.ndarray, List[float]], thresh: float) -> np.ndarray:
        y = np.array(y, dtype=float)
        eps = 1e-9
        return np.where(y <= thresh, y, thresh + np.log10(y + eps))

    @staticmethod
    def get_ticks(y_values: Union[np.ndarray, List[float]], thresh: float, nticks: int = 10) -> np.ndarray:
        y_values = np.array(y_values, dtype=float)
        y_min, y_max = y_values.min(), y_values.max()
        linear = np.linspace(min(y_min, thresh), thresh, max(2, nticks // 3))
        log = np.logspace(np.log10(thresh + 1), np.log10(max(y_max, thresh * 10)), max(2, nticks // 3 * 2))
        return np.unique(np.concatenate([linear, log]))

    def run(self, debug=True):
        print(f"Your dashboard is available at: http://{self.host_ip}:8050")
        self.app.run(debug=debug, host="0.0.0.0", port=8050)


if __name__ == "__main__":

    data_file: str = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/bullet_stats_by_index.parquet"
    data_file2: str = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/bullet_stats_by_index_pa.parquet"

    data: pd.DataFrame = read_local_cache(data_file)
    data2: pd.DataFrame = read_local_cache(data_file2)

    data = pd.concat([data, data2])

    # OOP dashboard launch
    dashboard = FishHunterDashboard(data)
    dashboard.run()
