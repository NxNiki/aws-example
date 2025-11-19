import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from bituslabs_ds.s3_utils import read_local_cache

box_style = {
    "border": "2px solid #377EB8",
    "borderRadius": "15px",
    "backgroundColor": "#F9F9F9",
    "padding": "18px 18px 15px 18px",
    "boxShadow": "0 4px 24px 0 rgba(55,126,184,0.12)",
    "marginBottom": "26px",
    "marginTop": "6px",
}

check_box_style = {
    "flex": "1 1 0",
    "marginRight": "3%",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "flex-start",
    "minWidth": "0",
}


def create_dashboard(df):
    # Ensure bet_index is numeric
    df["bet_index"] = pd.to_numeric(df["bet_index"], errors="coerce")
    df = df.dropna(subset=["bet_index"])

    sessions = sorted(df["session_start_date"].unique())
    strategies = ["BOOST", "DYNA_RTP", "DEFAULT", "PA"]
    metrics = [c for c in df.columns if c not in ["session_start_date", "session_group", "bet_index"]]

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Date:"),
                            dcc.Dropdown(
                                id="session-dropdown",
                                options=[{"label": s, "value": s} for s in sessions],
                                value=sessions[0],
                                clearable=False,
                                style={"marginTop": "10px", "marginBottom": "15px"},
                            ),
                            html.Label("Strategy A:", style={"marginTop": "15px"}),
                            dcc.Dropdown(
                                id="strategy-a-dropdown",
                                options=[{"label": s, "value": s} for s in strategies],
                                value=strategies[0],
                                clearable=False,
                                style={"marginTop": "10px", "marginBottom": "15px"},
                            ),
                            html.Label("Strategy B:", style={"marginTop": "15px"}),
                            dcc.Dropdown(
                                id="strategy-b-dropdown",
                                options=[{"label": s, "value": s} for s in strategies],
                                value=strategies[1] if len(strategies) > 1 else strategies[0],
                                clearable=False,
                                style={"marginTop": "10px", "marginBottom": "15px"},
                            ),
                            html.Label("Strategy C:", style={"marginTop": "15px"}),
                            dcc.Dropdown(
                                id="strategy-c-dropdown",
                                options=[{"label": s, "value": s} for s in strategies],
                                value=strategies[2] if len(strategies) > 1 else strategies[0],
                                clearable=False,
                                style={"marginTop": "10px", "marginBottom": "15px"},
                            ),
                            html.Label("Strategy D:", style={"marginTop": "15px"}),
                            dcc.Dropdown(
                                id="strategy-d-dropdown",
                                options=[{"label": s, "value": s} for s in strategies],
                                value=strategies[3] if len(strategies) > 1 else strategies[0],
                                clearable=False,
                                style={"marginTop": "10px", "marginBottom": "15px"},
                            ),
                        ],
                        style=box_style,
                    ),
                    html.Div(
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
                                            html.Label(
                                                "Plot on Right Axis", style={"marginTop": "15px", "display": "block"}
                                            ),
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
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="log-check",
                                        options=[{"label": "Hybrid log", "value": True}],
                                        value=[],
                                        style={"marginTop": "10px", "marginBottom": "5px"},
                                    ),
                                    html.Label("Linear threshold:", style={"marginTop": "15px", "marginRight": "5px"}),
                                    dcc.Input(
                                        id="linear-thresh",
                                        type="number",
                                        value=10,
                                        min=0,
                                        style={
                                            "width": "70px",
                                            "marginTop": "10px",
                                            "marginBottom": "15px",
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
                                        options=[{"label": "Max sessions", "value": True}],
                                        value=[],
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
                    ),
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

    def hybrid_transform(y, thresh):
        y = np.array(y, dtype=float)
        eps = 1e-9
        return np.where(y <= thresh, y, thresh + np.log10(y + eps))

    def get_ticks(y_values, thresh, nticks=10):
        y_values = np.array(y_values, dtype=float)
        y_min, y_max = y_values.min(), y_values.max()
        linear = np.linspace(min(y_min, thresh), thresh, max(2, nticks // 3))
        log = np.logspace(np.log10(thresh + 1), np.log10(max(y_max, thresh * 10)), max(2, nticks // 3 * 2))
        return np.unique(np.concatenate([linear, log]))

    # This callback enforces that a value can only exist in ONE list at a time.
    @app.callback(
        [Output("metric-checklist", "value"), Output("right-axis-checklist", "value")],
        [Input("metric-checklist", "value"), Input("right-axis-checklist", "value")],
        prevent_initial_call=True,
    )
    def enforce_mutual_exclusivity(left_values, right_values):
        ctx = callback_context

        # If no trigger (shouldn't happen due to prevent_initial_call), return current
        if not ctx.triggered:
            return no_update, no_update

        # Determine which list was just clicked
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Handle None types for safe list operations
        left_values = left_values or []
        right_values = right_values or []

        if trigger_id == "metric-checklist":
            # User clicked the LEFT list.
            # Remove any items selected on the left from the right list.
            new_right = [item for item in right_values if item not in left_values]
            return no_update, new_right

        elif trigger_id == "right-axis-checklist":
            # User clicked the RIGHT list.
            # Remove any items selected on the right from the left list.
            new_left = [item for item in left_values if item not in right_values]
            return new_left, no_update

        return no_update, no_update

    @app.callback(
        Output("combined-plot", "figure"),
        Input("session-dropdown", "value"),
        Input("strategy-a-dropdown", "value"),
        Input("strategy-b-dropdown", "value"),
        Input("strategy-c-dropdown", "value"),
        Input("strategy-d-dropdown", "value"),
        Input("metric-checklist", "value"),
        Input("right-axis-checklist", "value"),
        Input("log-check", "value"),
        Input("linear-thresh", "value"),
        Input("filter-check", "value"),
        Input("filter-thresh", "value"),
    )
    def update_plot(
        session,
        strat_a,
        strat_b,
        strat_c,
        strat_d,
        sel_metrics,
        right_axis_metrics,
        log_check,
        log_thresh,
        filter_check,
        filter_thresh,
    ):

        df_sess = df[df["session_start_date"] == session].sort_values("bet_index")
        if filter_check:
            df_sess = df_sess[(df_sess["bet_index"] <= filter_thresh)]

        df_session = [pd.DataFrame() for _ in range(4)]
        for i, strat in enumerate([strat_a, strat_b, strat_c, strat_d]):
            if strat:
                df_session[i] = df_sess[df_sess["session_group"] == strat]

        log_thresh = max(float(log_thresh), 1)

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(strat_a, strat_b, strat_c, strat_d),
            # Specify secondary_y for each row
            specs=[[{"secondary_y": True}] for _ in range(4)],
        )

        # Define a fixed color map for metrics
        colors_hex = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3", "#A65628", "#F781BF", "#999999"]
        metrics_all = sel_metrics + right_axis_metrics
        metric_colors = {metric: colors_hex[i % len(colors_hex)] for i, metric in enumerate(metrics_all)}

        def add_traces(df_group, row, metrics, right_axis=False, log_check=False):
            if df_group.empty or len(metrics) == 0:
                return

            for metric in metrics:
                if metric not in df_group.columns:
                    continue
                y = pd.to_numeric(df_group[metric], errors="coerce").ffill()
                y_plot = hybrid_transform(y, log_thresh) if log_check else y
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
                        # yaxis=f"y{row}2" if row > 1 else "y2",  # Just for completeness; Plotly takes care when using secondary_y
                    ),
                    row=row,
                    col=1,
                    secondary_y=right_axis,
                )

        for row in range(1, 5):
            # --- CRITICAL FIX: PREVENT JS CRASH ---
            # If Left Axis is empty but Right Axis is used, Plotly JS crashes with "rangemode undefined".
            # We inject an invisible dummy trace on the Primary Y-axis to initialize it.
            if not sel_metrics and not df_session[row - 1].empty:
                fig.add_trace(
                    go.Scatter(
                        x=[df_session[row - 1]["bet_index"].iloc[0]],
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
            else:
                add_traces(df_session[row - 1], row, sel_metrics, log_check)
                fig.update_yaxes(title="", showticklabels=True, showgrid=True, secondary_y=False, row=row, col=1)

            add_traces(df_session[row - 1], row, right_axis_metrics, right_axis=True, log_check=log_check)
            fig.update_yaxes(title="", showticklabels=True, showgrid=True, secondary_y=True, row=row, col=1)

        # Add session (date) as annotation at the top center of the plot
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
            height=950,  # increase figure height
            template="plotly_white",
            margin=dict(l=70, r=50, t=80, b=50),  # more margins
        )

        range_slider_row = len(df_session)
        # Show X-axis range slider
        # Only add rangeslider to the last subplot (row=3, col=1), and hide the sent-back curve (set visible=False for the graph)
        fig.update_xaxes(
            title_text="bet_index",
            rangeslider=dict(visible=True, thickness=0.05, bgcolor="white"),  # thinner, clean bg
            rangeslider_visible=True,
            row=range_slider_row,
            col=1,
        )
        # For other rows, set without slider
        for r in range(1, range_slider_row):
            fig.update_xaxes(title_text="bet_index", rangeslider=dict(visible=False), row=r, col=1)

        # Set nice Y ticks for hybrid
        if log_check:
            for i, df_sess in enumerate(df_session):
                if not df_sess.empty:
                    if sel_metrics:
                        yticks = get_ticks(df_sess[sel_metrics].values.flatten(), log_thresh)
                        fig.update_yaxes(
                            tickvals=hybrid_transform(yticks, log_thresh),
                            ticktext=[f"{v:.0f}" for v in yticks],
                            row=i + 1,
                            col=1,
                        )
                    if right_axis_metrics:
                        yticks = get_ticks(df_sess[right_axis_metrics].values.flatten(), log_thresh)
                        fig.update_yaxes(
                            tickvals=hybrid_transform(yticks, log_thresh),
                            ticktext=[f"{v:.0f}" for v in yticks],
                            row=i + 1,
                            col=1,
                            secondary_y=True,
                        )

        return fig

    # host_ip = socket.gethostbyname(socket.gethostname())
    host_ip = "192.168.2.115"
    print(f"Your dashboard is available at: http://{host_ip}:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":

    data_file = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/bullet_stats_by_index.parquet"
    data_file2 = "/Users/niuxin/Documents/aws-example/jobs/output_fish_hunter/bullet_stats_by_index_pa.parquet"

    data = read_local_cache(data_file)
    data2 = read_local_cache(data_file2)

    data = pd.concat([data, data2])

    create_dashboard(data)
