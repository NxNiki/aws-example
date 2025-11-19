import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from bituslabs_ds.s3_utils import read_local_cache


def create_dashboard(df):
    # Ensure bet_index is numeric
    df["bet_index"] = pd.to_numeric(df["bet_index"], errors="coerce")
    df = df.dropna(subset=["bet_index"])

    sessions = sorted(df["session_start_date"].unique())
    strategies = ["BOOST", "DYNA_RTP", "DEFAULT", "PA"]
    metrics = [c for c in df.columns if c not in ["session_start_date", "session_group", "bet_index"]]

    # Define a fixed color map for metrics
    colors_hex = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3"]
    metric_colors = {metric: colors_hex[i % len(colors_hex)] for i, metric in enumerate(metrics)}

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Label("Session:"),
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
                    html.Label("Metrics:", style={"marginTop": "15px"}),
                    dcc.Checklist(
                        id="metric-checklist",
                        options=[{"label": m, "value": m} for m in metrics],
                        value=[metrics[0]] if metrics else [],
                        style={"maxHeight": "40vh", "overflowY": "auto", "marginTop": "10px", "marginBottom": "15px"},
                    ),
                    html.Label("Use Hybrid Log:", style={"marginTop": "15px"}),
                    dcc.Checklist(
                        id="log-check",
                        options=[{"label": "Hybrid log", "value": "log"}],
                        value=[],
                        style={"marginTop": "10px", "marginBottom": "5px"},
                    ),
                    html.Label("Linear threshold:", style={"marginTop": "15px", "marginRight": "5px"}),
                    dcc.Input(
                        id="linear-thresh",
                        type="number",
                        value=10,
                        min=0,
                        style={"width": "10%", "marginTop": "10px", "marginBottom": "15px"},
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
                    dcc.Graph(id="combined-plot", style={"height": "1450px", "marginBottom": "10px"})
                ],  # increase figure height
                style={"width": "80%", "height": "150%", "display": "inline-block", "marginBottom": "30px"},
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

    @app.callback(
        Output("combined-plot", "figure"),
        Input("session-dropdown", "value"),
        Input("strategy-a-dropdown", "value"),
        Input("strategy-b-dropdown", "value"),
        Input("strategy-c-dropdown", "value"),
        Input("strategy-d-dropdown", "value"),
        Input("metric-checklist", "value"),
        Input("log-check", "value"),
        Input("linear-thresh", "value"),
    )
    def update_plot(session, strat_a, strat_b, strat_c, strat_d, sel_metrics, log_check, thresh):
        df_sess = df[df["session_start_date"] == session].sort_values("bet_index")
        df_a = df_sess[df_sess["session_group"] == strat_a]
        df_b = df_sess[df_sess["session_group"] == strat_b]
        df_c = df_sess[df_sess["session_group"] == strat_c]
        df_d = df_sess[df_sess["session_group"] == strat_d]

        use_hybrid = "log" in log_check
        thresh = max(float(thresh), 1)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=(strat_a, strat_b, strat_c, strat_d))

        def add_traces(df_group, row):
            if df_group.empty or len(sel_metrics) == 0:
                return
            for metric in sel_metrics:
                if metric not in df_group.columns:
                    continue
                y = pd.to_numeric(df_group[metric], errors="coerce").ffill()
                y_plot = hybrid_transform(y, thresh) if use_hybrid else y
                fig.add_trace(
                    go.Scatter(
                        x=df_group["bet_index"],
                        y=y_plot,
                        name=metric,
                        mode="lines",
                        line=dict(
                            width=1.5,
                            color=metric_colors.get(metric, "#000000"),
                        ),
                        opacity=0.6,
                        # For Scatter, overall trace opacity can also be controlled:
                        # mode="markers",
                        # marker=dict(size=2.5, color=metric_colors.get(metric, "#000000")),
                    ),
                    row=row,
                    col=1,
                )

        add_traces(df_a, 1)
        add_traces(df_b, 2)
        add_traces(df_c, 3)
        add_traces(df_d, 4)

        fig.update_layout(
            height=950,  # increase figure height
            template="plotly_white",
            margin=dict(l=70, r=50, t=80, b=50),  # more margins
        )

        # Show X-axis range slider
        # Only add rangeslider to the last subplot (row=3, col=1), and hide the sent-back curve (set visible=False for the graph)
        fig.update_xaxes(
            title_text="bet_index",
            rangeslider=dict(visible=True, thickness=0.05, bgcolor="white"),  # thinner, clean bg
            rangeslider_visible=True,
            row=4,
            col=1,
        )
        # For other rows, set without slider
        fig.update_xaxes(title_text="bet_index", rangeslider=dict(visible=False), row=1, col=1)
        fig.update_xaxes(title_text="bet_index", rangeslider=dict(visible=False), row=2, col=1)
        fig.update_xaxes(title_text="bet_index", rangeslider=dict(visible=False), row=3, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=4, col=1)

        # Set nice Y ticks for hybrid
        if use_hybrid:
            if not df_a.empty:
                yticks = get_ticks(df_a[sel_metrics].values.flatten(), thresh)
                fig.update_yaxes(
                    tickvals=hybrid_transform(yticks, thresh), ticktext=[f"{v:.0f}" for v in yticks], row=1, col=1
                )
            if not df_b.empty:
                yticks = get_ticks(df_b[sel_metrics].values.flatten(), thresh)
                fig.update_yaxes(
                    tickvals=hybrid_transform(yticks, thresh), ticktext=[f"{v:.0f}" for v in yticks], row=2, col=1
                )
            if not df_c.empty:
                yticks = get_ticks(df_c[sel_metrics].values.flatten(), thresh)
                fig.update_yaxes(
                    tickvals=hybrid_transform(yticks, thresh), ticktext=[f"{v:.0f}" for v in yticks], row=3, col=1
                )

            if not df_d.empty:
                yticks = get_ticks(df_d[sel_metrics].values.flatten(), thresh)
                fig.update_yaxes(
                    tickvals=hybrid_transform(yticks, thresh), ticktext=[f"{v:.0f}" for v in yticks], row=4, col=1
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
