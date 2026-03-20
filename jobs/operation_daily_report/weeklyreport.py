import base64
import io

import pandas as pd
import plotly.graph_objs as go

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State


# ========= 解析上传 CSV =========
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    if 'csv' in filename.lower():
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
        raise ValueError("目前只支持 CSV 文件")

    # 删掉 '中文' 和 '定义'
    for col in ['中文', '定义']:
        if col in df.columns:
            df = df.drop(columns=col)

    return df


# ========= 识别数值类型（金额 / 百分比 / 普通数字） =========
def convert_series(series):
    s = series.astype(str)

    # 百分比
    if s.str.endswith('%').all():
        v = s.str.replace('%', '', regex=False).astype(float)
        return v, 'percent'

    # 金额（含 ¥）
    if s.str.contains('¥').any():
        v = series.replace('[¥,]', '', regex=True).astype(float)
        return v, 'currency'

    # 其他当普通数字处理（顺手去掉逗号）
    v = pd.to_numeric(series.replace(',', '', regex=True), errors='coerce')
    return v, 'number'


# ========= 画 14 天图（Plotly 版） =========
def make_figure(df, row_name, start_date):
    if df is None or row_name is None or start_date is None:
        return go.Figure()

    # 日期列（除 INDEX 以外）
    date_cols = [c for c in df.columns if c != 'INDEX']

    if start_date not in date_cols:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_white',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': f"start_date {start_date} 不在列名里",
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]
        )
        return fig

    start_idx = date_cols.index(start_date)
    end_idx = start_idx + 14
    if end_idx > len(date_cols):
        fig = go.Figure()
        fig.update_layout(
            template='plotly_white',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': "从这个 start date 开始不足 14 天数据",
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]
        )
        return fig

    cols_14 = date_cols[start_idx:end_idx]

    if row_name not in df['INDEX'].values:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_white',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': f"INDEX 中找不到：{row_name}",
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]
        )
        return fig

    row = df.loc[df['INDEX'] == row_name, cols_14].iloc[0]

    # 数值转换
    values, vtype = convert_series(row)
    if vtype == 'percent':
        plot_values = values.replace(999, 0)
    else:
        plot_values = values
    first7 = plot_values.iloc[:7]
    last7 = plot_values.iloc[7:]


    first7_dates = [c.replace(', 2025', '') for c in cols_14[:7]]
    last7_dates = [c.replace(', 2025', '') for c in cols_14[7:]]

    if vtype == 'percent':
        first7_valid = values.iloc[:7].replace(999, pd.NA).dropna()
        last7_valid  = values.iloc[7:].replace(999, pd.NA).dropna()

        avg_first = first7_valid.mean() if not first7_valid.empty else 0
        avg_last  = last7_valid.mean() if not last7_valid.empty else 0
    else:
        avg_first = values.iloc[:7].mean()
        avg_last  = values.iloc[7:].mean()

    fig = go.Figure()

    # 前 7 天：橙色
    fig.add_trace(go.Scatter(
        x=first7_dates,
        y=first7,
        mode='lines+markers',
        name='last week',
        line=dict(color='orange')
    ))

    # 后 7 天：蓝色
    fig.add_trace(go.Scatter(
        x=last7_dates,
        y=last7,
        mode='lines+markers',
        name='this week',
        line=dict(color='blue')
    ))

    # 灰色虚线连接
    fig.add_trace(go.Scatter(
        x=[first7_dates[-1], last7_dates[0]],
        y=[first7.iloc[-1], last7.iloc[0]],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))

    # 均值线
    if vtype == 'percent':
        avg_label_1 = f"last week mean: {avg_first:.1f}%"
        avg_label_2 = f"this week mean: {avg_last:.1f}%"
    elif vtype == 'currency':
        avg_label_1 = f"last week mean: ¥{avg_first:,.0f}"
        avg_label_2 = f"this week mean: ¥{avg_last:,.0f}"
    else:
        avg_label_1 = f"last week mean: {avg_first:,.0f}"
        avg_label_2 = f"this week mean: {avg_last:,.0f}"

    fig.add_hline(
        y=avg_first,
        line=dict(color='orange', dash='dash'),
        annotation_text=avg_label_1,
        annotation_position='right'
    )
    fig.add_hline(
        y=avg_last,
        line=dict(color='blue', dash='dash'),
        annotation_text=avg_label_2,
        annotation_position='right'
    )

    # Y 轴标签/格式
    if vtype == 'percent':
        fig.update_yaxes(tickformat='.1f%')
        y_label = f"{row_name} (%)"
    elif vtype == 'currency':
        y_label = f"{row_name} (¥)"
    else:
        y_label = row_name

    fig.update_layout(
        title=f"{row_name} – from {start_date} two weeks",
        xaxis_title='Date',
        yaxis_title=y_label,
        template='plotly_white',
        height=600
    )

    return fig


# ========= Dash app =========
app = dash.Dash(__name__)

app.layout = html.Div(
    style={'width': '80%', 'margin': '0 auto'},
    children=[
        html.H3("14-day Metric Plotter"),

        # 上传 CSV
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV'),
            multiple=False
        ),
        html.Div(id='file-info', style={'marginTop': '8px', 'color': '#555'}),

        # 存 df 的地方
        dcc.Store(id='stored-df'),

        html.Hr(),

        # 控件：row 下拉 + start date 输入
        html.Div(
            style={'display': 'flex', 'gap': '40px', 'marginBottom': '10px'},
            children=[
                html.Div([
                    html.Label("Row name (INDEX):"),
                    dcc.Dropdown(
                        id='row-name',
                        placeholder='Upload CSV first',
                        style={'width': '350px'}
                    )
                ]),
                html.Div([
                    html.Label("Start date (must be one of the columns):"),
                    dcc.Input(
                        id='start-date',
                        type='text',
                        placeholder='e.g. Nov 3, 2025',
                        style={'width': '200px'}
                    )
                ])
            ]
        ),

        dcc.Graph(id='metric-graph')
    ]
)


# 上传后：保存 df，并填充 row 下拉选项
@app.callback(
    Output('stored-df', 'data'),
    Output('file-info', 'children'),
    Output('row-name', 'options'),
    Output('row-name', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def on_file_upload(contents, filename):
    if contents is None:
        return dash.no_update, "", [], None

    df = parse_contents(contents, filename)

    # 过滤掉 INDEX 里是空的 / NaN 的行
    idx_series = df['INDEX'].dropna()
    idx_values = [str(v) for v in idx_series.tolist() if str(v).strip() != ""]

    options = [{'label': v, 'value': v} for v in idx_values]

    return (
        df.to_json(date_format='iso', orient='split'),
        f"Loaded file: {filename}, shape = {df.shape}",
        options,
        idx_values[0] if idx_values else None
    )


# 根据 df + row + start date 画图
@app.callback(
    Output('metric-graph', 'figure'),
    Input('stored-df', 'data'),
    Input('row-name', 'value'),
    Input('start-date', 'value')
)
def update_graph(df_json, row_name, start_date):
    if df_json is None:
        return go.Figure()

    df = pd.read_json(df_json, orient='split')
    return make_figure(df, row_name, start_date)


if __name__ == '__main__':
    app.run(debug=True, port=8051)