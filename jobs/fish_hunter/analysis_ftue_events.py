"""
FTUE event-timing analysis for the FM01 first-session dataset.

Usage:
    python analysis_ftue_events.py [--bin-seconds 30] [--high-fish-value 500]

Reads the parquet dataset produced by etl_game_stats_ftue.py (one row per
bullet in each new user's first session) and generates an interactive HTML
report with, for each user, the timing of four milestone events:

    - first bet            (first bullet fired)
    - first kill           (first bullet with payout > 0)
    - first high-value kill (first bullet with payout > 0 and fish_value >= threshold)
    - stop play            (last bullet of the first session)

Event counts are bucketed into fixed-width time bins (default 30 s), measured
from two origins: the account-created time and the first-bet time.
"""

import argparse
import os
from datetime import datetime

import awswrangler as wr
import pandas as pd
import plotly.graph_objects as go

from bituslabs_ds.config import DEFAULT_ETL_OUTPUT, LOCAL_ROOT, setup_logging

DEFAULT_INPUT = f"{DEFAULT_ETL_OUTPUT}/jobs/output_fish_hunter/ftue_first_session/"
DEFAULT_OUTPUT = f"{LOCAL_ROOT}/jobs/output_fish_hunter/ftue_analysis_report.html"

COLUMNS = ["user_id", "account_created", "event_timestamp", "payout", "fish_value"]

EVENT_LABELS = {
    "first_bet": "First bet",
    "first_kill": "First kill (payout > 0)",
    "first_high_kill": "First high-value kill",
    "stop_play": "Stop play (last bet)",
}

EVENT_COLORS = {
    "first_bet": "#636efa",
    "first_kill": "#EF553B",
    "first_high_kill": "#00cc96",
    "stop_play": "#ab63fa",
}


def load_data(input_path: str) -> pd.DataFrame:
    df = wr.s3.read_parquet(path=input_path, dataset=True, columns=COLUMNS)
    print(f"Loaded {len(df)} bullet rows for {df['user_id'].nunique()} users from {input_path}")
    return df


def compute_user_events(df: pd.DataFrame, high_fish_value: float) -> pd.DataFrame:
    """One row per user with the timestamp of each milestone event (NaT if never reached)."""
    df = df.sort_values("event_timestamp")
    kills = df[df["payout"] > 0]
    high_kills = kills[kills["fish_value"] >= high_fish_value]

    events = df.groupby("user_id").agg(
        account_created=("account_created", "first"),
        first_bet=("event_timestamp", "min"),
        stop_play=("event_timestamp", "max"),
    )
    events["first_kill"] = kills.groupby("user_id")["event_timestamp"].min()
    events["first_high_kill"] = high_kills.groupby("user_id")["event_timestamp"].min()
    return events.reset_index()


def bin_event_counts(events: pd.DataFrame, origin_col: str, bin_seconds: int) -> pd.DataFrame:
    """Count users per (event, time bin), with bins measured in seconds since origin_col."""
    records = []
    for event in EVENT_LABELS:
        offsets = (events[event] - events[origin_col]).dt.total_seconds().dropna()
        bins = (offsets // bin_seconds).astype(int) * bin_seconds
        counts = bins.value_counts().sort_index()
        records.append(pd.DataFrame({"bin_start": counts.index, "count": counts.values, "event": event}))
    return pd.concat(records, ignore_index=True)


def make_figure(
    binned: pd.DataFrame, origin_label: str, bin_seconds: int, default_visible: tuple = ("first_bet", "stop_play")
) -> go.Figure:
    fig = go.Figure()
    for event, label in EVENT_LABELS.items():
        sub = binned[binned["event"] == event]
        fig.add_trace(
            go.Scatter(
                x=sub["bin_start"],
                y=sub["count"],
                mode="lines",
                name=label,
                line={"width": 1.5, "color": EVENT_COLORS[event]},
                visible=event in default_visible,
            )
        )
    fig.update_layout(
        title=f"Event counts per {bin_seconds}s bin — time since {origin_label}",
        xaxis_title=f"Seconds since {origin_label}",
        yaxis_title="Number of users",
        showlegend=False,
        height=500,
    )
    # Long-tail sessions run to 12h; default zoom to the first hour, the
    # rangeslider below the x-axis drags the window across the full range
    fig.update_xaxes(range=[0, 3600], rangeslider={"visible": True, "thickness": 0.08})
    return fig


def figure_controls(div_id: str, default_visible: tuple) -> str:
    """Checkbox per metric + a min/max time-range input wired to the plotly div via JS."""
    boxes = []
    for event, label in EVENT_LABELS.items():
        checked = "checked" if event in default_visible else ""
        boxes.append(
            f'<label style="color:{EVENT_COLORS[event]}">'
            f'<input type="checkbox" data-target="{div_id}" data-trace="{list(EVENT_LABELS).index(event)}"'
            f" {checked}> {label}</label>"
        )
    range_inputs = (
        f'<span class="range-ctl">Time range (s): '
        f'<input type="number" id="{div_id}-min" value="0" step="30" style="width:90px"> &ndash; '
        f'<input type="number" id="{div_id}-max" value="3600" step="30" style="width:90px"> '
        f"<button onclick=\"applyRange('{div_id}')\">Apply</button></span>"
    )
    return f'<div class="fig-controls">{" ".join(boxes)} {range_inputs}</div>'


CONTROLS_JS = """
<script>
document.querySelectorAll('.fig-controls input[type=checkbox]').forEach(function (cb) {
    cb.addEventListener('change', function () {
        Plotly.restyle(cb.dataset.target, {visible: cb.checked}, [parseInt(cb.dataset.trace)]);
    });
});
function applyRange(divId) {
    var lo = parseFloat(document.getElementById(divId + '-min').value);
    var hi = parseFloat(document.getElementById(divId + '-max').value);
    Plotly.relayout(divId, {'xaxis.range': [lo, hi]});
}
</script>
"""


def summary_table(events: pd.DataFrame, high_fish_value: float) -> pd.DataFrame:
    n_users = len(events)
    rows = []
    for origin_col, origin_label in [("account_created", "account created"), ("first_bet", "first bet")]:
        for event, label in EVENT_LABELS.items():
            offsets = (events[event] - events[origin_col]).dt.total_seconds().dropna()
            rows.append(
                {
                    "origin": origin_label,
                    "event": label,
                    "users reached": len(offsets),
                    "% of users": round(100 * len(offsets) / n_users, 1) if n_users else 0,
                    "median (s)": round(offsets.median(), 1) if len(offsets) else None,
                    "p90 (s)": round(offsets.quantile(0.9), 1) if len(offsets) else None,
                }
            )
    return pd.DataFrame(rows)


def build_report(events: pd.DataFrame, bin_seconds: int, high_fish_value: float, output_path: str) -> None:
    # First-bet trace is a degenerate spike at t=0 when the origin IS the first
    # bet, so default it off there; checkboxes let the reader enable any mix.
    origins = [
        ("account_created", "account created", ("first_bet", "stop_play")),
        ("first_bet", "first bet", ("first_kill", "stop_play")),
    ]
    figures = []
    for origin_col, origin_label, default_visible in origins:
        binned = bin_event_counts(events, origin_col, bin_seconds)
        figures.append((make_figure(binned, origin_label, bin_seconds, default_visible), default_visible))

    summary_html = summary_table(events, high_fish_value).to_html(index=False, border=0)

    parts = [
        "<html><head><title>FTUE Event Timing Report</title>",
        "<style>body{font-family:sans-serif;margin:24px} table{border-collapse:collapse}",
        "td,th{padding:4px 12px;border-bottom:1px solid #ddd;text-align:right}",
        "th{background:#f5f5f5} td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}",
        ".fig-controls{margin:24px 0 4px 0} .fig-controls label{margin-right:16px;font-weight:bold}",
        ".range-ctl{margin-left:24px;color:#333;font-weight:normal}</style></head><body>",
        "<h1>FTUE Event Timing Report (FM01 first session)</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} &middot; {len(events)} users &middot; "
        f"bin = {bin_seconds}s &middot; high-value fish threshold = {high_fish_value}</p>",
        "<h2>Summary</h2>",
        summary_html,
    ]
    for i, (fig, default_visible) in enumerate(figures):
        div_id = f"ftue-fig-{i}"
        parts.append(figure_controls(div_id, default_visible))
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False, div_id=div_id))
    parts.append(CONTROLS_JS)
    parts.append("</body></html>")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FTUE event-timing analysis report")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Parquet dataset path (S3 or local)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output HTML report path")
    parser.add_argument("--bin-seconds", type=int, default=30, help="Time bin width in seconds")
    parser.add_argument("--high-fish-value", type=float, default=500, help="fish_value threshold for high-value kill")
    args = parser.parse_args()

    df = load_data(args.input)
    events = compute_user_events(df, args.high_fish_value)
    build_report(events, args.bin_seconds, args.high_fish_value, args.output)


if __name__ == "__main__":
    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")
    main()
