"""
FTUE event-timing analysis for the FM01 first-session dataset.

Usage:
    python analysis_ftue_events.py [--bin-seconds 30] [--high-fish-value 500]

Reads the parquet dataset produced by etl_game_stats_ftue.py (one row per
bullet in each new user's first session, with attached CNY transactions) and
generates an interactive HTML report. For each user it finds the timing of
these milestone events, measured from both account-created and first-bet time:

    - first bet            (first bullet fired)
    - first kill           (first bullet with payout > 0)
    - first high-value kill (first kill with fish_value >= threshold)
    - stop play            (last bullet of the first session)
    - first deposit        (first deposit transaction, amount > 0)
    - second deposit       (second deposit transaction, amount > 0)
    - first withdrawal     (first withdrawal transaction, amount > 0)
    - change device        (first bullet whose device_type differs from the first bullet)
    - change IP            (first bullet whose ip differs from the first bullet)

The report is broken out per strategy group: each user is assigned the
strategy_name of their first bullet (the strategy at session start), and the
analysis is rendered separately for "All users" and for each strategy level.
"""

import argparse
import os
from datetime import datetime
from typing import cast

import awswrangler as wr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bituslabs_ds.config import DEFAULT_ETL_OUTPUT, LOCAL_ROOT, setup_logging

DEFAULT_INPUT = f"{DEFAULT_ETL_OUTPUT}/jobs/output_fish_hunter/ftue_first_session/"
DEFAULT_OUTPUT = f"{LOCAL_ROOT}/jobs/output_fish_hunter/ftue_analysis_report.html"

COLUMNS = [
    "user_id",
    "account_created",
    "event_timestamp",
    "payout",
    "fish_value",
    "device_type",
    "ip",
    "strategy_name",
    "transaction_type",
    "transaction_amount",
    "transaction_processed_at",
]

EVENT_LABELS = {
    "first_bet": "First bet",
    "first_kill": "First kill (payout > 0)",
    "first_high_kill": "First high-value kill",
    "stop_play": "Stop play (last bet)",
    "first_deposit": "First deposit",
    "second_deposit": "Second deposit",
    "first_withdrawal": "First withdrawal",
    "change_device": "Change device",
    "change_ip": "Change IP",
}

EVENT_COLORS = {
    "first_bet": "#636efa",
    "first_kill": "#EF553B",
    "first_high_kill": "#00cc96",
    "stop_play": "#ab63fa",
    "first_deposit": "#FFA15A",
    "second_deposit": "#19d3f3",
    "first_withdrawal": "#FF6692",
    "change_device": "#B6E880",
    "change_ip": "#FF97FF",
}

# (origin column, axis label, events visible by default in that figure)
ORIGINS = [
    ("account_created", "account created", ("first_bet", "stop_play")),
    ("first_bet", "first bet", ("first_kill", "stop_play")),
]


def load_data(input_path: str) -> pd.DataFrame:
    df = wr.s3.read_parquet(path=input_path, dataset=True, columns=COLUMNS)
    print(f"Loaded {len(df)} bullet rows for {df['user_id'].nunique()} users from {input_path}")
    return df


def compute_user_events(df: pd.DataFrame, high_fish_value: float) -> pd.DataFrame:
    """One row per user with the timestamp of each milestone event (NaT if never reached).

    Bullets fan out across attached transactions (a bullet with N transactions
    appears N times), but every per-user metric here is a min/max/first over
    timestamps, which is idempotent under that duplication. Each transaction
    attaches to exactly one bullet, so deposit/withdrawal ordering is unaffected.
    """
    df = df.sort_values("event_timestamp")
    g = df.groupby("user_id")

    events = cast(
        pd.DataFrame,
        g.agg(
            account_created=("account_created", "first"),
            first_bet=("event_timestamp", "min"),
            stop_play=("event_timestamp", "max"),
            strategy_group=("strategy_name", "first"),
        ),
    )

    kills = df.loc[df["payout"] > 0]
    high_kills = kills.loc[kills["fish_value"] >= high_fish_value]
    events["first_kill"] = kills.groupby("user_id")["event_timestamp"].min()
    events["first_high_kill"] = high_kills.groupby("user_id")["event_timestamp"].min()

    # Device / IP change: earliest bullet whose value differs from the user's
    # first bullet. notna guard avoids treating a missing value as a change.
    first_device = g["device_type"].transform("first")
    first_ip = g["ip"].transform("first")
    dev_chg = df.loc[df["device_type"].notna() & (df["device_type"] != first_device)]
    ip_chg = df.loc[df["ip"].notna() & (df["ip"] != first_ip)]
    events["change_device"] = dev_chg.groupby("user_id")["event_timestamp"].min()
    events["change_ip"] = ip_chg.groupby("user_id")["event_timestamp"].min()

    # Deposits / withdrawals with amount > 0, ordered by processed_at.
    txn = df.loc[df["transaction_type"].notna() & (df["transaction_amount"] > 0)]
    deposits = (
        txn.loc[txn["transaction_type"] == "deposit", ["user_id", "transaction_processed_at"]]
        .drop_duplicates()
        .sort_values(["user_id", "transaction_processed_at"])
    )
    dep_rank = deposits.groupby("user_id").cumcount()
    events["first_deposit"] = deposits.loc[dep_rank == 0].set_index("user_id")["transaction_processed_at"]
    events["second_deposit"] = deposits.loc[dep_rank == 1].set_index("user_id")["transaction_processed_at"]
    withdrawals = txn.loc[txn["transaction_type"] == "withdrawal"]
    events["first_withdrawal"] = withdrawals.groupby("user_id")["transaction_processed_at"].min()

    return events.reset_index()


DAY_SECONDS = 86400


def _n_fine_bins(bin_seconds: int) -> int:
    return DAY_SECONDS // bin_seconds


def bin_event_counts(events: pd.DataFrame, origin_col: str, bin_seconds: int) -> pd.DataFrame:
    """Count users per (event, time bin) since origin_col, on a compressed ordinal axis.

    Uneven bins: bin_seconds resolution within the first 24 h, one-day bins
    beyond (accounts can predate first FM01 play by months, and a uniform
    seconds axis would squash the first day into a sliver). bin_index is the
    ordinal position; the seconds->index mapping is mirrored in the JS range
    controls (controls_js).
    """
    n_fine = _n_fine_bins(bin_seconds)
    records = []
    for event in EVENT_LABELS:
        offsets = (events[event] - events[origin_col]).dt.total_seconds().dropna().clip(lower=0)
        idx = np.where(offsets < DAY_SECONDS, offsets // bin_seconds, n_fine + offsets // DAY_SECONDS - 1)
        counts = pd.Series(idx.astype(int)).value_counts().sort_index()
        records.append(pd.DataFrame({"bin_index": counts.index, "count": counts.values, "event": event}))
    return pd.concat(records, ignore_index=True)


def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _bin_label(idx: int, bin_seconds: int) -> str:
    n_fine = _n_fine_bins(bin_seconds)
    if idx < n_fine:
        start = idx * bin_seconds
        return f"{_fmt_hms(start)}–{_fmt_hms(start + bin_seconds)}"
    day = idx - n_fine + 1
    return f"day {day}–{day + 1}"


def _axis_ticks(bin_seconds: int, max_index: int) -> tuple:
    """Tick positions/labels: time-of-day marks across the fine region, day marks after."""
    fine_marks = [0, 300, 900, 1800, 3600, 2 * 3600, 4 * 3600, 8 * 3600, 12 * 3600, 18 * 3600]
    labels = ["0", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "18h"]
    vals = [s // bin_seconds for s in fine_marks]
    texts = list(labels)
    n_fine = _n_fine_bins(bin_seconds)
    max_day = max_index - n_fine + 2 if max_index >= n_fine else 0
    if max_day > 0:
        step = max(1, round(max_day / 8))
        for d in range(1, max_day + 1, step):
            vals.append(n_fine + d - 1)
            texts.append(f"{d}d")
    return vals, texts


def make_figure(binned: pd.DataFrame, origin_label: str, bin_seconds: int, default_visible: tuple) -> go.Figure:
    fig = go.Figure()
    for event, label in EVENT_LABELS.items():
        sub = binned[binned["event"] == event]
        fig.add_trace(
            go.Scatter(
                x=sub["bin_index"],
                y=sub["count"],
                mode="lines",
                name=label,
                line={"width": 1.5, "color": EVENT_COLORS[event]},
                visible=event in default_visible,
                customdata=[_bin_label(i, bin_seconds) for i in sub["bin_index"]],
                hovertemplate="%{customdata}<br>%{y} users<extra>%{fullData.name}</extra>",
            )
        )
    max_index = int(binned["bin_index"].max()) if not binned.empty else _n_fine_bins(bin_seconds)
    tickvals, ticktext = _axis_ticks(bin_seconds, max_index)
    fig.update_layout(
        title=f"Event counts — time since {origin_label} ({bin_seconds}s bins in first 24h, daily bins after)",
        xaxis_title=f"Time since {origin_label}",
        yaxis_title="Number of users",
        showlegend=False,
        height=500,
    )
    # Default zoom to the first hour; the rangeslider below the x-axis drags
    # the window across the full (compressed) range
    fig.update_xaxes(
        range=[0, 3600 // bin_seconds],
        rangeslider={"visible": True, "thickness": 0.08},
        tickvals=tickvals,
        ticktext=ticktext,
    )
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


def controls_js(bin_seconds: int) -> str:
    """JS for the checkbox/range controls; secToIdx mirrors bin_event_counts' uneven binning."""
    return f"""
<script>
function secToIdx(sec) {{
    if (sec < {DAY_SECONDS}) return sec / {bin_seconds};
    return {_n_fine_bins(bin_seconds)} + Math.floor(sec / {DAY_SECONDS}) - 1;
}}
document.querySelectorAll('.fig-controls input[type=checkbox]').forEach(function (cb) {{
    cb.addEventListener('change', function () {{
        Plotly.restyle(cb.dataset.target, {{visible: cb.checked}}, [parseInt(cb.dataset.trace)]);
    }});
}});
function applyRange(divId) {{
    var lo = parseFloat(document.getElementById(divId + '-min').value);
    var hi = parseFloat(document.getElementById(divId + '-max').value);
    Plotly.relayout(divId, {{'xaxis.range': [secToIdx(lo), secToIdx(hi)]}});
}}
</script>
"""


def summary_table(events: pd.DataFrame) -> pd.DataFrame:
    n_users = len(events)
    rows = []
    for origin_col, origin_label, _ in ORIGINS:
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


def _fmt_duration(seconds: float) -> str:
    if pd.isna(seconds):
        return "n/a"
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def interpretation_html(events: pd.DataFrame, high_fish_value: float) -> str:
    """Auto-generated reading of the headline numbers; recomputed from the data on every run."""
    n = len(events)
    if n == 0:
        return "<p>No users in dataset.</p>"

    bet_delay = (events["first_bet"] - events["account_created"]).dt.total_seconds()
    kill_after_bet = (events["first_kill"] - events["first_bet"]).dt.total_seconds().dropna()
    high_after_bet = (events["first_high_kill"] - events["first_bet"]).dt.total_seconds().dropna()
    play_time = (events["stop_play"] - events["first_bet"]).dt.total_seconds()
    dep_after_acct = (events["first_deposit"] - events["account_created"]).dt.total_seconds().dropna()

    pct_kill = 100 * len(kill_after_bet) / n
    pct_high = 100 * len(high_after_bet) / n
    pct_kill_10s = 100 * (kill_after_bet <= 10).mean() if len(kill_after_bet) else 0
    pct_quit_2min = 100 * (play_time <= 120).mean()
    pct_dep = 100 * events["first_deposit"].notna().mean()
    pct_dep2 = 100 * events["second_deposit"].notna().mean()
    pct_wd = 100 * events["first_withdrawal"].notna().mean()

    has_high = events["first_high_kill"].notna()
    play_with_high = play_time[has_high].median()
    play_without_high = play_time[~has_high].median()

    bullets = [
        f"<b>Onboarding is fast:</b> half of new users fire their first bullet within "
        f"{_fmt_duration(bet_delay.median())} of account creation "
        f"(90% within {_fmt_duration(bet_delay.quantile(0.9))}).",
        f"<b>First kill comes almost immediately:</b> {pct_kill:.0f}% of users kill at least one fish in their "
        f"first session, with a median of {_fmt_duration(kill_after_bet.median())} after the first bet; "
        f"{pct_kill_10s:.0f}% of those killers get it within 10 seconds. Early kill feedback is effectively "
        f"guaranteed by game design.",
        f"<b>High-value kills (fish value &ge; {high_fish_value:g}) are a rare, late event:</b> only "
        f"{pct_high:.1f}% of users reach one, at a median of {_fmt_duration(high_after_bet.median())} into play "
        f"&mdash; long after the median player has already stopped "
        f"({_fmt_duration(play_time.median())} of play).",
        f"<b>The first session is short for most:</b> median first-session play time is "
        f"{_fmt_duration(play_time.median())}, and {pct_quit_2min:.0f}% of users stop within 2 minutes of their "
        f"first bet; the p90 session runs {_fmt_duration(play_time.quantile(0.9))}.",
        f"<b>Deposit conversion:</b> {pct_dep:.1f}% of users make at least one deposit in the observed window "
        f"(median {_fmt_duration(dep_after_acct.median())} after account creation), {pct_dep2:.1f}% make a "
        f"second, and {pct_wd:.1f}% reach a first withdrawal.",
        f"<b>High-value kills coincide with much longer sessions:</b> users who hit one play a median of "
        f"{_fmt_duration(play_with_high)} vs {_fmt_duration(play_without_high)} for everyone else. This is "
        f"correlation, not causation &mdash; longer play also gives more chances to hit a big fish.",
        "<b>Caveat:</b> only the first bet session is observed, scanned up to 12 h after first play, so "
        "&ldquo;stop play&rdquo; means the end of the first session, not churn; users may return later.",
    ]
    return "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"


def _strategy_groups(events: pd.DataFrame) -> list:
    """('All users' + each strategy level), each as (key, label, subset)."""
    groups = [("__all__", "All users", events)]
    for grp in sorted(events["strategy_group"].dropna().unique()):
        groups.append((grp, str(grp), events.loc[events["strategy_group"] == grp]))
    return groups


def build_report(events: pd.DataFrame, bin_seconds: int, high_fish_value: float, output_path: str) -> None:
    groups = _strategy_groups(events)
    nav = " &middot; ".join(
        f'<a href="#grp-{gi}">{label} ({len(sub)})</a>' for gi, (_, label, sub) in enumerate(groups)
    )

    parts = [
        "<html><head><title>FTUE Event Timing Report</title>",
        "<style>body{font-family:sans-serif;margin:24px} table{border-collapse:collapse}",
        "td,th{padding:4px 12px;border-bottom:1px solid #ddd;text-align:right}",
        "th{background:#f5f5f5} td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}",
        ".fig-controls{margin:24px 0 4px 0} .fig-controls label{margin-right:16px;font-weight:bold}",
        ".range-ctl{margin-left:24px;color:#333;font-weight:normal}",
        "h2{margin-top:48px;border-top:2px solid #ccc;padding-top:16px}</style></head><body>",
        "<h1>FTUE Event Timing Report (FM01 first session)</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} &middot; {len(events)} users &middot; "
        f"bin = {bin_seconds}s &middot; high-value fish threshold = {high_fish_value}</p>",
        "<p>Users grouped by the strategy_name of their first bullet (session-start strategy).</p>",
        f"<p><b>Jump to:</b> {nav}</p>",
    ]

    plotly_included = False
    for gi, (_, label, sub) in enumerate(groups):
        parts.append(f'<h2 id="grp-{gi}">Strategy group: {label} ({len(sub)} users)</h2>')
        parts.append(summary_table(sub).to_html(index=False, border=0))
        parts.append("<h3>Interpretation</h3>")
        parts.append(interpretation_html(sub, high_fish_value))
        for oi, (origin_col, origin_label, default_visible) in enumerate(ORIGINS):
            binned = bin_event_counts(sub, origin_col, bin_seconds)
            fig = make_figure(binned, origin_label, bin_seconds, default_visible)
            div_id = f"ftue-{gi}-{oi}"
            parts.append(figure_controls(div_id, default_visible))
            include = "cdn" if not plotly_included else False
            parts.append(fig.to_html(full_html=False, include_plotlyjs=include, div_id=div_id))
            plotly_included = True

    parts.append(controls_js(bin_seconds))
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
