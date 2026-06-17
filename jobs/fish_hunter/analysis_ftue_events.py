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

Each user is assigned the strategy_name of their first bullet (the strategy at
session start). The report is a single interactive figure: pick the time
origin, up to two strategy groups to compare (Group A solid, Group B dashed),
and which metrics to draw; the summary table below rebuilds from the selection.
"""

import argparse
import json
import os
from datetime import datetime
from typing import cast

import awswrangler as wr
import numpy as np
import pandas as pd

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

# Lines are colored by strategy group (stable per group index); metric is
# encoded by dash style (or solid when a single metric is selected). High-
# contrast ColorBrewer Set1 palette so groups stay distinct.
GROUP_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]


def _group_color(i: int) -> str:
    return GROUP_PALETTE[i % len(GROUP_PALETTE)]


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


DEFAULT_METRICS = ("first_kill",)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# Pure JS (no f-string) — reads the embedded CFG object and redraws one figure
# from the selected origin / groups / metrics. secToIdx / binLabel mirror the
# uneven binning in bin_event_counts so the range inputs and hover read in real
# seconds while the axis stays on ordinal bin indices.
APP_JS = """
<script>
const DASH = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'];
function pad(n) { return n < 10 ? '0' + n : '' + n; }
function fmtHms(sec) {
    var h = Math.floor(sec / 3600), r = sec % 3600, m = Math.floor(r / 60), s = r % 60;
    return h ? h + ':' + pad(m) + ':' + pad(s) : m + ':' + pad(s);
}
function binLabel(idx) {
    if (idx < CFG.nFine) {
        var st = idx * CFG.binSeconds;
        return fmtHms(st) + '\\u2013' + fmtHms(st + CFG.binSeconds);
    }
    var d = idx - CFG.nFine + 1;
    return 'day ' + d + '\\u2013' + (d + 1);
}
function secToIdx(sec) {
    if (sec < 86400) return sec / CFG.binSeconds;
    return CFG.nFine + Math.floor(sec / 86400) - 1;
}
function selOrigin() { return document.querySelector('input[name=origin]:checked').value; }
function selGroups() {
    // preserve CFG.groups order so each group's dash style is stable
    return CFG.groups.filter(function (g) {
        var cb = document.querySelector('.group[value="' + g + '"]');
        return cb && cb.checked;
    });
}
function selMetrics() {
    return CFG.order.filter(function (m) {
        var cb = document.querySelector('.metric[value="' + m + '"]');
        return cb && cb.checked;
    });
}
function isChecked(id) { var cb = document.getElementById(id); return cb && cb.checked; }
function redraw() {
    var origin = selOrigin(), groups = selGroups(), metrics = selMetrics();
    var abs = isChecked('absToggle'), useLog = isChecked('logToggle'), traces = [];
    groups.forEach(function (g) {
        var gColor = CFG.groupColors[CFG.groups.indexOf(g)];
        metrics.forEach(function (m) {
            var s = CFG.data[origin][g][m];
            if (!s) return;
            var nm = CFG.events[m] + (groups.length > 1 ? ' [' + g + ']' : '');
            // dash distinguishes metrics; with a single metric it adds no info, so keep solid
            var dash = metrics.length > 1 ? DASH[CFG.order.indexOf(m) % DASH.length] : 'solid';
            var hov = abs
                ? '%{customdata[0]}<br>%{y} users<extra>' + nm + '</extra>'
                : '%{customdata[0]}<br>%{y:.2%} of group (%{customdata[1]} users)<extra>' + nm + '</extra>';
            traces.push({
                x: s.x, y: abs ? s.c : s.y, mode: 'lines+markers', type: 'scatter', name: nm,
                line: { color: gColor, width: 1.5, dash: dash },
                marker: { color: gColor, size: 4 },
                customdata: s.x.map(function (idx, i) { return [binLabel(idx), s.c[i]]; }),
                hovertemplate: hov
            });
        });
    });
    var sub = ' (' + CFG.binSeconds + 's bins in first 24h, daily bins after)';
    // Log y on a percentage axis: Plotly keeps the fractional values and applies
    // the percent tickformat to the 10^k tick positions (0.1% / 1% / 10% / 100%).
    // Non-positive (zero) bins simply drop out, which is correct on a log scale.
    var yaxis = abs
        ? { title: 'Number of users', tickformat: '' }
        : { title: 'Share of group users', tickformat: '.1%' };
    if (useLog) { yaxis.type = 'log'; }
    var layout = {
        title: (abs ? 'Number of users by event' : 'Share of users by event') + ' \\u2014 time since ' + origin + sub,
        xaxis: {
            title: 'Time since ' + origin, tickvals: CFG.ticks.vals, ticktext: CFG.ticks.text,
            rangeslider: { visible: true, thickness: 0.08 }, range: CFG.defaultRange.slice()
        },
        yaxis: yaxis, height: 700, showlegend: true, legend: { orientation: 'h', y: -0.35 }
    };
    Plotly.react('figure', traces, layout);
    rebuildTable(origin, groups, metrics);
}
function rebuildTable(origin, groups, metrics) {
    var rows = '';
    groups.forEach(function (g) {
        metrics.forEach(function (m) {
            var st = CFG.summary[origin][g][m];
            rows += '<tr><td>' + g + '</td><td>' + CFG.events[m] + '</td><td>' + st.reached
                + '</td><td>' + st.pct + '</td><td>' + (st.median == null ? '\\u2013' : st.median)
                + '</td><td>' + (st.p90 == null ? '\\u2013' : st.p90) + '</td></tr>';
        });
    });
    document.getElementById('tbl').innerHTML = rows || '<tr><td colspan="6">No metric selected.</td></tr>';
}
function applyRange() {
    var lo = parseFloat(document.getElementById('rng-min').value);
    var hi = parseFloat(document.getElementById('rng-max').value);
    Plotly.relayout('figure', { 'xaxis.range': [secToIdx(lo), secToIdx(hi)] });
}
function resetRange() { Plotly.relayout('figure', { 'xaxis.range': CFG.defaultRange.slice() }); }
document.querySelectorAll('input[name=origin], .group, .metric, #absToggle, #logToggle').forEach(function (el) {
    el.addEventListener('change', redraw);
});
redraw();
</script>
"""


def _report_config(events: pd.DataFrame, bin_seconds: int) -> dict:
    """Precompute every (origin, group, metric) series + summary stat for the client."""
    groups = _strategy_groups(events)
    data: dict = {}
    summary: dict = {}
    max_idx = _n_fine_bins(bin_seconds)
    for origin_col, origin_label, _ in ORIGINS:
        data[origin_label] = {}
        summary[origin_label] = {}
        for _, label, sub in groups:
            binned = bin_event_counts(sub, origin_col, bin_seconds)
            n_users = len(sub)
            dser: dict = {}
            sser: dict = {}
            for event in EVENT_LABELS:
                d = binned[binned["event"] == event]
                xs = [int(v) for v in d["bin_index"].tolist()]
                cs = [int(v) for v in d["count"].tolist()]
                # y is the share of the group's users per bin so groups of
                # different sizes are comparable; raw counts kept in c for hover.
                ys = [round(c / n_users, 6) if n_users else 0 for c in cs]
                if xs:
                    max_idx = max(max_idx, max(xs))
                dser[event] = {"x": xs, "y": ys, "c": cs}
                offsets = (sub[event] - sub[origin_col]).dt.total_seconds().dropna()
                sser[event] = {
                    "reached": int(len(offsets)),
                    "pct": round(100 * len(offsets) / n_users, 1) if n_users else 0,
                    "median": round(float(offsets.median()), 1) if len(offsets) else None,
                    "p90": round(float(offsets.quantile(0.9)), 1) if len(offsets) else None,
                }
            data[origin_label][label] = dser
            summary[origin_label][label] = sser

    tickvals, ticktext = _axis_ticks(bin_seconds, max_idx)
    group_labels = [label for _, label, _ in groups]
    return {
        "events": dict(EVENT_LABELS),
        "order": list(EVENT_LABELS),
        "groups": group_labels,
        "groupColors": [_group_color(i) for i in range(len(group_labels))],
        "ticks": {"vals": [int(v) for v in tickvals], "text": ticktext},
        "binSeconds": bin_seconds,
        "nFine": _n_fine_bins(bin_seconds),
        "defaultRange": [0, 900 // bin_seconds],
        "data": data,
        "summary": summary,
    }


def _controls_html(group_labels: list, bin_seconds: int) -> str:
    origin_radios = "".join(
        f'<label><input type="radio" name="origin" value="{ol}" {"checked" if i == 0 else ""}> {ol}</label>'
        for i, (_, ol, _) in enumerate(ORIGINS)
    )
    group_boxes = "".join(
        f'<label style="color:{_group_color(i)}">'
        f'<input type="checkbox" class="group" value="{g}" {"checked" if g == "All users" else ""}> {g}</label>'
        for i, g in enumerate(group_labels)
    )
    metric_boxes = "".join(
        f'<label><input type="checkbox" class="metric" value="{e}" {"checked" if e in DEFAULT_METRICS else ""}>'
        f" {lab}</label>"
        for e, lab in EVENT_LABELS.items()
    )
    return (
        f'<div class="ctl-row"><b>Origin:</b> {origin_radios}'
        f' &nbsp;&nbsp; <b>Y-axis:</b> <label><input type="checkbox" id="absToggle">'
        f" show absolute user count (default: ratio)</label>"
        f' &nbsp; <label><input type="checkbox" id="logToggle"> log scale</label></div>'
        f'<div class="ctl-row groups"><b>Groups (one color each):</b> {group_boxes}</div>'
        f'<div class="ctl-row metrics">{metric_boxes}</div>'
        f'<div class="ctl-row"><b>Time range (s):</b> '
        f'<input type="number" id="rng-min" value="0" step="{bin_seconds}" style="width:90px"> &ndash; '
        f'<input type="number" id="rng-max" value="900" step="{bin_seconds}" style="width:90px"> '
        f'<button onclick="applyRange()">Apply</button> '
        f'<button onclick="resetRange()">Reset</button></div>'
    )


def build_report(events: pd.DataFrame, bin_seconds: int, high_fish_value: float, output_path: str) -> None:
    cfg = _report_config(events, bin_seconds)
    group_labels = [label for _, label, _ in _strategy_groups(events)]

    table_head = (
        "<table><thead><tr><th>group</th><th>event</th><th>users reached</th>"
        "<th>% of users</th><th>median (s)</th><th>p90 (s)</th></tr></thead><tbody id='tbl'></tbody></table>"
    )

    parts = [
        "<html><head><title>FTUE Event Timing Report</title>",
        "<style>body{font-family:sans-serif;margin:24px} table{border-collapse:collapse}",
        "td,th{padding:4px 12px;border-bottom:1px solid #ddd;text-align:right}",
        "th{background:#f5f5f5} td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}",
        ".ctl-row{margin:10px 0} .ctl-row.metrics label{margin-right:14px;font-weight:bold}",
        "select{font-size:14px} h2{margin-top:40px}</style></head><body>",
        "<h1>FTUE Event Timing Report (FM01 first session)</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} &middot; {len(events)} users &middot; "
        f"bin = {bin_seconds}s &middot; high-value fish threshold = {high_fish_value}</p>",
        "<p>Users grouped by the strategy_name of their first bullet (session-start strategy).</p>",
        "<h2>Interpretation (All users)</h2>",
        interpretation_html(events, high_fish_value),
        "<h2>Interactive event timing</h2>",
        _controls_html(group_labels, bin_seconds),
        '<div id="figure" style="height:720px"></div>',
        "<h3>Selected metrics &mdash; summary</h3>",
        table_head,
        f'<script src="{PLOTLY_CDN}"></script>',
        "<script>const CFG = " + json.dumps(cfg) + ";</script>",
        APP_JS,
        "</body></html>",
    ]

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
