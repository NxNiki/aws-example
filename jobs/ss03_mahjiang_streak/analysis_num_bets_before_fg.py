"""
Fetch per-``(user_id, activity_date, ab_group_id)`` ``num_bets`` from Redshift and plot its distribution.

SQL semantics: cumulative FREE count per user-day; ``WHERE fg_count = 0`` keeps only rows **before**
the first FREE bet that day; ``GROUP BY`` yields one ``num_bets`` per user-day per AB arm.

Uses :class:`bituslabs_ds.etl.DataLoader` with :class:`RedshiftBackend`, same pattern as
``etl_game_stats_daily_by_user_group.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import cast

import pandas as pd

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, RedshiftBackend

OUTPUT_DIR = Path(LOCAL_ROOT) / "jobs" / "output" / "ss03_mahjiang_streak"
DEFAULT_HTML = OUTPUT_DIR / "num_bets_before_fg_histogram.html"


def _sql_num_bets_no_free() -> str:
    """Redshift SQL: count of BASE/non-FREE-prefix bets before first FREE that user-day (``fg_count = 0`` rows only)."""
    return dedent(
        """
        WITH ranked_bets AS (
            SELECT
                t.user_id,
                t.bet_amount,
                t.bet_type,
                t.partition_ab[0] AS ab_group_id,
                CAST(DATE_TRUNC('day', CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at)) AS DATE) AS activity_date,
                -- get the accumulative number of free game played for at each bet
                SUM(CASE WHEN t.bet_type = 'FREE' THEN 1 ELSE 0 END) OVER(
                    PARTITION BY t.user_id, CAST(DATE_TRUNC('day', CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at)) AS DATE)
                    ORDER BY t.created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS fg_count
            FROM
                public.fct_bet_orders AS t
            WHERE
                t.game_id = 'SS03'
            AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) >= '2026-03-20'
            AND t.currency_type = 'CNY'
            AND t.status = 'COMPLETED'
            AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
            AND t.partition_ab[0] IN ('4a04df21-c749-4808-8e55-3a0b74c084d2', '4f1a46ca-7baa-4452-9a40-ef21d9b33b57')
        )

        SELECT
            activity_date,
            user_id,
            COUNT(bet_amount) AS num_bets,
            ab_group_id
        FROM ranked_bets
        WHERE fg_count = 0
        GROUP BY
            activity_date,
            user_id,
            ab_group_id
        ORDER BY
            activity_date,
            ab_group_id
        ;

        """
    ).strip()


def _escape_json_for_html_script(s: str) -> str:
    return s.replace("</", "<\\/")


def _prepare_num_bets_histogram_payload(df: pd.DataFrame) -> tuple[list[dict], int]:
    """Return JSON-serializable rows ``{{num_bets, ab_group_id}}`` and sample count after dedupe.

    Each ``(user_id, activity_date, ab_group_id)`` is **one** sample; y-axis frequency counts such
    samples per ``num_bets`` bin. Dedupes when ``user_id`` and ``activity_date`` are present.
    """
    if df.empty or not {"num_bets", "ab_group_id"}.issubset(df.columns):
        return [], 0
    work = df.loc[:, [c for c in ("user_id", "activity_date", "num_bets", "ab_group_id") if c in df.columns]].copy()
    work["num_bets"] = pd.to_numeric(work["num_bets"], errors="coerce")
    work["ab_group_id"] = work["ab_group_id"].astype(str)
    work = work.dropna(subset=["num_bets"])
    if {"user_id", "activity_date", "ab_group_id"}.issubset(work.columns):
        work = cast(
            pd.DataFrame,
            work.drop_duplicates(subset=["user_id", "activity_date", "ab_group_id"], keep="first"),
        )
    n = len(work)
    payload_json = work[["num_bets", "ab_group_id"]].to_json(orient="records", date_format="iso")
    rows = cast(list[dict], json.loads(payload_json or "[]"))
    return rows, n


def write_num_bets_histogram_html(
    df: pd.DataFrame,
    out_path: Path,
    *,
    default_bins: int = 50,
    default_threshold: float = 50.0,
) -> tuple[Path, int]:
    """Self-contained HTML: one Plotly histogram per ``ab_group_id``; each point is one user-day sample."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload, n_samples = _prepare_num_bets_histogram_payload(df)

    records_json = json.dumps(payload)
    records_safe = _escape_json_for_html_script(records_json)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SS03 num_bets before first FREE (per user-day)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 1rem auto; padding: 0 1rem; }}
section {{ margin-bottom: 1.5rem; }}
.controls {{ margin-bottom: 0.75rem; display: flex; align-items: center; flex-wrap: wrap; gap: 0.75rem; }}
input[type="number"] {{ width: 5rem; padding: 0.35rem 0.5rem; }}
label {{ margin-right: 0.25rem; }}
h1 {{ font-size: 1.25rem; }}
h2 {{ font-size: 1.05rem; }}
code {{ font-size: 0.9em; }}
</style>
</head>
<body>
<h1>SS03 — Distribution of <code>num_bets</code> (before first FREE bet that day)</h1>
<section>
<p>Each <strong>sample</strong> is one <code>(user_id, activity_date, ab_group_id)</code> row from the query:
<code>num_bets</code> is the number of bet rows in the prefix where cumulative FREE count is still 0
(before the first <code>FREE</code> that calendar day). The histogram <strong>frequency</strong> is how
many user-days fall in each <code>num_bets</code> bin. One chart per <code>ab_group_id</code>.
Use the <strong>threshold</strong> to split counts into <code>num_bets &lt; T</code> (green) vs <code>num_bets ≥ T</code> (blue);
<strong>P</strong> is the fraction of samples with <code>num_bets &lt; T</code>.</p>
<div class="controls">
  <label for="bet-bins">Bins</label>
  <input type="number" id="bet-bins" value="{default_bins}" min="1" max="500" step="1" />
  <label for="bet-threshold">Threshold <code>num_bets</code> &lt;</label>
  <input type="number" id="bet-threshold" value="{default_threshold}" min="0" step="any" title="Bars are split by num_bets strictly below vs at/above this value."/>
  <label><input type="checkbox" id="bet-log" /> Log scale Y-axis (all panels)</label>
</div>
<p id="threshold-note" style="margin: 0.35rem 0 0.75rem; color: #333; font-size: 0.95rem;"></p>
<div id="bet-hist-container"></div>
</section>
<script>
const RAW = {records_safe};

(function () {{
  const binsInput = document.getElementById("bet-bins");
  const thresholdInput = document.getElementById("bet-threshold");
  const logCheck = document.getElementById("bet-log");
  const note = document.getElementById("threshold-note");
  const host = document.getElementById("bet-hist-container");
  const COLOR_BELOW = "#27ae60";
  const COLOR_AT_ABOVE = "#2980b9";
  if (!RAW.length) {{
    host.innerHTML = "<p><em>No rows to plot.</em></p>";
    return;
  }}
  const groups = Array.from(new Set(RAW.map(function (r) {{ return r.ab_group_id; }}))).sort();
  const plotIds = [];

  function draw() {{
    const nbins = Math.max(1, parseInt(binsInput.value, 10) || {default_bins});
    const T = parseFloat(thresholdInput.value);
    const useLog = logCheck.checked;
    const Tvalid = !isNaN(T) && isFinite(T);
    plotIds.forEach(function (id) {{
      const el = document.getElementById(id);
      if (el) {{ Plotly.purge(el); }}
    }});
    plotIds.length = 0;
    host.innerHTML = "";

    const summaryParts = [];
    if (Tvalid) {{
      summaryParts.push("Threshold: <code>num_bets &lt; " + T + "</code>. ");
    }}

    groups.forEach(function (g, idx) {{
      const x = RAW.filter(function (r) {{ return r.ab_group_id === g; }}).map(function (r) {{ return r.num_bets; }});
      const n = x.length;
      let xBelow = [];
      let xAbove = [];
      if (Tvalid) {{
        xBelow = x.filter(function (v) {{ return v < T; }});
        xAbove = x.filter(function (v) {{ return v >= T; }});
      }}
      const belowCount = Tvalid ? xBelow.length : 0;
      const propBelow = n > 0 && Tvalid ? (100 * belowCount / n) : null;

      const panel = document.createElement("div");
      panel.style.marginBottom = "2rem";
      const h = document.createElement("h2");
      h.style.fontSize = "1.05rem";
      h.appendChild(document.createTextNode("ab_group_id: " + g + " (n = " + n + " user-days)"));
      if (propBelow !== null) {{
        h.appendChild(document.createTextNode(" — "));
        const s = document.createElement("span");
        s.innerHTML = "P(<code>num_bets</code> &lt; " + T + ") = <strong>" + propBelow.toFixed(2) + "%</strong> (" + belowCount + "/" + n + ")";
        h.appendChild(s);
      }} else {{
        h.appendChild(document.createTextNode(" — set a numeric threshold to color bars and see proportion below T."));
      }}
      panel.appendChild(h);
      const gd = document.createElement("div");
      const gid = "bet-hist-" + idx;
      gd.id = gid;
      gd.style.width = "100%";
      gd.style.minHeight = "360px";
      panel.appendChild(gd);
      host.appendChild(panel);
      plotIds.push(gid);

      let data;
      const bingroup = "bg-" + idx;
      if (Tvalid) {{
        data = [
          {{
            type: "histogram",
            name: "num_bets < " + T,
            x: xBelow,
            nbinsx: nbins,
            bingroup: bingroup,
            histfunc: "count",
            marker: {{ color: COLOR_BELOW, line: {{ width: 1, color: "#1e8449" }} }},
            opacity: 0.88,
          }},
          {{
            type: "histogram",
            name: "num_bets ≥ " + T,
            x: xAbove,
            nbinsx: nbins,
            bingroup: bingroup,
            histfunc: "count",
            marker: {{ color: COLOR_AT_ABOVE, line: {{ width: 1, color: "#1f618d" }} }},
            opacity: 0.88,
          }},
        ];
      }} else {{
        data = [{{
          type: "histogram",
          name: "all",
          x: x,
          nbinsx: nbins,
          histfunc: "count",
          marker: {{ color: "#7f8c8d", line: {{ width: 1, color: "#333" }} }},
          opacity: 0.85,
        }}];
      }}

      const shapes = Tvalid ? [{{
        type: "line",
        xref: "x",
        yref: "paper",
        x0: T,
        x1: T,
        y0: 0,
        y1: 1,
        line: {{ color: "#555", width: 1.5, dash: "dash" }},
      }}] : [];

      const layout = {{
        title: "Frequency of user-days by num_bets",
        barmode: Tvalid ? "stack" : "overlay",
        xaxis: {{ title: "num_bets (bets before first FREE that day)" }},
        yaxis: {{
          title: "Count of user-days (frequency)",
          type: useLog ? "log" : "linear",
        }},
        showlegend: Tvalid,
        legend: {{ orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 }},
        shapes: shapes,
      }};
      Plotly.newPlot(gd, data, layout, {{ responsive: true }});

      if (propBelow !== null) {{
        summaryParts.push(
          "<strong>" + g + "</strong>: " + propBelow.toFixed(2) + "% &lt; " + T + " (" + belowCount + "/" + n + ")"
        );
      }}
    }});

    if (note) {{
      note.innerHTML = Tvalid
        ? summaryParts.join(" &nbsp;|&nbsp; ")
        : ('Enter a numeric threshold to split each bar into '
          + '<span style="color:' + COLOR_BELOW + '">num_bets &lt; T</span> vs '
          + '<span style="color:' + COLOR_AT_ABOVE + '">num_bets ≥ T</span>, '
          + 'and to show P(num_bets &lt; T).');
    }}
  }}

  binsInput.addEventListener("change", draw);
  binsInput.addEventListener("input", draw);
  thresholdInput.addEventListener("change", draw);
  thresholdInput.addEventListener("input", draw);
  logCheck.addEventListener("change", draw);
  draw();
}})();
</script>
</body>
</html>
"""
    out_path.write_text(page, encoding="utf-8")
    return out_path, n_samples


def main() -> None:
    log_path = Path(LOCAL_ROOT) / "jobs" / "log"
    setup_logging(str(log_path), log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(
        description="Query Redshift for per-user-day num_bets (before first FREE) and write Plotly HTML histogram."
    )
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP for SSH tunnel (default: {DEFAULT_BASTION_IP}); omit for direct Redshift connect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_HTML,
        help=f"Output HTML path (default: {DEFAULT_HTML})",
    )
    parser.add_argument(
        "--default-bins",
        type=int,
        default=50,
        help="Default number of histogram bins in the HTML control (default: 50).",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=50.0,
        help="Default threshold T for num_bets (strictly below T vs ≥ T); shown in HTML input (default: 50).",
    )
    args = parser.parse_args()

    loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="slot-machine",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=args.bastion_ip,
        )
    )
    try:
        df = cast(pd.DataFrame, loader.query_to_df(_sql_num_bets_no_free()))
    finally:
        loader.close()

    out = args.output.expanduser().resolve()
    _html_path, n_samples = write_num_bets_histogram_html(
        df, out, default_bins=args.default_bins, default_threshold=args.default_threshold
    )
    print(f"Query rows: {len(df)}; histogram samples (deduped user-day×AB): {n_samples}; HTML: {out}")


if __name__ == "__main__":
    main()
