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
from bituslabs_ds.reports import escape_json_for_html_script, load_template, render_page

OUTPUT_DIR = Path(LOCAL_ROOT) / "jobs" / "output" / "ss03_mahjiang_streak"
DEFAULT_HTML = OUTPUT_DIR / "num_bets_before_fg_histogram.html"
_JS_TEMPLATE_PATH = Path(__file__).with_suffix(".js")


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
                LOWER(LEFT(t.partition_ab[0], 4)) AS ab_arm,
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
            ab_arm
        FROM ranked_bets
        WHERE fg_count = 0
        GROUP BY
            activity_date,
            user_id,
            ab_arm
        ORDER BY
            activity_date,
            ab_arm
        ;

        """
    ).strip()


def _prepare_num_bets_histogram_payload(df: pd.DataFrame) -> tuple[list[dict], int]:
    """Return JSON-serializable rows ``{num_bets, ab_arm}`` and sample count after dedupe.

    Each ``(user_id, activity_date, ab_arm)`` is **one** sample; y-axis frequency counts such
    samples per ``num_bets`` bin. Dedupes when ``user_id`` and ``activity_date`` are present.
    """
    if df.empty or not {"num_bets", "ab_arm"}.issubset(df.columns):
        return [], 0
    work = df.loc[:, [c for c in ("user_id", "activity_date", "num_bets", "ab_arm") if c in df.columns]].copy()
    work["num_bets"] = pd.to_numeric(work["num_bets"], errors="coerce")
    work["ab_arm"] = work["ab_arm"].astype(str)
    work = work.dropna(subset=["num_bets"])
    if {"user_id", "activity_date", "ab_arm"}.issubset(work.columns):
        work = cast(
            pd.DataFrame,
            work.drop_duplicates(subset=["user_id", "activity_date", "ab_arm"], keep="first"),
        )
    n = len(work)
    payload_json = work[["num_bets", "ab_arm"]].to_json(orient="records", date_format="iso")
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

    records_safe = escape_json_for_html_script(json.dumps(payload))
    inline_js = (
        load_template(_JS_TEMPLATE_PATH)
        .replace('"__RAW__"', records_safe)
        .replace('"__DEFAULT_BINS__"', str(default_bins))
    )

    body = f"""<h1>SS03 — Distribution of <code>num_bets</code> (before first FREE bet that day)</h1>
<section>
<p>Each <strong>sample</strong> is one <code>(user_id, activity_date, ab_arm)</code> row from the query:
<code>num_bets</code> is the number of bet rows in the prefix where cumulative FREE count is still 0
(before the first <code>FREE</code> that calendar day). The histogram <strong>frequency</strong> is how
many user-days fall in each <code>num_bets</code> bin. One chart per <code>ab_arm</code>.
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
</section>"""

    page = render_page(
        title="SS03 num_bets before first FREE (per user-day)",
        body=body,
        inline_scripts=inline_js,
    )
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
