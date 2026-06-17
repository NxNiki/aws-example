"""Publish FTUE first-session strategy-comparison findings to Confluence.

Report job: renders one comparison figure per milestone metric (ratio of a
group's users reaching the event vs time, one line per strategy group) from the
ftue_first_session dataset, uploads them as page attachments, and APPENDS a
findings section to an existing Confluence page. The section is wrapped in
sentinel comments so re-runs replace it in place rather than stacking copies;
existing page content outside the section is never touched.

Usage:
    python report_ftue_confluence.py --page-url <url> [--origin "first bet"]
    python report_ftue_confluence.py --dry-run            # write PNGs + XHTML locally, no publish

Credentials come from CONFLUENCE_URL / CONFLUENCE_EMAIL / CONFLUENCE_TOKEN
(loaded from .env by bituslabs_ds.config).
"""

import argparse
import os

import plotly.graph_objects as go
from analysis_ftue_events import (
    DEFAULT_INPUT,
    EVENT_LABELS,
    ORIGINS,
    _axis_ticks,
    _fmt_duration,
    _n_fine_bins,
    _strategy_groups,
    bin_event_counts,
    compute_user_events,
    load_data,
)

from bituslabs_ds.config import LOCAL_ROOT, setup_logging

# One color per strategy group (figures encode group by color since each
# figure is a single metric). High-contrast ColorBrewer Set1, matching the
# interactive HTML report's GROUP_PALETTE.
GROUP_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]

SECTION_START = "<!-- ftue-strategy-report:start -->"
SECTION_END = "<!-- ftue-strategy-report:end -->"

DEFAULT_OUTPUT_DIR = f"{LOCAL_ROOT}/jobs/output_fish_hunter/ftue_confluence"

# Static figures show the first 15 min window, where the first-session
# bet/kill/onset action concentrates; later deposit/withdrawal tails are in the
# interactive HTML report.
FIGURE_WINDOW_SECONDS = 900


def _origin_spec(origin_label: str) -> tuple:
    for origin_col, label, _ in ORIGINS:
        if label == origin_label:
            return origin_col, label
    raise ValueError(f"Unknown origin {origin_label!r}; choose from {[o[1] for o in ORIGINS]}")


def _comparison_groups(events) -> list:
    """Strategy levels only (drop the 'All users' aggregate), each (label, subset)."""
    return [(label, sub) for key, label, sub in _strategy_groups(events) if key != "__all__"]


def _metric_png(events, groups: list, origin_col: str, origin_label: str, event: str, bin_seconds: int) -> bytes:
    fig = go.Figure()
    tick_max = _n_fine_bins(bin_seconds)
    for gi, (label, sub) in enumerate(groups):
        binned = bin_event_counts(sub, origin_col, bin_seconds)
        d = binned[binned["event"] == event]
        n = len(sub)
        xs = [int(v) for v in d["bin_index"].tolist()]
        ys = [c / n if n else 0 for c in d["count"].tolist()]
        if xs:
            tick_max = max(tick_max, max(xs))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"{label} (n={n})",
                line={"color": GROUP_COLORS[gi % len(GROUP_COLORS)], "width": 1.5},
                marker={"size": 4},
            )
        )
    tickvals, ticktext = _axis_ticks(bin_seconds, tick_max)
    fig.update_layout(
        title=f"{EVENT_LABELS[event]} — share of users vs time since {origin_label}",
        height=430,
        width=860,
        xaxis={
            "title": f"time since {origin_label}",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "range": [0, FIGURE_WINDOW_SECONDS // bin_seconds],
        },
        yaxis={"title": "share of group users", "tickformat": ".0%"},
        legend={"orientation": "h", "y": -0.25},
        margin={"t": 50, "b": 40},
    )
    return fig.to_image(format="png", scale=2)


def _comparison_table(events, groups: list, origin_col: str, high_fish_value: float) -> str:
    """Storage-format table: rows = metrics, cols = groups, cell = '<pct>% (med <t>)'."""
    head = "<th>metric</th>" + "".join(f"<th>{label} (n={len(sub)})</th>" for label, sub in groups)
    rows = [f"<tr>{head}</tr>"]
    for event, ev_label in EVENT_LABELS.items():
        cells = [f"<td>{ev_label}</td>"]
        for _, sub in groups:
            n = len(sub)
            offsets = (sub[event] - sub[origin_col]).dt.total_seconds().dropna()
            pct = round(100 * len(offsets) / n, 1) if n else 0
            med = _fmt_duration(offsets.median()) if len(offsets) else "n/a"
            cells.append(f"<td>{pct}% (med {med})</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def _findings_bullets(events, groups: list, origin_col: str) -> str:
    """Cross-group highlights computed from the data (extremes per key metric)."""

    def pct_reach(sub, event):
        n = len(sub)
        return 100 * sub[event].notna().sum() / n if n else 0

    def med_after(sub, event):
        return (sub[event] - sub[origin_col]).dt.total_seconds().dropna().median()

    lines = []
    for event in ("first_kill", "first_high_kill", "first_deposit", "second_deposit", "first_withdrawal"):
        ranked = sorted(groups, key=lambda g: pct_reach(g[1], event), reverse=True)
        hi_label, hi_sub = ranked[0]
        lo_label, lo_sub = ranked[-1]
        lines.append(
            f"<li><b>{EVENT_LABELS[event]}:</b> highest reach in <b>{hi_label}</b> "
            f"({pct_reach(hi_sub, event):.1f}% of users, median {_fmt_duration(med_after(hi_sub, event))}), "
            f"lowest in <b>{lo_label}</b> ({pct_reach(lo_sub, event):.1f}%)."
        )
    return "<ul>" + "".join(lines) + "</ul>"


def build_section(events, groups, origin_col, origin_label, high_fish_value, image_names, generated):
    """Assemble the storage-format XHTML section (uses unicode, not named entities)."""
    n_total = sum(len(sub) for _, sub in groups)
    parts = [
        SECTION_START,
        "<h1>FTUE First-Session Strategy Comparison (FM01)</h1>",
        f"<p>Generated {generated} from the ftue_first_session dataset "
        f"({n_total} users across {len(groups)} strategy groups). Each user is assigned the "
        f"strategy_name of their first bullet. Time is measured since {origin_label}; the y-axis is the "
        f"share of each group&#8217;s users reaching the event by that time. High-value fish threshold = "
        f"{high_fish_value:g}.</p>",
        "<h2>Cross-group highlights</h2>",
        _findings_bullets(events, groups, origin_col),
        "<h2>Summary (% of users reaching each event, median time)</h2>",
        _comparison_table(events, groups, origin_col, high_fish_value),
        "<h2>Per-metric comparison</h2>",
    ]
    for event, fname in image_names:
        parts.append(f"<h3>{EVENT_LABELS[event]}</h3>")
        parts.append(f'<ac:image ac:width="820"><ri:attachment ri:filename="{fname}" /></ac:image>')
    parts.append(
        "<p><em>Caveat: only the first bet session is observed (scanned up to 12 h after first play), "
        "so &#8220;stop play&#8221; is the end of the first session, not churn.</em></p>"
    )
    parts.append(SECTION_END)
    return "\n".join(parts)


def _splice(existing: str, section: str) -> str:
    """Replace an existing marked section in place, else append."""
    if SECTION_START in existing and SECTION_END in existing:
        end = existing.index(SECTION_END) + len(SECTION_END)
        pre = existing[: existing.index(SECTION_START)]
        post = existing[end:]
        return pre + section + post
    return existing + ("\n" if existing else "") + section


def main():
    parser = argparse.ArgumentParser(description="Publish FTUE strategy-comparison findings to Confluence")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Parquet dataset path (S3 or local)")
    parser.add_argument("--page-url", default=None, help="Confluence page URL or ID to update (append section)")
    parser.add_argument("--origin", default="first bet", choices=[o[1] for o in ORIGINS], help="Time origin")
    parser.add_argument("--bin-seconds", type=int, default=30)
    parser.add_argument("--high-fish-value", type=float, default=500)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where --dry-run writes PNGs + XHTML")
    parser.add_argument("--dry-run", action="store_true", help="Render figures + XHTML locally; do not publish")
    args = parser.parse_args()

    from datetime import datetime

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    origin_col, origin_label = _origin_spec(args.origin)

    df = load_data(args.input)
    events = compute_user_events(df, args.high_fish_value)
    groups = _comparison_groups(events)
    print(f"Comparison groups: {[(label, len(sub)) for label, sub in groups]}")

    images = []  # (event, filename, png_bytes)
    for event in EVENT_LABELS:
        png = _metric_png(events, groups, origin_col, origin_label, event, args.bin_seconds)
        images.append((event, f"ftue_{event}_{origin_col}.png", png))
    image_names = [(event, fname) for event, fname, _ in images]

    section = build_section(events, groups, origin_col, origin_label, args.high_fish_value, image_names, generated)

    if args.dry_run or not args.page_url:
        os.makedirs(args.output_dir, exist_ok=True)
        for _, fname, png in images:
            with open(os.path.join(args.output_dir, fname), "wb") as f:
                f.write(png)
        with open(os.path.join(args.output_dir, "section.xhtml"), "w") as f:
            f.write(section)
        print(f"[dry-run] Wrote {len(images)} figures + section.xhtml to {args.output_dir}")
        if not args.page_url:
            print("[dry-run] No --page-url given; nothing published.")
        return

    # Publish: resolve page, attach images, splice section into existing storage.
    from dashboards.confluence_client import _get_client, attach_file, extract_page_id_from_url, update_page_storage

    page_id = extract_page_id_from_url(args.page_url) or args.page_url
    page = _get_client().get_page_by_id(page_id, expand="body.storage,version")
    title = str(page.get("title", "Untitled"))
    existing = str(((page.get("body") or {}).get("storage") or {}).get("value", "") or "")
    print(f"Publishing to page {page_id} ({title!r}); existing storage {len(existing)} chars")

    for _, fname, png in images:
        attach_file(page_id, png, fname, content_type="image/png", comment="FTUE strategy comparison figure")
    print(f"Attached {len(images)} figures.")

    new_storage = _splice(existing, section)
    update_page_storage(page_id, title, new_storage)
    action = "Replaced existing section" if SECTION_START in existing else "Appended new section"
    print(f"{action} on page {page_id}.")


if __name__ == "__main__":
    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")
    main()
