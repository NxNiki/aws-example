"""Build & publish the SS02 AI-group mathtable-permutation report.

Renders the CSV outputs of ``analysis_ai_mathtable_permutation.py`` into a
Simplified-Chinese Confluence page (storage format) and publishes it under
the "SS02_DeepDive_AI调控" folder, attaching the CSVs.

Usage:
    python jobs/ss02_deepdive/generate_ai_permutation_report.py            # preview HTML only
    python jobs/ss02_deepdive/generate_ai_permutation_report.py --publish
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl
from analysis_ai_mathtable_permutation import (
    EXCLUDED_COHORT_DATES,
    MIN_BLOCK_BETS,
    MIN_PERMUTATION_USER_DAYS,
    OUTPUT_DIR,
    PRESENCE_END,
    RANGES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFLUENCE_FOLDER_ID = "319520772"  # SS02_DeepDive_AI调控
# Set after first publish so re-runs update instead of duplicating.
CONFLUENCE_PAGE_ID = "1353777263"
PAGE_TITLE = "SS02 AI组数学表排列分析：留存、下注与RTP（2026-04-08 ~ 2026-09-02）"

RANGE_LABELS = {"full": "2026-04-08 ~ 2026-09-02"}
RANGE_LABELS_ASCII = {"full": "SS02, 2026-04-08 to 2026-09-02"}
METRIC_LABELS = {
    "retention_d1": "次日留存率",
    "avg_user_total_bet": "人均日下注额（CNY）",
    "avg_user_ggr": "用户GGR（CNY/人）",
}
SHORT_NAMES: dict[str, str] = {}
BUCKET_ORDER = ["<10", "10-50", "50-500", "500+"]
ENTITY_COLORS = {
    "AI(全部排列)": "#2a78d6",
    "medium3": "#eb6834",
    "medium5": "#1baf7a",
    "fast1": "#eda100",
}
PANEL_SERIES = {"full": ["AI(全部排列)", "medium3", "medium5", "fast1"]}
CSV_ATTACHMENTS = (
    "fig_bucket_metrics.png",
    "fig_perm_full.png",
    "summary_range.csv",
    "summary_range_by_table.csv",
    "summary_range_by_table_bucket.csv",
    "daily_all_users.csv",
    "summary_permutation.csv",
    "top5_permutations.csv",
    "corr_rtp_activity.csv",
    "fig_corr_rtp.png",
)


def _short(perm: str) -> str:
    parts = [SHORT_NAMES.get(p, p.removeprefix("normal_")) for p in perm.split("|")]
    return "|".join(parts)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _num(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{x:,.{nd}f}"


def _table(headers: list[str], rows: list[list[str]], width: int = 1800) -> str:
    """Storage-format table. ``data-table-width``/``data-layout`` mirror the
    hand-set widths on the live page (2026-09-02) so a republish keeps them;
    per-column pixel widths are left to Confluence's auto-fit."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table data-table-width="{width}" data-layout="center"><tbody><tr>{head}</tr>{body}</tbody></table>'


def range_table(summary: pl.DataFrame) -> str:
    cols = [
        ("用户日 / 用户数", lambda r: f"{r['user_days']:,} / {r['n_users']:,}"),
        ("次日留存率（D1 样本）", lambda r: f"{_pct(r['retention_d1'])}（{r['d1_cohort']:,}）"),
        (
            "人均日下注额 均值/中位（CNY）",
            lambda r: f"{_num(r['avg_user_total_bet'])} / {_num(r['med_user_total_bet'])}",
        ),
        ("人均日下注次数 均值/中位", lambda r: f"{_num(r['avg_user_num_bets'])} / {_num(r['med_user_num_bets'], 0)}"),
        ("单注注金 均值/中位（CNY）", lambda r: f"{_num(r['avg_user_avg_bet'], 2)} / {_num(r['med_user_avg_bet'], 2)}"),
        ("用户RTP 均值/中位", lambda r: f"{_num(r['avg_user_rtp'], 3)} / {_num(r['med_user_rtp'], 3)}"),
        ("整体RTP（合并）", lambda r: _num(r["pooled_rtp"], 3)),
        ("单注间隔 均值/中位（秒）", lambda r: f"{_num(r['avg_bet_interval'])} / {_num(r['med_bet_interval'])}"),
        ("最长连注 均值/中位", lambda r: f"{_num(r['avg_max_streak'], 0)} / {_num(r['med_max_streak'], 0)}"),
        ("用户GGR 均值/中位（CNY/人）", lambda r: f"{_num(r['avg_user_ggr'])} / {_num(r['med_user_ggr'])}"),
    ]
    rows = [[RANGE_LABELS[rec["range"]]] + [f(rec) for _, f in cols] for rec in summary.sort("range").to_dicts()]
    return _table(["时间段"] + [c for c, _ in cols], rows)


def top5_tables(top5: pl.DataFrame) -> str:
    out = []
    for metric, label in METRIC_LABELS.items():
        rows = []
        for rec in top5.filter(pl.col("ranked_by") == metric).sort(["range", "perm_len", "rank"]).to_dicts():
            rows.append(
                [
                    RANGE_LABELS[rec["range"]],
                    str(rec["perm_len"]),
                    str(rec["rank"]),
                    f"<code>{_short(rec['permutation'])}</code>",
                    f"{rec['user_days']:,} / {rec['n_users']:,}",
                    f"{_pct(rec['retention_d1'])}（{rec['d1_cohort']:,}）",
                    f"{_num(rec['avg_user_total_bet'])} / {_num(rec['med_user_total_bet'])}",
                    f"{_num(rec['avg_user_num_bets'])} / {_num(rec['med_user_num_bets'], 0)}",
                    f"{_num(rec['avg_user_avg_bet'], 2)} / {_num(rec['med_user_avg_bet'], 2)}",
                    f"{_num(rec['avg_user_rtp'], 3)} / {_num(rec['med_user_rtp'], 3)}",
                    _num(rec["pooled_rtp"], 3),
                    f"{_num(rec['avg_max_streak'], 0)} / {_num(rec['med_max_streak'], 0)}",
                    f"{_num(rec['avg_user_ggr'])} / {_num(rec['med_user_ggr'])}",
                ]
            )
        out.append(
            f"<h3>按{label}排序</h3>"
            + _table(
                [
                    "时间段",
                    "长度",
                    "名次",
                    "排列",
                    "用户日 / 用户数",
                    "次日留存（样本）",
                    "人均日下注额 均值/中位",
                    "人均日下注次数 均值/中位",
                    "单注注金 均值/中位",
                    "用户RTP 均值/中位",
                    "整体RTP",
                    "最长连注 均值/中位",
                    "用户GGR 均值/中位",
                ],
                rows,
            )
        )
    return "".join(out)


def comparison_table(by_table: pl.DataFrame) -> str:
    out = []
    for rng in RANGE_LABELS:
        rows = []
        for rec in (
            by_table.filter((pl.col("range") == rng) & (pl.col("user_days") >= 100))
            .sort("user_days", descending=True)
            .to_dicts()
        ):
            name = rec["mathtable"]
            label = name if name.startswith("AI") else f"<code>{_short(name)}</code>"
            if name.startswith("AI"):
                label = f"<strong>{label}</strong>"
            rows.append(
                [
                    label,
                    f"{rec['user_days']:,}",
                    f"{_pct(rec['retention_d1'])}（{rec['d1_cohort']:,}）",
                    f"{_num(rec['avg_user_total_bet'])} / {_num(rec['med_user_total_bet'])}",
                    f"{_num(rec['avg_user_num_bets'])} / {_num(rec['med_user_num_bets'], 0)}",
                    f"{_num(rec['avg_user_avg_bet'], 2)} / {_num(rec['med_user_avg_bet'], 2)}",
                    f"{_num(rec['avg_user_rtp'], 3)} / {_num(rec['med_user_rtp'], 3)}",
                    _num(rec["pooled_rtp"], 3),
                    f"{_num(rec['avg_bet_interval'])} / {_num(rec['med_bet_interval'])}",
                    f"{_num(rec['avg_max_streak'], 0)} / {_num(rec['med_max_streak'], 0)}",
                    f"{_num(rec['avg_user_ggr'])} / {_num(rec['med_user_ggr'])}",
                ]
            )
        out.append(
            f"<h3>{RANGE_LABELS[rng]}</h3>"
            + _table(
                [
                    "数学表",
                    "用户日",
                    "次日留存（样本）",
                    "人均日下注额 均值/中位",
                    "人均日下注次数 均值/中位",
                    "单注注金 均值/中位",
                    "用户RTP 均值/中位",
                    "整体RTP",
                    "单注间隔 均值/中位（秒）",
                    "最长连注 均值/中位",
                    "用户GGR 均值/中位",
                ],
                rows,
            )
        )
    return "".join(out)


def daily_summary(daily: pl.DataFrame) -> str:
    agg = (
        daily.filter(pl.col("range").is_not_null())
        .group_by("range")
        .agg(
            pl.col("daily_rtp").mean().alias("mean"),
            pl.col("daily_rtp").min().alias("min"),
            pl.col("daily_rtp").max().alias("max"),
            pl.col("n_users").mean().alias("users"),
        )
        .sort("range")
    )
    rows = [
        [RANGE_LABELS[r["range"]], _num(r["mean"], 3), _num(r["min"], 3), _num(r["max"], 3), _num(r["users"], 0)]
        for r in agg.to_dicts()
    ]
    return _table(["时间段", "日RTP均值", "最小", "最大", "日均活跃用户"], rows, width=1300)


def _fig_label(name: str) -> str:
    return "AI (MAB rotation)" if name.startswith("AI") else _short(name)


METRICS_FIG = [
    ("retention_d1", "D1 retention (%)", "wilson", 100.0, "d1_cohort"),
    ("avg_user_total_bet", "Avg daily total bet (CNY)", "std_user_total_bet", 1.0, "user_days"),
    ("avg_user_num_bets", "Avg daily num bets (BASE)", "std_user_num_bets", 1.0, "user_days"),
    ("avg_user_avg_bet", "Avg bet size (CNY)", "std_user_avg_bet", 1.0, "user_days"),
    ("avg_user_ggr", "GGR per user (CNY, lifetime in range)", "std_user_ggr", 1.0, "n_users_ggr"),
    ("avg_bet_interval", "Median bet interval (s)", "std_bet_interval", 1.0, "n_bet_interval"),
    ("avg_max_streak", "Max bet streak (<200s gaps)", "std_max_streak", 1.0, "n_max_streak"),
]
METRICS_FIG_NAME = "fig_bucket_metrics.png"


def _metric_bars(ax, bkt, rng, series, col, err_kind, scale, n_col="user_days"):
    import numpy as np

    x = np.arange(len(BUCKET_ORDER))
    w = 0.8 / len(series)
    for i, name in enumerate(series):
        rows = {
            r["bet_bucket"]: r for r in bkt.filter((pl.col("range") == rng) & (pl.col("mathtable") == name)).to_dicts()
        }
        vals = np.array([scale * (rows.get(b, {}).get(col) or np.nan) for b in BUCKET_ORDER])
        if err_kind == "wilson":
            lo = vals - np.array([scale * (rows.get(b, {}).get("retention_ci_lo") or np.nan) for b in BUCKET_ORDER])
            hi = np.array([scale * (rows.get(b, {}).get("retention_ci_hi") or np.nan) for b in BUCKET_ORDER]) - vals
        else:
            half = np.array(
                [
                    1.96 * scale * (rows.get(b, {}).get(err_kind) or 0) / max(rows.get(b, {}).get(n_col) or 1, 1) ** 0.5
                    for b in BUCKET_ORDER
                ]
            )
            lo = hi = half
        ax.bar(
            x - 0.4 + (i + 0.5) * w,
            np.nan_to_num(vals),
            w * 0.9,
            color=ENTITY_COLORS[name],
            label=_fig_label(name),
            yerr=[np.nan_to_num(lo), np.nan_to_num(hi)],
            capsize=2,
            error_kw={"lw": 0.9, "ecolor": "#444444"},
        )
    ax.set_xticks(x, BUCKET_ORDER)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def build_bucket_metrics_figure(bkt: pl.DataFrame) -> None:
    """4-metric grid (retention / total bet / num bets / bet size) by daily-spend
    tier, AI vs single tables, 95% CI (Wilson for retention, normal approx on
    the winsorized means otherwise). Placed above the tier table in the doc."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = list(PANEL_SERIES.items())
    fig, axes = plt.subplots(
        len(METRICS_FIG), len(panels), figsize=(6.4 * len(panels), 3.5 * len(METRICS_FIG)), squeeze=False
    )
    for r, (col, ylabel, err_kind, scale, n_col) in enumerate(METRICS_FIG):
        for c, (rng, series) in enumerate(panels):
            ax = axes[r][c]
            _metric_bars(ax, bkt, rng, series, col, err_kind, scale, n_col)
            if r == 0:
                ax.set_title(RANGE_LABELS_ASCII[rng], fontsize=11)
                ax.legend(fontsize=8.5, frameon=False, loc="upper left")
            if r == len(METRICS_FIG) - 1:
                ax.set_xlabel("daily total bet tier (CNY)", fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / METRICS_FIG_NAME, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", OUTPUT_DIR / METRICS_FIG_NAME)


PERM_LEN_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
PERM_FIG_METRICS = [
    ("retention_d1", "D1 retention (%)", "wilson", 100.0, "d1_cohort"),
    ("avg_user_total_bet", "Avg daily total bet (CNY)", "std_user_total_bet", 1.0, "user_days"),
    ("avg_user_ggr", "GGR per user (CNY)", "std_user_ggr", 1.0, "n_users_ggr"),
    ("avg_bet_interval", "Median bet interval (s)", "std_bet_interval", 1.0, "n_bet_interval"),
    ("avg_max_streak", "Max bet streak (<200s gaps)", "std_max_streak", 1.0, "n_max_streak"),
]


def build_perm_figures(top5: pl.DataFrame) -> list[str]:
    """One figure per range: the top-5 permutations per length for each ranking
    metric (retention / total bet / GGR), bars colored by permutation length,
    95% CI whiskers (Wilson for retention, normal approx otherwise)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    names = []
    for rng in RANGE_LABELS:
        fig, axes = plt.subplots(len(PERM_FIG_METRICS), 1, figsize=(12.5, 3.6 * len(PERM_FIG_METRICS)))
        lengths_seen: set[int] = set()
        for ax, (col, ylabel, err_kind, scale, n_col) in zip(axes, PERM_FIG_METRICS):
            # Rhythm metrics have no ranking of their own: show them for the
            # retention-ranked cells (same x categories as the top panel).
            rank_col = (
                col
                if col in ("retention_d1", "avg_user_total_bet", "avg_user_ggr", "avg_max_streak")
                else "retention_d1"
            )
            rows = (
                top5.filter((pl.col("ranked_by") == rank_col) & (pl.col("range") == rng))
                .sort(["perm_len", "rank"])
                .to_dicts()
            )
            if not rows:
                ax.axis("off")
                continue
            xs = np.arange(len(rows))
            vals = np.array([scale * (r[col] if r[col] is not None else np.nan) for r in rows])
            if err_kind == "wilson":
                lo = vals - np.array([scale * (r["retention_ci_lo"] or np.nan) for r in rows])
                hi = np.array([scale * (r["retention_ci_hi"] or np.nan) for r in rows]) - vals
            else:
                half = np.array([1.96 * scale * (r[err_kind] or 0) / max(r[n_col] or 1, 1) ** 0.5 for r in rows])
                lo = hi = half
            colors = [PERM_LEN_COLORS[r["perm_len"]] for r in rows]
            lengths_seen.update(r["perm_len"] for r in rows)
            ax.bar(
                xs,
                np.nan_to_num(vals),
                0.72,
                color=colors,
                yerr=[np.nan_to_num(lo), np.nan_to_num(hi)],
                capsize=2.5,
                error_kw={"lw": 1, "ecolor": "#444444"},
            )
            for i in range(1, len(rows)):
                if rows[i]["perm_len"] != rows[i - 1]["perm_len"]:
                    ax.axvline(i - 0.5, color="#bbbbbb", lw=0.8, ls="--")
            if col == "avg_user_ggr":
                ax.axhline(0, color="#888888", lw=0.8)
            ax.set_xticks(xs, [_short(r["permutation"]) for r in rows], rotation=28, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(axis="y", alpha=0.3, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].legend(
            handles=[
                mpatches.Patch(color=c, label=f"length {n}") for n, c in PERM_LEN_COLORS.items() if n in lengths_seen
            ],
            fontsize=9,
            frameon=False,
            loc="upper right",
            ncols=len(lengths_seen),
        )
        axes[0].set_title(f"Top permutations per length — {RANGE_LABELS_ASCII[rng]}", fontsize=11)
        fig.tight_layout()
        name = f"fig_perm_{rng}.png"
        fig.savefig(OUTPUT_DIR / name, dpi=150, facecolor="white")
        plt.close(fig)
        logger.info("wrote %s", OUTPUT_DIR / name)
        names.append(name)
    return names


def bucket_table(bkt: pl.DataFrame) -> str:
    rows = []
    shown = bkt.filter(pl.col("user_days") >= 100)
    for b in BUCKET_ORDER:
        for rec in shown.filter(pl.col("bet_bucket") == b).sort("user_days", descending=True).to_dicts():
            name = rec["mathtable"]
            label = f"<strong>{name}</strong>" if name.startswith("AI") else f"<code>{_short(name)}</code>"
            rows.append(
                [
                    b,
                    label,
                    f"{rec['user_days']:,}",
                    f"{_pct(rec['retention_d1'])}［{_pct(rec['retention_ci_lo'])}, {_pct(rec['retention_ci_hi'])}］",
                    f"{_num(rec['avg_user_num_bets'])} / {_num(rec['med_user_num_bets'], 0)}",
                    f"{_num(rec['avg_user_total_bet'])} / {_num(rec['med_user_total_bet'])}",
                    f"{_num(rec['avg_user_avg_bet'], 2)} / {_num(rec['med_user_avg_bet'], 2)}",
                    _num(rec["pooled_rtp"], 3),
                    f"{_num(rec['avg_bet_interval'])} / {_num(rec['med_bet_interval'])}",
                    f"{_num(rec['avg_max_streak'], 0)} / {_num(rec['med_max_streak'], 0)}",
                    f"{_num(rec['avg_user_ggr'])} / {_num(rec['med_user_ggr'])}",
                ]
            )
    return _table(
        [
            "日下注额层级",
            "数学表",
            "用户日",
            "次日留存［95% CI］",
            "人均日下注次数 均值/中位",
            "人均日下注额 均值/中位",
            "单注注金 均值/中位",
            "整体RTP",
            "单注间隔 均值/中位（秒）",
            "最长连注 均值/中位",
            "用户GGR 均值/中位",
        ],
        rows,
    )


CORR_FIG = "fig_corr_rtp.png"


def build_corr_figure(ud: pl.DataFrame) -> None:
    """Analysis 4 figure: hexbin of log10(user_rtp+0.01) vs log10(activity)
    per user-day, with a median line over RTP bins."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [("user_total_bet", "Daily total bet (CNY)"), ("user_num_bets", "Daily num bets (BASE)")]
    panels = list(RANGE_LABELS)
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(6.6 * len(panels), 4.4 * len(metrics)), squeeze=False)
    for c, rng in enumerate(panels):
        g = ud.filter((pl.col("range") == rng) & pl.col("user_rtp").is_not_null())
        lx = np.log10(g["user_rtp"].to_numpy() + 0.01)
        for r, (col, ylabel) in enumerate(metrics):
            ax = axes[r][c]
            ly = np.log10(np.maximum(g[col].to_numpy(), 0.01))
            hb = ax.hexbin(lx, ly, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
            edges = np.quantile(lx, np.linspace(0, 1, 21))
            mids, meds = [], []
            for i in range(20):
                m = (lx >= edges[i]) & (lx <= edges[i + 1])
                if m.sum() >= 30:
                    mids.append(np.median(lx[m]))
                    meds.append(np.median(ly[m]))
            ax.plot(mids, meds, color="#eb6834", lw=2, marker="o", ms=3.5, label="median")
            xt = [0.03, 0.1, 0.3, 1, 3, 10]
            ax.set_xticks([np.log10(v + 0.01) for v in xt], [str(v) for v in xt])
            yt = [1, 10, 100, 1000, 10000]
            ax.set_yticks([np.log10(v) for v in yt], ["1", "10", "100", "1k", "10k"])
            if r == 0:
                ax.set_title(RANGE_LABELS_ASCII[rng], fontsize=11)
            if r == len(metrics) - 1:
                ax.set_xlabel("user-day RTP (log scale)", fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel + " (log)", fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / CORR_FIG, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", OUTPUT_DIR / CORR_FIG)


def corr_table(corr: pl.DataFrame) -> str:
    rows = []
    names = {"user_total_bet": "人均日下注额", "user_num_bets": "人均日下注次数"}
    for rec in corr.sort(["range", "metric"]).to_dicts():
        rows.append(
            [
                RANGE_LABELS[rec["range"]],
                names[rec["metric"]],
                f"{rec['n']:,}",
                f"{rec['spearman']:.3f}",
                f"{rec['pearson_loglog']:.3f}",
                f"{rec['pearson_raw']:.3f}",
            ]
        )
    return _table(
        ["时间段", "指标", "用户日 n", "Spearman ρ", "Pearson（log-log）", "Pearson（原始）"], rows, width=1300
    )


def build_page() -> str:
    summary = pl.read_csv(OUTPUT_DIR / "summary_range.csv")
    by_table = pl.read_csv(OUTPUT_DIR / "summary_range_by_table.csv")
    top5 = pl.read_csv(OUTPUT_DIR / "top5_permutations.csv")
    bkt = pl.read_csv(OUTPUT_DIR / "summary_range_by_table_bucket.csv")
    corr = pl.read_csv(OUTPUT_DIR / "corr_rtp_activity.csv")
    build_corr_figure(pl.read_csv(OUTPUT_DIR / "user_day_ai.csv"))
    build_bucket_metrics_figure(bkt)
    build_perm_figures(top5)
    daily = pl.read_csv(OUTPUT_DIR / "daily_all_users.csv", try_parse_dates=True)
    start, end = RANGES["full"]

    return f"""
<h2>摘要</h2>
<p>本报告对 SS02（Deep Dive，仅 CNY 用户，2026-04-08 ~ 09-02）做四部分分析。各部分显著结果：</p>
<ol>
<li><strong>AI 组 vs Default 单表（整体）</strong>（Default 的固定表随时间更换：4月 medium5 → 5月 fast1 →
5月底起 medium3，故有三张单表对照）：<strong>AI 各项活跃指标全面不弱于任一单表</strong>——
留存 5.9% vs medium3 3.2%（约 1.8 倍）/ medium5 5.5% / fast1 2.9%；
下注次数 53 vs 41~52；最长连注 49 ≈ fast1/medium5（48~49）&gt; medium3（39）；整体 RTP 相当（约 0.95）。
用户GGR：AI 23 vs medium3 13 / fast1 19，但 <strong>medium5 最高（37）</strong>——最强单表，
留存接近 AI 且变现更好。剔除 FG 播放后各组单注间隔几乎相同（约 4 秒）。</li>
<li><strong>按每日下注额分层</strong>（&lt;10 / 10-50 / 50-500 / 500+ CNY，按 SS02 分布定标）：
<strong>AI 在每一层都优于 medium3 与 fast1</strong>（如 50-500 层留存 6.1% vs 3.5% / 2.7%，GGR 27 vs 13 / 21）；
对 medium5 在 10-50 / 50-500 / 500+ 层留存领先（500+ 层 18.7% vs 16.9%，区间有重叠），
仅 &lt;10 层 medium5 更高（6.0% vs 2.6%）；500+ 层 GGR medium5（187）高于 AI（151）。图表见分层章节。</li>
<li><strong>AI 组内数学表排列（按长度分层，长度 = 30 注区段数 ≈ 下注量/30；重度玩家子集）</strong>：
同量级下：长度1 以 <code>medium1</code>（21.9%，34 用户日、区间宽）与 <code>fast3</code>（11.1%）居前；
长度2 以<strong>直接在 fast 表间轮换</strong>的 <code>fast2|fast1</code>（留存 19.0%、GGR 286）、
<code>fast1|fast1</code>（15.4%）最佳，medium5 开局组合仅 0%~5.3%。</li>
<li><strong>用户日 RTP 与活跃度的相关</strong>：重尾分布下取 log / 秩相关后，RTP 与当日下注额
ρ≈0.41、与下注次数 ρ≈0.37，弱于 SS03；中位数曲线在 RTP≈1 见顶后回落；系同日相关，不可作因果解读（详见分析4）。</li>
</ol>

<h2>背景</h2>
<p>与 <a href="https://bituslabs.atlassian.net/wiki/pages/viewpage.action?pageId=1353973761">SS03 AI组数学表排列分析</a>
同口径的 SS02（Deep Dive）分析：仅统计 <strong>CNY 币种下注</strong>（MAB 调控只作用于 CNY 用户），
单一时间窗 <strong>{start} ~ {end}</strong>（北京日期）。
SS02 全程只有 <strong>AI 与 Default 两组</strong>：Default 每期固定一张表（4月 <code>medium5</code>、
5月 <code>fast1</code>、5月底起 <code>medium3</code>，各期占其 BASE 下注约 75%~99%），
AI 组由 MAB 在 medium5 / fast1-4 / slow1-2 等表间轮换。
对比 AI 组与 Default 各单表用户日，并在 AI 组内做数学表排列分析。
注意 SS02 体量较小：窗口内 AI 组日均约 59 名 CNY 用户，逐日指标波动较大。</p>

<h2>数据来源与分组</h2>
<ul>
<li>Athena 表 <code>bituslabs_ds.slot_orders_ab_group</code>，底层为 S3 冷数据
<code>s3://bituslabs-team-ai/etl-results/jobs/output_slot_orders_ab_group/orders/game_id=SS02/period=YYYY-MM-DD/</code>，
由 <code>jobs/etl/sagemaker/slot_machine/etl_slot_orders_ab_group.py</code> 每日增量刷新；
源头已剔除未完成订单与测试单。本报告在查询中筛选 <code>currency_type='CNY'</code>。</li>
<li><strong>分组按原始 <code>partition_ab</code> 标签</strong>（首元素 = AI 组 id 记为 AI，其余为 Default）。
该表的全局 <code>ab_group</code> 列自 2026-08-04 起按 user_id 尾号分组——该切换仅适用于 SS03，
<strong>对 SS02 是错误标注</strong>（已验证：08-04 后按尾号标注的"AI"组数学表构成收敛为 Default 的
medium3，而按 partition_ab 标注的 AI 组仍保持 MAB 轮换）。SS02 从未有过 A/B 测试分组，
08-04 后出现的 AB_TEST_A/B 标签为尾号策略的产物。因分组全程连续，无需剔除切换日（08-03/04）。</li>
</ul>

<h2>口径与数据清洗</h2>
<ul>
<li><strong>会话与日期归属</strong>：按 30 分钟无下注间隔切分会话；一次会话的全部下注计入
会话开始时刻的北京日（跨零点不拆分）。留存判定同样按会话开始日。</li>
<li>下注次数 = BASE 下注次数；下注额 = BASE 注金；单注注金 = 当日 BASE 注金合计 / BASE 次数；RTP = 派彩 / BASE 注金（含免费游戏派彩）；仅 CNY。
<strong>用户RTP</strong>为单用户日的比值，<strong>整体RTP</strong>为全体用户合并（∑派彩/∑注金）。</li>
<li><strong>均值 99 分位截断（winsorize）</strong>：user_total_bet、user_num_bets 与单注注金（user_avg_bet_amount）的均值将超过 99 分位的样本
截断到阈值（保留而非剔除）；阈值按全人群用户日分布估计（约 3,711 CNY / 454 次 / 单注 151.1 CNY），统一应用于所有单元格。
中位数、留存率、用户RTP 与整体 RTP 不截断。</li>
<li><strong>用户GGR（lifetime，时段内）</strong>：单个用户在该时间段、该分析人群内全部下注的
（注金 − 派彩）合计，即用户净输给游戏的金额（负值 = 用户净赢），<strong>每人一个值</strong>。
用户在不同天属于不同单元格（数学表 / 下注额层级 / 排列）时，归入其<strong>出现天数最多</strong>的单元格，
并列时归入总规模更大的单元格。GGR 为有符号量，其均值做<strong>双侧</strong> 1/99 分位截断（阈值按全人群估计：约 [−1,106, 890] CNY）；中位数不截断。</li>
<li><strong>下注节奏</strong>：单注间隔 = 用户日内<strong>会话内</strong>相邻 BASE 下注时间差的当日中位数（秒），
<strong>剔除两注之间有免费游戏（FG）播放的间隔</strong>（FG 动画时间并入前一注，非玩家节奏）；
最长连注 = 相邻间隔 &lt;200 秒的连续 BASE 下注的当日最长次数，FG 播放不中断连注。
两者均按"均值（99 分位截断）/ 中位数"展示。</li>
<li><strong>次日留存</strong>：用户次日（北京日）在 SS02 有任意 CNY 下注（不限分组、不做清洗）即算留存。
数据完整至 {PRESENCE_END}，09-01 / 09-02 的次日留存为空；09-02 当日数据不完整。</li>
<li><strong>排列标签的构造与清洗（仅决定标签）</strong>：同一用户日内连续同表的一段下注按<strong>每 30 次 BASE</strong>
拆分为区段（连续 60 次 fast1 → <code>fast1|fast1</code>，长度 2——相邻重复不合并，标签长度即约 30 注一档的下注量）。
区段取 30 而非 SS03 的 50，因为<strong>SS02 的 MAB 约每 30 注轮换一次</strong>（AI 组连续同表 run 长度分布
在 30~34 注处有 34.7% 的峰值，仅 1.8% 的 run 达到 100 注）；
不足 {MIN_BLOCK_BETS} 次 BASE 的余段从排列标签中剔除，但该用户日的下注额、下注次数与 RTP 仍按全天全部下注计算。
SS02 打法较轻（AI 用户日 BASE 次数中位数仅 28），清洗后约 60% 的 AI 用户日可标注——
<strong>排列分析实际覆盖的是较重度的玩家子集</strong>，其留存水平天然高于 AI 组整体，排列之间横向可比，
与整体对比时需注意。</li>
<li><strong>FourScatter</strong> 为主表内触发/购买的功能旋转（全部为 BASE、平均注金约 154 CNY），
视为主表的子模式：不作为第二张表、不参与标注，但计入当日合计。</li>
<li>单表对比原则上剔除同日使用 ≥2 张主数学表的用户日；按 partition_ab 分组后
<strong>Default 组 100% 为单表用户日</strong>（6,954 / 6,954，锁定 medium3），实际无剔除。
<strong>AI(全部排列)</strong> 为 AI 组整体。</li>
</ul>

<h2>AI 组 vs Default 单表用户日（仅 CNY，未清洗）</h2>
{comparison_table(by_table)}
<p>要点：<strong>AI 轮换整体优于任一固定单表</strong>——次日留存 5.9%，vs <code>medium3</code> 3.2%
（约 1.8 倍）、<code>medium5</code> 5.5%、<code>fast1</code> 2.9%；下注次数（53）与下注额（213）不低于任一单表，
整体 RTP 相当（0.952 vs 0.90~0.96）。<code>medium5</code> 是最强单表（留存 5.5%、用户GGR 37）。
这与 SS03 相反（SS03 范围1 中 AI 略逊于头部单表）。SS02 整体留存水平远低于 SS03（约 6% vs 约 20%）。</p>

<h2>按日历日的全体用户 RTP</h2>
{daily_summary(daily)}
<p>日均 RTP 0.945；因 AI 组日均仅约 59 名 CNY 用户，日 RTP 波动大（0.49 ~ 2.41）。
逐日明细见附件 <code>daily_all_users.csv</code>。</p>

<h2>按每日下注额分层：AI 组 vs Default 单表（仅 CNY，未清洗）</h2>
<p>层级按用户日的<strong>原始</strong>（未截断）日下注总额划分：&lt;10 / 10-50 / 50-500 / 500+ CNY
（边界约在全人群的 p18 / 中位数 / p91；SS02 玩家日下注中位数仅 49 CNY，故门槛低于 SS03 的 10/100/1000）。
留存附 95% Wilson 置信区间；均值列仍为 99 分位截断口径。上方摘要图为本节的留存对比。</p>
<p><ac:image ac:align="center" ac:width="1100"><ri:attachment ri:filename="fig_bucket_metrics.png" /></ac:image></p>
{bucket_table(bkt)}

<h2>Top-5 数学表排列（每指标；仅统计 ≥ {MIN_PERMUTATION_USER_DAYS} 用户日的排列；
排列标签经清洗，指标为全天口径）</h2>
<p><strong>按排列长度分层排名</strong>：长度 = 标签中 30 注区段的个数（含重复，连续 60 次 fast1 记
<code>fast1|fast1</code> 长度 2），即长度 ≈ 当日清洗后 BASE 下注量 / 30。更活跃的用户下注更多、标签天然更长，
不同长度直接对比反映的是活跃度（反向因果）而非表序效应，故 Top-5 仅在<strong>相同长度</strong>（1/2/3/4，
即下注量相近）内比较；排名指标为次日留存、人均日下注额与用户GGR。长度 ≥5 的重度日不参与排名。</p>
<p>达到 {MIN_PERMUTATION_USER_DAYS} 用户日门槛的排列共 29 个（共 865 个）；20~30 用户日的单元格置信区间较宽，解读需谨慎。</p>
<p><ac:image ac:align="center" ac:width="1100"><ri:attachment ri:filename="fig_perm_full.png" /></ac:image></p>
{top5_tables(top5)}

<h2>分析4：用户日 RTP 与下注活跃度的相关性（AI 组，仅 CNY，未清洗）</h2>
<p>问题：当天体验到更高 RTP 的用户是否下注更多？逐用户日计算 user_rtp 与
user_total_bet / user_num_bets 的相关。三个变量都严重右偏（偏度：RTP 38、下注额 61、下注次数 18），
<strong>原始 Pearson 几乎为 0</strong>（被极值支配），故对两侧取 log10（RTP 加 0.01 处理零派彩日）后计算
Pearson，并以对单调变换不敏感的 <strong>Spearman 秩相关为主</strong>。</p>
{corr_table(corr)}
<p><ac:image ac:align="center" ac:width="760"><ri:attachment ri:filename="fig_corr_rtp.png" /></ac:image></p>
<p>结果：RTP 与下注额 ρ≈0.41、与下注次数 ρ≈0.37（log-log Pearson 相近），中位数曲线在 RTP 0~1 区间上升、RTP&gt;1 后回落（极端高 RTP 多为少量下注即中奖的日子），
整体弱于 SS03（ρ≈0.45~0.55）。<strong>注意不可作因果解读</strong>：同日相关混合了"赢→继续玩"与
"玩得多→更可能触发派彩"两个方向，且下注很少的日子 RTP 方差天然大。
如需因果结论应做随机化实验或滞后（次日）分析。</p>

<h2>结论</h2>
<ul>
<li><strong>SS02 上 AI 轮换整体优于任一固定单表</strong>：留存 5.9% vs medium3 3.2%、medium5 5.5%、
fast1 2.9%，下注次数与下注额不低于任一单表，RTP 基本持平——与 SS03 的结论方向相反。
单表之间差异也大：medium5 明显强于 medium3 与 fast1。</li>
<li>排列分析（重度玩家子集，<strong>按长度分层</strong>，长度 = 30 注区段数——跨长度对比受活跃度反向因果影响）：
长度1 以 <code>medium1</code>（21.9%，34 用户日、区间宽）、<code>fast3</code>（11.1%）居前；
长度2 中<strong>直接在 fast 表间轮换</strong>最佳——<code>fast2|fast1</code>（留存 19.0%、GGR 286）、
<code>fast1|fast1</code>（15.4%），medium5 开局组合仅 0%~5.3%；
长度3 以 <code>medium5|medium5|fast4</code>（10.0%）居首，长度4 各单元格 0%~5.1%。</li>
<li>整体 RTP 约 0.95，无以派彩换活跃的迹象；但样本量小（排列单元格多在 30~170 用户日），结论需谨慎外推。</li>
<li><strong>用户GGR</strong>（时段内人均净输额，双侧截断均值）：AI 组整体 23，vs medium3 13、fast1 19、
medium5 37——medium5 单表用户的人均净输最高；分层看 AI 在 50-500 层领先 medium3（27 vs 13）但低于
medium5（47），500+ 层 medium5（187）&gt; fast1（157）&gt; AI（151）&gt; medium3（132），区间较宽。
AI 轮换在提升留存与活跃的同时收入不降，但对中高投入用户 medium5 的变现更强。</li>
<li><strong>数据质量提示</strong>：<code>slot_orders_ab_group</code> 的全局 <code>ab_group</code> 列对
SS02 在 2026-08-04 之后是错误标注（尾号切换仅适用于 SS03）；SS02 相关分析请改用
<code>partition_ab_label</code> 分组。</li>
</ul>

<h2>数据与脚本</h2>
<ul>
<li>分析脚本：<code>jobs/ss02_deepdive/analysis_ai_mathtable_permutation.py</code>（--extract / --analyze）。</li>
<li>输出目录：<code>jobs/output_ss02_deepdive/ai_mathtable_permutation/</code>；汇总 CSV 已作为附件上传。</li>
<li>报告生成时间：{__import__("datetime").datetime.now():%Y-%m-%d %H:%M}（本页由脚本生成，手工修改会在下次发布时被覆盖）。</li>
</ul>
"""


def publish(body: str) -> None:
    from bituslabs_ds.confluence.client import _v2_session, attach_file, get_page_storage, update_page_storage

    if CONFLUENCE_PAGE_ID:
        from bituslabs_ds.confluence.client import _get_client

        confluence = _get_client()
        # Update attachments IN PLACE (stable attachment ids) and BEFORE the
        # body save: Confluence re-pins each image's ri:version-at-save to the
        # attachment's current version when the body is saved, so this order
        # keeps user-edited bodies pointing at renderable versions. Delete +
        # re-create only as a fallback (it resets the id and needs the body
        # save afterwards to heal the pins).
        for name in CSV_ATTACHMENTS:
            ctype = "image/png" if name.endswith(".png") else "text/csv"
            data = (OUTPUT_DIR / name).read_bytes()
            try:
                attach_file(CONFLUENCE_PAGE_ID, data, name, content_type=ctype)
            except Exception as exc:
                logger.warning("in-place update failed for %s (%s); replacing", name, exc)
                try:
                    confluence.delete_attachment(CONFLUENCE_PAGE_ID, name)
                except Exception:
                    pass
                attach_file(CONFLUENCE_PAGE_ID, data, name, content_type=ctype)
            logger.info("attached %s", name)
        current = get_page_storage(CONFLUENCE_PAGE_ID)
        update_page_storage(CONFLUENCE_PAGE_ID, PAGE_TITLE, body, expected_version=current["version"])
        logger.info("updated page id=%s", CONFLUENCE_PAGE_ID)
        return

    session, base = _v2_session()
    resp = session.get(f"{base}/folders/{CONFLUENCE_FOLDER_ID}", timeout=15)
    resp.raise_for_status()
    space_id = resp.json()["spaceId"]
    resp = session.post(
        f"{base}/pages",
        json={
            "spaceId": space_id,
            "status": "current",
            "title": PAGE_TITLE,
            "parentId": CONFLUENCE_FOLDER_ID,
            "body": {"representation": "storage", "value": body},
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code >= 300:
        raise SystemExit(f"page creation failed: {resp.status_code} {resp.text[:500]}")
    page = resp.json()
    for name in CSV_ATTACHMENTS:
        attach_file(page["id"], (OUTPUT_DIR / name).read_bytes(), name)
        logger.info("attached %s", name)
    link = page.get("_links", {})
    logger.info("created page id=%s: %s%s", page["id"], link.get("base", ""), link.get("webui", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="create/update the Confluence page")
    args = parser.parse_args()
    body = build_page()
    preview = OUTPUT_DIR / "report_storage.html"
    preview.write_text(body, encoding="utf-8")
    logger.info("wrote preview %s", preview)
    if args.publish:
        publish(body)


if __name__ == "__main__":
    main()
