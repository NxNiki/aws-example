"""Build & publish the SS03 AI-group mathtable-permutation report (AIP-575).

Renders the CSV outputs of ``analysis_ai_mathtable_permutation.py`` into a
Simplified-Chinese Confluence page (storage format) and publishes it under
the "SS03_MahjiangStreak_数学表分析" folder, attaching the CSVs.

Usage:
    python jobs/ss03_mahjiang_streak/generate_ai_permutation_report.py            # preview HTML only
    python jobs/ss03_mahjiang_streak/generate_ai_permutation_report.py --publish
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

CONFLUENCE_FOLDER_ID = "1267040280"  # SS03_MahjiangStreak_数学表分析
# Set after first publish so re-runs update instead of duplicating.
CONFLUENCE_PAGE_ID = "1353973761"
PAGE_TITLE = "SS03 AI组数学表排列分析：留存、下注与RTP（2026-06-19~08-17 vs 2026-08-19~09-02）"

RANGE_LABELS = {"range1_pre": "范围1（06-19 ~ 08-17）", "range2_post": "范围2（08-19 ~ 09-02）"}
RANGE_LABELS_ASCII = {"range1_pre": "Range 1 (06-19 to 08-17)", "range2_post": "Range 2 (08-19 to 09-02)"}
METRIC_LABELS = {
    "retention_d1": "次日留存率",
    "avg_user_total_bet": "人均日下注额（CNY）",
    "avg_user_ggr": "用户GGR（CNY/人）",
}
SHORT_NAMES = {
    "normal_zero_95_kai": "95_kai",
    "normal_Zero_BGTR95_Saitekika_BGadj_v3": "BGTR95_v3",
    "normal_Zero_BGTR97_Saitekika_BGadj_v2": "BGTR97_v2",
    "normal_Zero_BG97_Saitekika_BGadj": "BG97_adj",
}
BUCKET_ORDER = ["<10", "10-100", "100-1000", "1000+"]
# Fixed entity->color map (dataviz palette, adjacency validated per panel);
# color follows the entity across both panels.
ENTITY_COLORS = {
    "AI(全部排列)": "#2a78d6",
    "normal_zero_95_kai": "#eb6834",
    "normal_zero": "#1baf7a",
    "normal_zero93kai": "#eda100",
    "normal_Zero_BG97_Saitekika_BGadj": "#e87ba4",
    "normal_Zero_BGTR97_Saitekika_BGadj_v2": "#008300",
    "normal_Zero_BGTR95_Saitekika_BGadj_v3": "#4a3aa7",
    "normal_shi": "#e34948",
}
# Every table shown in the tier table (>=100 user-days per cell) is plotted;
# bar order per panel follows the validated palette adjacency.
PANEL_SERIES = {
    "range1_pre": [
        "AI(全部排列)",
        "normal_zero_95_kai",
        "normal_zero",
        "normal_zero93kai",
        "normal_Zero_BG97_Saitekika_BGadj",
        "normal_Zero_BGTR95_Saitekika_BGadj_v3",
        "normal_shi",
    ],
    "range2_post": [
        "AI(全部排列)",
        "normal_zero_95_kai",
        "normal_Zero_BGTR95_Saitekika_BGadj_v3",
        "normal_Zero_BGTR97_Saitekika_BGadj_v2",
    ],
}
CSV_ATTACHMENTS = (
    "fig_bucket_metrics.png",
    "fig_perm_range1_pre.png",
    "fig_perm_range2_post_v2.png",
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
    for rng in ("range1_pre", "range2_post"):
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
    return "AI (all tables)" if name.startswith("AI") else _short(name)


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
        len(METRICS_FIG), len(panels), figsize=(7.6 * len(panels), 3.6 * len(METRICS_FIG)), squeeze=False
    )
    for r, (col, ylabel, err_kind, scale, n_col) in enumerate(METRICS_FIG):
        for c, (rng, series) in enumerate(panels):
            ax = axes[r][c]
            _metric_bars(ax, bkt, rng, series, col, err_kind, scale, n_col)
            if r == 0:
                ax.set_title(RANGE_LABELS_ASCII[rng], fontsize=11)
                ax.legend(fontsize=8, frameon=False, loc="upper left", ncols=2)
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
        # range2's original filename is retired: a broken media reference in the
        # user-edited page body kept re-materializing it as a 0-byte attachment.
        name = "fig_perm_range2_post_v2.png" if rng == "range2_post" else f"fig_perm_{rng}.png"
        fig.savefig(OUTPUT_DIR / name, dpi=150, facecolor="white")
        plt.close(fig)
        logger.info("wrote %s", OUTPUT_DIR / name)
        names.append(name)
    return names


def bucket_table(bkt: pl.DataFrame) -> str:
    out = []
    for rng in RANGE_LABELS:
        rows = []
        shown = bkt.filter((pl.col("range") == rng) & (pl.col("user_days") >= 100))
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
        out.append(
            f"<h3>{RANGE_LABELS[rng]}</h3>"
            + _table(
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
        )
    return "".join(out)


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
    (r1s, r1e), (r2s, r2e) = RANGES["range1_pre"], RANGES["range2_post"]

    return f"""
<h2>摘要</h2>
<p>本报告对 SS03 AI 组（MAB 数学表轮换，仅 CNY 用户）做四部分分析，
对比 08-18 轮换表更换前后两个时间段。各部分显著结果：</p>
<ol>
<li><strong>AI 组 vs 固定单表（整体）</strong>：留存——范围1 AI（18.4%）略逊于头部单表（最高 19.9%），
范围2 持平（21.4% vs 20.7%~22.5%）；整体 RTP AI 略高（范围2 0.992 vs 0.935~1.013）；
用户GGR 与单表相当（范围1 55 vs 43~59，范围2 51 vs 35~55 CNY/人）。
节奏——各表单注间隔相近（约 3.5 秒；<code>BG97_adj</code> 例外，约 7 秒），
但 <strong>AI 组最长连注短于单表</strong>（范围1 124 vs 131~173，范围2 157 vs 160~203）——轮换会打断连注。</li>
<li><strong>按每日下注额分层</strong>（&lt;10 / 10-100 / 100-1000 / 1000+ CNY）：
<strong>AI 的优势集中在高投入用户</strong>——1000+ 层留存两个范围都最高（48.5% / 50.2%，单表 40%~47%），
且范围1 该层人均 GGR 约 415，明显高于单表的 285~320（对鲸鱼用户留存与收入双优）；
单注注金也更大（8.1 vs 约 7.0 CNY）。&lt;100 的低投入层 AI 不占优
（范围2 10-100 层 10.2%，低于 BGTR95_v3 单表的 14.0%）。图表见分层章节。</li>
<li><strong>AI 组内数学表排列（按长度分层，长度 = 50 注区段数 ≈ 下注量/50）</strong>：
同量级下 <strong>95_kai 单表日留存显著最差</strong>（范围2 长度1 仅 6.0%（745 用户日），
vs <code>shi</code> 29.4%、<code>ni</code> 27.8%、<code>BGTR95_v3</code> 21.1%）；
含 BGTR95_v3 区段的排列在各长度上普遍优于 95_kai 为主的排列（长度2 17.1% vs 7.1%，长度3 25.9% vs 9.3%），
GGR 也更高（长度2 BGTR95_v3|san GGR 294）。排列间单注节奏差异小（约 3 秒）——留存差异并非节奏所致。</li>
<li><strong>用户日 RTP 与活跃度的相关</strong>：重尾分布下取 log / 秩相关后，RTP 与当日下注次数
ρ≈0.54~0.55、与下注额 ρ≈0.45~0.47；中位数曲线在 RTP≈1 处见顶、之后回落（极端高 RTP 多为少量下注即中大奖的日子）；
系同日相关，不可作因果解读（详见分析4）。</li>
</ol>

<h2>背景</h2>
<p>Jira 工单：<a href="https://bituslabs.atlassian.net/browse/AIP-575">AIP-575</a>（Epic AIP-21 Data Analysis）。
本报告仅针对 <strong>AI 组</strong>、且<strong>仅统计 CNY 币种下注</strong>（MAB 调控只作用于 CNY 用户），
对比两个时间段（北京日期）的表现：
<strong>范围1：{r1s} ~ {r1e}</strong>，<strong>范围2：{r2s} ~ {r2e}</strong>。
两段之间（08-18）AI 轮换的数学表清单发生了变化：范围1 基本在 8 张 <code>normal_*</code> 数字表间均匀轮换；
范围2 以 <code>normal_zero_95_kai</code> 和 08-18 上线的新表
<code>normal_Zero_BGTR95_Saitekika_BGadj_v3</code> 为主，数字表大幅退场。</p>

<h2>数据来源</h2>
<ul>
<li>Athena 表 <code>bituslabs_ds.slot_orders_ab_group</code>，底层为 S3 冷数据
<code>s3://bituslabs-team-ai/etl-results/jobs/output_slot_orders_ab_group/orders/game_id=SS03/period=YYYY-MM-DD/</code>，
由 <code>jobs/etl/sagemaker/slot_machine/etl_slot_orders_ab_group.py</code> 每日增量刷新（滚动最近 3 个北京日）。</li>
<li>该表是订单级 <code>bet_order</code> 的拷贝，并补充了<strong>策略校正后的 <code>ab_group</code></strong>
（2026-08-04 05:30 北京时间切换前按 <code>partition_ab</code>，切换后按 user_id 尾号），
本报告的 AI/固定表分组均取自该列；源头已剔除未完成订单（status ≠ COMPLETED）与测试单
（op_code B26/TST/TSB/TSO）。</li>
<li>该表本身不做币种过滤（含全部币种）；<strong>本报告在查询中筛选
<code>currency_type='CNY'</code></strong>，因为 MAB 调控只作用于 CNY 用户——与 DS 仪表盘
<code>*_stats</code> 数据集及特征工程数据集的 CNY-only 口径一致。金额单位即 CNY，无汇率折算。</li>
</ul>

<h2>口径与数据清洗</h2>
<ul>
<li>筛选条件：<code>game_id='SS03'</code>，时间窗内全部四个分组（AI 与固定表组）。</li>
<li><strong>会话与日期归属</strong>：按 30 分钟无下注间隔切分会话；一次会话的<strong>全部</strong>下注计入
会话开始时刻的北京日（跨零点不拆分），与既有 AB测试数学表分析口径一致。留存判定同样按会话开始日。</li>
<li><strong>区块（block）</strong>：同一用户同一北京日内，按时间排序后连续使用同一数学表的一段下注。</li>
<li><strong>排列标签的构造与清洗</strong>：同一用户日内连续同表的一段下注按<strong>每 50 次 BASE</strong>
拆分为区段（连续 100 次 shi → <code>shi|shi</code>，长度 2——相邻重复不合并，标签长度即约 50 注一档的下注量）；
不足 {MIN_BLOCK_BETS} 次 BASE 的余段从<strong>排列标签</strong>中剔除
（按区段而非当日合计：如某用户当日 A 20 注 → B 50 注 → A 20 注，则标签只保留 B）。
但该用户日的<strong>下注额、下注次数与 RTP 仍按全天全部下注计算</strong>（含被剔除区块）；
无存留区块的用户日整日剔除（无法标注，AI 组保留 76% 的用户日）。
<strong>整体对比、逐日 RTP 与单表对比完全不受清洗影响。</strong></li>
<li><strong>与 DS 仪表盘口径的对照</strong>：统计人群一致（均为 CNY 用户），底层数据已逐项核对。
以范围2 AI 组人均日下注额为例：按仪表盘口径（下注日历日归属、不作极值处理）均值/中位为 785/128；
仪表盘开 99.9 分位过滤（<strong>剔除</strong>）为 667/127（同一 S3 数据复算得 666.7/127.5，完全一致）；
本报告为 596/128（会话开始日归属 + <strong>99 分位截断</strong>）。差异均来自口径（日期归属与极值处理），非数据本身。
另注意仪表盘按<strong>下注</strong>的日历日归属，本报告按<strong>会话开始</strong>日归属，人群有细微差异。另外仪表盘的 user_num_bets 含 FREE 免费旋转
（本报告仅计 BASE；FREE 注金为 0，不影响下注额）。</li>
<li><strong>排列（permutation）</strong>：存留区块的数学表按时间顺序排列（顺序敏感，A-B ≠ B-A），
相邻重复合并（小区块被剔除后相邻同表会出现重复），以 <code>-</code> 连接。</li>
<li><strong>均值 99 分位截断（winsorize）</strong>：user_total_bet、user_num_bets 与单注注金（user_avg_bet_amount）的<strong>均值</strong>
将超过 99 分位的样本<strong>截断到阈值</strong>（保留而非剔除——鲸鱼用户日是真实收入，只是不让单个样本主导均值）。
阈值按每个时间段的<strong>全人群</strong>用户日分布估计（线性插值；小单元内重新估计分位数噪声太大），
统一应用于所有单元格：范围1 约 13,028 CNY / 2,132 次 / 单注 23.5 CNY，范围2 约 13,071 CNY / 2,226 次 / 单注 19.6 CNY。
中位数、留存率、用户RTP 与整体 RTP 不截断。</li>
<li>下注次数 = BASE 下注次数；下注额 = BASE 注金；单注注金 = 当日 BASE 注金合计 / BASE 次数；RTP = 派彩 / BASE 注金（含免费游戏派彩）；仅 CNY，金额无需折算。
<strong>用户RTP</strong>为单用户日的比值，<strong>整体RTP</strong>为全体用户合并（∑派彩/∑注金）。</li>
<li><strong>用户GGR（lifetime，时段内）</strong>：单个用户在该时间段、该分析人群内全部下注的
（注金 − 派彩）合计，即用户净输给游戏的金额（负值 = 用户净赢），<strong>每人一个值</strong>。
用户在不同天属于不同单元格（数学表 / 下注额层级 / 排列）时，归入其<strong>出现天数最多</strong>的单元格，
并列时归入总规模更大的单元格。GGR 为有符号量，其均值做<strong>双侧</strong> 1/99 分位截断（阈值按全人群估计：范围1 约 [−1,191, 1,634]，范围2 约 [−1,303, 1,694] CNY）；中位数不截断。</li>
<li><strong>下注节奏</strong>：单注间隔 = 用户日内<strong>会话内</strong>相邻 BASE 下注时间差的当日中位数（秒），
<strong>剔除两注之间有免费游戏（FG）播放的间隔</strong>（FG 动画时间并入前一注，非玩家节奏）；
最长连注 = 相邻间隔 &lt;200 秒的连续 BASE 下注的当日最长次数，FG 播放不中断连注。
两者均按"均值（99 分位截断）/ 中位数"展示。</li>
<li><strong>次日留存</strong>：用户次日（北京日）在 SS03 有任意 CNY 下注（不限分组、不做清洗）即算留存。
数据完整至 {PRESENCE_END}，故 09-01 / 09-02 的次日留存为空；09-02 当日数据不完整（分析时仅入库约 1/3）。</li>
<li>剔除 {EXCLUDED_COHORT_DATES[0]} / {EXCLUDED_COHORT_DATES[1]}（AB 分组策略切换日，AI 归属不明确），
与 <a href="https://bituslabs.atlassian.net/wiki/pages/viewpage.action?pageId=1319829528">AB测试数学表分析</a> 口径一致。</li>
</ul>

<h2>两时间段整体对比（AI 组，仅 CNY，未清洗）</h2>
{range_table(summary)}
<p>范围2 相对范围1：次日留存 +3.0pp（18.4% → 21.4%），人均日下注次数 +23%（166 → 205），
人均日下注额均值 +9%（547 → 596，中位数 97 → 128 为 +32%），整体 RTP 提高约 6pp（0.932 → 0.992）。</p>

<h2>AI 组 vs 单一数学表用户日（同期对比，仅 CNY，未清洗）</h2>
<p>作为参照，纳入同期使用<strong>固定数学表</strong>的用户（不做区块清洗），按当日游玩的<strong>唯一</strong>
主数学表标注（时间段内表有更换时分别统计）。同日使用 ≥2 张主数学表的用户日<strong>整日剔除</strong>
（单表标注会低估其当日下注次数/下注额；仅 102 / 92,431 个用户日，占 0.11%）。
<code>normal_kakuteiB/C</code> 为主表触发的单旋转奖励子表（中位数 1 次 BASE 旋转），
不视为第二张表、也不参与标注，但其下注/派彩计入当日合计。少于 100 用户日的表不列出。
表名缩写同下节；<strong>AI(全部排列)</strong> 为 AI 组整体。</p>
{comparison_table(by_table)}
<p>要点：<strong>范围1 中 AI 组整体略逊于头部单表用户日</strong>——次日留存 18.4%，低于
<code>BGTR95_v3</code>（19.9%）、<code>95_kai</code>（19.2%）、<code>zero</code>（18.9%）、<code>shi</code>（18.8%），
高于 <code>zero93kai</code>（15.9%）与 <code>BG97_adj</code>（15.0%），
人均日下注次数（166）也低于多数单表用户日（142~226）；
<strong>范围2 中 AI 组与单表用户日基本持平</strong>（留存 21.4% vs 20.7%~22.5%，
其中 <code>BGTR97_v2</code> 22.5% 最高、下注额均值 912 也显著领先），但 AI 组整体 RTP（0.992）高于
<code>95_kai</code>（0.935）与 <code>BGTR95_v3</code>（0.969）。
另外 <code>BGTR95_v3</code> 早在范围1 即被固定表组使用，08-18 起才进入 AI 轮换。</p>

<h2>按日历日的全体用户 RTP</h2>
{daily_summary(daily)}
<p>范围2 日均 RTP 达 1.023，多天日 RTP &gt; 1（如 08-22 为 1.237、08-26 为 1.150），即当日 AI 组派彩超过注金；
范围1 也有零星高 RTP 日（07-23 为 2.667）。逐日明细（RTP、留存、活跃用户、下注量）见附件
<code>daily_all_users.csv</code>。</p>

<h2>按每日下注额分层：AI 组 vs 单一数学表（仅 CNY，未清洗）</h2>
<p>层级按用户日的<strong>原始</strong>（未截断）日下注总额划分：&lt;10 / 10-100 / 100-1000 / 1000+ CNY。
留存附 95% Wilson 置信区间；均值列仍为 99 分位截断口径；少于 100 用户日的单元格不列出。
上方摘要图为本节 AI 与头部单表的留存对比。</p>
<p><ac:image ac:align="center" ac:width="1100"><ri:attachment ri:filename="fig_bucket_metrics.png" /></ac:image></p>
{bucket_table(bkt)}

<h2>Top-5 数学表排列（每指标、每时间段；仅统计 ≥ {MIN_PERMUTATION_USER_DAYS} 用户日的排列；
排列标签经清洗，指标为全天口径）</h2>
<p><strong>按排列长度分层排名</strong>：长度 = 标签中 50 注区段的个数（含重复，连续 100 次 shi 记
<code>shi|shi</code> 长度 2），即长度 ≈ 当日清洗后 BASE 下注量 / 50。更活跃的用户下注更多、标签天然更长，
不同长度直接对比反映的是活跃度（反向因果）而非表序效应，故 Top-5 仅在<strong>相同长度</strong>（1/2/3/4，
即下注量相近）内比较；排名指标为次日留存、人均日下注额与用户GGR。长度 ≥5 的重度日不参与排名。
该期 AI 在约 9 张表间均匀轮换、长序列高度分散，范围1 长度3 仅 2 个单元格达到门槛、长度4 没有
（最大 13 用户日）；范围2 主要在两张表间轮换，长序列更集中。20~30 用户日的单元格置信区间较宽，解读需谨慎。</p>
<p>表名缩写：去掉 <code>normal_</code> 前缀；<code>95_kai</code> = <code>normal_zero_95_kai</code>；
<code>BGTR95_v3</code> = <code>normal_Zero_BGTR95_Saitekika_BGadj_v3</code>。</p>
<p><ac:image ac:align="center" ac:width="1100"><ri:attachment ri:filename="fig_perm_range1_pre.png" /></ac:image></p>
<p><ac:image ac:align="center" ac:width="1100"><ri:attachment ri:filename="fig_perm_range2_post_v2.png" /></ac:image></p>
{top5_tables(top5)}

<h2>分析4：用户日 RTP 与下注活跃度的相关性（AI 组，仅 CNY，未清洗）</h2>
<p>问题：当天体验到更高 RTP 的用户是否下注更多？逐用户日计算 user_rtp 与
user_total_bet / user_num_bets 的相关。三个变量都严重右偏（偏度：RTP 21、下注额 90、下注次数 8），
<strong>原始 Pearson 几乎为 0</strong>（被极值支配），故对两侧取 log10（RTP 加 0.01 处理零派彩日）后计算
Pearson，并以对单调变换不敏感的 <strong>Spearman 秩相关为主</strong>。</p>
{corr_table(corr)}
<p><ac:image ac:align="center" ac:width="1100"><ri:attachment ri:filename="fig_corr_rtp.png" /></ac:image></p>
<p>结果：两个时间段一致——RTP 与下注次数 ρ≈0.54~0.55、与下注额 ρ≈0.45~0.47（log-log Pearson 相近），
中位数曲线在 RTP 0~1 区间单调上升、在 RTP≈1 处见顶，RTP&gt;1 后回落——
极端高 RTP 的日子多为少量下注即中大奖的用户日（机械关系），活跃度峰值出现在"接近回本"的日子。
<strong>注意不可作因果解读</strong>：这是同日相关——大赢后继续下注（赢→玩）与下注多才可能触发大派彩（玩→赢）
混在一起；且下注很少的日子 RTP 方差天然大（低 RTP 端多为几十注即离场的用户日），属机械性关联。
如需因果结论应做随机化实验或滞后（次日）分析。</p>

<h2>结论</h2>
<ul>
<li>AI 组内排列（清洗后，<strong>按长度分层</strong>，长度 = 50 注区段数——跨长度对比受活跃度反向因果影响，
原先"多表日优于单表日"的表述主要反映活跃度差异，已按此口径修正）：同量级下<strong>表组合的影响显著</strong>——
范围1 长度1 以 <code>shi</code>（24.3%）、<code>go</code>（23.6%）最佳，长度2 中数字表组合居前
（<code>shi|ni</code> 50.0%、<code>san|san</code> 30.0%，均 20~29 用户日、区间较宽），
长度3 仅两个 95_kai 开头的单元格达标且留存差（8.3% / 3.7%）；
范围2 长度1 中 <code>shi</code>/<code>ni</code>（29.4%/27.8%）与 <code>BGTR95_v3</code>（21.1%）
远高于 <code>95_kai</code>（6.0%，745 用户日）。</li>
<li>跨组对比：<strong>范围1 中 AI 组整体略逊于头部固定单表用户日</strong>（留存 18.4% vs 最高 19.9%，下注次数也更低）；
范围2 改用 95_kai + BGTR95_v3 轮换后与单表用户日持平（21.4% vs 20.7%~22.5%），但整体 RTP 更高（0.992）。</li>
<li>范围2 中<strong>含 BGTR95_v3 区段的排列在同量级下普遍优于以 95_kai 为主的排列</strong>：
长度2 的 <code>BGTR95_v3|95_kai</code>（40.0%）与 <code>BGTR95_v3|BGTR95_v3</code>（17.1%）
vs <code>95_kai|95_kai</code>（7.1%）；长度3/4 的 BGTR95_v3 连打（25.9%；30.4%、GGR 434）
vs 95_kai 开头（9.3%；5.7%）——同量级的一天，落在 BGTR95_v3 上的用户留存与 GGR 都更好。</li>
<li><code>shi-ni</code>（四 → 二）是唯一在两个时间段都进入多个榜单的排列。</li>
<li>范围2 整体留存、下注次数与剔除极值后的下注额均值都优于范围1，但同期 RTP 也更高，评估收益时需一并考虑。</li>
<li><strong>用户GGR</strong>（时段内人均净输额，双侧截断均值）：AI 组整体与单表相当
（范围1 55 vs 43~59，范围2 51 vs 35~55 CNY/人）；分层看，范围1 高投入层（1000+）AI 组
人均 GGR 约 415，明显高于单表的 285~320——对高投入用户 AI 轮换既提升留存也提升收入；
范围2 该层 AI（约 235）与 95_kai（约 292）相当、高于 BGTR97_v2（约 112），置信区间较宽。</li>
</ul>

<h2>数据与脚本</h2>
<ul>
<li>分析脚本：<code>jobs/ss03_mahjiang_streak/analysis_ai_mathtable_permutation.py</code>（--extract / --analyze）。</li>
<li>输出目录：<code>jobs/output_ss03_mahjiang_streak/ai_mathtable_permutation/</code>；
汇总 CSV（含全部 2,235 个排列单元格的 <code>summary_permutation.csv</code>）已作为附件上传。</li>
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
