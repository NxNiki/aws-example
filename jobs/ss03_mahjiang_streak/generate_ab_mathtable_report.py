"""Render the SS03 AB-test mathtable/retention analysis as a Chinese Confluence report.

Companion to ``analysis_ab_mathtable_retention.py`` (run its --extract/--analyze
first): reads the CSV tables it writes, renders PNG charts, builds a
Confluence storage-format page in Chinese, and (with ``--publish``) creates a
new page under the given Confluence folder with the charts attached.

Usage:
    poetry run python jobs/ss03_mahjiang_streak/generate_ab_mathtable_report.py            # build only
    poetry run python jobs/ss03_mahjiang_streak/generate_ab_mathtable_report.py --publish
"""

from __future__ import annotations

import argparse
import html
import logging
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from analysis_ab_mathtable_retention import (
    EXCLUDED_COHORT_DATES,
    FX_AS_OF,
    FX_TO_CNY,
    OUTPUT_DIR,
    POST_SWITCH_START,
    PRESENCE_END,
    SESSION_GAP_SECONDS,
    WINDOW_END,
    WINDOW_START,
    bucket_labels,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFLUENCE_FOLDER_ID = "1267040280"
# Existing page to update in place; set to "" to create a new page instead.
CONFLUENCE_PAGE_ID = "1319829528"
PAGE_TITLE = f"SS03 AB测试数学表分析：留存与下注指标（{WINDOW_START} ~ {WINDOW_END}）"

FIG_DIR = OUTPUT_DIR / "figures"

GROUP_ORDER = ["AB_TEST_A", "AB_TEST_B", "Default", "AI"]
GROUP_COLORS = {"AB_TEST_A": "#2a78d6", "AB_TEST_B": "#eb6834", "Default": "#1baf7a", "AI": "#eda100"}
GROUP_CN = {"AB_TEST_A": "A组", "AB_TEST_B": "B组", "Default": "默认组", "AI": "AI组"}

# Post-switch (07-31+) each fixed group runs exactly one math table; the AI
# group rotates tables dynamically.
GROUP_TABLE_CN = {
    "AB_TEST_A": "A组·normal_shi",
    "AB_TEST_B": "B组·BGTR95_v3",
    "Default": "默认组·zero_95_kai",
    "AI": "AI组·轮换多表",
}

# Mathtable display labels; A/B switched tables on 2026-07-30.
MT_CN = {
    ("AB_TEST_A", "normal_zero"): "A组·旧表 normal_zero",
    ("AB_TEST_A", "normal_shi"): "A组·新表 normal_shi",
    ("AB_TEST_B", "normal_zero93kai"): "B组·旧表 normal_zero93kai",
    ("AB_TEST_B", "normal_Zero_BGTR95_Saitekika_BGadj_v3"): "B组·新表 BGTR95_v3",
    ("Default", "normal_zero_95_kai"): "默认组 normal_zero_95_kai",
}
MT_ORDER = list(MT_CN)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#eceae6"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Hiragino Sans GB", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d8d6d0",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def _read(name: str) -> pl.DataFrame:
    return pl.read_csv(OUTPUT_DIR / f"{name}.csv", try_parse_dates=True)


def _with_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """Insert null rows on the excluded cutover dates so lines break there."""
    fillers = pl.DataFrame({"bet_date": list(EXCLUDED_COHORT_DATES)}).join(
        df.select(pl.col("ab_group").unique()), how="cross"
    )
    return pl.concat([df, fillers], how="diagonal").sort("bet_date")


def _date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, axis="y")
    ax.margins(x=0.02)


def _mark_events(ax, y=None) -> None:
    ax.axvline(date(2026, 7, 30), color=TEXT_SECONDARY, lw=1, ls="--", zorder=1)
    ax.axvspan(date(2026, 8, 3), date(2026, 8, 4), color="#f0efec", zorder=0)


def _line_by_group(ax, df: pl.DataFrame, ycol: str, scale: float = 1.0) -> None:
    for g in GROUP_ORDER:
        part = df.filter(pl.col("ab_group") == g)
        ax.plot(
            part["bet_date"],
            part[ycol] * scale,
            color=GROUP_COLORS[g],
            lw=2,
            marker="o",
            ms=4,
            label=GROUP_CN[g],
        )


def fig_daily_users_volume(daily: pl.DataFrame) -> Path:
    daily = _with_gaps(daily)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [
        ("n_users", 1.0, "日活跃用户数（人）"),
        ("total_spin", 1e-4, "日下注次数（万次）"),
        ("total_bet_cny", 1e-4, "日下注额（万CNY）"),
    ]
    for ax, (col, scale, title) in zip(axes, specs):
        _line_by_group(ax, daily, col, scale)
        _mark_events(ax)
        _date_axis(ax)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower left", ncols=3)
    axes[0].annotate(
        "7-30 A/B组换表(虚线)；8-3/8-4 分组策略切换(剔除)",
        xy=(0.01, 1.08),
        xycoords="axes fraction",
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig1_daily_users_volume.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_daily_retention(daily: pl.DataFrame) -> Path:
    daily = _with_gaps(daily)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (col, title) in zip(axes, [("retention_d1", "次日留存率 D1"), ("retention_d3", "第3日留存率 D3")]):
        _line_by_group(ax, daily, col, 100.0)
        _mark_events(ax)
        _date_axis(ax)
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower left", ncols=3)
    fig.tight_layout()
    out = FIG_DIR / "fig2_daily_retention.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_retention_by_mathtable(gm: pl.DataFrame) -> Path:
    rows = [
        gm.row(by_predicate=(pl.col("ab_group") == g) & (pl.col("mathtable") == m), named=True) for g, m in MT_ORDER
    ]
    labels = [MT_CN[k] for k in MT_ORDER]
    colors = [GROUP_COLORS[g] for g, _ in MT_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, col, title in [(axes[0], "retention_d1", "次日留存率 D1"), (axes[1], "retention_d3", "第3日留存率 D3")]:
        vals = [r[col] * 100 for r in rows]
        y = range(len(rows))[::-1]
        ax.barh(list(y), vals, color=colors, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.15, yi, f"{v:.1f}%", va="center", fontsize=9, color=TEXT_PRIMARY)
        ax.set_yticks(list(y), labels)
        ax.set_title(title)
        ax.grid(True, axis="x")
        ax.margins(x=0.15)
    fig.tight_layout()
    out = FIG_DIR / "fig3_retention_by_mathtable.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _ci95(p: float, n: int) -> float:
    return 1.96 * (p * (1 - p) / n) ** 0.5 if n else 0.0


def fig_metrics_by_bucket(ps: pl.DataFrame) -> Path:
    """2x2 overview per spend bucket × group (post-switch): D1, D3, and the
    per-user-day AVERAGE bets and bet amount (not group totals, which mostly
    reflect group size)."""
    from matplotlib.ticker import FuncFormatter

    buckets = bucket_labels()
    x = range(len(buckets))
    width = 0.2
    ps = ps.with_columns((pl.col("total_spin") / pl.col("user_days")).alias("avg_spin_per_user_day"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    panels = [
        (axes[0][0], "retention_d1", "d1_cohort", 100.0, "次日留存率 D1（%，误差线=95%CI）", "{:.1f}", False, None),
        (axes[0][1], "retention_d3", "d3_cohort", 100.0, "第3日留存率 D3（%，误差线=95%CI）", "{:.1f}", False, None),
        (
            axes[1][0],
            "avg_spin_per_user_day",
            None,
            1.0,
            "人日均下注次数（误差线=bootstrap 95%CI）",
            "{:,.0f}",
            False,
            "avg_spin",
        ),
        (
            axes[1][1],
            "avg_bet_cny_per_user_day",
            None,
            1.0,
            "人日均下注额（CNY，对数刻度，误差线=bootstrap 95%CI）",
            "{:,.0f}",
            True,
            "avg_bet_cny",
        ),
    ]
    for ax, col, ncol, scale, title, fmt, log, ci_prefix in panels:
        for i, g in enumerate(GROUP_ORDER):
            part = ps.filter(pl.col("ab_group") == g)
            vals, err_lo, err_up = [], [], []
            for b in buckets:
                r = part.filter(pl.col("bet_bucket") == b)
                v = r[col][0] * scale
                vals.append(v)
                if ncol:
                    e = _ci95(r[col][0], r[ncol][0]) * scale
                    err_lo.append(e)
                    err_up.append(e)
                elif ci_prefix:
                    err_lo.append(max(v - r[f"{ci_prefix}_ci_lo"][0] * scale, 0.0))
                    err_up.append(max(r[f"{ci_prefix}_ci_hi"][0] * scale - v, 0.0))
                else:
                    err_lo.append(0.0)
                    err_up.append(0.0)
            pos = [xi + (i - (len(GROUP_ORDER) - 1) / 2) * width for xi in x]
            has_ci = any(e > 0 for e in err_up)
            ax.bar(
                pos,
                vals,
                width=width * 0.92,
                color=GROUP_COLORS[g],
                label=GROUP_TABLE_CN[g],
                yerr=[err_lo, err_up] if has_ci else None,
                error_kw={"ecolor": TEXT_SECONDARY, "lw": 1, "capsize": 2},
            )
            top = max(vals)
            for p, v, e in zip(pos, vals, err_up):
                label = _fmt_adaptive(v) if log else fmt.format(v)
                y_text = (v + e) * 1.12 if log else v + e + top * 0.015
                ax.text(p, y_text, label, ha="center", fontsize=8, color=TEXT_SECONDARY)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_adaptive(v)))
        ax.set_xticks(list(x), buckets)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.margins(y=0.12)
    axes[1][0].set_xlabel("当日下注额分组（CNY）")
    axes[1][1].set_xlabel("当日下注额分组（CNY）")
    axes[0][0].legend(loc="upper left")
    fig.suptitle(f"各组 × 下注额层级：留存与人日均下注强度（{POST_SWITCH_START} 起）", y=0.995)
    fig.tight_layout()
    out = FIG_DIR / "fig6_metrics_by_bucket.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_bucket_mix(gb: pl.DataFrame) -> Path:
    """Share of user-days per bucket within each group, plus per-bucket bet share."""
    buckets = bucket_labels()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, col, title in [
        (axes[0], "user_days", "用户日占比（按当日下注额分组）"),
        (axes[1], "total_bet_cny", "下注额占比（按当日下注额分组）"),
    ]:
        y = range(len(GROUP_ORDER))[::-1]
        shades = ["#c7dbf4", "#8db6e8", "#5597df", "#2a78d6"]
        left = [0.0] * len(GROUP_ORDER)
        for bi, b in enumerate(buckets):
            vals = []
            for g in GROUP_ORDER:
                part = gb.filter(pl.col("ab_group") == g)
                total = part[col].sum()
                v = part.filter(pl.col("bet_bucket") == b)[col][0]
                vals.append(100.0 * v / total)
            ax.barh(list(y), vals, left=left, color=shades[bi], height=0.55, label=b, edgecolor="#ffffff", linewidth=2)
            for yi, l, v in zip(y, left, vals):
                if v > 7:
                    ax.text(
                        l + v / 2,
                        yi,
                        f"{v:.0f}%",
                        va="center",
                        ha="center",
                        fontsize=8,
                        color="#ffffff" if bi >= 2 else TEXT_PRIMARY,
                    )
            left = [l + v for l, v in zip(left, vals)]
        ax.set_yticks(list(y), [GROUP_CN[g] for g in GROUP_ORDER])
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.grid(True, axis="x")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="日下注额(CNY)")
    fig.tight_layout()
    out = FIG_DIR / "fig5_bucket_mix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


AI_FIG_MIN_USER_DAYS = 30  # cells below this are hidden (kakuteiB)

BUCKET_SHADES = ["#c7dbf4", "#8db6e8", "#5597df", "#2a78d6"]


def _fmt_adaptive(v: float) -> str:
    if v >= 10:
        return f"{v:,.0f}"
    return f"{v:.1f}" if v >= 0.1 else f"{v:.2f}"


def _ai_bucket_panel(
    ax,
    cells: dict,
    tables: list[str],
    col: str,
    ncol: str | None,
    scale: float,
    fmt: str,
    log: bool = False,
    ci_cols: tuple[str, str] | None = None,
) -> None:
    """Grouped bars: x = math tables, one bar per spend bucket.

    CI whiskers come from ``ncol`` (normal-approx on a proportion) or from
    ``ci_cols`` (absolute lo/hi columns, e.g. bootstrap percentiles).
    """
    buckets = bucket_labels()
    x = range(len(tables))
    width = 0.2
    for i, b in enumerate(buckets):
        vals, err_lo, err_up = [], [], []
        for m in tables:
            r = cells.get((m, b))
            if r is None or r["user_days"] < AI_FIG_MIN_USER_DAYS or r[col] is None:
                vals.append(0.0)
                err_lo.append(0.0)
                err_up.append(0.0)
                continue
            v = r[col] * scale
            vals.append(v)
            if ncol:
                e = _ci95(r[col], r[ncol]) * scale
                err_lo.append(e)
                err_up.append(e)
            elif ci_cols and r[ci_cols[0]] is not None:
                err_lo.append(max(v - r[ci_cols[0]] * scale, 0.0))
                err_up.append(max(r[ci_cols[1]] * scale - v, 0.0))
            else:
                err_lo.append(0.0)
                err_up.append(0.0)
        pos = [xi + (i - (len(buckets) - 1) / 2) * width for xi in x]
        has_ci = ncol or ci_cols
        ax.bar(
            pos,
            vals,
            width=width * 0.92,
            color=BUCKET_SHADES[i],
            label=b,
            yerr=[err_lo, err_up] if has_ci else None,
            error_kw={"ecolor": TEXT_SECONDARY, "lw": 0.8, "capsize": 2},
        )
        top = max(vals) or 1
        for pxi, v, e in zip(pos, vals, err_up):
            if v > 0:
                label = _fmt_adaptive(v) if log else fmt.format(v)
                y_text = (v + e) * 1.12 if log else v + e + top * 0.02
                ax.text(pxi, y_text, label, ha="center", fontsize=8, color=TEXT_SECONDARY)
    if log:
        from matplotlib.ticker import FuncFormatter

        ax.set_yscale("log")
        # The CJK font lacks the superscript-minus glyph mathtext uses for
        # 10^-1 tick labels; plain decimal labels sidestep it.
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_adaptive(v)))
    ax.set_xticks(list(x), [m.removeprefix("normal_") for m in tables])
    ax.grid(True, axis="y")
    ax.margins(y=0.14)


def fig_ai_retention_by_bucket(aib: pl.DataFrame, order: list[str]) -> Path:
    """AI per-mathtable D1 and D3 retention, grouped by daily-spend bucket."""
    tables = [m for m in order if m != "normal_kakuteiB"]
    cells = {(r["mathtable"], r["bet_bucket"]): r for r in aib.iter_rows(named=True)}
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4))
    _ai_bucket_panel(axes[0], cells, tables, "retention_d1", "d1_cohort", 100.0, "{:.0f}")
    axes[0].set_title("次日留存率 D1（%）")
    axes[0].set_ylabel("%")
    _ai_bucket_panel(axes[1], cells, tables, "retention_d3", "d3_cohort", 100.0, "{:.0f}")
    axes[1].set_title("第3日留存率 D3（%）")
    axes[1].set_ylabel("%")
    axes[0].legend(title="日下注额(CNY)", loc="upper left", ncols=4)
    fig.suptitle(
        "AI组各数学表留存，按当日下注额分组（全天归因，≥50次/日，误差线=95%CI；kakuteiB 样本过小未画出）", y=0.995
    )
    fig.tight_layout()
    out = FIG_DIR / "fig10_ai_retention_by_bucket.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_ai_total_bet_by_bucket(aib: pl.DataFrame, order: list[str]) -> Path:
    """AI per-mathtable attributed total bet (CNY), grouped by daily-spend bucket."""
    tables = [m for m in order if m != "normal_kakuteiB"]
    cells = {(r["mathtable"], r["bet_bucket"]): r for r in aib.iter_rows(named=True)}
    fig, ax = plt.subplots(figsize=(13, 4.6))
    _ai_bucket_panel(
        ax,
        cells,
        tables,
        "avg_bet_cny_per_user_day",
        None,
        1.0,
        "{:,.0f}",
        log=True,
        ci_cols=("avg_bet_cny_ci_lo", "avg_bet_cny_ci_hi"),
    )
    ax.set_ylabel("CNY / 用户日（对数刻度）")
    ax.set_title("AI组各数学表人日均下注额（CNY/用户日，对数刻度，误差线=bootstrap 95%CI；全天归因）")
    ax.legend(title="日下注额(CNY)", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    out = FIG_DIR / "fig11_ai_total_bet_by_bucket.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _wan(x) -> str:
    return f"{x / 1e4:,.1f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th><p><strong>{html.escape(h)}</strong></p></th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td><p>{html.escape(str(c))}</p></td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tbody><tr>{head}</tr>{body}</tbody></table>"


def _img(filename: str, width: int = 1000) -> str:
    return f'<ac:image ac:width="{width}"><ri:attachment ri:filename="{filename}"/></ac:image>'


def _ztest_p(p1: float, n1: int, p2: float, n2: int) -> float:
    from math import erf, sqrt

    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def build_page(figs: dict[str, Path]) -> str:
    gm = _read("summary_group_mathtable")
    gb = _read("summary_group_bucket")
    gmb = _read("summary_group_mathtable_bucket")
    ps = _read("summary_group_bucket_postswitch")
    ai = _read("summary_ai_mathtable").sort("user_days", descending=True)
    aib = _read("summary_ai_mathtable_bucket")

    overall = (
        gm.group_by("ab_group")
        .agg(
            pl.col("user_days").sum(),
            pl.col("total_spin").sum(),
            pl.col("total_bet_cny").sum(),
            ((pl.col("retention_d1") * pl.col("d1_cohort")).sum() / pl.col("d1_cohort").sum()).alias("retention_d1"),
            ((pl.col("retention_d3") * pl.col("d3_cohort")).sum() / pl.col("d3_cohort").sum()).alias("retention_d3"),
        )
        .sort("ab_group")
    )

    t_overall = _table(
        [
            "组",
            "用户日数",
            "总下注次数(万)",
            "总下注额(万CNY)",
            "人日均下注次数",
            "人日均下注额(CNY)",
            "D1留存",
            "D3留存",
        ],
        [
            [
                GROUP_CN[r["ab_group"]],
                f"{r['user_days']:,}",
                _wan(r["total_spin"]),
                _wan(r["total_bet_cny"]),
                f"{r['total_spin'] / r['user_days']:.0f}",
                f"{r['total_bet_cny'] / r['user_days']:.1f}",
                _pct(r["retention_d1"]),
                _pct(r["retention_d3"]),
            ]
            for r in overall.iter_rows(named=True)
        ],
    )

    t_mathtable = _table(
        [
            "组·数学表",
            "用户日数",
            "独立用户",
            "总下注次数(万)",
            "总下注额(万CNY)",
            "人日均下注额(CNY)",
            "单次均注(CNY)",
            "D1留存",
            "D3留存",
        ],
        [
            [
                MT_CN[(r["ab_group"], r["mathtable"])],
                f"{r['user_days']:,}",
                f"{r['n_users']:,}",
                _wan(r["total_spin"]),
                _wan(r["total_bet_cny"]),
                f"{r['avg_bet_cny_per_user_day']:,.1f}",
                f"{r['avg_bet_cny']:.2f}",
                _pct(r["retention_d1"]),
                _pct(r["retention_d3"]),
            ]
            for gk in MT_ORDER
            for r in gm.filter((pl.col("ab_group") == gk[0]) & (pl.col("mathtable") == gk[1])).iter_rows(named=True)
        ],
    )

    buckets = bucket_labels()
    t_bucket = _table(
        [
            "组",
            "日下注额(CNY)",
            "用户日数",
            "总下注次数(万)",
            "总下注额(万CNY)",
            "人日均下注额(CNY)",
            "D1留存",
            "D3留存",
        ],
        [
            [
                GROUP_CN[r["ab_group"]],
                r["bet_bucket"],
                f"{r['user_days']:,}",
                _wan(r["total_spin"]),
                _wan(r["total_bet_cny"]),
                f"{r['avg_bet_cny_per_user_day']:,.1f}",
                _pct(r["retention_d1"]),
                _pct(r["retention_d3"]),
            ]
            for g in GROUP_ORDER
            for b in buckets
            for r in gb.filter((pl.col("ab_group") == g) & (pl.col("bet_bucket") == b)).iter_rows(named=True)
        ],
    )

    t_full = _table(
        [
            "组·数学表",
            "日下注额(CNY)",
            "用户日数",
            "总下注次数",
            "总下注额(万CNY)",
            "人日均下注额(CNY)",
            "D1留存",
            "D3留存",
        ],
        [
            [
                MT_CN[(r["ab_group"], r["mathtable"])],
                r["bet_bucket"],
                f"{r['user_days']:,}",
                f"{r['total_spin']:,}",
                _wan(r["total_bet_cny"]),
                f"{r['avg_bet_cny_per_user_day']:,.1f}",
                _pct(r["retention_d1"]),
                _pct(r["retention_d3"]),
            ]
            for gk in MT_ORDER
            for b in buckets
            for r in gmb.filter(
                (pl.col("ab_group") == gk[0]) & (pl.col("mathtable") == gk[1]) & (pl.col("bet_bucket") == b)
            ).iter_rows(named=True)
        ],
    )

    t_fx = _table(
        ["币种", "对CNY汇率"],
        [[c, f"{r:.6f}"] for c, r in FX_TO_CNY.items()],
    )

    # Per-tier comparison table (post-switch): D1/D3 ± 95% CI per table, with
    # the A-vs-B / A-vs-Default / B-vs-Default z-test p-values.
    buckets_order = bucket_labels()
    psr = {(r["ab_group"], r["bet_bucket"]): r for r in ps.iter_rows(named=True)}
    tier_rows = []
    pvals: dict[str, dict[str, float]] = {}
    for b in buckets_order:
        row = [b]
        for g in GROUP_ORDER:
            r = psr[(g, b)]
            row.append(
                f"{r['retention_d1'] * 100:.1f}% ±{_ci95(r['retention_d1'], r['d1_cohort']) * 100:.1f} "
                f"(n={r['d1_cohort']:,})"
            )
        for g in GROUP_ORDER:
            r = psr[(g, b)]
            row.append(f"{r['retention_d3'] * 100:.1f}% ±{_ci95(r['retention_d3'], r['d3_cohort']) * 100:.1f}")
        pvals[b] = {
            "AB_d1": _ztest_p(
                psr[("AB_TEST_A", b)]["retention_d1"],
                psr[("AB_TEST_A", b)]["d1_cohort"],
                psr[("AB_TEST_B", b)]["retention_d1"],
                psr[("AB_TEST_B", b)]["d1_cohort"],
            ),
            "AB_d3": _ztest_p(
                psr[("AB_TEST_A", b)]["retention_d3"],
                psr[("AB_TEST_A", b)]["d3_cohort"],
                psr[("AB_TEST_B", b)]["retention_d3"],
                psr[("AB_TEST_B", b)]["d3_cohort"],
            ),
        }
        tier_rows.append(row)
    t_tier = _table(
        ["日下注额(CNY)"]
        + [f"D1 {GROUP_TABLE_CN[g]}" for g in GROUP_ORDER]
        + [f"D3 {GROUP_TABLE_CN[g]}" for g in GROUP_ORDER],
        tier_rows,
    )

    # D1 pivot: rows = AI math tables (ordered by D1 in the actionable
    # ¥100-1000 tier), columns = spend tiers. kakuteiB excluded (n<=13/cell).
    aib_cells = {(r["mathtable"], r["bet_bucket"]): r for r in aib.iter_rows(named=True)}
    pivot_tables = sorted(
        {m for m in aib["mathtable"] if m != "normal_kakuteiB"},
        key=lambda m: -(aib_cells.get((m, "100-1000"), {}).get("retention_d1") or 0),
    )
    t_ai_d1_pivot = _table(
        ["数学表"] + [f"{b} CNY" for b in bucket_labels()],
        [
            [m.removeprefix("normal_")]
            + [
                (
                    _pct(aib_cells[(m, b)]["retention_d1"])
                    if (m, b) in aib_cells and aib_cells[(m, b)]["user_days"] >= 30
                    else "—"
                )
                for b in bucket_labels()
            ]
            for m in pivot_tables
        ],
    )

    t_ai_bucket = _table(
        [
            "数学表",
            "日下注额(CNY)",
            "用户日数",
            "总下注次数(万)",
            "总下注额(万CNY)",
            "人日均下注额(CNY)",
            "D1留存",
            "D3留存",
        ],
        [
            [
                r["mathtable"].removeprefix("normal_"),
                r["bet_bucket"],
                f"{r['user_days']:,}",
                _wan(r["total_spin"]),
                _wan(r["total_bet_cny"]),
                f"{r['avg_bet_cny_per_user_day']:,.1f}",
                _pct(r["retention_d1"]),
                _pct(r["retention_d3"]),
            ]
            for m in ai["mathtable"]
            for bkt in bucket_labels()
            for r in aib.filter((pl.col("mathtable") == m) & (pl.col("bet_bucket") == bkt)).iter_rows(named=True)
        ],
    )

    ov = {r["ab_group"]: r for r in overall.iter_rows(named=True)}
    a, b, d = ov["AB_TEST_A"], ov["AB_TEST_B"], ov["Default"]

    p_mid_d1 = pvals["100-1000"]["AB_d1"]
    p_mid_d3 = pvals["100-1000"]["AB_d3"]
    psr_avg = {(r["ab_group"], r["bet_bucket"]): r["avg_bet_cny_per_user_day"] for r in ps.iter_rows(named=True)}
    body = f"""
<h1>摘要</h1>
<p><strong>分析目的：以留存和投注额为指标，识别不同日下注额层级的用户更适合哪张数学表。</strong>
窗口 {WINDOW_START} ~ {WINDOW_END}（北京时间，按会话首注日期归属，全币种折算CNY）；A/B组 07-30 换表，
分层对比使用 <strong>{POST_SWITCH_START} 起</strong>的数据（A=normal_shi，B=BGTR95_v3，默认=zero_95_kai）。</p>
<ul>
<li><strong>留存</strong>：低额层级（&lt;¥100，约78%用户日）三表无显著差异；<strong>¥100-1000 层级 normal_shi 最优</strong>
（D1 21.6% vs 20.3%/19.0%，D3 15.9% vs 13.7%/12.4%，A vs B p={p_mid_d1:.3f}/{p_mid_d3:.3f}）；¥1000+ 无一致赢家（差异不显著）。</li>
<li><strong>投注额</strong>：低中层级三表人日均下注额基本相同（&lt;10：≈¥2.8-3.0；10-100：≈¥36-38；100-1000：≈¥313-320）；
<strong>¥1000+ 层级分化</strong>：默认表 ¥{psr_avg[("Default", "1000+")]:,.0f} &gt; BGTR95_v3 ¥{psr_avg[("AB_TEST_B", "1000+")]:,.0f}
&gt; normal_shi ¥{psr_avg[("AB_TEST_A", "1000+")]:,.0f}（见分层对比）。</li>
<li><strong>注意</strong>：整体口径受构成效应影响（B组整体 D1 最高仅因高额用户占比更高），不能用于选表；
默认组跨 08-04 策略切换前后是不同人群；AI组表间差异含强选择效应（见专节）。</li>
</ul>

<h1>总体指标（按组）</h1>
{t_overall}
<p>注：D1/D3 为用户日口径的加权平均（分母为该组该日活跃用户，见口径说明）。</p>

<h1>按日趋势</h1>
{_img(figs["fig1"].name)}
{_img(figs["fig2"].name)}
<p>灰色区间为 08-03/08-04 分组策略切换期（成员归属不明确，已从指标与留存分母中剔除，但仍计入“次日是否回访”的判断）；虚线为 07-30 A/B组数学表切换。默认组在切换后规模近乎减半：切换前默认组为"未分配 partition_ab"的全部用户，切换后为 user_id 尾号 0-3 的用户（约 40%），属人群构成变化而非流失。</p>

<h1>按数学表（A/B/默认组）</h1>
{_img(figs["fig3"].name)}
{t_mathtable}
<p>注：旧表数据仅覆盖 07-29（及 07-30 切换当日的部分用户），样本量小（各约 900 用户日），留存对比仅供参考；切换前后跨越不同日期，存在时间混杂。AI组按数学表的拆分见下方专节（口径不同）。</p>

<h1>分层对比：不同下注额层级 × 数学表（{POST_SWITCH_START} 起）</h1>
<p>本节为报告核心：{POST_SWITCH_START} 起 A/B/默认组各自固定使用一张数学表，组间对比即表间对比
（用户按 user 维度随机分组，层内可比）。AI组动态轮换多表，此处作为整体参照系列展示。
分层维度为<strong>当日下注总额（CNY，全币种折算）</strong>。</p>
{_img(figs["fig6"].name)}
<p>上图汇总各层级的留存与人日均下注强度：留存差异集中在 ¥100-1000 层级；人日均下注次数/下注额在低中层级
三表基本一致，¥1000+ 层级分化（默认表人日均下注额最高，normal_shi 最低；AI组被极少数鲸鱼拉高、CI 极宽）。
各组总量（受组规模影响：默认组用户约为 A/B 组的 2.4 倍）见下方汇总表。</p>
{t_tier}
<p>注：±为95%置信区间半宽；n 为该层级留存分母（用户日）。加粗结论见摘要；除 ¥100-1000 层级外，
表间差异均不显著。</p>
<h2>用户构成</h2>
{_img(figs["fig5"].name)}
<h2>全窗口（含换表前）按层级汇总</h2>
{t_bucket}

<h1>专节：AI组各数学表</h1>
<p><strong>口径（与上文不同，请注意）：</strong></p>
<ul>
<li><strong>全天归因（多重计入）</strong>：AI组会在一天内（甚至一个会话内）为同一用户轮换多张数学表
（用户日平均触及约 2.1 张表）。因此本节对"用户日 × 数学表"做全天归因：用户当日下注涉及的<strong>每一张</strong>表，
都计入该用户当日的<strong>全部</strong>下注次数与下注总额。例如某用户当日在表1和表2上下注，则表1和表2
各计入该用户当日全部下注次数/金额，并各自将该用户日纳入自己的留存分母。由此各表数值<strong>相加会超过</strong>AI组总量，
表间为重叠样本而非互斥划分。</li>
<li><strong>表的识别按每笔下注的 math_table_id</strong>（bet 级），而非会话首注标注——因为AI在会话内即会换表；
日期归属仍按<strong>会话开始</strong>的北京日期（跨零点会话整体归属开始当天）。</li>
<li><strong>样本过滤</strong>：剔除当日 BASE 下注次数 &lt; 50 的用户日（含其涉及的全部表行）。
理由：AI按行为轮换表，低频用户日对单张表的暴露过少，无法支撑对该表的判断，且噪声大。
本窗口内共保留 17,460 个"用户日×表"行，剔除 6,829 行（约 28%）。</li>
<li>留存口径与主分析一致：D+1 / D+3 在 SS03 有任意下注即算留存；08-03/04 剔除。</li>
</ul>
<p>在上述口径下，按<strong>当日下注总额</strong>分层（层级同主分析：&lt;10 / 10-100 /
100-1000 / 1000+ CNY；多重归因的用户日在其涉及的每张表下落入同一层级）。</p>
{_img(figs["fig10"].name)}
<ul>
<li><strong>zero_95_kai 在每一个层级都显著垫底</strong>（D1：6.3% / 6.5% / 10.4% / 26.9%）——
即使在 ¥1000+ 高额层级也不足其他表的一半，"初始表聚集轻度用户"的选择效应无法完全解释层内的差距，
该表本身对留存可能确有拖累。</li>
<li><strong>ichi 几乎在所有层级领先</strong>（D1：12.3% / 18.4% / 31.8% / 69.4%），但其样本集中于被策略
挑选的高粘性用户（总计仅 584 用户日），外推需谨慎。</li>
<li><strong>低额层级（&lt;¥100）：shi 最稳</strong>（&lt;10 层 D1 16.0%，明显高于其余表的 6%~12%）；
<strong>中高额（¥100-1000）：ichi / shi / ni 居前</strong>（31.8% / 28.6% / 27.1%）；
<strong>高额（¥1000+）：ichi、go 领先</strong>（69.4% / 63.2%），shi 反而回落到 52.3%。</li>
<li>各层级内的表间差异远大于固定组（A/B/默认）之间的差异，但请结合下方选择效应说明解读；
完整数值表见附录。</li>
</ul>
<h2>人日均下注额（各表 × 下注额层级）</h2>
{_img(figs["fig11"].name)}
<p>注：数值为该单元格的<strong>人日均</strong>下注额（CNY/用户日）；误差线为 <strong>bootstrap 95% 置信区间</strong>
（500 次重采样，层内金额分布右偏，故不用正态近似）。纵轴为对数刻度（层级间相差数个数量级）。
金额为全天归因口径（某表的金额 = 其所涉用户日的<strong>全天</strong>下注额，表间有重叠）。
层内均值主要反映用户落在该层级的位置：如 ¥1000+ 层级中 ichi 的人日均显著更高，说明策略把最重度的用户
轮换到了该表；各单元格总额见附录完整交叉表。</p>
<h2>D1 汇总表（各表 × 下注额层级）</h2>
{t_ai_d1_pivot}
<p>注：行按 ¥100-1000 层级 D1 降序；"—"表示该单元格用户日数 &lt; 30；kakuteiB 因样本过小（每格 ≤13 用户日）未列入。
置信区间见上方留存图误差线；各单元格样本量见附录完整交叉表。</p>
<p><strong>解读注意（选择效应）</strong>：AI组的表分配由策略根据用户实时行为决定——高粘性/高价值用户更可能被
轮换到特定表，低活跃用户更可能停留在默认表。因此表间留存差异主要反映<strong>被分配人群的差异</strong>，
不能直接解读为数学表本身的因果效应。两个可见的模式：
（1）normal_zero_95_kai 的 4,548 个用户日全部来自不同用户（每用户仅出现一次），D1 仅 8.2%——
它更像用户接触AI组的"初始/一次性"表，聚集了大量当天即流失的轻度用户；
（2）ichi / ni / shi 等表的用户日留存高（D1 27%~36%）且人日均下注强度大，说明策略将其用于高粘性用户。
若要评估某张表的因果效应，需要在固定人群上做随机实验（如 A/B 组换表）。</p>

<h1>附录</h1>
<h2>完整交叉表（组 × 数学表 × 日下注额，全窗口）</h2>
{t_full}
<h2>AI组完整交叉表（数学表 × 日下注额，全天归因，≥50次/日）</h2>
{t_ai_bucket}
<h2>口径说明</h2>
<ul>
<li>数据源：Athena <code>bituslabs_ds.slot_orders_ab_group</code>（冷数据 + 按策略时间轴修正的 ab_group；已预过滤 status/op_code 测试单）。</li>
<li>会话：同一用户相邻下注间隔 &gt; {SESSION_GAP_SECONDS // 60} 分钟即断开为新会话。</li>
<li>下注日期（bet_date）：会话<strong>首注</strong>的北京日期——跨零点的会话整体归属开始当天。</li>
<li>数学表：以会话首注的数学表标注整个会话（忽略会话内换表）；用户当日出现多个会话数学表时，取当日 BASE 下注次数最多者（winner-take-all；仅 110/72,070 个用户日受影响）。</li>
<li>total_spin：BASE 下注次数（FREE 免费旋转不计次，但其金额不产生下注额）；总下注额仅含 BASE 注金。</li>
<li>下注额分层：按用户<strong>当日下注总额</strong>（CNY，全币种折算）分为 &lt;10 / 10-100 / 100-1000 / 1000+，
约占用户日 42% / 36% / 18% / 5%（分位数 p50≈16，p75≈80，p90≈360）。</li>
<li>留存：D 日某单元格活跃用户中，D+1（D1）/ D+3（D3）在 SS03 有<strong>任意</strong>下注（不限组/表/币种）者的占比；数据完整至 {PRESENCE_END}（北京），故 08-17 的 D3 为空。</li>
<li>剔除 {EXCLUDED_COHORT_DATES[0]} / {EXCLUDED_COHORT_DATES[1]}（北京，按会话开始日）：AB 分组策略当日从 partition_ab 切换为按 user_id 尾号，归属不明确；与特征工程 ETL 口径一致。</li>
<li>AI 组已纳入整体对比（黄色系列）；其"胜者表"口径与其他组一致（当日 BASE 下注最多的会话表）。
AI 组按单张数学表的拆分采用专节所述的全天归因口径。</li>
<li>明细数据已按北京日期分区保存：<code>jobs/output_ss03_mahjiang_streak/ab_mathtable_retention/user_day/</code>；汇总 CSV 同目录。分析脚本：<code>jobs/ss03_mahjiang_streak/analysis_ab_mathtable_retention.py</code>。</li>
</ul>
<h2>汇率（{FX_AS_OF}，exchangerate-api.com）</h2>
{t_fx}
"""
    return body


def publish(body: str, image_paths: list[Path]) -> None:
    from bituslabs_ds.confluence.client import _v2_session, attach_file, get_page_storage, update_page_storage

    if CONFLUENCE_PAGE_ID:
        current = get_page_storage(CONFLUENCE_PAGE_ID)
        for p in image_paths:
            attach_file(CONFLUENCE_PAGE_ID, p.read_bytes(), p.name)
            logger.info("attached %s", p.name)
        update_page_storage(CONFLUENCE_PAGE_ID, PAGE_TITLE, body, expected_version=current["version"])
        logger.info("updated page id=%s (was version %s)", CONFLUENCE_PAGE_ID, current["version"])
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
    page_id = page["id"]
    for p in image_paths:
        attach_file(page_id, p.read_bytes(), p.name)
        logger.info("attached %s", p.name)
    link = page.get("_links", {})
    logger.info("created page id=%s: %s%s", page_id, link.get("base", ""), link.get("webui", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="create the Confluence page")
    args = parser.parse_args()

    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    daily = _read("daily_group")
    figs = {
        "fig1": fig_daily_users_volume(daily),
        "fig2": fig_daily_retention(daily),
        "fig3": fig_retention_by_mathtable(_read("summary_group_mathtable")),
        "fig6": fig_metrics_by_bucket(_read("summary_group_bucket_postswitch")),
        "fig10": fig_ai_retention_by_bucket(
            _read("summary_ai_mathtable_bucket"),
            _read("summary_ai_mathtable").sort("user_days", descending=True)["mathtable"].to_list(),
        ),
        "fig11": fig_ai_total_bet_by_bucket(
            _read("summary_ai_mathtable_bucket"),
            _read("summary_ai_mathtable").sort("user_days", descending=True)["mathtable"].to_list(),
        ),
        "fig5": fig_bucket_mix(_read("summary_group_bucket")),
    }
    body = build_page(figs)
    preview = OUTPUT_DIR / "report_storage.html"
    preview.write_text(body, encoding="utf-8")
    logger.info("storage-format body written to %s", preview)
    if args.publish:
        publish(body, list(figs.values()))


if __name__ == "__main__":
    main()
