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
METRIC_LABELS = {
    "retention_d1": "次日留存率",
    "avg_user_total_bet": "人均日下注额（CNY）",
    "avg_user_num_bets": "人均日下注次数",
}
SHORT_NAMES = {
    "normal_zero_95_kai": "95_kai",
    "normal_Zero_BGTR95_Saitekika_BGadj_v3": "BGadj_v3",
    "normal_Zero_BGTR97_Saitekika_BGadj_v2": "BGTR97_v2",
    "normal_Zero_BG97_Saitekika_BGadj": "BG97_adj",
}
CSV_ATTACHMENTS = (
    "summary_range.csv",
    "summary_range_by_table.csv",
    "daily_all_users.csv",
    "summary_permutation.csv",
    "top5_permutations.csv",
)


def _short(perm: str) -> str:
    parts = [SHORT_NAMES.get(p, p.removeprefix("normal_")) for p in perm.split("-")]
    return "-".join(parts)


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
        ("用户RTP 均值/中位", lambda r: f"{_num(r['avg_user_rtp'], 3)} / {_num(r['med_user_rtp'], 3)}"),
        ("整体RTP（合并）", lambda r: _num(r["pooled_rtp"], 3)),
    ]
    rows = [[RANGE_LABELS[rec["range"]]] + [f(rec) for _, f in cols] for rec in summary.sort("range").to_dicts()]
    return _table(["时间段"] + [c for c, _ in cols], rows)


def top5_tables(top5: pl.DataFrame) -> str:
    out = []
    for metric, label in METRIC_LABELS.items():
        rows = []
        for rec in top5.filter(pl.col("ranked_by") == metric).sort(["range", "rank"]).to_dicts():
            rows.append(
                [
                    RANGE_LABELS[rec["range"]],
                    str(rec["rank"]),
                    f"<code>{_short(rec['permutation'])}</code>",
                    f"{rec['user_days']:,} / {rec['n_users']:,}",
                    f"{_pct(rec['retention_d1'])}（{rec['d1_cohort']:,}）",
                    f"{_num(rec['avg_user_total_bet'])} / {_num(rec['med_user_total_bet'])}",
                    f"{_num(rec['avg_user_num_bets'])} / {_num(rec['med_user_num_bets'], 0)}",
                    f"{_num(rec['avg_user_rtp'], 3)} / {_num(rec['med_user_rtp'], 3)}",
                    _num(rec["pooled_rtp"], 3),
                ]
            )
        out.append(
            f"<h3>按{label}排序</h3>"
            + _table(
                [
                    "时间段",
                    "名次",
                    "排列",
                    "用户日 / 用户数",
                    "次日留存（样本）",
                    "人均日下注额 均值/中位",
                    "人均日下注次数 均值/中位",
                    "用户RTP 均值/中位",
                    "整体RTP",
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
                    f"{_num(rec['avg_user_rtp'], 3)} / {_num(rec['med_user_rtp'], 3)}",
                    _num(rec["pooled_rtp"], 3),
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
                    "用户RTP 均值/中位",
                    "整体RTP",
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


def build_page() -> str:
    summary = pl.read_csv(OUTPUT_DIR / "summary_range.csv")
    by_table = pl.read_csv(OUTPUT_DIR / "summary_range_by_table.csv")
    top5 = pl.read_csv(OUTPUT_DIR / "top5_permutations.csv")
    daily = pl.read_csv(OUTPUT_DIR / "daily_all_users.csv", try_parse_dates=True)
    (r1s, r1e), (r2s, r2e) = RANGES["range1_pre"], RANGES["range2_post"]

    return f"""
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
<li><strong>清洗规则（仅决定排列标签）</strong>：BASE 下注次数 &lt; {MIN_BLOCK_BETS} 的区块从<strong>排列标签</strong>中剔除
（按区块而非当日合计：如某用户当日 A 20 注 → B 50 注 → A 20 注，则标签只保留 B）。
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
<li><strong>均值 99 分位截断（winsorize）</strong>：user_total_bet 与 user_num_bets 的<strong>均值</strong>
将超过 99 分位的样本<strong>截断到阈值</strong>（保留而非剔除——鲸鱼用户日是真实收入，只是不让单个样本主导均值）。
阈值按每个时间段的<strong>全人群</strong>用户日分布估计（线性插值；小单元内重新估计分位数噪声太大），
统一应用于所有单元格：范围1 约 13,028 CNY / 2,132 次，范围2 约 13,071 CNY / 2,226 次。
中位数、留存率、用户RTP 与整体 RTP 不截断。</li>
<li>下注次数 = BASE 下注次数；下注额 = BASE 注金；RTP = 派彩 / BASE 注金（含免费游戏派彩）；仅 CNY，金额无需折算。
<strong>用户RTP</strong>为单用户日的比值，<strong>整体RTP</strong>为全体用户合并（∑派彩/∑注金）。</li>
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
<code>BGadj_v3</code>（19.9%）、<code>95_kai</code>（19.2%）、<code>zero</code>（18.9%）、<code>shi</code>（18.8%），
高于 <code>zero93kai</code>（15.9%）与 <code>BG97_adj</code>（15.0%），
人均日下注次数（166）也低于多数单表用户日（142~226）；
<strong>范围2 中 AI 组与单表用户日基本持平</strong>（留存 21.4% vs 20.7%~22.5%，
其中 <code>BGTR97_v2</code> 22.5% 最高、下注额均值 912 也显著领先），但 AI 组整体 RTP（0.992）高于
<code>95_kai</code>（0.935）与 <code>BGadj_v3</code>（0.969）。
另外 <code>BGadj_v3</code> 早在范围1 即被固定表组使用，08-18 起才进入 AI 轮换。</p>

<h2>按日历日的全体用户 RTP</h2>
{daily_summary(daily)}
<p>范围2 日均 RTP 达 1.023，多天日 RTP &gt; 1（如 08-22 为 1.237、08-26 为 1.150），即当日 AI 组派彩超过注金；
范围1 也有零星高 RTP 日（07-23 为 2.667）。逐日明细（RTP、留存、活跃用户、下注量）见附件
<code>daily_all_users.csv</code>。</p>

<h2>Top-5 数学表排列（每指标、每时间段；仅统计 ≥ {MIN_PERMUTATION_USER_DAYS} 用户日的排列；
排列标签经清洗，指标为全天口径）</h2>
<p>表名缩写：去掉 <code>normal_</code> 前缀；<code>95_kai</code> = <code>normal_zero_95_kai</code>；
<code>BGadj_v3</code> = <code>normal_Zero_BGTR95_Saitekika_BGadj_v3</code>。</p>
{top5_tables(top5)}

<h2>结论</h2>
<ul>
<li>AI 组内部（清洗后）：两个时间段内，<strong>多表切换的用户日在留存、下注额、下注次数上都优于 AI 组内的单表用户日</strong>。</li>
<li>跨组对比：<strong>范围1 中 AI 组整体略逊于头部固定单表用户日</strong>（留存 18.4% vs 最高 19.9%，下注次数也更低）；
范围2 改用 95_kai + BGadj_v3 轮换后与单表用户日持平（21.4% vs 20.7%~22.5%），但整体 RTP 更高（0.992）。</li>
<li>范围2 中 <strong>BGadj_v3 → 95_kai</strong> 的排列最突出：次日留存 55.0%（约为范围2 AI 组整体 21.4% 的 2.6 倍）、
人均日下注额 833 CNY；但其全天整体 RTP 高达 1.80，相当一部分活跃度是用派彩换来的。</li>
<li><code>shi-ni</code>（四 → 二）是唯一在两个时间段都进入多个榜单的排列。</li>
<li>范围2 整体留存、下注次数与剔除极值后的下注额均值都优于范围1，但同期 RTP 也更高，评估收益时需一并考虑。</li>
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

        current = get_page_storage(CONFLUENCE_PAGE_ID)
        update_page_storage(CONFLUENCE_PAGE_ID, PAGE_TITLE, body, expected_version=current["version"])
        confluence = _get_client()
        for name in CSV_ATTACHMENTS:
            # Updating an existing attachment's data 400s (content-type
            # mismatch on the update endpoint), so replace by delete+create.
            try:
                confluence.delete_attachment(CONFLUENCE_PAGE_ID, name)
            except Exception:
                pass
            attach_file(CONFLUENCE_PAGE_ID, (OUTPUT_DIR / name).read_bytes(), name)
            logger.info("attached %s", name)
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
