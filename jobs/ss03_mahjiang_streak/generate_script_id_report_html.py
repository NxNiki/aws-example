"""
Build a self-contained HTML report for the SS03 script_id distribution check.

Reads the JSON summaries and figure PNGs produced by
``analysis_script_id_normality.py`` and emits a single HTML file (figures
embedded as base64) with two selectors — math table and bet type
(BASE/FREE) — defaulting to normal_zero + BASE.

Run manually after (re)running the analysis:

    poetry run python jobs/ss03_mahjiang_streak/generate_script_id_report_html.py
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from bituslabs_ds.config import LOCAL_ROOT

DEFAULT_ANALYSIS_DIR = Path(LOCAL_ROOT) / "jobs" / "output_ss03_mahjiang_streak" / "script_id_normality"
METRICS = ["num_bets", "num_users"]
BET_TYPES = ["BASE", "FREE"]
DEFAULT_TABLE = "normal_zero"
DEFAULT_BET_TYPE = "BASE"


def normality_verdict(r: dict) -> str:
    if not r["anderson_darling"]["reject_at_5pct"]:
        return "未拒绝 —— 符合正态 ✓"
    if abs(r["skewness"]) < 0.1 and abs(r["excess_kurtosis"]) < 0.2:
        return "统计上拒绝,但效应量极小,实际接近正态"
    if r["mean"] < 5:
        return "拒绝 —— 均值过小,呈 Poisson 右偏,正态近似不适用"
    return "统计上拒绝 —— 轻微右偏"


def uniformity_verdict(u: dict) -> str:
    if not u["reject_at_5pct"]:
        return "符合均匀 ✓"
    return f"拒绝(χ² p={u['p_value']:.3g},离散指数 {u['dispersion_index']:.2f})"


def build_data(analysis_dir: Path) -> tuple[dict, list[str]]:
    summaries = {m: json.loads((analysis_dir / f"{m}_normality_summary.json").read_text()) for m in METRICS}

    data: dict[str, dict] = {}
    tables: set[str] = set()
    for cell, groups in summaries["num_bets"].items():
        math_table, bet_type = cell.rsplit("_", 1)
        if bet_type not in BET_TYPES or "pooled" not in groups:
            continue
        tables.add(math_table)

        cell_data: dict[str, dict] = {}
        for metric in METRICS:
            r = summaries[metric][cell]["pooled"]
            u = r["uniformity"]
            png = analysis_dir / f"{metric}_{cell}_pooled.png"
            uni_png = analysis_dir / f"{metric}_{cell}_uniformity.png"
            cell_data[metric] = {
                "img": "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode(),
                "uni_img": "data:image/png;base64," + base64.b64encode(uni_png.read_bytes()).decode(),
                "n": r["n"],
                "mean_std": f"{r['mean']:.1f} ± {r['std']:.1f}",
                "skew": f"{r['skewness']:.3f}",
                "kurt": f"{r['excess_kurtosis']:.3f}",
                "k2_p": f"{r['dagostino_k2']['p_value']:.3g}",
                "ad": f"{r['anderson_darling']['stat']:.2f}(临界 {r['anderson_darling']['crit_5pct']:.2f})",
                "shapiro": f"{r['shapiro_wilk']['p_value']:.3g}(抽样 {r['shapiro_wilk']['n_subsample']})",
                "uni_p": f"{u['p_value']:.3g}",
                "dispersion": f"{u['dispersion_index']:.3f}",
                "normality": normality_verdict(r),
                "uniformity": uniformity_verdict(u),
            }
        data[cell] = cell_data

    return data, sorted(tables)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SS03 script_id 分布检验报告</title>
<style>
  body {{ font-family: "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f5f6f8; color: #24292f; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; margin: 8px 0 4px; }}
  .meta {{ color: #57606a; font-size: 13px; margin-bottom: 16px; }}
  .card {{ background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 18px 20px; margin-bottom: 16px; }}
  .finding {{ border-left: 4px solid #0969da; }}
  .selectors {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }}
  .selectors label {{ font-weight: 600; margin-right: 8px; }}
  select {{ font-size: 15px; padding: 6px 10px; border-radius: 6px; border: 1px solid #d0d7de; background: #fff; }}
  table.stats {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  table.stats th, table.stats td {{ border: 1px solid #d8dee4; padding: 6px 10px; text-align: left; }}
  table.stats th {{ background: #f6f8fa; }}
  .verdict {{ font-weight: 600; }}
  .ok {{ color: #1a7f37; }}
  .warn {{ color: #9a6700; }}
  img.figure {{ max-width: 100%; border: 1px solid #d8dee4; border-radius: 6px; margin-top: 10px; }}
  h2 {{ font-size: 18px; margin: 4px 0 12px; }}
  h3 {{ font-size: 15px; margin: 16px 0 4px; color: #57606a; }}
  .foot {{ color: #57606a; font-size: 12px; margin-top: 8px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>SS03 script_id 分布检验报告</h1>
  <div class="meta">数据窗口 2026-06-10 ~ 2026-07-28(Asia/Shanghai)· fct_bet_orders,COMPLETED,剔除 B26/TST/TSB/TSO,仅 UUID 格式 script_id,全币种</div>

  <div class="card finding">
    <h2>核心发现</h2>
    <p>每个 math table 为 <b>BASE(普通旋转)与 FREE(免费游戏)各维护一套互不相交的 script 池</b>(主力表各
    10,000 个,kakuteiB 各 1,000 个)。此前观察到的双峰分布完全由两套池的服务频率差异造成;按
    (math table × bet type) 拆分后,<b>池内 script 分配高度符合均匀随机抽取</b>,计数在均值较大时收敛于正态
    (如 normal_zero BASE:σ 观测 31.7 ≈ 均匀随机预测 √1030 ≈ 32),均值较小时呈 Poisson 右偏 —— 这是离散计数的
    数学性质而非机制异常。22 个单元格中仅 normal_zero FREE(χ² p=0.003)与 normal_zero93kai BASE(χ² p=0.030)
    出现轻微均匀性偏离。</p>
  </div>

  <div class="card">
    <div class="selectors">
      <div><label for="sel-table">Math Table</label>
        <select id="sel-table">{table_options}</select></div>
      <div><label for="sel-bettype">Game Type</label>
        <select id="sel-bettype">
          <option value="BASE">BASE(普通旋转)</option>
          <option value="FREE">FREE(免费游戏)</option>
        </select></div>
    </div>
  </div>

  <div class="card">
    <h2 id="cell-title"></h2>
    <table class="stats">
      <tr><th style="width:220px">统计量</th><th>num_bets(每 script 下注数)</th><th>num_users(每 script 用户数)</th></tr>
      <tr><td>script 数 n</td><td id="v-n-0"></td><td id="v-n-1"></td></tr>
      <tr><td>均值 ± 标准差</td><td id="v-mean_std-0"></td><td id="v-mean_std-1"></td></tr>
      <tr><td>偏度</td><td id="v-skew-0"></td><td id="v-skew-1"></td></tr>
      <tr><td>超额峰度</td><td id="v-kurt-0"></td><td id="v-kurt-1"></td></tr>
      <tr><td>D'Agostino–Pearson K² p 值</td><td id="v-k2_p-0"></td><td id="v-k2_p-1"></td></tr>
      <tr><td>Anderson–Darling 统计量</td><td id="v-ad-0"></td><td id="v-ad-1"></td></tr>
      <tr><td>Shapiro–Wilk p 值</td><td id="v-shapiro-0"></td><td id="v-shapiro-1"></td></tr>
      <tr><td>均匀性 χ² p 值</td><td id="v-uni_p-0"></td><td id="v-uni_p-1"></td></tr>
      <tr><td>离散指数(方差/均值)</td><td id="v-dispersion-0"></td><td id="v-dispersion-1"></td></tr>
      <tr><td>正态性结论</td><td class="verdict" id="v-normality-0"></td><td class="verdict" id="v-normality-1"></td></tr>
      <tr><td>均匀性结论</td><td class="verdict" id="v-uniformity-0"></td><td class="verdict" id="v-uniformity-1"></td></tr>
    </table>

    <h3>script_id 均匀性检查:每个 script_id 的 num_bets(按 script_id 字典序)</h3>
    <p class="foot" style="margin:2px 0 0">每个点是一个 script;若服务均匀,点应落在红线(均匀期望)附近、
    ±3·√均值 的 Poisson 带内,且无随 script_id 变化的模式。</p>
    <img class="figure" id="img-uni-num_bets" alt="num_bets per script_id">
    <h3>num_bets:直方图 + 正态拟合 / QQ 图</h3>
    <img class="figure" id="img-num_bets" alt="num_bets figure">
    <h3>num_users:直方图 + 正态拟合 / QQ 图</h3>
    <img class="figure" id="img-num_users" alt="num_users figure">
    <h3>script_id 均匀性检查(num_users 口径)</h3>
    <img class="figure" id="img-uni-num_users" alt="num_users per script_id">

    <div class="foot">检验方法:D'Agostino–Pearson K² 与 Anderson–Darling(大样本正态性),Shapiro–Wilk 于固定种子
    5000 抽样;均匀性为卡方拟合优度(期望 = 各 script 均等计数)。n≈10000 时检验对微小偏离极为敏感,
    实际判断请结合偏度/峰度与 QQ 图。生成脚本:jobs/ss03_mahjiang_streak/generate_script_id_report_html.py</div>
  </div>
</div>

<script>
const DATA = {data_json};
const FIELDS = ["n","mean_std","skew","kurt","k2_p","ad","shapiro","uni_p","dispersion","normality","uniformity"];
const selTable = document.getElementById("sel-table");
const selBet = document.getElementById("sel-bettype");

function render() {{
  const cell = selTable.value + "_" + selBet.value;
  const d = DATA[cell];
  document.getElementById("cell-title").textContent =
    selTable.value + " · " + (selBet.value === "BASE" ? "BASE(普通旋转)" : "FREE(免费游戏)");
  ["num_bets","num_users"].forEach((metric, i) => {{
    FIELDS.forEach(f => {{
      const el = document.getElementById("v-" + f + "-" + i);
      el.textContent = d[metric][f];
      if (f === "normality" || f === "uniformity") {{
        el.className = "verdict " + (String(d[metric][f]).includes("✓") ? "ok" : "warn");
      }}
    }});
    document.getElementById("img-" + metric).src = d[metric].img;
    document.getElementById("img-uni-" + metric).src = d[metric].uni_img;
  }});
}}

selTable.value = "{default_table}";
selBet.value = "{default_bet_type}";
selTable.addEventListener("change", render);
selBet.addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the SS03 script_id distribution HTML report.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Default: <analysis-dir>/report_zh.html")
    args = parser.parse_args()

    data, tables = build_data(args.analysis_dir)
    table_options = "".join(
        f'<option value="{t}"{" selected" if t == DEFAULT_TABLE else ""}>{t}</option>' for t in tables
    )
    html = HTML_TEMPLATE.format(
        table_options=table_options,
        data_json=json.dumps(data),
        default_table=DEFAULT_TABLE,
        default_bet_type=DEFAULT_BET_TYPE,
    )
    output = args.output or (args.analysis_dir / "report_zh.html")
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(data)} cells)")


if __name__ == "__main__":
    main()
