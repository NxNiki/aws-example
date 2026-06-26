import { fmtNum, fmtPct, fmtPValue, pctColor, pctScale, pctValue } from "../../lib/format";
import { STAT_LABEL, metricNote } from "./SummaryGrid";
import type { SummaryCell, SummaryColumn, SummaryRow, SummaryStat, SummaryTableFigureSource } from "../../api/types";

// Confluence storage-format <table> for a Summary-table report figure. Tables
// have no chart to rasterize, so the report export ships this HTML (rendered
// inline) instead of a PNG. Mirrors SummaryGrid's layout: metric rows sectioned
// by group, cohort×range columns, one reference column, value (±%) per stat,
// optional p-value column.

const esc = (s: string): string => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const compactRange = (label: string): string =>
  label
    .split(" → ")
    .map((d) => d.slice(2))
    .join(":");

// Confluence accepts inline text color as `<span style="color: rgb(...)">`.
// The ±% uses the per-metric diverging colormap (pctColor); "(ref)" stays gray.
const REF_GRAY = "rgb(150,150,150)";
const colored = (text: string, color: string): string => `<span style="color: ${color};">${esc(text)}</span>`;

// Per-row colormap scale = largest |Δ%| among a metric's non-reference cells.
function rowPctScale(cells: (SummaryCell | null)[], refCell: SummaryCell | null, refIndex: number, stats: SummaryStat[]): number {
  const shown = stats.length ? stats : (["mean"] as SummaryStat[]);
  const pcts: Array<number | null> = [];
  cells.forEach((cell, ci) => {
    if (ci === refIndex || !cell) return;
    for (const st of shown) pcts.push(pctValue(cell[st], refCell ? refCell[st] : null));
  });
  return pctScale(pcts);
}

function cellHtml(
  cell: SummaryCell | null,
  refCell: SummaryCell | null,
  stats: SummaryStat[],
  isRef: boolean,
  scale: number,
): string {
  const shown = stats.length ? stats : (["mean"] as SummaryStat[]);
  return shown
    .map((st) => {
      if (!cell) return "—";
      const val = cell[st];
      const refVal = refCell ? refCell[st] : null;
      let s = esc(fmtNum(val));
      if (isRef) {
        s += ` ${colored("(ref)", REF_GRAY)}`;
      } else {
        const p = pctValue(val, refVal);
        if (p != null) s += ` ${colored(`(${fmtPct(val, refVal)})`, pctColor(p, scale) ?? REF_GRAY)}`;
      }
      return s;
    })
    .join("<br/>");
}

export function buildSummaryTableHtml(
  source: SummaryTableFigureSource,
  columns: SummaryColumn[],
  rows: SummaryRow[],
): string {
  const { stats, reference_key: refKey, metrics: selected, metric_options: opts } = source;
  const filtered = rows.filter((r) => (selected[r.group_id] ?? []).includes(r.metric));
  if (columns.length === 0 || filtered.length === 0) return "";

  const refIndex = columns.findIndex((c) => c.key === refKey);
  const showP = source.pvalues && filtered.some((r) => r.test);
  // With >1 stat, a "Stat" column carries the labels so they aren't repeated in
  // every cell (mirrors SummaryGrid).
  const shownStats = stats.length ? stats : (["mean"] as SummaryStat[]);
  const showStatCol = shownStats.length > 1;
  const nCols = 1 + (showStatCol ? 1 : 0) + columns.length + (showP ? 1 : 0);

  const head =
    "<th>Metric</th>" +
    (showStatCol ? "<th>Stat</th>" : "") +
    columns
      .map(
        (c, i) =>
          `<th>${esc(c.cohort)}<br/>${esc(compactRange(c.range_label))}${i === refIndex ? "<br/>(ref)" : ""}</th>`,
      )
      .join("") +
    (showP ? "<th>p-value</th>" : "");

  const body: string[] = [];
  filtered.forEach((r, i) => {
    if (i === 0 || filtered[i - 1].group_id !== r.group_id) {
      body.push(`<tr><th colspan="${nCols}">${esc(r.group_label)}</th></tr>`);
    }
    const refCell = refIndex >= 0 ? r.cells[refIndex] : null;
    const scale = rowPctScale(r.cells, refCell, refIndex, stats);
    const statCell = showStatCol ? `<td>${shownStats.map((st) => esc(STAT_LABEL[st])).join("<br/>")}</td>` : "";
    const cells = columns.map((_, ci) => `<td>${cellHtml(r.cells[ci], refCell, stats, ci === refIndex, scale)}</td>`).join("");
    const pCell = showP ? `<td>${esc(fmtPValue(r.pvalue))}</td>` : "";
    body.push(`<tr><td>${esc(r.metric + metricNote(opts[r.metric]))}</td>${statCell}${cells}${pCell}</tr>`);
  });

  return `<table><tbody><tr>${head}</tr>${body.join("")}</tbody></table>`;
}
