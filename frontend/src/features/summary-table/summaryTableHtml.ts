import { fmtNum, fmtPct, fmtPValue } from "../../lib/format";
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

// Confluence accepts inline text color as `<span style="color: rgb(...)">`
// (matches SummaryGrid's red/green ±% and gray "(ref)").
const PCT_DOWN = "rgb(220,38,38)"; // text-red-600
const PCT_UP = "rgb(21,128,61)"; // text-green-700
const REF_GRAY = "rgb(150,150,150)";
const colored = (text: string, color: string): string => `<span style="color: ${color};">${esc(text)}</span>`;

function cellHtml(cell: SummaryCell | null, refCell: SummaryCell | null, stats: SummaryStat[], isRef: boolean): string {
  if (!cell) return "—";
  const shown = stats.length ? stats : (["mean"] as SummaryStat[]);
  return shown
    .map((st) => {
      const val = cell[st];
      let s = esc(`${shown.length > 1 ? `${STAT_LABEL[st]} ` : ""}${fmtNum(val)}`);
      if (isRef) {
        s += ` ${colored("(ref)", REF_GRAY)}`;
      } else {
        const pct = fmtPct(val, refCell ? refCell[st] : null);
        if (pct) s += ` ${colored(`(${pct})`, pct.startsWith("-") ? PCT_DOWN : PCT_UP)}`;
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
  const nCols = 1 + columns.length + (showP ? 1 : 0);

  const head =
    "<th>Metric</th>" +
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
    const cells = columns.map((_, ci) => `<td>${cellHtml(r.cells[ci], refCell, stats, ci === refIndex)}</td>`).join("");
    const pCell = showP ? `<td>${esc(fmtPValue(r.pvalue))}</td>` : "";
    body.push(`<tr><td>${esc(r.metric + metricNote(opts[r.metric]))}</td>${cells}${pCell}</tr>`);
  });

  return `<table><tbody><tr>${head}</tr>${body.join("")}</tbody></table>`;
}
