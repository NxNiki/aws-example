import { Fragment } from "react";
import { fmtNum, fmtPct, fmtPValue, pctColor, pctScale, pctValue } from "../../lib/format";
import type { SummaryCell, SummaryColumn, SummaryMetricOption, SummaryRow, SummaryStat } from "../../api/types";

// Stat → short header label. Used by the controls (full label) and cells (short).
export const STAT_LABEL: Record<SummaryStat, string> = {
  n: "count",
  mean: "mean",
  median: "median",
  q1: "25%",
  q3: "75%",
  min: "min",
  max: "max",
};

// Column date range as a compact "YY-MM-DD:YY-MM-DD" (the backend label is
// "YYYY-MM-DD → YYYY-MM-DD"). Full range stays in the header's title tooltip.
function compactRange(rangeLabel: string): string {
  return rangeLabel
    .split(" → ")
    .map((d) => d.slice(2)) // 2026-06-19 → 26-06-19
    .join(":");
}

// A dim suffix noting a metric's own clip / log (its values are reshaped, so the
// label has to say so). Empty when the metric has no options set.
export function metricNote(opt: SummaryMetricOption | undefined): string {
  if (!opt) return "";
  const parts: string[] = [];
  if (opt.clip.enable && (opt.clip.min != null || opt.clip.max != null)) {
    const pct = opt.clip.percentile ? "%" : "";
    parts.push(`clip ${opt.clip.min == null ? "−∞" : `${opt.clip.min}${pct}`}:${opt.clip.max == null ? "∞" : `${opt.clip.max}${pct}`}`);
  }
  if (opt.log) parts.push("log");
  return parts.length ? `  ·  ${parts.join(", ")}` : "";
}

// One metric's value in one column: each selected stat as `value (±% vs reference)`.
// The reference column shows raw values only. Multiple stats stack as rows.
function CellView(props: {
  cell: SummaryCell | null;
  refCell: SummaryCell | null;
  stats: SummaryStat[];
  isRef: boolean;
  scale: number; // per-row max |Δ%| for the colormap
}) {
  const { cell, refCell, stats, isRef, scale } = props;
  // Stat labels live in the dedicated "Stat" column; cells show only values,
  // one line per stat (in the same order) so they align with that column.
  const shown = stats.length ? stats : (["mean"] as SummaryStat[]);
  return (
    <div className="flex flex-col gap-0.5">
      {shown.map((st) => {
        if (!cell) {
          return (
            <div key={st} className="text-gray-300">
              —
            </div>
          );
        }
        const val = cell[st];
        const refVal = refCell ? refCell[st] : null;
        const p = isRef ? null : pctValue(val, refVal);
        return (
          <div key={st} className="whitespace-nowrap tabular-nums">
            <span className="text-gray-800">{fmtNum(val)}</span>
            {p != null && (
              <span className="ml-1" style={{ color: pctColor(p, scale) ?? undefined }}>
                ({fmtPct(val, refVal)})
              </span>
            )}
            {isRef && <span className="text-gray-400 ml-1">(ref)</span>}
          </div>
        );
      })}
    </div>
  );
}

// Per-row colormap normalization: the largest |Δ%| among a metric's non-reference
// cells (across every shown stat), so each metric is colored on its own scale.
function rowPctScale(
  row: SummaryRow,
  columns: SummaryColumn[],
  refIndex: number,
  stats: SummaryStat[],
): number {
  const shown = stats.length ? stats : (["mean"] as SummaryStat[]);
  const refCell = refIndex >= 0 ? row.cells[refIndex] : null;
  const pcts: Array<number | null> = [];
  columns.forEach((_, ci) => {
    if (ci === refIndex) return;
    const cell = row.cells[ci];
    if (!cell) return;
    for (const st of shown) pcts.push(pctValue(cell[st], refCell ? refCell[st] : null));
  });
  return pctScale(pcts);
}

// The summary grid: rows are metrics (sectioned by metric group), columns are
// cohort × date-range, one column flagged as the reference. Shared by the
// Summary-table tab (live, with a reference radio) and the Report tab's figure
// card (read-only). `rows` should already be filtered to the selected metrics.
export function SummaryGrid(props: {
  columns: SummaryColumn[];
  rows: SummaryRow[];
  stats: SummaryStat[];
  referenceKey: string | null;
  showPValues: boolean;
  metricOptions: Record<string, SummaryMetricOption>;
  onSetReference?: (key: string) => void; // when set, the header shows ref radios
}) {
  const { columns, rows, stats, referenceKey, metricOptions, onSetReference } = props;
  if (columns.length === 0 || rows.length === 0) return null;
  const refIndex = columns.findIndex((col) => col.key === referenceKey);
  const showPCol = props.showPValues && rows.some((r) => r.test);
  // With >1 stat, a dedicated "Stat" column carries the labels (count/mean/…)
  // so they aren't repeated inside every cell; with one stat it's omitted.
  const shownStats = stats.length ? stats : (["mean"] as SummaryStat[]);
  const showStatCol = shownStats.length > 1;
  const totalCols = 1 + (showStatCol ? 1 : 0) + columns.length + (showPCol ? 1 : 0);

  return (
    <div className="overflow-x-auto border rounded">
      <table className="border-collapse text-base">
        <thead>
          <tr className="bg-gray-100">
            <th className="sticky left-0 z-10 bg-gray-100 border-b border-r px-3 py-2 text-left font-semibold text-gray-700">
              Metric
            </th>
            {showStatCol && (
              <th className="border-b border-r px-3 py-2 text-left font-semibold text-gray-700">Stat</th>
            )}
            {columns.map((col) => {
              const isRef = col.key === referenceKey;
              return (
                <th key={col.key} className={"border-b border-r px-3 py-2 text-left align-top " + (isRef ? "bg-blue-50" : "")}>
                  {/* One line per group dimension (lifecycle | daily, …) to keep columns narrow. */}
                  {col.cohort.split(" | ").map((dim) => (
                    <div key={dim} className="font-semibold text-gray-700 whitespace-nowrap">
                      {dim}
                    </div>
                  ))}
                  <div className="text-gray-500 text-sm whitespace-nowrap tabular-nums" title={col.range_label}>
                    {compactRange(col.range_label)}
                  </div>
                  {onSetReference ? (
                    <label className="mt-1 flex items-center gap-1 text-sm font-normal text-gray-500 cursor-pointer">
                      <input
                        type="checkbox"
                        name="summary-reference"
                        checked={isRef}
                        onChange={() => onSetReference(col.key)}
                      />
                      reference
                    </label>
                  ) : (
                    isRef && <div className="mt-1 text-sm font-normal text-gray-400">reference</div>
                  )}
                </th>
              );
            })}
            {showPCol && (
              <th className="border-b px-3 py-2 text-left font-semibold text-gray-700 whitespace-nowrap">p-value</th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const newGroup = i === 0 || rows[i - 1].group_id !== row.group_id;
            const refCell = refIndex >= 0 ? row.cells[refIndex] : null;
            const note = metricNote(metricOptions[row.metric]);
            const scale = rowPctScale(row, columns, refIndex, stats);
            return (
              <Fragment key={`${row.group_id}:${row.metric}`}>
                {newGroup && (
                  <tr className="bg-gray-50">
                    <td
                      className="sticky left-0 z-10 bg-gray-50 border-b border-t px-3 py-1.5 font-semibold text-gray-600"
                      colSpan={totalCols}
                    >
                      {row.group_label}
                    </td>
                  </tr>
                )}
                <tr className="hover:bg-gray-50">
                  <td className="sticky left-0 z-10 bg-white border-b border-r px-3 py-2 font-medium text-gray-700 whitespace-nowrap">
                    {row.metric}
                    {note && <span className="text-sm font-normal text-gray-400">{note}</span>}
                  </td>
                  {showStatCol && (
                    <td className="border-b border-r px-3 py-2 text-sm text-gray-500 align-top">
                      <div className="flex flex-col gap-0.5">
                        {shownStats.map((st) => (
                          <div key={st}>{STAT_LABEL[st]}</div>
                        ))}
                      </div>
                    </td>
                  )}
                  {columns.map((col, ci) => (
                    <td key={col.key} className={"border-b border-r px-3 py-2 " + (ci === refIndex ? "bg-blue-50/50" : "")}>
                      <CellView cell={row.cells[ci]} refCell={refCell} stats={stats} isRef={ci === refIndex} scale={scale} />
                    </td>
                  ))}
                  {showPCol && (
                    <td className="border-b px-3 py-2 tabular-nums whitespace-nowrap" title={row.test ?? undefined}>
                      {fmtPValue(row.pvalue)}
                    </td>
                  )}
                </tr>
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
