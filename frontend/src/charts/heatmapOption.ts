import type { EChartsOption } from "echarts";
import type { CorrMatrix } from "../api/types";

// Deep Dive correlation heatmap: a Pearson correlation matrix for one
// (cohort × range). RdBu-style diverging scale centered at 0; cell labels show
// the coefficient. p-value significance stars are deferred (need scipy).

// Margins reserve room for the (long, rotated) metric labels; the inner plot
// area is square when width and height each = margin + N·cell.
const HM_MARGIN = { left: 150, right: 30, top: 30, bottom: 130 };
const HM_CELL = 60; // px per cell (was 46, +30%) — figure grows with the number of variables

// Square figure size for an N×N matrix, so cells are always square.
export function heatmapSize(n: number): { width: number; height: number } {
  const plot = Math.max(1, n) * HM_CELL;
  return { width: HM_MARGIN.left + plot + HM_MARGIN.right, height: HM_MARGIN.top + plot + HM_MARGIN.bottom };
}

export function buildHeatmapOption(matrix: CorrMatrix): EChartsOption {
  const { metrics, corr } = matrix;
  const data: [number, number, number | null][] = [];
  for (let i = 0; i < metrics.length; i++) {
    for (let j = 0; j < metrics.length; j++) {
      data.push([j, i, corr[i]?.[j] ?? null]);
    }
  }

  return {
    tooltip: {
      position: "top",
      formatter: (p) => {
        const [x, y, v] = (p as unknown as { data: [number, number, number | null] }).data;
        return `${metrics[y]} × ${metrics[x]}<br/>r = ${v == null ? "–" : v.toFixed(3)}`;
      },
    },
    // Square cells: the caller sizes the figure via heatmapSize() so that the
    // inner plot area (width/height minus these margins) is exactly square.
    grid: HM_MARGIN,
    xAxis: { type: "category", data: metrics, splitArea: { show: true }, axisLabel: { rotate: 45, fontSize: 14 } },
    yAxis: { type: "category", data: metrics, splitArea: { show: true }, axisLabel: { fontSize: 14 } },
    visualMap: {
      min: -1,
      max: 1,
      show: false, // cells are labelled with the coefficient, so the colorbar is redundant
      inRange: { color: ["#B2182B", "#F7F7F7", "#2166AC"] }, // RdBu (neg → 0 → pos)
    },
    series: [
      {
        type: "heatmap",
        data: data.map(([x, y, v]) => [x, y, v as number]),
        label: {
          show: metrics.length <= 12,
          formatter: (p) => {
            const v = (p as unknown as { data: [number, number, number | null] }).data[2];
            return v == null ? "" : v.toFixed(2);
          },
          fontSize: 13,
        },
      },
    ],
  };
}
