import type { EChartsOption } from "echarts";
import type { GroupStat } from "../api/types";
import { colorForIndex } from "./styles";

// Stats-by-Group: one box (5-number summary) or bar (mean ± bootstrap CI) per
// (cohort × date-range). Color encodes the cohort; ranges 2/3 are dimmed
// (parity with the legacy opacity/hatch). Backend returns the summary stats;
// this just lays them out.

function cohortColors(stats: GroupStat[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const s of stats) if (!map.has(s.cohort)) map.set(s.cohort, colorForIndex(map.size));
  return map;
}

const opacityFor = (rangeIndex: number) => (rangeIndex === 0 ? 1 : 0.45);
// Two-line category label: cohort on top, the exact date range beneath.
const label = (s: GroupStat) => `${s.cohort}\n${s.range_label}`;

function tooltipText(s: GroupStat): string {
  const f = (v: number | null) => (v == null ? "–" : v.toFixed(2));
  return [
    `<b>${s.cohort}</b> · ${s.range_label}`,
    `n=${s.n} μ=${f(s.mean)} med=${f(s.median)}`,
    `min=${f(s.min)} max=${f(s.max)}`,
    `95% CI [${f(s.ci_lower)}, ${f(s.ci_upper)}]`,
  ].join("<br/>");
}

export function buildGroupDistributionOption(stats: GroupStat[], mode: "box" | "bar", metric: string): EChartsOption {
  const colors = cohortColors(stats);
  const categories = stats.map(label);
  const tips = stats.map(tooltipText);
  const base: EChartsOption = {
    tooltip: { trigger: "axis", formatter: (p) => tips[(Array.isArray(p) ? p[0] : p).dataIndex] ?? "" },
    grid: { left: 96, right: 24, top: 24, bottom: 88 },
    xAxis: { type: "category", data: categories, axisLabel: { interval: 0, fontSize: 14, lineHeight: 18 } },
    yAxis: {
      type: "value",
      scale: true,
      name: metric,
      nameLocation: "middle",
      nameGap: 72,
      nameTextStyle: { fontWeight: "bold", fontSize: 15 },
      axisLabel: { fontSize: 14 },
    },
  };

  if (mode === "box") {
    const data = stats.map((s) => ({
      // ECharts boxplot needs numbers; missing summary values render as gaps.
      value: [s.min, s.q1, s.median, s.q3, s.max].map((v) => (v == null ? NaN : v)),
      itemStyle: { color: colors.get(s.cohort), opacity: opacityFor(s.range_index), borderColor: "#333" },
    }));
    return { ...base, series: [{ type: "boxplot", data }] };
  }

  // Bar (mean) + a custom error-bar overlay for the bootstrap CI. The CI
  // whiskers are a custom series that does NOT contribute to the axis extent,
  // so set the y-limits explicitly from the CI bounds — otherwise an upper CI
  // above the tallest bar gets clipped.
  const extent = stats
    .flatMap((s) => [s.mean, s.ci_lower, s.ci_upper])
    .filter((v): v is number => v != null && Number.isFinite(v));
  const lo = Math.min(0, ...extent);
  const hi = Math.max(0, ...extent);
  const pad = (hi - lo) * 0.05 || 1;
  const yAxis = {
    ...(base.yAxis as object),
    min: lo === 0 ? 0 : Math.floor((lo - pad) * 100) / 100,
    max: Math.ceil((hi + pad) * 100) / 100,
  };

  const bars = stats.map((s) => ({
    value: s.mean,
    itemStyle: { color: colors.get(s.cohort), opacity: opacityFor(s.range_index) },
  }));
  return {
    ...base,
    yAxis,
    series: [
      { type: "bar", data: bars },
      {
        type: "custom",
        // Draw a vertical whisker from ci_lower to ci_upper at each category.
        renderItem: (_params, api) => {
          const idx = api.value(0) as number;
          const s = stats[idx];
          if (s.ci_lower == null || s.ci_upper == null) return { type: "group", children: [] };
          const lo = api.coord([idx, s.ci_lower]);
          const hi = api.coord([idx, s.ci_upper]);
          const stroke = { stroke: "#333", lineWidth: 1.5 };
          const cap = 5;
          return {
            type: "group",
            children: [
              { type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: stroke },
              { type: "line", shape: { x1: lo[0] - cap, y1: lo[1], x2: lo[0] + cap, y2: lo[1] }, style: stroke },
              { type: "line", shape: { x1: hi[0] - cap, y1: hi[1], x2: hi[0] + cap, y2: hi[1] }, style: stroke },
            ],
          };
        },
        data: stats.map((_, i) => [i]),
        z: 3,
        silent: true,
      },
    ],
  };
}
