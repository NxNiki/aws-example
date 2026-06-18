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

// Per-bar summary drawn beside each bar (parity with the legacy plotly tab's
// annotation). Kept compact so it fits the inter-bar gap: the "95%" qualifier
// and full precision live in the tooltip; the CI line uses an en-dash range.
function statText(s: GroupStat): string {
  const f = (v: number | null) => (v == null ? "–" : v.toFixed(2));
  return [
    `n=${s.n}`,
    `μ=${f(s.mean)}`,
    `med=${f(s.median)}`,
    `max=${f(s.max)}`,
    `min=${f(s.min)}`,
    `CI ${f(s.ci_lower)}–${f(s.ci_upper)}`,
  ].join("\n");
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
  const range = hi - lo || 1;
  // Headroom for the per-bar stat block, which sits just above each bar's top
  // and grows upward; without it the upper lines clip.
  const yAxis = {
    ...(base.yAxis as object),
    min: lo === 0 ? 0 : Math.floor((lo - range * 0.05) * 100) / 100,
    max: Math.ceil((hi + range * 0.9) * 100) / 100,
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
        // Per category: the CI whisker (ci_lower→ci_upper) plus the summary
        // stat text beside the bar. clip:false so the multi-line text isn't
        // cut off at the grid edge.
        clip: false,
        renderItem: (_params, api) => {
          const idx = api.value(0) as number;
          const s = stats[idx];
          const stroke = { stroke: "#333", lineWidth: 1.5 };
          const cap = 5;
          // ECharts custom-element children are awkward to type precisely; the
          // renderItem return is the documented escape hatch.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const children: any[] = [];

          if (s.ci_lower != null && s.ci_upper != null) {
            const lo = api.coord([idx, s.ci_lower]);
            const hi = api.coord([idx, s.ci_upper]);
            children.push(
              { type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: stroke },
              { type: "line", shape: { x1: lo[0] - cap, y1: lo[1], x2: lo[0] + cap, y2: lo[1] }, style: stroke },
              { type: "line", shape: { x1: hi[0] - cap, y1: hi[1], x2: hi[0] + cap, y2: hi[1] }, style: stroke },
            );
          }

          // Left-aligned stat block to the LEFT of the CI whisker, sitting just
          // above the bar's top (mean) and growing upward. Right edge clears the
          // whisker's cap; left edge is offset by the full block width
          // (estimated from the longest line so wide values still clear it).
          const text = statText(s);
          const fontSize = 18;
          const longest = Math.max(...text.split("\n").map((l) => l.length));
          const blockWidth = longest * fontSize * 0.5;
          const at = api.coord([idx, s.mean ?? 0]);
          children.push({
            type: "text",
            style: {
              text,
              x: at[0] - cap - 6 - blockWidth,
              y: at[1] - 8,
              textAlign: "left",
              textVerticalAlign: "bottom",
              fontSize,
              lineHeight: 22,
              fill: "#333",
            },
          });

          return { type: "group", children };
        },
        data: stats.map((_, i) => [i]),
        z: 3,
        silent: true,
      },
    ],
  };
}
