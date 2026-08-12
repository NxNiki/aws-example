import type { EChartsOption } from "echarts";
import type { GroupStat } from "../api/types";
import { infoGraphic } from "./clipFilterInfo";
import { colorForIndex, wrapCohort } from "./styles";

// Stats-by-Group: one box (5-number summary) or bar (mean ± bootstrap CI) per
// (cohort × date-range). Color encodes the cohort; ranges 2/3 are dimmed
// (parity with the legacy opacity/hatch). Backend returns the summary stats;
// this just lays them out.

// Bar-density tiers: the per-bar width budget steps down as cohorts are
// added, so large selections — e.g. the ss03_ai mathtable-combo picker —
// don't produce a multi-screen-wide chart. The x-axis date range always
// wraps its end date onto its own line (~95px instead of ~190px), so the
// label never dictates bar spacing; the real per-bar constraint is the
// left-side stat block, whose font already shrinks with bar count.
export const COMPACT_BARS_FROM = 8; // 1-7 bars = base tier
export const DENSE_BARS_FROM = 15; // "more than 14"

export function groupChartWidth(nBars: number): number {
  const perBar = nBars >= DENSE_BARS_FROM ? 120 : nBars >= COMPACT_BARS_FROM ? 150 : 180;
  return Math.max(500, 120 + nBars * perBar);
}

function cohortColors(stats: GroupStat[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const s of stats) if (!map.has(s.cohort)) map.set(s.cohort, colorForIndex(map.size));
  return map;
}

const opacityFor = (rangeIndex: number) => (rangeIndex === 0 ? 1 : 0.45);
// Multi-line category label: cohort on top (one line per group dimension when
// two are selected, e.g. lifecycle | daily), then the date range with its end
// date on its own line — half the width of the one-line form, so bars can sit
// close together at every density tier.
const label = (s: GroupStat) => `${wrapCohort(s.cohort)}\n${s.range_label.replace(" → ", " →\n")}`;

function tooltipText(s: GroupStat): string {
  const f = (v: number | null) => (v == null ? "–" : v.toFixed(3));
  return [
    `<b>${s.cohort}</b> · ${s.range_label}`,
    `n=${s.n} μ=${f(s.mean)} med=${f(s.median)}`,
    `min=${f(s.min)} max=${f(s.max)}`,
    `95% CI [${f(s.ci_lower)}, ${f(s.ci_upper)}]`,
  ].join("<br/>");
}

// Abbreviate large magnitudes to save width: ≥1e9 → "B", ≥1e6 → "M", ≥1e3 →
// "k", each with three decimals (1,234 → "1.234k", 10,234 → "10.234k"). Below
// 1,000 stays as-is — 2 decimals for floats, the bare integer for counts.
// Full precision stays in the tooltip.
function compact(v: number | null, integer = false): string {
  if (v == null) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(3)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(3)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(3)}k`;
  return integer ? String(v) : v.toFixed(3);
}

// Per-bar summary drawn beside each bar (parity with the legacy plotly tab's
// annotation). The "95%" qualifier and full precision live in the tooltip.
// (name, value) pairs: the renderer lays them out in two columns so the
// metric names are left-aligned and every '=' lands in the same column.
function statLines(s: GroupStat): [string, string][] {
  return [
    ["n", compact(s.n, true)],
    ["μ", compact(s.mean)],
    ["med", compact(s.median)],
    ["max", compact(s.max)],
    ["min", compact(s.min)],
    ["CI lo", compact(s.ci_lower)],
    ["CI hi", compact(s.ci_upper)],
  ];
}

export function buildGroupDistributionOption(
  stats: GroupStat[],
  mode: "box" | "bar",
  metric: string,
  info = "", // active clip/filter tag, e.g. "clip: [0, 100], filter: [1%, 99%]"
): EChartsOption {
  const colors = cohortColors(stats);
  const categories = stats.map(label);
  const tips = stats.map(tooltipText);
  const base: EChartsOption = {
    graphic: infoGraphic(info),
    tooltip: { trigger: "axis", formatter: (p) => tips[(Array.isArray(p) ? p[0] : p).dataIndex] ?? "" },
    // Bottom gutter sized for the three-line label (cohort + wrapped range).
    grid: { left: 96, right: 24, top: 24, bottom: 108 },
    xAxis: { type: "category", data: categories, axisLabel: { interval: 0, fontSize: 16, lineHeight: 20 } },
    yAxis: {
      type: "value",
      scale: true,
      name: metric,
      nameLocation: "middle",
      nameGap: 72,
      nameTextStyle: { fontWeight: "bold", fontSize: 17 },
      axisLabel: { fontSize: 16 },
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
    max: Math.ceil((hi + range * 0.95) * 100) / 100,
  };

  // Hollow bars: cohort color on the outline only, so overlapping context
  // near the bar's base (whiskers, gridlines, neighboring stat text) stays
  // visible through the body.
  const bars = stats.map((s) => ({
    value: s.mean,
    itemStyle: {
      color: "transparent",
      borderColor: colors.get(s.cohort),
      borderWidth: 2,
      opacity: opacityFor(s.range_index),
    },
  }));

  // Per-bar stat text shrinks as bars get more crowded: 19px at ≤3 bars down to
  // 14px at ≥9 bars, linear in between (hardcoded range, clamped at both ends).
  const n = stats.length;
  const t = Math.max(0, Math.min(1, (n - 3) / (9 - 3)));
  const statFontSize = Math.round(19 - t * (19 - 14));
  const statLineHeight = Math.round(statFontSize * 1.25);
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
          const stroke = { stroke: "rgba(128, 128, 128, 0.6)", lineWidth: 1.5 };
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

          // Two-column stat block centered on the bar: a fixed-width name
          // column (names left-aligned, sized to snugly fit "CI lo") followed
          // by the aligned '=' column, positioned so each '=' ends just LEFT
          // of the error bar; values extend rightward from the whisker.
          // Anchored just above the bar's top (mean), growing upward.
          const nameColWidth = Math.round(statFontSize * 2.9);
          const eqGap = Math.round(statFontSize * 0.75); // '=' glyph + clearance before the whisker
          // The leading space nudges the names right by the same gap that
          // separates '=' from the values.
          const text = statLines(s)
            .map(([name, value]) => `{k| ${name}}{v|= ${value}}`)
            .join("\n");
          const at = api.coord([idx, s.mean ?? 0]);
          children.push({
            type: "text",
            style: {
              text,
              x: at[0] - nameColWidth - eqGap,
              y: at[1] - 8,
              textAlign: "left",
              textVerticalAlign: "bottom",
              rich: {
                k: { width: nameColWidth, align: "left", fontSize: statFontSize, lineHeight: statLineHeight, fill: "#333" },
                v: { fontSize: statFontSize, lineHeight: statLineHeight, fill: "#333" },
              },
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
