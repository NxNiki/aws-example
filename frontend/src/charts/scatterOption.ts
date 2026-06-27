import type { EChartsOption } from "echarts";
import type { ScatterSeries } from "../api/types";
import { colorForIndex } from "./styles";

// Deep Dive scatter: plot metric[xi] vs metric[yi] from the server's
// outlier-filtered, down-sampled points, overlaying each (cohort × range)
// series. Reused for the single pair plot and for each off-diagonal cell of the
// scatter matrix. log x/y use ECharts' native log axes (non-positive values are
// dropped by the axis).
export function buildScatterOption(
  scatters: ScatterSeries[],
  xi: number,
  yi: number,
  opts: { logX: boolean; logY: boolean; xLabel?: string; yLabel?: string; showLegend?: boolean; compact?: boolean },
): EChartsOption {
  const cohorts: string[] = [];
  for (const s of scatters) if (!cohorts.includes(s.cohort)) cohorts.push(s.cohort);
  const compact = opts.compact ?? false;

  return {
    tooltip: { trigger: "item" },
    legend: opts.showLegend ? { type: "scroll", top: 8, right: 8, textStyle: { fontSize: 13 } } : undefined,
    // Square GRID (plot rectangle) via explicit equal width/height — not derived
    // from margins — so the data area is exactly square. Sizes assume the fixed
    // square canvas DeepDive passes (260 compact / 600 full); left/top leave room
    // for axis labels and the top-right legend.
    // containLabel:false (default, set explicitly) so axis labels NEVER shrink the
    // grid — width/height are the exact plot size, guaranteeing a square data area.
    grid: compact
      ? { left: 42, top: 14, width: 190, height: 190, containLabel: false }
      : { left: 70, top: 50, width: 450, height: 450, containLabel: false },
    xAxis: {
      type: opts.logX ? "log" : "value",
      scale: true,
      name: opts.xLabel ?? "",
      nameLocation: "middle",
      nameGap: 26,
      nameTextStyle: { fontSize: compact ? 11 : 14 },
      axisLabel: { fontSize: compact ? 10 : 13 },
    },
    yAxis: {
      type: opts.logY ? "log" : "value",
      scale: true,
      name: opts.yLabel ?? "",
      nameLocation: "middle",
      nameGap: compact ? 32 : 48,
      nameTextStyle: { fontSize: compact ? 11 : 14 },
      axisLabel: { fontSize: compact ? 10 : 13 },
    },
    series: scatters.map((s) => ({
      type: "scatter",
      name: `${s.cohort} · ${s.range_label}`,
      data: s.points.map((p) => [p[xi], p[yi]]),
      symbolSize: compact ? 3 : 5,
      large: true,
      largeThreshold: 2000,
      itemStyle: { color: colorForIndex(Math.max(0, cohorts.indexOf(s.cohort))), opacity: 0.6 },
    })),
  };
}
