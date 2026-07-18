import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import type { HistogramSeries } from "../api/types";
import { infoGraphic } from "./clipFilterInfo";
import { colorForIndex, wrapCohort } from "./styles";

// Deep Dive histogram: real bars (one per server-computed bin) that tile by bin
// width, overlaid across (cohort × range) with transparency. Each bar is drawn
// as a custom rectangle from bin_edges[i]→bin_edges[i+1] grounded at the axis
// bottom, so it works on a value x-axis and a linear or log count axis.

export function buildHistogramOption(
  series: HistogramSeries[],
  opts: { logY: boolean; normalize: boolean; info?: string },
): EChartsOption {
  const cohorts: string[] = [];
  for (const s of series) if (!cohorts.includes(s.cohort)) cohorts.push(s.cohort);

  return {
    graphic: infoGraphic(opts.info ?? ""),
    tooltip: { trigger: "item" },
    legend: { type: "scroll", bottom: 0, textStyle: { fontSize: 13 }, formatter: wrapCohort },
    grid: { left: 76, right: 24, top: 24, bottom: 48 },
    xAxis: {
      type: "value",
      scale: true,
      name: series[0]?.metric ?? "",
      nameTextStyle: { fontSize: 14 },
      axisLabel: { fontSize: 13 },
    },
    yAxis: {
      type: opts.logY ? "log" : "value",
      // Rotated axis label at the middle of the axis (not perched on top of the
      // figure, where it crowded the per-metric title above the chart).
      name: opts.normalize ? "density" : "count",
      nameLocation: "middle",
      nameGap: 52,
      nameTextStyle: { fontSize: 14 },
      axisLabel: { fontSize: 13 },
    },
    series: series.map((s) => {
      const color = colorForIndex(Math.max(0, cohorts.indexOf(s.cohort)));
      const isPrimary = s.range_index === 0;
      // On a log y-axis, empty (0-count) bins would poison ECharts' extent
      // computation (log(0) is invalid → it falls back to a 1–10 range, glaring
      // once normalized to <1). Drop them so the axis ranges over the real
      // positive values; on a linear axis they're invisible anyway.
      const data = s.counts
        .map((c, i) => [s.bin_edges[i], s.bin_edges[i + 1], c])
        .filter((d) => !opts.logY || d[2] > 0);
      return {
        type: "custom",
        name: `${s.cohort} · ${s.range_label}`,
        dimensions: ["x0", "x1", "count"],
        encode: { x: [0, 1], y: 2 },
        data,
        itemStyle: { color, opacity: isPrimary ? 0.55 : 0.3, borderColor: color, borderWidth: 0.5 },
        z: isPrimary ? 2 : 1,
        renderItem: (params, api) => {
          const x0 = api.value(0) as number;
          const x1 = api.value(1) as number;
          const count = api.value(2) as number;
          const left = api.coord([x0, 0])[0];
          const top = api.coord([x1, count]);
          const sys = params.coordSys as unknown as { x: number; y: number; width: number; height: number };
          const baseY = sys.y + sys.height;
          const rect = echarts.graphic.clipRectByRect(
            { x: Math.min(left, top[0]), y: top[1], width: Math.abs(top[0] - left), height: baseY - top[1] },
            sys,
          );
          return rect ? { type: "rect", shape: rect, style: api.style() } : undefined;
        },
      };
    }),
  };
}
