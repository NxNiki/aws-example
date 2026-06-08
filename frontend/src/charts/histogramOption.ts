import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import type { HistogramSeries } from "../api/types";
import { colorForIndex } from "./styles";

// Deep Dive histogram: real bars (one per server-computed bin) that tile by bin
// width, overlaid across (cohort × range) with transparency. Each bar is drawn
// as a custom rectangle from bin_edges[i]→bin_edges[i+1] grounded at the axis
// bottom, so it works on a value x-axis and a linear or log count axis.

export function buildHistogramOption(series: HistogramSeries[], opts: { logY: boolean; normalize: boolean }): EChartsOption {
  const cohorts: string[] = [];
  for (const s of series) if (!cohorts.includes(s.cohort)) cohorts.push(s.cohort);

  return {
    tooltip: { trigger: "item" },
    legend: { type: "scroll", bottom: 0 },
    grid: { left: 56, right: 24, top: 24, bottom: 48 },
    xAxis: {
      type: "value",
      scale: true,
      name: series[0]?.metric ?? "",
      nameTextStyle: { fontSize: 12 },
      axisLabel: { fontSize: 11 },
    },
    yAxis: {
      type: opts.logY ? "log" : "value",
      name: opts.normalize ? "density" : "count",
      nameTextStyle: { fontSize: 12 },
      axisLabel: { fontSize: 11 },
    },
    series: series.map((s) => {
      const color = colorForIndex(Math.max(0, cohorts.indexOf(s.cohort)));
      const isPrimary = s.range_index === 0;
      const data = s.counts.map((c, i) => [s.bin_edges[i], s.bin_edges[i + 1], c]);
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
