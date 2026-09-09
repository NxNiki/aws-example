import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import type { HistogramSeries } from "../api/types";
import { infoGraphic } from "./clipFilterInfo";
import { colorForIndex, wrapCohort } from "./styles";

// Deep Dive histogram: real bars (one per server-computed bin) that tile by bin
// width, overlaid across (cohort × range) with transparency. Each bar is drawn
// as a custom rectangle from bin_edges[i]→bin_edges[i+1] grounded at the axis
// bottom, so it works on a value x-axis with either count scale.
//
// "Log y" plots log1p(count) on a LINEAR axis rather than using a true log
// axis: a log axis cannot include 0 and draws 1-sample bins with zero height
// (log(1) = 0), making lone outlier bins invisible. log1p keeps the axis
// anchored at 0 and gives count 1 a visible bar; tick labels are
// back-transformed (expm1) so they still read as real counts.

const fmtNum = (n: number): string => Number(n.toPrecision(3)).toLocaleString();

export function buildHistogramOption(
  series: HistogramSeries[],
  opts: { logY: boolean; normalize: boolean; info?: string },
): EChartsOption {
  const cohorts: string[] = [];
  for (const s of series) if (!cohorts.includes(s.cohort)) cohorts.push(s.cohort);

  return {
    graphic: infoGraphic(opts.info ?? ""),
    tooltip: {
      trigger: "item",
      // The plotted y is log1p-transformed under logY, so read the raw count
      // from the extra data dimension instead of the encoded value.
      formatter: (p) => {
        const { seriesName, data } = p as unknown as { seriesName: string; data: [number, number, number, number] };
        const [x0, x1, , count] = data;
        return `${seriesName}<br/>${fmtNum(x0)} – ${fmtNum(x1)}: ${fmtNum(count)}`;
      },
    },
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
      type: "value",
      // Rotated axis label at the middle of the axis (not perched on top of the
      // figure, where it crowded the per-metric title above the chart).
      name: opts.normalize ? "density" : "count",
      nameLocation: "middle",
      nameGap: 52,
      nameTextStyle: { fontSize: 14 },
      axisLabel: opts.logY
        ? { fontSize: 13, formatter: (v: number) => fmtNum(Math.expm1(v)) }
        : { fontSize: 13 },
    },
    series: series.map((s) => {
      const color = colorForIndex(Math.max(0, cohorts.indexOf(s.cohort)));
      const isPrimary = s.range_index === 0;
      const data = s.counts.map((c, i) => [
        s.bin_edges[i],
        s.bin_edges[i + 1],
        opts.logY ? Math.log1p(c) : c,
        c,
      ]);
      return {
        type: "custom",
        name: `${s.cohort} · ${s.range_label}`,
        dimensions: ["x0", "x1", "y", "count"],
        encode: { x: [0, 1], y: 2 },
        data,
        itemStyle: { color, opacity: isPrimary ? 0.55 : 0.3, borderColor: color, borderWidth: 0.5 },
        z: isPrimary ? 2 : 1,
        renderItem: (params, api) => {
          const x0 = api.value(0) as number;
          const x1 = api.value(1) as number;
          const yVal = api.value(2) as number;
          const left = api.coord([x0, 0])[0];
          const top = api.coord([x1, yVal]);
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
