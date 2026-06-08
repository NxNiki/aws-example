import type { EChartsOption, LineSeriesOption } from "echarts";
import type { Granularity, Series } from "../api/types";
import { colorForIndex, dashForIndex } from "./styles";
import { hybridInverse, hybridValue } from "./scale";

export interface BuildOpts {
  leftMetrics: string[];
  rightMetrics: string[];
  log: boolean;
  threshold: number;
  granularity: Granularity;
}

type Point = [string, number | null];

function transform(values: (number | null)[], log: boolean, thresh: number): (number | null)[] {
  if (!log) return values;
  return values.map((v) => (v === null || !Number.isFinite(v) ? null : hybridValue(v, thresh)));
}

function points(x: string[], ys: (number | null)[]): Point[] {
  return x.map((d, i) => [d, ys[i] ?? null]);
}

// Alternating Mon→Sun week bands for daily granularity (port of _add_week_stripes):
// the first visible week is left unshaded so the alternation is obvious.
type Band = [{ xAxis: string }, { xAxis: string }];

function weekStripes(dates: string[]): Band[] {
  if (dates.length === 0) return [];
  const sorted = [...dates].sort();
  const max = new Date(sorted[sorted.length - 1]);
  const monday = new Date(sorted[0]);
  monday.setUTCDate(monday.getUTCDate() - ((monday.getUTCDay() + 6) % 7)); // back to Monday
  const bands: Band[] = [];
  let idx = 0;
  while (monday <= max) {
    const next = new Date(monday);
    next.setUTCDate(next.getUTCDate() + 7);
    if (idx % 2 === 1) {
      bands.push([{ xAxis: monday.toISOString().slice(0, 10) }, { xAxis: next.toISOString().slice(0, 10) }]);
    }
    monday.setUTCDate(monday.getUTCDate() + 7);
    idx += 1;
  }
  return bands;
}

function axisLabelFormatter(log: boolean, thresh: number) {
  if (!log) return undefined;
  return (v: number) => {
    const orig = hybridInverse(v, thresh);
    return Math.abs(orig) >= 1000 ? orig.toExponential(1) : String(Math.round(orig * 100) / 100);
  };
}

/**
 * Build the ECharts option for one Stats-by-Date panel from the backend's data
 * series. The backend returns raw values; everything visual — dual-axis
 * assignment, cohort color / metric dash, bootstrap-CI bands, hybrid-log
 * compression, and weekend stripes — is applied here.
 */
export function buildStatsByDateOption(series: Series[], opts: BuildOpts): EChartsOption {
  const { leftMetrics, rightMetrics, log, threshold, granularity } = opts;

  const metricOrder = [...leftMetrics, ...rightMetrics];
  const cohorts: string[] = [];
  for (const s of series) if (!cohorts.includes(s.cohort)) cohorts.push(s.cohort);

  const axisIndexFor = (metric: string) => (rightMetrics.includes(metric) && !leftMetrics.includes(metric) ? 1 : 0);

  const echSeries: LineSeriesOption[] = [];
  const legendNames: string[] = [];
  let bandId = 0;

  series.forEach((s) => {
    const color = colorForIndex(Math.max(0, cohorts.indexOf(s.cohort)));
    const dash = dashForIndex(Math.max(0, metricOrder.indexOf(s.metric)));
    const yAxisIndex = axisIndexFor(s.metric);
    const name = `${s.cohort}: ${s.metric}`;
    legendNames.push(name);

    // CI band (user_* metrics only): invisible lower line + stacked diff area.
    if (s.kind === "user") {
      const lo = transform(s.lower, log, threshold);
      const hi = transform(s.upper, log, threshold);
      const diff = hi.map((h, i) => (h === null || lo[i] === null ? null : h - (lo[i] as number)));
      const stack = `band-${bandId++}`;
      echSeries.push({
        type: "line",
        name: `${name} (ci-lo)`,
        xAxisIndex: 0,
        yAxisIndex,
        stack,
        data: points(s.x, lo),
        lineStyle: { opacity: 0 },
        showSymbol: false,
        silent: true,
        z: 1,
      });
      echSeries.push({
        type: "line",
        name: `${name} (ci)`,
        xAxisIndex: 0,
        yAxisIndex,
        stack,
        data: points(s.x, diff),
        lineStyle: { opacity: 0 },
        areaStyle: { color, opacity: 0.15 },
        showSymbol: false,
        silent: true,
        z: 1,
      });
    }

    echSeries.push({
      type: "line",
      name,
      xAxisIndex: 0,
      yAxisIndex,
      data: points(s.x, transform(s.y, log, threshold)),
      showSymbol: true,
      symbolSize: 4,
      lineStyle: { color, type: dash, width: 2 },
      itemStyle: { color },
      z: 2,
    });
  });

  // Weekend stripes (daily only) — attach to the first real line so they render behind.
  if (granularity === "day") {
    const bands = weekStripes(series.flatMap((s) => s.x));
    const firstLine = echSeries.find((s) => s.z === 2);
    if (firstLine && bands.length) {
      firstLine.markArea = {
        silent: true,
        itemStyle: { color: "rgba(128,128,128,0.18)" },
        data: bands,
      };
    }
  }

  const mkYAxis = (metrics: string[], position: "left" | "right") => ({
    type: "value" as const,
    name: metrics.join(", "),
    position,
    scale: true,
    nameTextStyle: { fontSize: 12 },
    axisLabel: { formatter: axisLabelFormatter(log, threshold), fontSize: 12 },
  });

  return {
    tooltip: { trigger: "axis" },
    legend: { type: "scroll", bottom: 0, data: legendNames },
    grid: { left: 64, right: 64, top: 24, bottom: 56 },
    xAxis: { type: "time", axisLabel: { fontSize: 12 } },
    yAxis: [mkYAxis(leftMetrics, "left"), mkYAxis(rightMetrics, "right")],
    series: echSeries,
  };
}
