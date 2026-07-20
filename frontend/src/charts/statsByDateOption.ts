import type { EChartsOption, LineSeriesOption } from "echarts";
import type { Granularity, Series } from "../api/types";
import { colorForIndex, dashForIndex, wrapCohort } from "./styles";
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

// Category-axis week bands for the concatenated multi-range mode: per segment,
// group its dates by their week's Monday and shade alternate weeks (first week
// of each segment unshaded). Categories are `${range}|${date}` ids.
function segmentWeekStripes(segDates: string[], cat: (d: string) => string): Band[] {
  const bands: Band[] = [];
  let currentMonday = "";
  let weekIdx = -1;
  let first = "";
  let last = "";
  const flush = () => {
    if (weekIdx % 2 === 1 && first) bands.push([{ xAxis: cat(first) }, { xAxis: cat(last) }]);
  };
  for (const d of [...segDates].sort()) {
    const dt = new Date(`${d}T00:00:00Z`);
    dt.setUTCDate(dt.getUTCDate() - ((dt.getUTCDay() + 6) % 7));
    const monday = dt.toISOString().slice(0, 10);
    if (monday !== currentMonday) {
      flush();
      currentMonday = monday;
      weekIdx += 1;
      first = d;
    }
    last = d;
  }
  flush();
  return bands;
}

function axisLabelFormatter(log: boolean, thresh: number) {
  if (!log) return undefined;
  return (v: number) => {
    const orig = hybridInverse(v, thresh);
    return Math.abs(orig) >= 1000 ? orig.toExponential(1) : String(Math.round(orig * 100) / 100);
  };
}

const fmtVal = (v: number | null | undefined): string => {
  if (v == null || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  return v.toLocaleString("en-US", { maximumFractionDigits: a >= 100 ? 0 : a >= 1 ? 2 : 4 });
};

type Tip = { y: number | null; lo: number | null; hi: number | null; ci: boolean };

/**
 * Build the ECharts option for one Stats-by-Date panel from the backend's data
 * series. The backend returns raw values; everything visual — dual-axis
 * assignment, cohort color / metric dash, bootstrap-CI bands, hybrid-log
 * compression, and weekend stripes — is applied here.
 *
 * Dashboard feature: the global "Date groups" picker. When the series carry
 * range_index (several date windows selected), the windows are concatenated
 * horizontally on a category axis with a small gap between them; one legend
 * entry per cohort × metric spans all windows, its line broken at the gaps.
 * Series without range_index (older report recipes) keep the plain time axis.
 */
export function buildStatsByDateOption(series: Series[], opts: BuildOpts): EChartsOption {
  const { leftMetrics, rightMetrics, log, threshold, granularity } = opts;

  const metricOrder = [...leftMetrics, ...rightMetrics];
  const cohorts: string[] = [];
  for (const s of series) if (!cohorts.includes(s.cohort)) cohorts.push(s.cohort);

  const multiRange = series.some((s) => s.range_index != null);

  const echSeries: LineSeriesOption[] = [];
  const legendNames: string[] = [];
  let bandId = 0;

  // Raw (untransformed) values per series per x-key, for the tooltip — hover
  // should show real mean/CI numbers even when the hybrid-log display is on.
  const tipValues = new Map<string, Map<string, Tip>>();

  const axisAssignment = (metric: string): 0 | 1 | null => {
    // Assign left-first; drop series whose metric is no longer selected on
    // either axis. Without this, a metric just removed from one axis would,
    // until the (debounced) refetch drops it, fall through to the left axis
    // instead of disappearing — the "unselect right → jumps to left" glitch.
    if (leftMetrics.includes(metric)) return 0;
    if (rightMetrics.includes(metric)) return 1;
    return null;
  };

  const styleFor = (cohort: string, metric: string) => ({
    color: colorForIndex(Math.max(0, cohorts.indexOf(cohort))),
    dash: dashForIndex(Math.max(0, metricOrder.indexOf(metric))),
  });

  const pushBand = (
    name: string,
    yAxisIndex: 0 | 1,
    color: string,
    xs: string[] | undefined,
    lo: (number | null)[],
    hi: (number | null)[],
  ) => {
    const diff = hi.map((h, i) => (h === null || lo[i] === null ? null : h - (lo[i] as number)));
    const stack = `band-${bandId++}`;
    const base = { type: "line" as const, xAxisIndex: 0, yAxisIndex, stack, showSymbol: false, silent: true, z: 1 };
    echSeries.push(
      {
        ...base,
        name: `${name} (ci-lo)`,
        data: xs ? points(xs, lo) : lo,
        lineStyle: { opacity: 0 },
        itemStyle: { color },
      },
      {
        ...base,
        name: `${name} (ci)`,
        data: xs ? points(xs, diff) : diff,
        lineStyle: { opacity: 0 },
        itemStyle: { color },
        areaStyle: { color, opacity: 0.15 },
      },
    );
  };

  let xAxis: EChartsOption["xAxis"];
  let stripeBands: Band[] = [];

  if (!multiRange) {
    // ── Single-window time axis (legacy path) ──────────────────────────────
    series.forEach((s) => {
      const yAxisIndex = axisAssignment(s.metric);
      if (yAxisIndex === null) return;
      const { color, dash } = styleFor(s.cohort, s.metric);
      const name = `${s.cohort}: ${s.metric}`;
      legendNames.push(name);

      const perDate = new Map<string, Tip>();
      s.x.forEach((d, i) =>
        perDate.set(d, { y: s.y[i] ?? null, lo: s.lower[i] ?? null, hi: s.upper[i] ?? null, ci: s.kind === "user" }),
      );
      tipValues.set(name, perDate);

      if (s.kind === "user") {
        pushBand(name, yAxisIndex, color, s.x, transform(s.lower, log, threshold), transform(s.upper, log, threshold));
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
    if (granularity === "day") stripeBands = weekStripes(series.flatMap((s) => s.x));
    xAxis = { type: "time", axisLabel: { fontSize: 14 } };
  } else {
    // ── Concatenated multi-range category axis ──────────────────────────────
    // Category ids are `${range}|${date}` (unique even when ranges overlap),
    // with a `gap-N` spacer between consecutive ranges for the visual break.
    const segIndexes = [...new Set(series.map((s) => s.range_index ?? 0))].sort((a, b) => a - b);
    const segDates = new Map<number, string[]>(
      segIndexes.map((ri) => [
        ri,
        [...new Set(series.filter((s) => (s.range_index ?? 0) === ri).flatMap((s) => s.x))].sort(),
      ]),
    );
    const categories: string[] = [];
    const catIndex = new Map<string, number>();
    segIndexes.forEach((ri, k) => {
      if (k > 0) categories.push(`gap-${ri}`);
      for (const d of segDates.get(ri) ?? []) {
        catIndex.set(`${ri}|${d}`, categories.length);
        categories.push(`${ri}|${d}`);
      }
    });

    // One merged entry per cohort × metric across all ranges.
    const groups = new Map<string, Series[]>();
    for (const s of series) {
      if (axisAssignment(s.metric) === null) continue;
      const key = `${s.cohort}: ${s.metric}`;
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(s);
    }

    for (const [name, parts] of groups) {
      const { cohort, metric } = { cohort: parts[0].cohort, metric: parts[0].metric };
      const yAxisIndex = axisAssignment(metric) as 0 | 1;
      const { color, dash } = styleFor(cohort, metric);
      legendNames.push(name);

      const y: (number | null)[] = categories.map(() => null);
      const lo: (number | null)[] = categories.map(() => null);
      const hi: (number | null)[] = categories.map(() => null);
      const perCat = new Map<string, Tip>();
      const hasCI = parts.some((p) => p.kind === "user");
      for (const p of parts) {
        const ri = p.range_index ?? 0;
        p.x.forEach((d, i) => {
          const idx = catIndex.get(`${ri}|${d}`);
          if (idx === undefined) return;
          y[idx] = p.y[i] ?? null;
          lo[idx] = p.lower[i] ?? null;
          hi[idx] = p.upper[i] ?? null;
          perCat.set(`${ri}|${d}`, {
            y: p.y[i] ?? null,
            lo: p.lower[i] ?? null,
            hi: p.upper[i] ?? null,
            ci: p.kind === "user",
          });
        });
      }
      tipValues.set(name, perCat);

      if (hasCI) {
        pushBand(name, yAxisIndex, color, undefined, transform(lo, log, threshold), transform(hi, log, threshold));
      }
      echSeries.push({
        type: "line",
        name,
        xAxisIndex: 0,
        yAxisIndex,
        data: transform(y, log, threshold),
        showSymbol: true,
        symbolSize: 4,
        connectNulls: false, // breaks the line at the gap between ranges
        lineStyle: { color, type: dash, width: 2 },
        itemStyle: { color },
        z: 2,
      });
    }

    if (granularity === "day") {
      for (const ri of segIndexes) {
        stripeBands.push(...segmentWeekStripes(segDates.get(ri) ?? [], (d) => `${ri}|${d}`));
      }
    }
    xAxis = {
      type: "category",
      data: categories,
      axisLabel: {
        fontSize: 14,
        formatter: (v: string) => (v.startsWith("gap-") ? "" : v.split("|")[1] ?? v),
      },
    };
  }

  // Weekend stripes (daily only) — attach to the first real line so they render behind.
  const firstLine = echSeries.find((s) => s.z === 2);
  if (firstLine && stripeBands.length) {
    firstLine.markArea = {
      silent: true,
      itemStyle: { color: "rgba(128,128,128,0.18)" },
      data: stripeBands,
    };
  }

  // Anchor the axis title toward the plot (left axis → extends right, right
  // axis → extends left) so long metric names aren't clipped at the canvas edge.
  const mkYAxis = (metrics: string[], position: "left" | "right") => ({
    type: "value" as const,
    name: metrics.join(", "),
    position,
    scale: true,
    nameTextStyle: { fontSize: 14, align: position },
    axisLabel: { formatter: axisLabelFormatter(log, threshold), fontSize: 14 },
  });

  // One tooltip row per real series: line-colored marker, mean, and the actual
  // CI bounds (not the band's stacked diff), always in RAW values even when the
  // hybrid-log display transform is on. The "(ci-lo)"/"(ci)" helper series that
  // draw the band are filtered out.
  const tooltipFormatter = (params: unknown): string => {
    const list = (Array.isArray(params) ? params : [params]) as {
      seriesName?: string;
      marker?: string;
      axisValue?: string;
      data?: [string, number | null] | number | null;
    }[];
    const main = list.filter((p) => p.seriesName && !p.seriesName.includes("(ci"));
    if (main.length === 0) return "";
    const xKey = multiRange ? (main[0].axisValue ?? "") : String((main[0].data as Point | undefined)?.[0] ?? "");
    if (multiRange && xKey.startsWith("gap-")) return "";
    const date = multiRange ? (xKey.split("|")[1] ?? xKey) : xKey;
    const rangeTag = multiRange && xKey.includes("|") ? ` (R${Number(xKey.split("|")[0]) + 1})` : "";
    const rows = main.flatMap((p) => {
      const v = tipValues.get(p.seriesName as string)?.get(xKey);
      if (!v || v.y == null) return [];
      const ci = v.ci ? ` <span style="color:#888">CI [${fmtVal(v.lo)} – ${fmtVal(v.hi)}]</span>` : "";
      return [`${p.marker ?? ""} ${p.seriesName}: <b>${fmtVal(v.y)}</b>${ci}`];
    });
    return [`<b>${date}${rangeTag}</b>`, ...rows].join("<br/>");
  };

  return {
    tooltip: { trigger: "axis", formatter: tooltipFormatter },
    legend: { type: "scroll", bottom: 0, data: legendNames, textStyle: { fontSize: 13 }, formatter: wrapCohort },
    grid: { left: 80, right: 80, top: 44, bottom: 56 },
    xAxis,
    yAxis: [mkYAxis(leftMetrics, "left"), mkYAxis(rightMetrics, "right")],
    series: echSeries,
  };
}
