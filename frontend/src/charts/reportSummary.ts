import type {
  CorrMatrix,
  GroupStat,
  HistogramSeries,
  ReportFigure,
  ScatterSeries,
  Series,
  SummaryColumn,
  SummaryMetricOption,
  SummaryRow,
} from "../api/types";
import type { FigureData } from "../store/reportStore";
import { clipFilterInfo } from "./clipFilterInfo";

// Build the compact JSON "data_summary" the LLM description/summary endpoints
// consume, from the RAW data a report figure was rendered from (the legacy app
// derived this from Plotly fig_dicts; the endpoints treat it as opaque JSON).
// Values are downsampled to ≤400 points per trace, like the legacy extractor.

const MAX_POINTS = 400;

function downsample<T>(items: T[]): T[] {
  if (items.length <= MAX_POINTS) return items;
  const step = items.length / MAX_POINTS;
  const out: T[] = [];
  for (let i = 0; i < MAX_POINTS; i++) out.push(items[Math.floor(i * step)]);
  return out;
}

const round = (v: number | null | undefined): number | null =>
  v == null || !Number.isFinite(v) ? null : Math.round(v * 10000) / 10000;

function dateTraces(series: Series[]) {
  return series.map((s) => {
    const idx = downsample(s.x.map((_, i) => i));
    return {
      name: `${s.cohort}: ${s.metric}`,
      x: idx.map((i) => s.x[i]),
      y: idx.map((i) => round(s.y[i])),
      ...(s.kind === "user"
        ? { error_y: { lower: idx.map((i) => round(s.lower[i])), upper: idx.map((i) => round(s.upper[i])) } }
        : {}),
    };
  });
}

function groupTraces(stats: GroupStat[]) {
  return stats.map((s) => ({
    name: `${s.cohort} (${s.range_label})`,
    n: s.n,
    mean: round(s.mean),
    median: round(s.median),
    q1: round(s.q1),
    q3: round(s.q3),
    min: round(s.min),
    max: round(s.max),
    ci_95: [round(s.ci_lower), round(s.ci_upper)],
  }));
}

function histogramTraces(hists: HistogramSeries[]) {
  return hists.map((h) => ({
    name: `${h.metric} — ${h.cohort} (${h.range_label})`,
    bin_edges: downsample(h.bin_edges).map(round),
    counts: downsample(h.counts).map(round),
  }));
}

function heatmapTraces(maps: CorrMatrix[]) {
  return maps.map((m) => ({
    name: `correlation — ${m.cohort} (${m.range_label})`,
    metrics: m.metrics,
    corr: m.corr.map((row) => row.map(round)),
  }));
}

function scatterTraces(scatters: ScatterSeries[]) {
  // Points are heavy and the LLM doesn't need them — summarize counts + ranges.
  return scatters.map((s) => ({
    name: `scatter — ${s.cohort} (${s.range_label})`,
    metrics: s.metrics,
    n_points: s.points.length,
  }));
}

// The summary table for the LLM: column headers (with which is the reference),
// then per metric its value in each column + the significance test. Reuses
// `mean` as the representative value (what the ±% in the UI is built from).
function summaryTableData(
  columns: SummaryColumn[],
  rows: SummaryRow[],
  selected: Record<string, string[]>,
  referenceKey: string | null,
  metricOptions: Record<string, SummaryMetricOption>,
) {
  const colLabels = columns.map((c) => `${c.cohort} · ${c.range_label}`);
  const refIndex = columns.findIndex((c) => c.key === referenceKey);
  const reshape = (m: string): string | undefined => {
    const o = metricOptions[m];
    if (!o) return undefined;
    const parts: string[] = [];
    if (o.clip.enable && (o.clip.min != null || o.clip.max != null)) parts.push(`clip ${o.clip.min ?? "-inf"}:${o.clip.max ?? "inf"}`);
    if (o.log) parts.push("log");
    return parts.length ? parts.join(", ") : undefined;
  };
  const metricRows = rows
    .filter((r) => (selected[r.group_id] ?? []).includes(r.metric))
    .map((r) => ({
      metric: r.metric,
      group: r.group_label,
      ...(reshape(r.metric) ? { reshape: reshape(r.metric) } : {}),
      mean_by_column: r.cells.map((c) => (c ? round(c.mean) : null)),
      ...(r.test ? { pvalue: round(r.pvalue), test: r.test } : {}),
    }));
  return {
    columns: colLabels,
    reference: refIndex >= 0 ? colLabels[refIndex] : null,
    metrics: metricRows,
  };
}

export function buildDataSummary(fig: ReportFigure, data: FigureData): Record<string, unknown> {
  const src = fig.source;
  const base = {
    title: fig.title || undefined,
    chart_kind: src.kind,
    config: src.config,
    granularity: src.granularity,
  };
  if (src.kind === "stats-by-date" && data.series) {
    return { ...base, period: { from: src.date_from, to: src.date_to }, traces: dateTraces(data.series) };
  }
  if (src.kind === "stats-by-group" && data.stats) {
    const reshape = clipFilterInfo(src.clip, src.filter);
    return { ...base, metric: src.metric, display: src.mode, ...(reshape ? { reshape } : {}), groups: groupTraces(data.stats) };
  }
  if (src.kind === "summary-table" && data.columns && data.rows) {
    return { ...base, ...summaryTableData(data.columns, data.rows, src.metrics, src.reference_key, src.metric_options) };
  }
  if (src.kind === "stats-deepdive") {
    const reshape = clipFilterInfo(src.clip, src.filter);
    const dd = { ...base, ...(reshape ? { reshape } : {}) };
    if (src.mode === "histogram" && data.histograms) return { ...dd, traces: histogramTraces(data.histograms) };
    if (src.mode === "heatmap" && data.heatmaps) return { ...dd, matrices: heatmapTraces(data.heatmaps) };
    if (src.mode === "scatter" && data.scatters) return { ...dd, traces: scatterTraces(data.scatters) };
  }
  return base;
}
