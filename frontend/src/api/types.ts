// Friendly aliases over the generated OpenAPI schema (./schema.d.ts), so the
// rest of the app imports stable names while the underlying types stay
// single-sourced from dashboard_api's Pydantic models.
//
// Regenerate schema.d.ts with `npm run gen:types` (or `make openapi` from the
// repo root, which also refreshes openapi.json from the backend).
import type { components } from "./schema";

type Schemas = components["schemas"];

export type ConfigSummary = Schemas["ConfigSummary"];
export type ConfigList = Schemas["ConfigList"];
export type MetricGroup = Schemas["MetricGroup"];
export type ConfigDetail = Schemas["ConfigDetail"];
export type SeriesRequest = Schemas["SeriesRequest"];
export type Series = Schemas["Series"];
export type SeriesKind = Series["kind"];
export type SeriesResponse = Schemas["SeriesResponse"];
export type Granularity = SeriesResponse["granularity"];
export type GroupValues = Schemas["GroupValues"];
export type DateBounds = Schemas["DateBounds"];

export type DateRange = Schemas["DateRange"];
export type ClipOpts = Schemas["ClipOpts"];
export type GroupDistributionRequest = Schemas["GroupDistributionRequest"];
export type GroupStat = Schemas["GroupStat"];
export type GroupDistributionResponse = Schemas["GroupDistributionResponse"];
export type DeepdiveRequest = Schemas["DeepdiveRequest"];
export type DeepdivePanel = DeepdiveRequest["panel"];
export type DeepdiveMode = DeepdiveRequest["mode"];
export type HistogramSeries = Schemas["HistogramSeries"];
export type CorrMatrix = Schemas["CorrMatrix"];
export type ScatterSeries = Schemas["ScatterSeries"];
export type DeepdiveResponse = Schemas["DeepdiveResponse"];
export type DeepdiveMetrics = Schemas["DeepdiveMetrics"];

// Report Spec (Phase 4). Hand-written mirrors of schemas/report.py rather than
// the generated aliases: pydantic marks default-bearing fields optional in the
// wire schema, but the server ALWAYS fills them — requiring them here avoids
// null-guards on every access. The client casts at the API boundary.
export type ReportLanguage = "en" | "zh-Hans" | "zh-Hant";

export interface DateFigureSource {
  kind: "stats-by-date";
  config: string;
  granularity: Granularity;
  date_from: string | null;
  date_to: string | null;
  cohort_selection: Record<string, string[]>;
  panel_id: string;
  left: string[];
  right: string[];
  log: boolean;
  threshold: number;
}

export interface GroupFigureSource {
  kind: "stats-by-group";
  config: string;
  granularity: Granularity;
  ranges: DateRange[];
  cohort_selection: Record<string, string[]>;
  panel_id: string;
  metric: string;
  mode: "box" | "bar";
  clip: ClipOpts;
}

export interface DeepdiveFigureSource {
  kind: "stats-deepdive";
  config: string;
  granularity: Granularity;
  ranges: DateRange[];
  cohort_selection: Record<string, string[]>;
  panel: "derived" | "user";
  mode: "histogram" | "heatmap" | "scatter";
  metrics: string[];
  nbins: number;
  normalize: boolean;
  outliers_std: number | null;
  scatter_log_x: boolean;
  scatter_log_y: boolean;
  log_y: boolean;
  clip: ClipOpts;
}

export interface SummaryTableFigureSource {
  kind: "summary-table";
  config: string;
  granularity: Granularity;
  ranges: DateRange[];
  cohort_selection: Record<string, string[]>;
  metrics: Record<string, string[]>; // per metric-group (group id → selected metrics)
  stats: SummaryStat[];
  reference_key: string | null;
  metric_options: Record<string, SummaryMetricOption>;
  pvalues: boolean;
}

export type FigureSource = DateFigureSource | GroupFigureSource | DeepdiveFigureSource | SummaryTableFigureSource;

export interface ReportFigure {
  id: string;
  title: string;
  description: string;
  inherit_period: boolean;
  source: FigureSource;
}

export interface ReportReference {
  id: string;
  url: string;
  title?: string | null;
  text?: string | null;
  status?: string | null;
  error?: string | null;
}

export interface ReportSpec {
  version: number;
  id: string;
  title: string;
  language: ReportLanguage;
  period: DateRange | null;
  // Linked saved dashboard view (one view ↔ many reports). Loading the report
  // also restores it; figures still embed their own recipes.
  view: string | null;
  figures: ReportFigure[];
  references: ReportReference[];
  summary: string;
  confluence_url: string;
}

// Summary-table tab. Hand-written (not yet in the generated schema) so the new
// endpoint works without regenerating schema.d.ts — run `make openapi` later to
// fold these into the typed client. Mirror schemas/data.py:Summary*.
// Per-metric display reshaping applied before stats: clip to [min,max] then a
// signed log1p. Independent per metric.
export interface SummaryMetricOption {
  log: boolean;
  clip: ClipOpts;
}

export interface SummaryTableRequest {
  config: string;
  granularity: Granularity;
  ranges: DateRange[];
  group_values: Record<string, string[]>;
  metric_options: Record<string, SummaryMetricOption>; // keyed by metric name
  pvalues: boolean;
}

export interface SummaryColumn {
  key: string; // stable id for the reference-column selection
  cohort: string;
  range_index: number;
  range_label: string;
}

export interface SummaryCell {
  n: number;
  mean: number | null;
  median: number | null;
  q1: number | null;
  q3: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
}

export interface SummaryRow {
  metric: string;
  group_id: string;
  group_label: string;
  cells: (SummaryCell | null)[]; // aligned to SummaryTableResponse.columns; null = no data
  pvalue: number | null;
  test: string | null; // "t-test" | "anova" | null
  missing: boolean;
}

export interface SummaryTableResponse {
  config: string;
  granularity: Granularity;
  columns: SummaryColumn[];
  rows: SummaryRow[];
}

// The cell stats the user can toggle on; each shows its own ±% vs the reference.
export type SummaryStat = "n" | "mean" | "median" | "q1" | "q3" | "min" | "max";

export type GenerateProxyResponse = Schemas["GenerateProxyResponse"];
export type ExportRequest = Schemas["ExportRequest"];
export type ExportResponse = Schemas["ExportResponse"];

// Hand-written export input: a figure ships EITHER a chart PNG or pre-rendered
// HTML (Summary-table grid). Kept separate from the generated ExportRequest so
// the optional png/html don't need a schema regen.
export interface ExportFigureInput {
  title: string;
  description: string;
  png_base64?: string;
  html?: string;
}

export interface ExportRequestInput {
  confluence_url: string;
  summary: string;
  references: ReportReference[];
  figures: ExportFigureInput[];
}
