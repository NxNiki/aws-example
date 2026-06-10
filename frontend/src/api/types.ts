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

export type FigureSource = DateFigureSource | GroupFigureSource | DeepdiveFigureSource;

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

export type GenerateProxyResponse = Schemas["GenerateProxyResponse"];
export type ExportRequest = Schemas["ExportRequest"];
export type ExportResponse = Schemas["ExportResponse"];
