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
export type DeepdiveResponse = Schemas["DeepdiveResponse"];
export type DeepdiveMetrics = Schemas["DeepdiveMetrics"];
