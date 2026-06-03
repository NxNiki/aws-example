// Hand-written in Phase 0 to mirror dashboard_api's Pydantic models
// (src/dashboard_api/schemas/data.py). Phase 1+ replaces this file with a
// client generated from the FastAPI OpenAPI schema (openapi-typescript), so
// the contract is single-sourced and compiler-checked end to end.

export type Granularity = "day" | "week" | "month";

export interface ConfigSummary {
  id: string;
  title: string;
}

export interface ConfigList {
  configs: ConfigSummary[];
}

export interface MetricGroup {
  id: string;
  label: string;
  metrics: string[];
}

export interface ConfigDetail {
  id: string;
  title: string;
  date_col: string;
  group_col: string;
  user_group_cols: string[];
  granularities: Granularity[];
  groups: MetricGroup[];
  tabs: string[];
}

export interface SeriesRequest {
  config: string;
  granularity?: Granularity;
  metrics: string[];
  date_from?: string | null;
  date_to?: string | null;
}

export interface Series {
  name: string;
  y: (number | null)[];
}

export interface SeriesResponse {
  config: string;
  granularity: Granularity;
  date_col: string;
  x: (string | null)[];
  series: Series[];
  missing: string[];
}
