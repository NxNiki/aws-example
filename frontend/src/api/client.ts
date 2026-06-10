import createClient from "openapi-fetch";
import type { components, paths } from "./schema";
import type {
  ConfigDetail,
  ConfigList,
  DateBounds,
  DeepdiveMetrics,
  DeepdiveRequest,
  DeepdiveResponse,
  ExportRequest,
  ExportResponse,
  GenerateProxyResponse,
  GroupDistributionRequest,
  GroupDistributionResponse,
  GroupValues,
  Granularity,
  ReportSpec,
  SeriesRequest,
  SeriesResponse,
} from "./types";

// Typed client generated from the OpenAPI schema: paths, params, request
// bodies, and responses are all checked against dashboard_api's contract.
const BASE = import.meta.env.VITE_API_BASE ?? "";
const client = createClient<paths>({ baseUrl: BASE });

function fail(path: string, error: unknown): never {
  throw new Error(`${path} failed: ${JSON.stringify(error)}`);
}

export const api = {
  health: async (): Promise<Record<string, string>> => {
    const { data, error } = await client.GET("/api/health");
    if (error || !data) fail("GET /api/health", error);
    return data as Record<string, string>;
  },

  listConfigs: async (): Promise<ConfigList> => {
    const { data, error } = await client.GET("/api/data/configs");
    if (error || !data) fail("GET /api/data/configs", error);
    return data;
  },

  getConfig: async (id: string): Promise<ConfigDetail> => {
    const { data, error } = await client.GET("/api/data/config/{config_id}", {
      params: { path: { config_id: id } },
    });
    if (error || !data) fail(`GET /api/data/config/${id}`, error);
    return data;
  },

  series: async (body: SeriesRequest, signal?: AbortSignal): Promise<SeriesResponse> => {
    const { data, error } = await client.POST("/api/data/series", { body, signal });
    if (error || !data) fail("POST /api/data/series", error);
    return data;
  },

  groupValues: async (config: string, granularity: Granularity): Promise<GroupValues> => {
    const { data, error } = await client.GET("/api/data/group-values", {
      params: { query: { config, granularity } },
    });
    if (error || !data) fail("GET /api/data/group-values", error);
    return data;
  },

  dateBounds: async (config: string, granularity: Granularity): Promise<DateBounds> => {
    const { data, error } = await client.GET("/api/data/date-bounds", {
      params: { query: { config, granularity } },
    });
    if (error || !data) fail("GET /api/data/date-bounds", error);
    return data;
  },

  groupDistribution: async (
    body: GroupDistributionRequest,
    signal?: AbortSignal,
  ): Promise<GroupDistributionResponse> => {
    const { data, error } = await client.POST("/api/data/group-distribution", { body, signal });
    if (error || !data) fail("POST /api/data/group-distribution", error);
    return data;
  },

  deepdive: async (body: DeepdiveRequest, signal?: AbortSignal): Promise<DeepdiveResponse> => {
    const { data, error } = await client.POST("/api/data/deepdive", { body, signal });
    if (error || !data) fail("POST /api/data/deepdive", error);
    return data;
  },

  deepdiveMetrics: async (config: string, granularity: Granularity): Promise<DeepdiveMetrics> => {
    const { data, error } = await client.GET("/api/data/deepdive-metrics", {
      params: { query: { config, granularity } },
    });
    if (error || !data) fail("GET /api/data/deepdive-metrics", error);
    return data;
  },

  // Saved view snapshots (legacy "save/load config"). The snapshot is opaque
  // frontend-owned JSON, so these use plain fetch rather than the typed client.
  listViews: async (): Promise<string[]> => {
    const r = await fetch(`${BASE}/api/views`);
    if (!r.ok) fail("GET /api/views", await r.text());
    return ((await r.json()) as { views: string[] }).views;
  },

  loadView: async (name: string): Promise<unknown> => {
    const r = await fetch(`${BASE}/api/views/${encodeURIComponent(name)}`);
    if (!r.ok) fail(`GET /api/views/${name}`, await r.text());
    return r.json();
  },

  saveView: async (name: string, snapshot: unknown): Promise<{ name: string; path: string; views: string[] }> => {
    const r = await fetch(`${BASE}/api/views/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(snapshot),
    });
    if (!r.ok) fail(`PUT /api/views/${name}`, await r.text());
    return (await r.json()) as { name: string; path: string; views: string[] };
  },

  // Report Spec (Phase 4)
  listReportSpecs: async (): Promise<string[]> => {
    const { data, error } = await client.GET("/api/report/specs");
    if (error || !data) fail("GET /api/report/specs", error);
    return data.specs;
  },

  loadReportSpec: async (name: string): Promise<ReportSpec> => {
    const { data, error } = await client.GET("/api/report/spec/{name}", { params: { path: { name } } });
    if (error || !data) fail(`GET /api/report/spec/${name}`, error);
    // The wire schema marks default-bearing fields optional; the server always
    // fills them (Pydantic defaults), so the app-facing required type is safe.
    return data as unknown as ReportSpec;
  },

  saveReportSpec: async (name: string, spec: ReportSpec): Promise<{ name: string; path: string; specs: string[] }> => {
    const { data, error } = await client.PUT("/api/report/spec/{name}", {
      params: { path: { name } },
      body: spec as unknown as components["schemas"]["ReportSpec-Input"],
    });
    if (error || !data) fail(`PUT /api/report/spec/${name}`, error);
    return data;
  },

  reportReferences: async (urls: string[]): Promise<Record<string, string>[]> => {
    const { data, error } = await client.POST("/api/report/references", { body: { urls } });
    if (error || !data) fail("POST /api/report/references", error);
    return data.references;
  },

  generateDescription: async (payload: Record<string, unknown>): Promise<GenerateProxyResponse> => {
    const { data, error } = await client.POST("/api/report/description", { body: payload });
    if (error || !data) fail("POST /api/report/description", error);
    return data;
  },

  generateSummary: async (payload: Record<string, unknown>): Promise<GenerateProxyResponse> => {
    const { data, error } = await client.POST("/api/report/summary", { body: payload });
    if (error || !data) fail("POST /api/report/summary", error);
    return data;
  },

  exportReport: async (req: ExportRequest): Promise<ExportResponse> => {
    const { data, error } = await client.POST("/api/report/export", { body: req });
    if (error || !data) fail("POST /api/report/export", error);
    return data;
  },
};
