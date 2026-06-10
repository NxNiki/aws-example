import createClient from "openapi-fetch";
import type { paths } from "./schema";
import type {
  ConfigDetail,
  ConfigList,
  DateBounds,
  DeepdiveMetrics,
  DeepdiveRequest,
  DeepdiveResponse,
  GroupDistributionRequest,
  GroupDistributionResponse,
  GroupValues,
  Granularity,
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
};
