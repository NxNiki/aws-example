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
const client = createClient<paths>({ baseUrl: import.meta.env.VITE_API_BASE ?? "" });

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

  series: async (body: SeriesRequest): Promise<SeriesResponse> => {
    const { data, error } = await client.POST("/api/data/series", { body });
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

  groupDistribution: async (body: GroupDistributionRequest): Promise<GroupDistributionResponse> => {
    const { data, error } = await client.POST("/api/data/group-distribution", { body });
    if (error || !data) fail("POST /api/data/group-distribution", error);
    return data;
  },

  deepdive: async (body: DeepdiveRequest): Promise<DeepdiveResponse> => {
    const { data, error } = await client.POST("/api/data/deepdive", { body });
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
};
