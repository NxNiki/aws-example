import createClient from "openapi-fetch";
import type { paths } from "./schema";
import type { ConfigDetail, ConfigList, SeriesRequest, SeriesResponse } from "./types";

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
};
