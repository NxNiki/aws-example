import type { ConfigDetail, ConfigList, SeriesRequest, SeriesResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => getJson<{ status: string; service: string }>("/api/health"),
  listConfigs: () => getJson<ConfigList>("/api/data/configs"),
  getConfig: (id: string) => getJson<ConfigDetail>(`/api/data/config/${encodeURIComponent(id)}`),
  series: (req: SeriesRequest) => postJson<SeriesResponse>("/api/data/series", req),
};
