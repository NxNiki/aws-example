import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the typed API client so the store's logic is tested without network.
vi.mock("../api/client", () => ({
  api: {
    listConfigs: vi.fn(),
    getConfig: vi.fn(),
    series: vi.fn(),
  },
}));

import { api } from "../api/client";
import { useDashboardStore } from "./dashboardStore";

const mockApi = api as unknown as {
  listConfigs: ReturnType<typeof vi.fn>;
  getConfig: ReturnType<typeof vi.fn>;
  series: ReturnType<typeof vi.fn>;
};

const initialState = {
  configs: [],
  configId: null,
  config: null,
  series: null,
  loading: false,
  error: null,
};

describe("dashboardStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDashboardStore.setState(initialState);
  });

  it("loadConfigs populates configs and auto-selects the first", async () => {
    mockApi.listConfigs.mockResolvedValue({ configs: [{ id: "ss01", title: "SS01" }] });
    mockApi.getConfig.mockResolvedValue({ id: "ss01", title: "SS01", granularities: ["day"], groups: [] });

    await useDashboardStore.getState().loadConfigs();

    const s = useDashboardStore.getState();
    expect(s.configs).toHaveLength(1);
    expect(s.configId).toBe("ss01");
    expect(s.config?.id).toBe("ss01");
    expect(mockApi.getConfig).toHaveBeenCalledWith("ss01");
    expect(s.loading).toBe(false);
    expect(s.error).toBeNull();
  });

  it("records an error and clears loading when the API fails", async () => {
    mockApi.listConfigs.mockRejectedValue(new Error("boom"));

    await useDashboardStore.getState().loadConfigs();

    const s = useDashboardStore.getState();
    expect(s.error).toContain("boom");
    expect(s.loading).toBe(false);
  });

  it("loadSeries is a no-op until a config is selected", async () => {
    await useDashboardStore.getState().loadSeries(["active_users"]);
    expect(mockApi.series).not.toHaveBeenCalled();
  });
});
