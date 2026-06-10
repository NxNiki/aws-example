import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the typed API client so the store's logic is tested without network.
vi.mock("../api/client", () => ({
  api: {
    listConfigs: vi.fn(),
    getConfig: vi.fn(),
    groupValues: vi.fn(),
    dateBounds: vi.fn(),
    deepdiveMetrics: vi.fn(),
    series: vi.fn(),
  },
}));

import { api } from "../api/client";
import { useDashboardStore } from "./dashboardStore";

const mockApi = api as unknown as {
  listConfigs: ReturnType<typeof vi.fn>;
  getConfig: ReturnType<typeof vi.fn>;
  groupValues: ReturnType<typeof vi.fn>;
  dateBounds: ReturnType<typeof vi.fn>;
  deepdiveMetrics: ReturnType<typeof vi.fn>;
  series: ReturnType<typeof vi.fn>;
};

const CONFIG = {
  id: "ss01",
  title: "SS01",
  date_col: "activity_date",
  group_col: "",
  user_group_cols: ["user_group"],
  granularities: ["day"],
  groups: [
    { id: "group1", label: "DAU", metrics: ["num_active_users", "rtp"] },
    { id: "group2", label: "Bets", metrics: ["total_bet"] },
  ],
  tabs: ["stats-by-date"],
};

const initialState = {
  configs: [],
  configId: null,
  config: null,
  granularity: "day" as const,
  dateFrom: null,
  dateTo: null,
  groupValues: {},
  cohortSelection: {},
  panels: {},
  loading: false,
  error: null,
};

describe("dashboardStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDashboardStore.setState(initialState);
    mockApi.getConfig.mockResolvedValue(CONFIG);
    mockApi.groupValues.mockResolvedValue({ config: "ss01", granularity: "day", values: { user_group: ["new", "old"] } });
    mockApi.dateBounds.mockResolvedValue({ config: "ss01", granularity: "day", min: "2025-01-01", max: "2025-01-30" });
    mockApi.deepdiveMetrics.mockResolvedValue({ config: "ss01", granularity: "day", derived: ["rtp"], user: ["user_total_bet"] });
    mockApi.series.mockResolvedValue({ config: "ss01", granularity: "day", date_col: "activity_date", series: [], missing: [] });
  });

  it("loadConfigs auto-selects the first config, builds panels, and fetches cohort values + series", async () => {
    mockApi.listConfigs.mockResolvedValue({ configs: [{ id: "ss01", title: "SS01" }] });

    await useDashboardStore.getState().loadConfigs();

    const s = useDashboardStore.getState();
    expect(s.configId).toBe("ss01");
    // panels are seeded from config.groups, first metric on the left axis.
    expect(Object.keys(s.panels)).toEqual(["group1", "group2"]);
    expect(s.panels.group1.left).toEqual(["num_active_users"]);
    expect(s.groupValues).toEqual({ user_group: ["new", "old"] });
    // Default window is the last 30 days ending at the data's max date.
    expect(s.dateTo).toBe("2025-01-30");
    expect(s.dateFrom).toBe("2025-01-01");
    expect(mockApi.series).toHaveBeenCalledTimes(2); // one fetch per panel
    expect(s.error).toBeNull();
  });

  it("records an error and clears loading when listing configs fails", async () => {
    mockApi.listConfigs.mockRejectedValue(new Error("boom"));
    await useDashboardStore.getState().loadConfigs();
    const s = useDashboardStore.getState();
    expect(s.error).toContain("boom");
    expect(s.loading).toBe(false);
  });

  it("setPanelMetrics updates the targeted panel/axis only", () => {
    useDashboardStore.setState({
      panels: { group1: { left: ["a"], right: [], log: false, threshold: 10, series: [] } },
    });
    useDashboardStore.getState().setPanelMetrics("group1", "right", ["rtp"]);
    expect(useDashboardStore.getState().panels.group1.right).toEqual(["rtp"]);
    expect(useDashboardStore.getState().panels.group1.left).toEqual(["a"]);
  });

  it("loadAllSeries requests each panel's combined metric set with the cohort selection", async () => {
    useDashboardStore.setState({
      configId: "ss01",
      cohortSelection: { user_group: ["new"] },
      panels: { group1: { left: ["num_active_users"], right: ["rtp"], log: false, threshold: 10, series: [] } },
    });
    await useDashboardStore.getState().loadAllSeries();
    expect(mockApi.series).toHaveBeenCalledWith(
      {
        config: "ss01",
        granularity: "day",
        metrics: ["num_active_users", "rtp"],
        date_from: null,
        date_to: null,
        group_values: { user_group: ["new"] },
      },
      expect.anything(), // AbortSignal threaded for cancellation
    );
  });
});
