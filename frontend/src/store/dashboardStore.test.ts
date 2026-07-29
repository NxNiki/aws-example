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
import type { RangeState } from "../components/DateRanges";

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
  range_group_defaults: [],
  granularities: ["day"],
  groups: [
    { id: "group1", label: "DAU", metrics: ["num_active_users", "rtp"] },
    { id: "group2", label: "Bets", metrics: ["total_bet"] },
  ],
  tabs: ["stats-by-date"],
};

const freshRanges = (): RangeState[] => [
  { start: null, end: null, show: true },
  { start: null, end: null, show: false },
  { start: null, end: null, show: false },
];

// Cohorts are per tab; granularity + date windows are the global Date groups.
const freshControls = () => ({
  date: { cohortSelection: {}, rangeSelection: [] },
  group: { cohortSelection: {}, rangeSelection: [] },
  viz: { cohortSelection: {}, rangeSelection: [] },
  summaryTable: { cohortSelection: {}, rangeSelection: [] },
});

const initialState = () => ({
  configs: [],
  configId: null,
  config: null,
  controls: freshControls(),
  dateGroups: { granularity: "day" as const, ranges: freshRanges() },
  groupValuesByGran: {},
  panels: {},
  loading: false,
  error: null,
});

describe("dashboardStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDashboardStore.setState(initialState());
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
    // panels are seeded from config.groups; only the FIRST panel starts with
    // a metric (one series request on first launch), the rest start empty.
    expect(Object.keys(s.panels)).toEqual(["group1", "group2"]);
    expect(s.panels.group1.left).toEqual(["num_active_users"]);
    expect(s.panels.group2.left).toEqual([]);
    // cohort values are cached per granularity now.
    expect(s.groupValuesByGran.day).toEqual({ user_group: ["new", "old"] });
    // Default window: last 30 days ending at the data's max date, seeded into
    // the global Date-groups R1.
    expect(s.dateGroups.ranges[0].start).toBe("2025-01-01");
    expect(s.dateGroups.ranges[0].end).toBe("2025-01-30");
    expect(mockApi.series).toHaveBeenCalledTimes(1); // only the first panel fetches
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

  it("date groups are global while cohort selections stay per-tab", () => {
    useDashboardStore.getState().setDateRange(1, { start: "2025-02-01", end: "2025-02-10", show: true });
    useDashboardStore.getState().setTabCohort("viz", "user_group", ["new"]);
    const s = useDashboardStore.getState();
    expect(s.dateGroups.ranges[1]).toEqual({ start: "2025-02-01", end: "2025-02-10", show: true });
    expect(s.controls.viz.cohortSelection).toEqual({ user_group: ["new"] });
    expect(s.controls.group.cohortSelection).toEqual({});
    expect(s.controls.date.cohortSelection).toEqual({});
  });

  it("loadAllSeries sends the active Date-groups windows with the DATE tab's cohorts", async () => {
    const ranges = freshRanges();
    ranges[0] = { start: "2025-01-05", end: "2025-01-12", show: true };
    ranges[2] = { start: "2025-01-20", end: "2025-01-25", show: true };
    useDashboardStore.setState({
      configId: "ss01",
      controls: { ...freshControls(), date: { cohortSelection: { user_group: ["new"] }, rangeSelection: [] } },
      dateGroups: { granularity: "day", ranges },
      panels: { group1: { left: ["num_active_users"], right: ["rtp"], log: false, threshold: 10, series: [] } },
    });
    await useDashboardStore.getState().loadAllSeries();
    expect(mockApi.series).toHaveBeenCalledWith(
      {
        config: "ss01",
        granularity: "day",
        metrics: ["num_active_users", "rtp"],
        ranges: [
          { start: "2025-01-05", end: "2025-01-12" },
          { start: "2025-01-20", end: "2025-01-25" },
        ],
        group_values: { user_group: ["new"] },
        lifecycle_groups: undefined,
      },
      expect.anything(), // AbortSignal threaded for cancellation
    );
  });
});

describe("cohort selection defaults", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDashboardStore.setState(initialState());
    mockApi.getConfig.mockResolvedValue(CONFIG);
    mockApi.groupValues.mockResolvedValue({ config: "ss01", granularity: "day", values: { user_group: ["new", "old"] } });
    mockApi.dateBounds.mockResolvedValue({ config: "ss01", granularity: "day", min: "2025-01-01", max: "2025-01-30" });
    mockApi.deepdiveMetrics.mockResolvedValue({ config: "ss01", granularity: "day", derived: [], user: [] });
    mockApi.series.mockResolvedValue({ config: "ss01", granularity: "day", date_col: "activity_date", series: [], missing: [] });
  });

  it("seeds 'all' explicitly in every visible picker when cohort values load", async () => {
    mockApi.listConfigs.mockResolvedValue({ configs: [{ id: "ss01", title: "SS01" }] });
    await useDashboardStore.getState().loadConfigs();

    const s = useDashboardStore.getState();
    // The UI now states what the request means: 'all' selected, not an
    // ambiguous empty picker (which used to render an empty-looking state
    // while the backend silently treated it as 'all').
    expect(s.controls.date.cohortSelection).toEqual({ user_group: ["all"] });
    expect(s.controls.group.cohortSelection).toEqual({ user_group: ["all"] });
    expect(s.controls.viz.cohortSelection).toEqual({ user_group: ["all"] });
    expect(s.controls.summaryTable.cohortSelection).toEqual({ user_group: ["all"] });
  });

  it("an explicitly emptied picker clears the plots and skips the fetch", async () => {
    mockApi.listConfigs.mockResolvedValue({ configs: [{ id: "ss01", title: "SS01" }] });
    await useDashboardStore.getState().loadConfigs();
    vi.clearAllMocks();

    useDashboardStore.getState().setTabCohort("date", "user_group", []);
    await useDashboardStore.getState().loadAllSeries();

    const s = useDashboardStore.getState();
    expect(mockApi.series).not.toHaveBeenCalled();
    expect(Object.values(s.panels).every((p) => p.series.length === 0)).toBe(true);
  });
});

describe("range-group and lifecycle selection consistency", () => {
  const RANGE_CONFIG = {
    ...CONFIG,
    user_group_cols: ["daily_group", "fish_value"],
    range_group_col: "fish_value",
    range_group_defaults: [{ label: "small", min: 0, max: 10 }],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    useDashboardStore.setState(initialState());
    mockApi.getConfig.mockResolvedValue(RANGE_CONFIG);
    mockApi.groupValues.mockResolvedValue({ config: "ss01", granularity: "day", values: { daily_group: ["a", "b"] } });
    mockApi.dateBounds.mockResolvedValue({ config: "ss01", granularity: "day", min: "2025-01-01", max: "2025-01-30" });
    mockApi.deepdiveMetrics.mockResolvedValue({ config: "ss01", granularity: "day", derived: [], user: [] });
    mockApi.series.mockResolvedValue({ config: "ss01", granularity: "day", date_col: "activity_date", series: [], missing: [] });
  });

  it("seeds rangeSelection to ['all'] on configs with a range dimension", async () => {
    mockApi.listConfigs.mockResolvedValue({ configs: [{ id: "ss01", title: "SS01" }] });
    await useDashboardStore.getState().loadConfigs();
    const s = useDashboardStore.getState();
    expect(s.controls.date.rangeSelection).toEqual(["all"]);
    expect(s.controls.group.rangeSelection).toEqual(["all"]);
    expect(s.lifecycleAll).toBe(true); // lifecycle bar 'all' checked by default
  });

  it("an emptied range-group selection means no data, same as cohorts", async () => {
    mockApi.listConfigs.mockResolvedValue({ configs: [{ id: "ss01", title: "SS01" }] });
    await useDashboardStore.getState().loadConfigs();
    vi.clearAllMocks();

    useDashboardStore.getState().setTabRangeSelection("date", []);
    await useDashboardStore.getState().loadAllSeries();

    expect(mockApi.series).not.toHaveBeenCalled();
    expect(Object.values(useDashboardStore.getState().panels).every((p) => p.series.length === 0)).toBe(true);
  });
});
