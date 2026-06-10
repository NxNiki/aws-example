import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  api: {
    listConfigs: vi.fn(),
    getConfig: vi.fn(),
    groupValues: vi.fn(),
    dateBounds: vi.fn(),
    deepdiveMetrics: vi.fn(),
    series: vi.fn().mockResolvedValue({ series: [], missing: [] }),
  },
}));

import { useDashboardStore } from "./dashboardStore";
import { dispatchAction } from "./actionDispatcher";

const CONFIG = {
  id: "ss01",
  title: "SS01",
  date_col: "activity_date",
  group_col: "",
  user_group_cols: [] as string[],
  granularities: ["day" as const],
  groups: [{ id: "group1", label: "DAU and Retention", metrics: ["num_active_users", "rtp"] }],
  tabs: ["stats-by-date"],
};

describe("dispatchAction", () => {
  beforeEach(() => {
    useDashboardStore.setState({
      configs: [{ id: "ss01", title: "SS01" }],
      configId: "ss01",
      config: CONFIG,
      activeTab: "stats-by-date",
      panels: { group1: { left: [], right: [], log: false, threshold: 10, series: [] } },
      notifications: [],
    });
  });

  it("navigate_tab switches the active tab", () => {
    expect(dispatchAction({ type: "navigate_tab", tab: "stats-by-group" })).toBe(true);
    expect(useDashboardStore.getState().activeTab).toBe("stats-by-group");
  });

  it("set_metrics resolves the panel by label and filters unknown metrics", () => {
    const ok = dispatchAction({
      type: "set_metrics",
      panel_id: "DAU and Retention",
      left_metrics: ["num_active_users", "not_a_metric"],
      right_metrics: ["rtp"],
    });
    expect(ok).toBe(true);
    const panel = useDashboardStore.getState().panels.group1;
    expect(panel.left).toEqual(["num_active_users"]);
    expect(panel.right).toEqual(["rtp"]);
  });

  it("set_date_range patches the date tab's window", () => {
    expect(dispatchAction({ type: "set_date_range", date_from: "2026-05-01", date_to: "2026-06-01" })).toBe(true);
    const c = useDashboardStore.getState().controls.date;
    expect(c.dateFrom).toBe("2026-05-01");
    expect(c.dateTo).toBe("2026-06-01");
  });

  it("rejects unknown actions / tabs / configs with an error toast, never throws", () => {
    expect(dispatchAction({ type: "explode" })).toBe(false);
    expect(dispatchAction({ type: "navigate_tab", tab: "nope" })).toBe(false);
    expect(dispatchAction({ type: "load_config", config_id: "nope" })).toBe(false);
    const toasts = useDashboardStore.getState().notifications;
    expect(toasts.length).toBe(3);
    expect(toasts.every((t) => t.kind === "error")).toBe(true);
    // and the dashboard state is untouched
    expect(useDashboardStore.getState().activeTab).toBe("stats-by-date");
  });
});
