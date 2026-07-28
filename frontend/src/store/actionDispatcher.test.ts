import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  api: {
    listConfigs: vi.fn(),
    getConfig: vi.fn(),
    groupValues: vi.fn(),
    dateBounds: vi.fn(),
    deepdiveMetrics: vi.fn(),
    series: vi.fn().mockResolvedValue({ series: [], missing: [] }),
    groupDistribution: vi.fn().mockResolvedValue({ stats: [] }),
    deepdive: vi.fn().mockResolvedValue({ histograms: [], heatmaps: [], scatters: [] }),
    listReportSpecs: vi.fn().mockResolvedValue([]),
    loadReportSpec: vi.fn(),
    saveReportSpec: vi.fn(),
  },
}));

import { api } from "../api/client";
import { useDashboardStore } from "./dashboardStore";
import { useReportStore } from "./reportStore";
import { dispatchAction, reportActionsSettled } from "./actionDispatcher";
import type { ReportSpec } from "../api/types";

const CONFIG = {
  id: "ss01",
  title: "SS01",
  date_col: "activity_date",
  group_col: "",
  user_group_cols: [] as string[],
  range_group_defaults: [],
  range_group_values: [] as number[],
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

  it("set_date_range sets the global Date-groups R1 window", () => {
    expect(dispatchAction({ type: "set_date_range", date_from: "2026-05-01", date_to: "2026-06-01" })).toBe(true);
    const r1 = useDashboardStore.getState().dateGroups.ranges[0];
    expect(r1).toEqual({ start: "2026-05-01", end: "2026-06-01", show: true });
  });

  it("agent report actions: period + figure add/patch/remove, queued in order", async () => {
    const blankSpec = (over: Partial<ReportSpec> = {}): ReportSpec => ({
      version: 1,
      id: "",
      title: "",
      language: "en",
      period: null,
      view: null,
      figures: [],
      references: [],
      summary: "",
      confluence_url: "",
      ...over,
    });
    useReportStore.setState({ spec: blankSpec(), figureData: {}, specs: [] });

    expect(dispatchAction({ type: "set_report_period", date_from: "2026-05-01", date_to: "2026-05-31" })).toBe(true);
    expect(dispatchAction({ type: "set_report_period", date_from: "May 1", date_to: "2026-05-31" })).toBe(false);

    const source = {
      kind: "stats-by-date",
      config: "ss01",
      granularity: "day",
      date_from: null,
      date_to: null,
      cohort_selection: {},
      panel_id: "group1",
      left: ["num_active_users"],
      right: [],
      log: false,
      threshold: 10,
    };
    expect(dispatchAction({ type: "add_report_figure", title: "DAU", source_json: JSON.stringify(source) })).toBe(true);
    expect(dispatchAction({ type: "add_report_figure", title: "bad", source_json: "{not json" })).toBe(false);
    expect(
      dispatchAction({ type: "add_report_figure", title: "bad", source_json: JSON.stringify({ kind: "nope" }) }),
    ).toBe(false);

    await reportActionsSettled();
    const st = useReportStore.getState();
    expect(st.spec.period).toEqual({ start: "2026-05-01", end: "2026-05-31" });
    expect(st.spec.figures).toHaveLength(1);
    const figId = st.spec.figures[0].id;
    // inherit_period figure rendered with the report period substituted in
    expect(st.figureData[figId]?.series).toEqual([]);
    expect(api.series).toHaveBeenLastCalledWith(expect.objectContaining({ date_from: "2026-05-01", date_to: "2026-05-31" }));

    // patch: title-only edit keeps the fetched data (no re-render), unknown id toasts
    vi.mocked(api.series).mockClear();
    expect(dispatchAction({ type: "patch_report_figure", figure_id: figId, patch_json: '{"title": "DAU v2"}' })).toBe(true);
    expect(dispatchAction({ type: "patch_report_figure", figure_id: "ghost", patch_json: "{}" })).toBe(true); // queued; fails at execution
    await reportActionsSettled();
    expect(useReportStore.getState().spec.figures[0].title).toBe("DAU v2");
    expect(api.series).not.toHaveBeenCalled();
    const toasts = useDashboardStore.getState().notifications;
    const last = toasts[toasts.length - 1];
    expect(last?.kind).toBe("error");
    expect(last?.message).toContain("ghost");

    expect(dispatchAction({ type: "remove_report_figure", figure_id: figId })).toBe(true);
    await reportActionsSettled();
    expect(useReportStore.getState().spec.figures).toHaveLength(0);
  });

  it("load_report_spec then set_report_period: the period lands on the LOADED spec", async () => {
    useReportStore.setState({
      spec: {
        version: 1,
        id: "",
        title: "stale",
        language: "en",
        period: null,
        view: null,
        figures: [],
        references: [],
        summary: "",
        confluence_url: "",
      },
      figureData: {},
      specs: ["monthly"],
    });
    vi.mocked(api.loadReportSpec).mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({
                version: 1,
                id: "monthly",
                title: "Monthly KPIs",
                language: "en",
                period: { start: "2026-04-01", end: "2026-04-30" },
                view: null,
                figures: [],
                references: [],
                summary: "",
                confluence_url: "",
              }),
            10,
          ),
        ),
    );
    expect(dispatchAction({ type: "load_report_spec", name: "monthly" })).toBe(true);
    expect(dispatchAction({ type: "set_report_period", date_from: "2026-05-01", date_to: "2026-05-31" })).toBe(true);
    await reportActionsSettled();
    const spec = useReportStore.getState().spec;
    expect(spec.title).toBe("Monthly KPIs"); // load completed first…
    expect(spec.period).toEqual({ start: "2026-05-01", end: "2026-05-31" }); // …then the period patch
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
