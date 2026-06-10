import { useDashboardStore } from "./dashboardStore";
import type { DashboardTab } from "./dashboardStore";

// Applies agent-emitted DashboardActions to the store — the "agent drives the
// dashboard" loop (docs/frontend_redesign.md §4). Every action is validated
// against the live store (known config/tab/panel/metric names) before it
// mutates anything; failures surface as an error toast and never throw, so a
// bad action can't break the chat stream. Applied actions emit an info toast.

export type DashboardAction =
  | { type: "load_config"; config_id: string }
  | { type: "navigate_tab"; tab: DashboardTab }
  | { type: "set_metrics"; panel_id: string; left_metrics: string[]; right_metrics: string[] }
  | { type: "set_date_range"; date_from: string; date_to: string }
  | { type: "set_granularity"; granularity: "day" | "week" | "month" };

const TABS: DashboardTab[] = ["stats-by-date", "stats-by-group", "stats-deepdive", "weekly-report", "report"];
const GRANULARITIES = ["day", "week", "month"] as const;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

const toStrArray = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);

export function dispatchAction(raw: Record<string, unknown>): boolean {
  const s = useDashboardStore.getState();
  const type = String(raw.type ?? "");
  try {
    switch (type) {
      case "load_config": {
        const id = String(raw.config_id ?? "");
        if (!s.configs.some((c) => c.id === id)) throw new Error(`unknown config "${id}"`);
        void s.selectConfig(id);
        s.notify("info", `Agent: switched to config “${id}”`);
        return true;
      }

      case "navigate_tab": {
        const tab = String(raw.tab ?? "") as DashboardTab;
        if (!TABS.includes(tab)) throw new Error(`unknown tab "${tab}"`);
        s.setActiveTab(tab);
        s.notify("info", `Agent: opened ${tab}`);
        return true;
      }

      case "set_metrics": {
        const wanted = String(raw.panel_id ?? "");
        // Resolve by group id or label; keep only metrics the panel offers.
        const group = (s.config?.groups ?? []).find((g) => g.id === wanted || g.label === wanted);
        if (!group || !s.panels[group.id]) throw new Error(`unknown panel "${wanted}"`);
        const allowed = new Set(group.metrics);
        const left = toStrArray(raw.left_metrics).filter((m) => allowed.has(m));
        const right = toStrArray(raw.right_metrics).filter((m) => allowed.has(m) && !left.includes(m));
        if (left.length === 0 && right.length === 0) {
          throw new Error(`none of the requested metrics exist on panel "${group.id}"`);
        }
        s.setPanelMetrics(group.id, "left", left);
        s.setPanelMetrics(group.id, "right", right);
        void s.loadAllSeries(); // incremental loader; cheap if the tab effect also fires
        s.notify("info", `Agent: ${group.label} now plots ${[...left, ...right].join(", ")}`);
        return true;
      }

      case "set_date_range": {
        const from = String(raw.date_from ?? "");
        const to = String(raw.date_to ?? "");
        if (!ISO_DATE.test(from) || !ISO_DATE.test(to)) throw new Error(`invalid date range ${from} → ${to}`);
        s.patchControls("date", { dateFrom: from, dateTo: to });
        void s.loadAllSeries();
        s.notify("info", `Agent: date range set to ${from} → ${to}`);
        return true;
      }

      case "set_granularity": {
        const gran = String(raw.granularity ?? "") as (typeof GRANULARITIES)[number];
        if (!GRANULARITIES.includes(gran)) throw new Error(`unknown granularity "${gran}"`);
        s.patchControls("date", { granularity: gran });
        void s.ensureGroupValues(gran);
        void s.loadAllSeries();
        s.notify("info", `Agent: granularity set to ${gran}`);
        return true;
      }

      default:
        throw new Error(`unknown action type "${type}"`);
    }
  } catch (e) {
    s.notify("error", `Agent action failed: ${e instanceof Error ? e.message : e}`);
    return false;
  }
}
