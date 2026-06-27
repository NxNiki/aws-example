import { uid } from "../lib/uid";
import { useDashboardStore } from "./dashboardStore";
import { useReportStore } from "./reportStore";
import type { DashboardTab } from "./dashboardStore";
import type { FigureSource, ReportFigure } from "../api/types";

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
  | { type: "set_granularity"; granularity: "day" | "week" | "month" }
  | { type: "set_report_period"; date_from: string; date_to: string }
  | { type: "regenerate_report" }
  | { type: "load_report_spec"; name: string }
  | { type: "save_report_spec"; name: string }
  | { type: "add_report_figure"; title: string; source_json: string }
  | { type: "patch_report_figure"; figure_id: string; patch_json: string }
  | { type: "remove_report_figure"; figure_id: string };

const TABS: DashboardTab[] = ["stats-by-date", "stats-by-group", "stats-deepdive", "report"];
const GRANULARITIES = ["day", "week", "month"] as const;
const FIGURE_KINDS = ["stats-by-date", "stats-by-group", "stats-deepdive"];
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

const toStrArray = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);

// Report actions execute async (S3 loads, figure re-fetches) but must land in
// the order the agent emitted them — load_report_spec → set_report_period
// would otherwise patch the OLD spec and be clobbered by the fetched one.
// "applied" in the chat chip means validated-and-queued, like load_config's
// fire-and-forget; execution failures surface as error toasts.
let reportQueue: Promise<unknown> = Promise.resolve();
const enqueueReport = (run: () => Promise<unknown> | void) => {
  reportQueue = reportQueue.then(run).catch((e) => {
    useDashboardStore.getState().notify("error", `Agent report action failed: ${e instanceof Error ? e.message : e}`);
  });
};

// Test hook: resolves when every queued report action has settled.
export const reportActionsSettled = (): Promise<unknown> => reportQueue;

// Tool args arrive as JSON strings (source_json/patch_json); be forgiving if a
// model passes a literal object instead.
const parseJsonObj = (v: unknown, what: string): Record<string, unknown> => {
  const obj = typeof v === "string" ? JSON.parse(v) : v;
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error(`${what} must be a JSON object`);
  return obj as Record<string, unknown>;
};

// Run a spec edit through applySpec so normalize + render + the staleness rule
// apply uniformly; applySpec reports errors as a return value, so re-throw for
// the queue's toast.
const applyFigures = async (figures: ReportFigure[]) => {
  const st = useReportStore.getState();
  const err = await st.applySpec({ ...st.spec, figures });
  if (err) throw new Error(err);
};

const requireFigure = (figureId: string): ReportFigure => {
  const fig = useReportStore.getState().spec.figures.find((f) => f.id === figureId);
  if (!fig) throw new Error(`unknown report figure "${figureId}"`);
  return fig;
};

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

      case "set_report_period": {
        const from = String(raw.date_from ?? "");
        const to = String(raw.date_to ?? "");
        if (!ISO_DATE.test(from) || !ISO_DATE.test(to)) throw new Error(`invalid report period ${from} → ${to}`);
        enqueueReport(() => useReportStore.getState().patchSpec({ period: { start: from, end: to } }));
        s.notify("info", `Agent: report period set to ${from} → ${to}`);
        return true;
      }

      case "regenerate_report": {
        enqueueReport(() => useReportStore.getState().regenerate());
        s.notify("info", "Agent: regenerating the report…");
        return true;
      }

      case "load_report_spec": {
        const name = String(raw.name ?? "");
        if (!name.trim()) throw new Error("missing report spec name");
        enqueueReport(() => useReportStore.getState().loadSpec(name));
        return true;
      }

      case "save_report_spec": {
        const name = String(raw.name ?? "");
        if (!name.trim()) throw new Error("missing report spec name");
        enqueueReport(() => useReportStore.getState().saveSpec(name));
        return true;
      }

      case "add_report_figure": {
        const title = String(raw.title ?? "");
        const source = parseJsonObj(raw.source_json, "source_json") as unknown as FigureSource;
        if (!FIGURE_KINDS.includes(source.kind)) throw new Error(`unknown figure kind "${source.kind}"`);
        enqueueReport(() => {
          const fig: ReportFigure = {
            id: uid(),
            title,
            description: "",
            inherit_period: true,
            source,
          };
          return applyFigures([...useReportStore.getState().spec.figures, fig]);
        });
        s.notify("info", `Agent: added “${title}” to the report`);
        return true;
      }

      case "patch_report_figure": {
        const figureId = String(raw.figure_id ?? "");
        const patch = parseJsonObj(raw.patch_json, "patch_json") as Partial<ReportFigure>;
        if (patch.source && !FIGURE_KINDS.includes(patch.source.kind)) {
          throw new Error(`unknown figure kind "${patch.source.kind}"`);
        }
        // Figure existence is checked at execution time, not dispatch time —
        // the figure may belong to a spec still loading earlier in the queue.
        enqueueReport(() => {
          requireFigure(figureId);
          return applyFigures(
            useReportStore.getState().spec.figures.map((f) => (f.id === figureId ? { ...f, ...patch, id: f.id } : f)),
          );
        });
        s.notify("info", `Agent: editing report figure “${figureId}”`);
        return true;
      }

      case "remove_report_figure": {
        const figureId = String(raw.figure_id ?? "");
        enqueueReport(() => {
          requireFigure(figureId);
          return applyFigures(useReportStore.getState().spec.figures.filter((f) => f.id !== figureId));
        });
        s.notify("info", `Agent: removed report figure “${figureId}”`);
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
