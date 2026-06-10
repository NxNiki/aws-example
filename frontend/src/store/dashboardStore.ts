import { create } from "zustand";
import { api } from "../api/client";
import { streamAgentChat } from "../api/agentStream";
import { dispatchAction } from "./actionDispatcher";
import type { RangeState } from "../components/DateRanges";
import type {
  ClipOpts,
  ConfigDetail,
  ConfigSummary,
  CorrMatrix,
  DeepdiveMode,
  DeepdivePanel,
  Granularity,
  GroupStat,
  HistogramSeries,
  ScatterSeries,
  Series,
} from "../api/types";

// ── Per-tab shared controls ────────────────────────────────────────────────
// Each tab owns its granularity, date window(s), and cohort selection — they
// are NOT shared across tabs (changing the Deep Dive range must not move the
// Stats-by-Group comparison).

export interface DateTabControls {
  granularity: Granularity;
  dateFrom: string | null;
  dateTo: string | null;
  cohortSelection: Record<string, string[]>;
}

export interface RangeTabControls {
  granularity: Granularity;
  ranges: RangeState[]; // up to 3 comparable windows
  cohortSelection: Record<string, string[]>;
}

export interface TabControls {
  date: DateTabControls;
  group: RangeTabControls;
  viz: RangeTabControls;
}
export type TabKey = keyof TabControls;

// Top-level dashboard tabs. Lives in the store (not component state) so the
// agent's navigate_tab action can drive it.
export type DashboardTab = "stats-by-date" | "stats-by-group" | "stats-deepdive" | "weekly-report";

const defaultRanges = (): RangeState[] => [
  { start: null, end: null, show: true },
  { start: null, end: null, show: false },
  { start: null, end: null, show: false },
];

const defaultControls = (): TabControls => ({
  date: { granularity: "day", dateFrom: null, dateTo: null, cohortSelection: {} },
  group: { granularity: "day", ranges: defaultRanges(), cohortSelection: {} },
  viz: { granularity: "day", ranges: defaultRanges(), cohortSelection: {} },
});

// ── Per-panel state ────────────────────────────────────────────────────────

// Stats-by-Date: which metrics go on each axis + the hybrid-log toggle.
export interface PanelState {
  left: string[];
  right: string[];
  log: boolean;
  threshold: number;
  series: Series[];
}

// Stats-by-Group: one metric, box-or-bar, clip; results are per (cohort×range).
export interface GroupPanelState {
  metric: string | null;
  mode: "box" | "bar";
  clip: ClipOpts;
  stats: GroupStat[];
  missing: boolean;
}

// Deep Dive: a panel (derived/user) with a mode + metric multi-select.
export interface DeepdivePanelState {
  mode: DeepdiveMode;
  metrics: string[];
  nbins: number;
  logY: boolean; // histogram count axis (display)
  normalize: boolean;
  clip: ClipOpts;
  outliersStd: number | null; // scatter: drop rows beyond N std (null = off)
  scatterLogX: boolean; // scatter x axis (display)
  scatterLogY: boolean; // scatter y axis (display)
  histograms: HistogramSeries[];
  heatmaps: CorrMatrix[];
  scatters: ScatterSeries[];
  missing: string[];
}

const DEFAULT_THRESHOLD = 10;
const noClip = (): ClipOpts => ({ enable: false, min: null, max: null });

function defaultPanels(config: ConfigDetail): Record<string, PanelState> {
  const panels: Record<string, PanelState> = {};
  for (const g of config.groups) {
    panels[g.id] = { left: g.metrics.slice(0, 1), right: [], log: false, threshold: DEFAULT_THRESHOLD, series: [] };
  }
  return panels;
}

function defaultGroupPanels(config: ConfigDetail): Record<string, GroupPanelState> {
  const panels: Record<string, GroupPanelState> = {};
  for (const g of config.groups) {
    panels[g.id] = { metric: g.metrics[0] ?? null, mode: "bar", clip: noClip(), stats: [], missing: false };
  }
  return panels;
}

const emptyDeepdivePanel = (): DeepdivePanelState => ({
  mode: "histogram",
  metrics: [],
  nbins: 50,
  logY: false,
  normalize: false,
  clip: noClip(),
  outliersStd: null,
  scatterLogX: false,
  scatterLogY: false,
  histograms: [],
  heatmaps: [],
  scatters: [],
  missing: [],
});

// A serializable snapshot of the dashboard's UI selections (NOT fetched data /
// figures) — the React-era "save/load config". Persisted opaquely to S3.
// v2 stores per-tab `controls`; v1 snapshots (global granularity/dates/cohorts)
// are migrated on load by `snapshotControls`.
export interface ViewSnapshot {
  version: number;
  configId: string | null;
  controls?: TabControls;
  // v1 legacy fields (still read for migration):
  granularity?: Granularity;
  dateFrom?: string | null;
  dateTo?: string | null;
  ranges?: RangeState[];
  cohortSelection?: Record<string, string[]>;
  panels: Record<string, PanelState>;
  group: Record<string, GroupPanelState>;
  deepdive: Record<DeepdivePanel, DeepdivePanelState>;
}

function snapshotControls(snap: ViewSnapshot): TabControls {
  const dc = defaultControls();
  if (snap.controls) {
    return {
      date: { ...dc.date, ...(snap.controls.date ?? {}) },
      group: { ...dc.group, ...(snap.controls.group ?? {}) },
      viz: { ...dc.viz, ...(snap.controls.viz ?? {}) },
    };
  }
  // v1: one global set of controls — apply it to every tab.
  const granularity = snap.granularity ?? "day";
  const cohortSelection = snap.cohortSelection ?? {};
  const ranges = snap.ranges ?? defaultRanges();
  return {
    date: { granularity, dateFrom: snap.dateFrom ?? null, dateTo: snap.dateTo ?? null, cohortSelection },
    group: { granularity, ranges, cohortSelection },
    viz: { granularity, ranges, cohortSelection },
  };
}

// Last-30-days window (inclusive) ending at the data's max date.
function lastThirtyDays(maxIso: string): { from: string; to: string } {
  const d = new Date(`${maxIso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() - 29);
  return { from: d.toISOString().slice(0, 10), to: maxIso };
}

// ── Store ────────────────────────────────────────────────────────────────

interface DashboardState {
  configs: ConfigSummary[];
  configId: string | null;
  config: ConfigDetail | null;

  activeTab: DashboardTab;
  setActiveTab: (tab: DashboardTab) => void;

  controls: TabControls;
  // Available cohort values per granularity (the data can differ by
  // granularity, and tabs can sit on different granularities).
  groupValuesByGran: Partial<Record<Granularity, Record<string, string[]>>>;

  panels: Record<string, PanelState>; // Stats-by-Date
  group: Record<string, GroupPanelState>; // Stats-by-Group
  deepdiveMetrics: { derived: string[]; user: string[] };
  deepdive: Record<DeepdivePanel, DeepdivePanelState>;

  views: string[]; // names of saved view snapshots
  notifications: Toast[]; // transient status messages (toasts)
  loading: boolean;
  status: string | null; // what the dashboard is currently doing (header indicator)
  error: string | null;

  loadConfigs: () => Promise<void>;
  selectConfig: (id: string) => Promise<void>;
  patchControls: <K extends TabKey>(tab: K, patch: Partial<TabControls[K]>) => void;
  setTabCohort: (tab: TabKey, col: string, values: string[]) => void;
  setTabRange: (tab: "group" | "viz", index: number, range: RangeState) => void;
  ensureGroupValues: (gran: Granularity) => Promise<void>;

  // Stats-by-Date
  setPanelMetrics: (panelId: string, side: "left" | "right", metrics: string[]) => void;
  setPanelLog: (panelId: string, log: boolean, threshold?: number) => void;
  loadDateBounds: () => Promise<void>;
  loadAllSeries: () => Promise<void>;

  // Stats-by-Group
  setGroupMetric: (panelId: string, metric: string) => void;
  setGroupMode: (panelId: string, mode: "box" | "bar") => void;
  setGroupClip: (panelId: string, clip: ClipOpts) => void;
  loadGroupDistribution: () => Promise<void>;

  // Deep Dive
  loadDeepdiveMetrics: () => Promise<void>;
  patchDeepdive: (panel: DeepdivePanel, patch: Partial<DeepdivePanelState>) => void;
  loadDeepdive: () => Promise<void>;

  // Saved views (save/load the dashboard setting)
  captureView: () => ViewSnapshot;
  applyView: (snap: ViewSnapshot) => Promise<boolean>;
  loadViews: () => Promise<void>;
  saveView: (name: string) => Promise<void>;
  loadViewByName: (name: string) => Promise<void>;

  // Status notifications (toasts)
  notify: (kind: Toast["kind"], message: string) => void;
  dismissNotification: (id: string) => void;

  // Agent chat
  chat: ChatState;
  toggleChat: () => void;
  setChatWidth: (width: number) => void;
  setChatModel: (model: string) => void;
  clearChat: () => void;
  stopChat: () => void;
  sendChat: (text: string) => Promise<void>;
}

export interface Toast {
  id: string;
  kind: "success" | "error" | "info";
  message: string;
}

// ── Agent chat (Phase 3) ───────────────────────────────────────────────────

// One tool call or dashboard action rendered inline in an assistant message.
export interface ChatEvent {
  kind: "tool" | "action";
  name: string;
  detail: string;
  done: boolean;
}

export interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  events: ChatEvent[];
}

export interface ChatState {
  open: boolean;
  width: number; // panel width in px (user-resizable via the drag handle)
  model: string; // catalog key, e.g. "gemini:gemini-2.5-flash"
  messages: ChatMsg[];
  streaming: boolean;
  pendingClarify: { question: string; options: string[] } | null;
}

export const CHAT_MIN_WIDTH = 360;
export const CHAT_MAX_WIDTH = 900;

// Mirrors ai_agent's MODEL_CATALOG keys (the legacy chat dropdown).
export const CHAT_MODELS = [
  "gemini:gemini-2.5-flash",
  "gemini:gemini-2.5-pro",
  "openai:gpt-4o-mini",
  "openai:gpt-4o",
  "openai:gpt-4.1-mini",
  "openai:gpt-4.1",
];

const emptyChat = (): ChatState => ({
  open: false,
  width: 520,
  model: CHAT_MODELS[0],
  messages: [],
  streaming: false,
  pendingClarify: null,
});

let _chatAbort: AbortController | null = null;

function activeRanges(ranges: RangeState[]): { start: string; end: string }[] {
  return ranges.filter((r) => r.show && r.start && r.end).map((r) => ({ start: r.start as string, end: r.end as string }));
}

// In-flight request management. Each data loader runs through this: it aborts the
// loader's previous request, ignores a response if a newer request has since
// superseded it, and tracks a global in-flight count so the loading indicator
// reflects any active fetch. Fixes the race where rapid control changes let an
// older response overwrite newer data (or leave the spinner stuck).
type Setter = (partial: Partial<DashboardState> | ((s: DashboardState) => Partial<DashboardState>)) => void;
const _abort: Record<string, AbortController> = {};
let _inflight = 0;
const _isAbort = (e: unknown) => e instanceof DOMException && e.name === "AbortError";
// Last-fetched input signature per panel (keyed "series:g1", "group:g2",
// "deepdive:derived", …). A panel whose signature is unchanged is skipped, so
// changing one panel doesn't re-fetch the others. configId is part of every
// signature, so a config switch naturally invalidates all panels.
const _panelSig: Record<string, string> = {};
// Stats-by-Date does finer-grained, metric-level loading: this caches each
// panel's CONTEXT signature (config/granularity/dates/cohorts — NOT metrics). If
// the context is unchanged, only newly-added metrics are fetched and appended;
// removed metrics are hidden by the option-builder (no fetch) and stay cached so
// re-adding is instant.
const _panelCtx: Record<string, string> = {};

async function runExclusive<T>(
  key: string,
  label: string,
  set: Setter,
  run: (signal: AbortSignal) => Promise<T>,
  apply: (result: T, set: Setter) => void,
  onError: (e: unknown, set: Setter) => void,
): Promise<void> {
  _abort[key]?.abort(); // cancel the previous fetch for this loader
  const ctl = new AbortController();
  _abort[key] = ctl;
  _inflight += 1;
  set({ loading: true, status: label, error: null });
  try {
    const result = await run(ctl.signal);
    if (_abort[key] === ctl) apply(result, set); // still the latest → apply
  } catch (e) {
    if (_abort[key] === ctl && !_isAbort(e)) onError(e, set); // ignore aborted / superseded
  } finally {
    if (_abort[key] === ctl) delete _abort[key];
    _inflight = Math.max(0, _inflight - 1);
    // Clear the status only when nothing else is in flight; otherwise leave the
    // other in-flight loader's label showing.
    set(_inflight > 0 ? { loading: true } : { loading: false, status: null });
  }
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  configs: [],
  configId: null,
  config: null,
  activeTab: "stats-by-date",
  setActiveTab: (tab) => set({ activeTab: tab }),
  controls: defaultControls(),
  groupValuesByGran: {},
  panels: {},
  group: {},
  deepdiveMetrics: { derived: [], user: [] },
  deepdive: { derived: emptyDeepdivePanel(), user: emptyDeepdivePanel() },
  views: [],
  notifications: [],
  chat: emptyChat(),
  loading: false,
  status: null,
  error: null,

  loadConfigs: async () => {
    set({ loading: true, error: null });
    try {
      const { configs } = await api.listConfigs();
      set({ configs });
      if (configs.length && !get().configId) {
        await get().selectConfig(configs[0].id);
      }
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  selectConfig: async (id: string) => {
    set((s) => ({
      loading: true,
      error: null,
      configId: id,
      groupValuesByGran: {},
      deepdiveMetrics: { derived: [], user: [] },
      deepdive: { derived: emptyDeepdivePanel(), user: emptyDeepdivePanel() },
      // Keep each tab's granularity/windows; cohort values are config-specific.
      controls: {
        date: { ...s.controls.date, cohortSelection: {} },
        group: { ...s.controls.group, cohortSelection: {} },
        viz: { ...s.controls.viz, cohortSelection: {} },
      },
    }));
    try {
      const config = await api.getConfig(id);
      set({ config, panels: defaultPanels(config), group: defaultGroupPanels(config) });
      const c = get().controls;
      const grans = [...new Set([c.date.granularity, c.group.granularity, c.viz.granularity])];
      await Promise.all(grans.map((g) => get().ensureGroupValues(g)));
      await get().loadDeepdiveMetrics();
      await get().loadDateBounds();
      await get().loadAllSeries();
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  patchControls: (tab, patch) =>
    set((s) => ({ controls: { ...s.controls, [tab]: { ...s.controls[tab], ...patch } } })),

  setTabCohort: (tab, col, values) =>
    set((s) => ({
      controls: {
        ...s.controls,
        [tab]: { ...s.controls[tab], cohortSelection: { ...s.controls[tab].cohortSelection, [col]: values } },
      },
    })),

  setTabRange: (tab, index, range) =>
    set((s) => ({
      controls: {
        ...s.controls,
        [tab]: { ...s.controls[tab], ranges: s.controls[tab].ranges.map((r, i) => (i === index ? range : r)) },
      },
    })),

  ensureGroupValues: async (gran) => {
    const { configId, groupValuesByGran } = get();
    if (!configId || groupValuesByGran[gran]) return;
    try {
      const { values } = await api.groupValues(configId, gran);
      set((s) => ({ groupValuesByGran: { ...s.groupValuesByGran, [gran]: values } }));
    } catch (e) {
      // Toast (not just the transient error field): a later successful load
      // clears `error`, so the toast is what reliably surfaces this failure.
      set({ error: String(e) });
      get().notify("error", `Failed to load cohort values: ${e}`);
    }
  },

  setPanelMetrics: (panelId, side, metrics) =>
    set((s) => {
      const panel = s.panels[panelId];
      if (!panel) return {};
      return { panels: { ...s.panels, [panelId]: { ...panel, [side]: metrics } } };
    }),

  setPanelLog: (panelId, log, threshold) =>
    set((s) => {
      const panel = s.panels[panelId];
      if (!panel) return {};
      return { panels: { ...s.panels, [panelId]: { ...panel, log, threshold: threshold ?? panel.threshold } } };
    }),

  loadDateBounds: async () => {
    const { configId, controls } = get();
    if (!configId) return;
    try {
      const { max } = await api.dateBounds(configId, controls.date.granularity);
      if (max) {
        const { from, to } = lastThirtyDays(max);
        const seed = (ranges: RangeState[]) => ranges.map((r, i) => (i === 0 ? { ...r, start: from, end: to } : r));
        set((s) => ({
          controls: {
            date: { ...s.controls.date, dateFrom: from, dateTo: to },
            group: { ...s.controls.group, ranges: seed(s.controls.group.ranges) },
            viz: { ...s.controls.viz, ranges: seed(s.controls.viz.ranges) },
          },
        }));
      }
    } catch (e) {
      set({ error: String(e) });
      get().notify("error", `Failed to load date bounds: ${e}`);
    }
  },

  // Fetch each Stats-by-Date panel independently AND incrementally: when only the
  // metric set changed (same context), fetch just the newly-added metrics and
  // append them; removed metrics are hidden by the option-builder and kept cached
  // (instant re-add). A context change (config/granularity/dates/cohorts) reloads
  // the whole panel. Each panel has its own cancellation key.
  loadAllSeries: async () => {
    const { configId, controls, panels } = get();
    if (!configId) return;
    const { granularity, dateFrom, dateTo, cohortSelection } = controls.date;
    const ctxSig = JSON.stringify({ configId, granularity, dateFrom, dateTo, cohortSelection });
    await Promise.all(
      Object.entries(panels).map(([panelId, panel]) => {
        const key = `series:${panelId}`;
        const ctxChanged = _panelCtx[key] !== ctxSig;
        const desired = [...new Set([...panel.left, ...panel.right])];
        const loaded = ctxChanged ? new Set<string>() : new Set(panel.series.map((s) => s.metric));
        const toFetch = desired.filter((m) => !loaded.has(m));

        if (toFetch.length === 0) {
          // Nothing new. On a context change with no selected metrics, clear the
          // now-stale series; otherwise the builder already hides unselected ones.
          if (ctxChanged) {
            _panelCtx[key] = ctxSig;
            set((s) =>
              s.panels[panelId] ? { panels: { ...s.panels, [panelId]: { ...s.panels[panelId], series: [] } } } : {},
            );
          }
          return;
        }

        return runExclusive(
          key,
          "Loading metrics…",
          set,
          async (signal): Promise<Series[]> => {
            const resp = await api.series(
              { config: configId, granularity, metrics: toFetch, date_from: dateFrom, date_to: dateTo, group_values: cohortSelection },
              signal,
            );
            return resp.series;
          },
          (fetched, set) => {
            _panelCtx[key] = ctxSig;
            set((s) => {
              const cur = s.panels[panelId];
              if (!cur) return {};
              // Replace on context change; append the new metrics otherwise.
              const series = ctxChanged ? fetched : [...cur.series, ...fetched];
              return { panels: { ...s.panels, [panelId]: { ...cur, series } } };
            });
          },
          (e, set) => {
            set({ error: String(e) });
            get().notify("error", `Failed to load series: ${e}`);
          },
        );
      }),
    );
  },

  setGroupMetric: (panelId, metric) =>
    set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], metric } } } : {})),
  setGroupMode: (panelId, mode) =>
    set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], mode } } } : {})),
  setGroupClip: (panelId, clip) =>
    set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], clip } } } : {})),

  loadGroupDistribution: async () => {
    const { configId, controls, group } = get();
    if (!configId) return;
    const { granularity, ranges, cohortSelection } = controls.group;
    const reqRanges = activeRanges(ranges);
    if (reqRanges.length === 0) return;
    await Promise.all(
      Object.entries(group).map(([panelId, p]) => {
        const key = `group:${panelId}`;
        const sig = JSON.stringify({ configId, granularity, reqRanges, cohortSelection, metric: p.metric, clip: p.clip });
        if (_panelSig[key] === sig) return;
        return runExclusive(
          key,
          "Computing distributions (bootstrap CI)…",
          set,
          async (signal) => {
            if (!p.metric) return { stats: [] as GroupStat[], missing: false };
            const resp = await api.groupDistribution(
              { config: configId, granularity, metric: p.metric, ranges: reqRanges, group_values: cohortSelection, clip: p.clip },
              signal,
            );
            return { stats: resp.stats, missing: resp.missing };
          },
          (r, set) => {
            _panelSig[key] = sig;
            set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], ...r } } } : {}));
          },
          (e, set) => {
            set({ error: String(e) });
            get().notify("error", `Failed to load group stats: ${e}`);
          },
        );
      }),
    );
  },

  loadDeepdiveMetrics: async () => {
    const { configId, controls } = get();
    if (!configId) return;
    try {
      const { derived, user } = await api.deepdiveMetrics(configId, controls.viz.granularity);
      set({ deepdiveMetrics: { derived, user } });
    } catch (e) {
      set({ error: String(e) });
      get().notify("error", `Failed to load Deep Dive metrics: ${e}`);
    }
  },

  patchDeepdive: (panel, patch) =>
    set((s) => ({ deepdive: { ...s.deepdive, [panel]: { ...s.deepdive[panel], ...patch } } })),

  // Capture the UI selections (drop fetched data / figures — they re-fetch).
  captureView: () => {
    const s = get();
    return {
      version: 2,
      configId: s.configId,
      controls: s.controls,
      panels: Object.fromEntries(Object.entries(s.panels).map(([k, v]) => [k, { ...v, series: [] }])),
      group: Object.fromEntries(Object.entries(s.group).map(([k, v]) => [k, { ...v, stats: [], missing: false }])),
      deepdive: {
        derived: { ...s.deepdive.derived, histograms: [], heatmaps: [], scatters: [], missing: [] },
        user: { ...s.deepdive.user, histograms: [], heatmaps: [], scatters: [], missing: [] },
      },
    };
  },

  // Restore a snapshot: load the config (for group/granularity metadata), merge
  // saved selections over fresh defaults (so renamed/added groups still work),
  // then let the tabs refetch. v1 snapshots are migrated by snapshotControls.
  applyView: async (snap) => {
    if (!snap?.configId) return false;
    set({ loading: true, error: null });
    try {
      const config = await api.getConfig(snap.configId);
      const controls = snapshotControls(snap);
      set({
        configId: snap.configId,
        config,
        controls,
        groupValuesByGran: {},
        panels: { ...defaultPanels(config), ...(snap.panels ?? {}) },
        group: { ...defaultGroupPanels(config), ...(snap.group ?? {}) },
        deepdive: {
          derived: { ...emptyDeepdivePanel(), ...(snap.deepdive?.derived ?? {}) },
          user: { ...emptyDeepdivePanel(), ...(snap.deepdive?.user ?? {}) },
        },
      });
      const grans = [...new Set([controls.date.granularity, controls.group.granularity, controls.viz.granularity])];
      await Promise.all(grans.map((g) => get().ensureGroupValues(g)));
      await get().loadDeepdiveMetrics();
      await get().loadAllSeries();
      return true;
    } catch (e) {
      set({ error: String(e) });
      get().notify("error", `Failed to apply view: ${e}`);
      return false;
    } finally {
      set({ loading: false });
    }
  },

  loadViews: async () => {
    try {
      set({ views: await api.listViews() });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  saveView: async (name) => {
    try {
      const r = await api.saveView(name, get().captureView());
      set({ views: r.views });
      get().notify("success", `Saved view “${r.name}” → ${r.path}`);
    } catch (e) {
      set({ error: String(e) });
      get().notify("error", `Failed to save view: ${e}`);
    }
  },

  loadViewByName: async (name) => {
    let snap: ViewSnapshot;
    try {
      snap = (await api.loadView(name)) as ViewSnapshot;
    } catch (e) {
      set({ error: String(e) });
      get().notify("error", `Failed to load view “${name}”: ${e}`);
      return;
    }
    if (await get().applyView(snap)) get().notify("info", `Loaded view “${name}”`);
  },

  notify: (kind, message) =>
    set((s) => ({
      notifications: [...s.notifications, { id: crypto.randomUUID(), kind, message }].slice(-5),
    })),

  dismissNotification: (id) => set((s) => ({ notifications: s.notifications.filter((n) => n.id !== id) })),

  loadDeepdive: async () => {
    const { configId, controls, deepdive } = get();
    if (!configId) return;
    const { granularity, ranges, cohortSelection } = controls.viz;
    const reqRanges = activeRanges(ranges);
    if (reqRanges.length === 0) return;
    const panels: DeepdivePanel[] = ["derived", "user"];
    await Promise.all(
      panels.map((panel) => {
        const p = deepdive[panel];
        const key = `deepdive:${panel}`;
        const sig = JSON.stringify({
          configId,
          granularity,
          reqRanges,
          cohortSelection,
          mode: p.mode,
          metrics: p.metrics,
          nbins: p.nbins,
          normalize: p.normalize,
          clip: p.clip,
          outliersStd: p.outliersStd,
        });
        if (_panelSig[key] === sig) return;
        const label =
          p.mode === "histogram"
            ? "Binning histograms…"
            : p.mode === "heatmap"
              ? "Computing correlations…"
              : "Sampling points…";
        return runExclusive(
          key,
          label,
          set,
          async (signal) => {
            if (p.metrics.length === 0) {
              return {
                histograms: [] as HistogramSeries[],
                heatmaps: [] as CorrMatrix[],
                scatters: [] as ScatterSeries[],
                missing: [] as string[],
              };
            }
            const resp = await api.deepdive(
              {
                config: configId,
                granularity,
                panel,
                mode: p.mode,
                metrics: p.metrics,
                ranges: reqRanges,
                group_values: cohortSelection,
                clip: p.clip,
                nbins: p.nbins,
                normalize: p.normalize,
                outliers_std: p.outliersStd,
              },
              signal,
            );
            return { histograms: resp.histograms, heatmaps: resp.heatmaps, scatters: resp.scatters, missing: resp.missing };
          },
          (r, set) => {
            _panelSig[key] = sig;
            set((s) => ({ deepdive: { ...s.deepdive, [panel]: { ...s.deepdive[panel], ...r } } }));
          },
          (e, set) => {
            set({ error: String(e) });
            get().notify("error", `Failed to load deep dive: ${e}`);
          },
        );
      }),
    );
  },

  // ── Agent chat ───────────────────────────────────────────────────────────

  toggleChat: () => set((s) => ({ chat: { ...s.chat, open: !s.chat.open } })),
  setChatWidth: (width) =>
    set((s) => ({ chat: { ...s.chat, width: Math.min(CHAT_MAX_WIDTH, Math.max(CHAT_MIN_WIDTH, width)) } })),
  setChatModel: (model) => set((s) => ({ chat: { ...s.chat, model } })),
  clearChat: () => {
    _chatAbort?.abort();
    set((s) => ({ chat: { ...emptyChat(), open: s.chat.open, width: s.chat.width, model: s.chat.model } }));
  },
  stopChat: () => {
    _chatAbort?.abort();
    set((s) => ({ chat: { ...s.chat, streaming: false } }));
  },

  sendChat: async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || get().chat.streaming) return;

    const patch = (fn: (c: ChatState) => Partial<ChatState>) => set((s) => ({ chat: { ...s.chat, ...fn(s.chat) } }));
    const patchLast = (fn: (m: ChatMsg) => ChatMsg) =>
      patch((c) => ({ messages: c.messages.map((m, i) => (i === c.messages.length - 1 ? fn(m) : m)) }));

    // History = prior turns only (the new user message travels separately).
    const history = get().chat.messages.map((m) => ({ role: m.role, content: m.content }));

    patch((c) => ({
      messages: [...c.messages, { role: "user", content: trimmed, events: [] }, { role: "assistant", content: "", events: [] }],
      streaming: true,
      pendingClarify: null,
    }));

    // Snapshot of what's on screen, so the agent can drive it (see _ACTION_GUIDE
    // on the backend). Panel entries pair available metrics with the selection.
    const s = get();
    const dashboard_state = {
      configs: s.configs.map((c) => c.id),
      configId: s.configId,
      activeTab: s.activeTab,
      granularities: s.config?.granularities ?? [],
      dateWindow: { from: s.controls.date.dateFrom, to: s.controls.date.dateTo, granularity: s.controls.date.granularity },
      panels: (s.config?.groups ?? []).map((g) => ({
        id: g.id,
        label: g.label,
        available_metrics: g.metrics,
        left: s.panels[g.id]?.left ?? [],
        right: s.panels[g.id]?.right ?? [],
      })),
    };

    _chatAbort?.abort();
    const ctl = new AbortController();
    _chatAbort = ctl;
    try {
      await streamAgentChat(
        {
          message: trimmed,
          history,
          model: get().chat.model,
          dashboard_config: s.configId ? `dashboard_config-${s.configId}.yaml` : null,
          dashboard_state,
        },
        {
          onToken: (t) => patchLast((m) => ({ ...m, content: m.content + t })),
          onToolStart: (name, input) =>
            patchLast((m) => ({ ...m, events: [...m.events, { kind: "tool", name, detail: input, done: false }] })),
          onToolEnd: (name) =>
            patchLast((m) => {
              // Mark the most recent unfinished call of this tool as done.
              let idx = -1;
              m.events.forEach((e, i) => {
                if (e.kind === "tool" && e.name === name && !e.done) idx = i;
              });
              if (idx < 0) return m;
              return { ...m, events: m.events.map((e, i) => (i === idx ? { ...e, done: true } : e)) };
            }),
          onAction: (action) => {
            const ok = dispatchAction(action);
            patchLast((m) => ({
              ...m,
              events: [...m.events, { kind: "action", name: String(action.type ?? "?"), detail: ok ? "applied" : "failed", done: true }],
            }));
          },
          onClarify: (question, options) => patch(() => ({ pendingClarify: { question, options } })),
          onDone: () => patch(() => ({ streaming: false })),
          onError: (msg) => {
            patchLast((m) => ({ ...m, content: m.content + (m.content ? "\n\n" : "") + `⚠️ ${msg}` }));
            patch(() => ({ streaming: false }));
          },
        },
        ctl.signal,
      );
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        patchLast((m) => ({
          ...m,
          content: m.content + (m.content ? "\n\n" : "") + `⚠️ Agent unreachable: ${e}. Is the ai_agent service running on :8051?`,
        }));
      }
    } finally {
      if (_chatAbort === ctl) _chatAbort = null;
      patch(() => ({ streaming: false }));
    }
  },
}));
