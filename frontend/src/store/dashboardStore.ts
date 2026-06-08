import { create } from "zustand";
import { api } from "../api/client";
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
const defaultRanges = (): RangeState[] => [
  { start: null, end: null, show: true },
  { start: null, end: null, show: false },
  { start: null, end: null, show: false },
];

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

  granularity: Granularity;
  dateFrom: string | null; // Stats-by-Date single window
  dateTo: string | null;
  ranges: RangeState[]; // Stats-by-Group + Deep Dive: up to 3 windows
  groupValues: Record<string, string[]>;
  cohortSelection: Record<string, string[]>;

  panels: Record<string, PanelState>; // Stats-by-Date
  group: Record<string, GroupPanelState>; // Stats-by-Group
  deepdiveMetrics: { derived: string[]; user: string[] };
  deepdive: Record<DeepdivePanel, DeepdivePanelState>;

  loading: boolean;
  error: string | null;

  loadConfigs: () => Promise<void>;
  selectConfig: (id: string) => Promise<void>;
  setGranularity: (g: Granularity) => void;
  setDateRange: (from: string | null, to: string | null) => void;
  setRange: (index: number, range: RangeState) => void;
  setCohort: (col: string, values: string[]) => void;

  // Stats-by-Date
  setPanelMetrics: (panelId: string, side: "left" | "right", metrics: string[]) => void;
  setPanelLog: (panelId: string, log: boolean, threshold?: number) => void;
  loadGroupValues: () => Promise<void>;
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
}

function activeRanges(ranges: RangeState[]): { start: string; end: string }[] {
  return ranges.filter((r) => r.show && r.start && r.end).map((r) => ({ start: r.start as string, end: r.end as string }));
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  configs: [],
  configId: null,
  config: null,
  granularity: "day",
  dateFrom: null,
  dateTo: null,
  ranges: defaultRanges(),
  groupValues: {},
  cohortSelection: {},
  panels: {},
  group: {},
  deepdiveMetrics: { derived: [], user: [] },
  deepdive: { derived: emptyDeepdivePanel(), user: emptyDeepdivePanel() },
  loading: false,
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
    set({
      loading: true,
      error: null,
      configId: id,
      cohortSelection: {},
      groupValues: {},
      deepdiveMetrics: { derived: [], user: [] },
      deepdive: { derived: emptyDeepdivePanel(), user: emptyDeepdivePanel() },
    });
    try {
      const config = await api.getConfig(id);
      set({ config, panels: defaultPanels(config), group: defaultGroupPanels(config) });
      await get().loadGroupValues();
      await get().loadDeepdiveMetrics();
      await get().loadDateBounds();
      await get().loadAllSeries();
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  setGranularity: (g) => set({ granularity: g }),
  setDateRange: (from, to) => set({ dateFrom: from, dateTo: to }),
  setRange: (index, range) =>
    set((s) => ({ ranges: s.ranges.map((r, i) => (i === index ? range : r)) })),
  setCohort: (col, values) => set((s) => ({ cohortSelection: { ...s.cohortSelection, [col]: values } })),

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

  loadGroupValues: async () => {
    const { configId, granularity } = get();
    if (!configId) return;
    try {
      const { values } = await api.groupValues(configId, granularity);
      set({ groupValues: values });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  loadDateBounds: async () => {
    const { configId, granularity } = get();
    if (!configId) return;
    try {
      const { max } = await api.dateBounds(configId, granularity);
      if (max) {
        const { from, to } = lastThirtyDays(max);
        set((s) => ({
          dateFrom: from,
          dateTo: to,
          ranges: s.ranges.map((r, i) => (i === 0 ? { ...r, start: from, end: to } : r)),
        }));
      }
    } catch (e) {
      set({ error: String(e) });
    }
  },

  loadAllSeries: async () => {
    const { configId, granularity, dateFrom, dateTo, cohortSelection, panels } = get();
    if (!configId) return;
    set({ loading: true, error: null });
    try {
      const entries = await Promise.all(
        Object.entries(panels).map(async ([panelId, panel]) => {
          const metrics = [...new Set([...panel.left, ...panel.right])];
          if (metrics.length === 0) return [panelId, [] as Series[]] as const;
          const resp = await api.series({
            config: configId,
            granularity,
            metrics,
            date_from: dateFrom,
            date_to: dateTo,
            group_values: cohortSelection,
          });
          return [panelId, resp.series] as const;
        }),
      );
      set((s) => {
        const next = { ...s.panels };
        for (const [panelId, series] of entries) if (next[panelId]) next[panelId] = { ...next[panelId], series };
        return { panels: next };
      });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  setGroupMetric: (panelId, metric) =>
    set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], metric } } } : {})),
  setGroupMode: (panelId, mode) =>
    set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], mode } } } : {})),
  setGroupClip: (panelId, clip) =>
    set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], clip } } } : {})),

  loadGroupDistribution: async () => {
    const { configId, granularity, ranges, cohortSelection, group } = get();
    if (!configId) return;
    const reqRanges = activeRanges(ranges);
    if (reqRanges.length === 0) return;
    set({ loading: true, error: null });
    try {
      const entries = await Promise.all(
        Object.entries(group).map(async ([panelId, p]) => {
          if (!p.metric) return [panelId, { stats: [] as GroupStat[], missing: false }] as const;
          const resp = await api.groupDistribution({
            config: configId,
            granularity,
            metric: p.metric,
            ranges: reqRanges,
            group_values: cohortSelection,
            clip: p.clip,
          });
          return [panelId, { stats: resp.stats, missing: resp.missing }] as const;
        }),
      );
      set((s) => {
        const next = { ...s.group };
        for (const [panelId, r] of entries) if (next[panelId]) next[panelId] = { ...next[panelId], ...r };
        return { group: next };
      });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  loadDeepdiveMetrics: async () => {
    const { configId, granularity } = get();
    if (!configId) return;
    try {
      const { derived, user } = await api.deepdiveMetrics(configId, granularity);
      set({ deepdiveMetrics: { derived, user } });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  patchDeepdive: (panel, patch) =>
    set((s) => ({ deepdive: { ...s.deepdive, [panel]: { ...s.deepdive[panel], ...patch } } })),

  loadDeepdive: async () => {
    const { configId, granularity, ranges, cohortSelection, deepdive } = get();
    if (!configId) return;
    const reqRanges = activeRanges(ranges);
    if (reqRanges.length === 0) return;
    set({ loading: true, error: null });
    try {
      const panels: DeepdivePanel[] = ["derived", "user"];
      const entries = await Promise.all(
        panels.map(async (panel) => {
          const p = deepdive[panel];
          if (p.metrics.length === 0) {
            return [
              panel,
              { histograms: [] as HistogramSeries[], heatmaps: [] as CorrMatrix[], scatters: [] as ScatterSeries[], missing: [] as string[] },
            ] as const;
          }
          const resp = await api.deepdive({
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
          });
          return [
            panel,
            { histograms: resp.histograms, heatmaps: resp.heatmaps, scatters: resp.scatters, missing: resp.missing },
          ] as const;
        }),
      );
      set((s) => {
        const next = { ...s.deepdive };
        for (const [panel, r] of entries) next[panel] = { ...next[panel], ...r };
        return { deepdive: next };
      });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },
}));
