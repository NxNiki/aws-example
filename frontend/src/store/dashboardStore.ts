import { create } from "zustand";
import { uid } from "../lib/uid";
import { api } from "../api/client";
import { streamAgentChat } from "../api/agentStream";
import { dispatchAction } from "./actionDispatcher";
// Cycle-safe: reportStore only references this store inside function bodies.
import { useReportStore } from "./reportStore";
import type { RangeState } from "../components/DateRanges";
import type { LifecycleGroupState } from "../components/LifecycleGroups";
import type { RangeGroupDef } from "../components/RangeGroupSelect";
import { type ClipFilterRows, noClip, noFilter } from "../components/clipFilter";
import type {
  ClipOpts,
  ConfigDetail,
  FilterOpts,
  ConfigSummary,
  CorrMatrix,
  DeepdiveMode,
  DeepdivePanel,
  Granularity,
  GroupStat,
  HistogramSeries,
  LifecycleGroup,
  RangeGroup,
  ScatterSeries,
  Series,
  SummaryColumn,
  SummaryMetricOption,
  SummaryRow,
  SummaryStat,
} from "../api/types";

// ── Global date groups + per-tab cohort controls ──────────────────────────
// Granularity and the (up to three) date windows are dashboard-wide — the
// "Date groups" bar above the tab selector — so every tab reads the same
// windows. Cohort selection stays per tab.

export interface DateGroupsState {
  granularity: Granularity;
  ranges: RangeState[]; // up to 3 comparable windows
}

export interface TabCohortControls {
  cohortSelection: Record<string, string[]>;
  // Per-tab checked labels of the value-range picker (e.g. Fish level); the
  // group DEFINITIONS are global (store.rangeGroups) so ranges mean the same
  // thing on every tab.
  rangeSelection: string[];
  // Per-tab checked labels of the "Total bet" (period-total) picker; the
  // definitions are global and per granularity (store.periodTotalGroups).
  periodTotalSelection: string[];
}

export interface TabControls {
  date: TabCohortControls;
  group: TabCohortControls;
  viz: TabCohortControls;
  summaryTable: TabCohortControls;
}
export type TabKey = keyof TabControls;

// Top-level dashboard tabs. Lives in the store (not component state) so the
// agent's navigate_tab action can drive it.
export type DashboardTab = "stats-by-date" | "stats-by-group" | "summary-table" | "stats-deepdive" | "report";

const defaultRanges = (): RangeState[] => [
  { start: null, end: null, show: true },
  { start: null, end: null, show: false },
  { start: null, end: null, show: false },
];

const defaultDateGroups = (): DateGroupsState => ({ granularity: "day", ranges: defaultRanges() });

const defaultControls = (): TabControls => ({
  date: { cohortSelection: {}, rangeSelection: [], periodTotalSelection: [] },
  group: { cohortSelection: {}, rangeSelection: [], periodTotalSelection: [] },
  viz: { cohortSelection: {}, rangeSelection: [], periodTotalSelection: [] },
  summaryTable: { cohortSelection: {}, rangeSelection: [], periodTotalSelection: [] },
});

// Seed the per-granularity "Total bet" definitions from the config; a
// granularity missing from the config's defaults falls back to the day list
// (edges rarely differ — most users play ~1 day per week/month).
export function periodTotalDefaults(config: ConfigDetail): Partial<Record<Granularity, RangeGroupDef[]>> {
  if (!config.period_total_col) return {};
  const d = config.period_total_defaults ?? {};
  const toDefs = (gs?: RangeGroup[] | null): RangeGroupDef[] =>
    (gs ?? []).map((g) => ({ label: g.label, min: g.min, max: g.max ?? null }));
  const day = toDefs(d["day"]);
  return {
    day,
    week: d["week"] ? toDefs(d["week"]) : day,
    month: d["month"] ? toDefs(d["month"]) : day,
  };
}

// ── Global lifecycle groups ────────────────────────────────────────────────
// UNLIKE the per-tab controls above, the lifecycle-group definitions are
// dashboard-wide (one definition per game, shared by every tab). There is one
// definition PER UNIT (day/week/month): ranges mean periods since first bet in
// the fetching tab's granularity, and the bar shows/edits the active tab's
// unit. Defaults mirror the ETL's stored day cohorts (week/month analogs pick
// whole periods); nothing shown → no lifecycle filter, so the dashboard
// behaves exactly as before the picker existed.
export type LifecycleByUnit = Record<Granularity, LifecycleGroupState[]>;

// Ranges are half-open [start, end) — see LifecycleGroups.tsx.
const defaultLifecycle = (): LifecycleByUnit => ({
  day: [
    { label: "new", start: 0, end: 3, show: false },
    { label: "beginner", start: 3, end: 7, show: false },
    { label: "old", start: 7, end: null, show: false },
  ],
  week: [
    { label: "new", start: 0, end: 1, show: false },
    { label: "beginner", start: 1, end: 2, show: false },
    { label: "old", start: 2, end: null, show: false },
  ],
  month: [
    { label: "new", start: 0, end: 1, show: false },
    { label: "beginner", start: 1, end: 2, show: false },
    { label: "old", start: 2, end: null, show: false },
  ],
});

// Range suffix appended to each group's label in half-open interval notation —
// "new[0, 3)", "old[7, max)" — so chart legends / table columns show the
// definition, not just the name. The label is what the backend uses as the
// cohort key, so the tag flows everywhere (series, summary columns, report
// figures) for free.
function rangeTag(g: LifecycleGroupState): string {
  return `[${g.start}, ${g.end ?? "max"})`;
}

// The lifecycle_groups request payload for a tab fetching at `granularity`, or
// undefined when the config has no lifecycle dimension / nothing is toggled on
// (→ the backend falls back to plain "all", i.e. pre-picker behavior).
export function activeLifecycleGroups(
  s: Pick<DashboardState, "config" | "lifecycle" | "lifecycleAll">,
  granularity: Granularity,
): LifecycleGroup[] | undefined {
  if (!s.config?.lifecycle_col) return undefined;
  const groups = (s.lifecycle[granularity] ?? [])
    .filter((g) => g.show && g.label.trim() && (g.end === null || g.end > g.start))
    .map((g) => ({ label: `${g.label.trim()}${rangeTag(g)}`, start: g.start, end: g.end }));
  if (groups.length === 0) return undefined;
  return s.lifecycleAll ? [{ label: "all", start: 0, end: null }, ...groups] : groups;
}

// An explicitly EMPTIED picker (key present, no values) means "show nothing" —
// the user unchecked everything, so tabs render no data rather than silently
// falling back to 'all' (an ABSENT key still means 'all' for direct API
// calls). Pickers seed to ['all'] when cohort values load.
export function hasEmptiedCohort(selection: Record<string, string[]>): boolean {
  return Object.values(selection).some((v) => Array.isArray(v) && v.length === 0);
}

// One rule for every dimension picker — cohorts, range groups (fish/bet
// level), lifecycle groups: each starts with an explicit 'all' selected, and
// fully unselecting any of them means "show no data" on that tab.
export function nothingSelected(
  s: Pick<DashboardState, "config" | "controls" | "lifecycle" | "lifecycleAll" | "dateGroups">,
  tab: TabKey,
): boolean {
  if (hasEmptiedCohort(s.controls[tab].cohortSelection)) return true;
  if (s.config?.range_group_col && s.controls[tab].rangeSelection.length === 0) return true;
  if (s.config?.period_total_col && (s.controls[tab].periodTotalSelection ?? []).length === 0) return true;
  if (s.config?.lifecycle_col && !s.lifecycleAll) {
    const shown = (s.lifecycle[s.dateGroups.granularity] ?? []).some((g) => g.show && g.label.trim());
    if (!shown) return true;
  }
  return false;
}

// The union window of the visible date ranges, for range-scoped cohort
// availability; null until at least one shown range is fully specified.
export function overallDateRange(dg: DateGroupsState): { start: string; end: string } | null {
  const complete = dg.ranges.filter((r) => r.show && r.start && r.end);
  if (!complete.length) return null;
  const start = complete.map((r) => r.start!).sort()[0];
  const end = complete.map((r) => r.end!).sort().slice(-1)[0];
  return { start, end };
}

// Cohort columns for the per-tab pickers: the lifecycle column is owned by
// the global picker and the range column by RangeGroupSelect, so both are
// hidden from CohortSelect.
export function visibleGroupValues(
  config: ConfigDetail | null,
  values: Record<string, string[]>,
): Record<string, string[]> {
  const rest = { ...values };
  for (const col of [config?.lifecycle_col, config?.range_group_col, config?.period_total_col]) {
    if (col && col in rest) delete rest[col];
  }
  return rest;
}

// Options for the range-group [min, max] pickers: the config-declared value
// ladder when present (the game's stable stake menu), else the distinct
// values of the range column present in the loaded data.
export function rangeGroupValues(
  s: Pick<DashboardState, "config" | "groupValuesByGran">,
  granularity: Granularity,
): number[] {
  const col = s.config?.range_group_col;
  if (!col) return [];
  const declared = (s.config?.range_group_values ?? []).filter(Number.isFinite);
  if (declared.length) return [...declared].sort((a, b) => a - b);
  return (s.groupValuesByGran[granularity]?.[col] ?? [])
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
}

// The range_groups request payload for one tab: its checked labels resolved
// against the global definitions, labels tagged with the inclusive range
// ("low[0, 10]"); undefined when the config has no range column or nothing
// is checked (backend treats that as plain "all").
export function activeRangeGroups(
  s: Pick<DashboardState, "config" | "rangeGroups">,
  selection: string[],
): RangeGroup[] | undefined {
  if (!s.config?.range_group_col || selection.length === 0) return undefined;
  const groups: RangeGroup[] = [];
  for (const label of selection) {
    if (label === "all") {
      groups.push({ label: "all", min: 0, max: null });
      continue;
    }
    const def = s.rangeGroups.find((g) => g.label === label);
    if (def && def.label.trim() && (def.max === null || def.max >= def.min)) {
      groups.push({ label: `${def.label.trim()}[${def.min}, ${def.max ?? "max"}]`, min: def.min, max: def.max });
    }
  }
  return groups.length ? groups : undefined;
}

// The "Total bet" picker's request payload for one tab: checked labels
// resolved against the granularity's global definitions, each entry tagged
// with the derived column so the backend routes it to the period-total
// dimension (entries without a column address the stored range column).
// Period-total ranges are HALF-OPEN [min, max) — adjacent tiers partition
// exactly, like lifecycle groups — so labels are tagged "label[min, max)".
export function activePeriodTotalGroups(
  s: Pick<DashboardState, "config" | "periodTotalGroups">,
  selection: string[],
  granularity: Granularity,
): RangeGroup[] | undefined {
  const col = s.config?.period_total_col;
  if (!col || selection.length === 0) return undefined;
  const defs = s.periodTotalGroups[granularity] ?? [];
  const groups: RangeGroup[] = [];
  for (const label of selection) {
    if (label === "all") {
      groups.push({ label: "all", column: col, min: 0, max: null });
      continue;
    }
    const def = defs.find((g) => g.label === label);
    // Strict max > min: with a right-exclusive bound, max == min is empty.
    // Legend label is just the range — the tier names (spend_mid, ...) are
    // picker handles, and dropping them keeps chart legends short.
    if (def && def.label.trim() && (def.max === null || def.max > def.min)) {
      groups.push({
        label: `[${def.min}, ${def.max ?? "\u221e"})`,
        column: col,
        min: def.min,
        max: def.max,
      });
    }
  }
  return groups.length ? groups : undefined;
}

// Merge the payloads of both range-style pickers for a tab's request.
export function activeRangeDimensions(
  s: Pick<DashboardState, "config" | "rangeGroups" | "periodTotalGroups">,
  c: TabCohortControls,
  granularity: Granularity,
): RangeGroup[] | undefined {
  const stored = activeRangeGroups(s, c.rangeSelection) ?? [];
  const derived = activePeriodTotalGroups(s, c.periodTotalSelection ?? [], granularity) ?? [];
  const all = [...stored, ...derived];
  return all.length ? all : undefined;
}

// ── Per-panel state ────────────────────────────────────────────────────────

// Stats-by-Date: which metrics go on each axis + the hybrid-log toggle.
export interface PanelState {
  left: string[];
  right: string[];
  log: boolean;
  threshold: number;
  series: Series[];
}

// Stats-by-Group: one metric, box-or-bar, clip/filter; results are per (cohort×range).
export interface GroupPanelState {
  metric: string | null;
  mode: "box" | "bar";
  clip: ClipOpts;
  filter: FilterOpts;
  clipFilterRows: ClipFilterRows; // inactive rows' bound memory (see components/clipFilter.ts)
  stats: GroupStat[];
  missing: boolean;
}

// Summary table: all metrics × all (cohort×range) columns in one grid. The
// fetched data is `columns`/`rows`; the rest are display options. `stats` and
// `referenceKey` are pure display (no refetch); `showPValues` triggers a refetch
// because the significance column is computed server-side.
export interface SummaryTableState {
  metrics: Record<string, string[]>; // per metric-group (group id → selected metrics); display filter
  metricOptions: Record<string, SummaryMetricOption>; // per-metric clip + log (refetch)
  stats: SummaryStat[]; // which cell stats to show (count / mean / median / quartiles / min / max)
  showPValues: boolean;
  referenceKey: string | null; // column the ±% values are measured against
  columns: SummaryColumn[];
  rows: SummaryRow[];
  missing: boolean; // every metric came back empty for the current selection
}

// Deep Dive: a panel (derived/user) with a mode + metric multi-select.
export interface DeepdivePanelState {
  mode: DeepdiveMode;
  metrics: string[];
  nbins: number;
  logY: boolean; // histogram count axis (display)
  normalize: boolean;
  clip: ClipOpts;
  filter: FilterOpts;
  clipFilterRows: ClipFilterRows; // inactive rows' bound memory (see components/clipFilter.ts)
  outliersStd: number | null; // scatter: drop rows beyond N std (null = off)
  scatterLogX: boolean; // scatter x axis (display)
  scatterLogY: boolean; // scatter y axis (display)
  histograms: HistogramSeries[];
  heatmaps: CorrMatrix[];
  scatters: ScatterSeries[];
  missing: string[];
}

const DEFAULT_THRESHOLD = 10;

function defaultPanels(config: ConfigDetail): Record<string, PanelState> {
  // Only the FIRST panel starts with a metric selected, so first launch fires
  // one series request instead of one per panel. Metrics picked later live in
  // the store and survive tab switches; a config switch resets to defaults.
  const panels: Record<string, PanelState> = {};
  config.groups.forEach((g, i) => {
    panels[g.id] = {
      left: i === 0 ? g.metrics.slice(0, 1) : [],
      right: [],
      log: false,
      threshold: DEFAULT_THRESHOLD,
      series: [],
    };
  });
  return panels;
}

const defaultGroupPanel = (): GroupPanelState => ({
  metric: null,
  mode: "bar",
  clip: noClip(),
  filter: noFilter(),
  clipFilterRows: {},
  stats: [],
  missing: false,
});

function defaultGroupPanels(config: ConfigDetail): Record<string, GroupPanelState> {
  // Same first-launch policy as Stats-by-Date: only the FIRST panel computes
  // by default (num_active_users when the panel offers it), later panels
  // start at <None> — each selected group metric costs a per-(cohort×range)
  // CI computation server-side.
  const panels: Record<string, GroupPanelState> = {};
  config.groups.forEach((g, i) => {
    const first = g.metrics.includes("num_active_users") ? "num_active_users" : (g.metrics[0] ?? null);
    panels[g.id] = { ...defaultGroupPanel(), metric: i === 0 ? first : null };
  });
  return panels;
}

const emptySummaryTable = (): SummaryTableState => ({
  metrics: {},
  metricOptions: {},
  stats: ["mean"],
  showPValues: false,
  referenceKey: null,
  columns: [],
  rows: [],
  missing: false,
});

// First metric of each group — the summary table's default (keep it small; the
// per-group selectors let the user add more).
function defaultSummaryMetrics(config: ConfigDetail): Record<string, string[]> {
  const metrics: Record<string, string[]> = {};
  for (const g of config.groups) metrics[g.id] = g.metrics.slice(0, 1);
  return metrics;
}

const emptyDeepdivePanel = (): DeepdivePanelState => ({
  mode: "histogram",
  metrics: [],
  nbins: 50,
  logY: false,
  normalize: false,
  clip: noClip(),
  filter: noFilter(),
  clipFilterRows: {},
  outliersStd: null,
  scatterLogX: false,
  scatterLogY: false,
  histograms: [],
  heatmaps: [],
  scatters: [],
  missing: [],
});

function normalizeDeepdivePanel(v?: Partial<DeepdivePanelState>): DeepdivePanelState {
  return {
    ...emptyDeepdivePanel(),
    ...(v ?? {}),
    // pre-percentile snapshots lack the flag; defaults keep the old meanings
    clip: { ...noClip(), ...(v?.clip ?? {}) },
    filter: { ...noFilter(), ...(v?.filter ?? {}) },
  };
}

// A serializable snapshot of the dashboard's UI selections (NOT fetched data /
// figures) — the React-era "save/load config". Persisted opaquely to S3.
// v3 stores global `dateGroups` + cohorts-only `controls`; v2 (per-tab
// granularity/dates) and v1 (one global set) are migrated by
// `snapshotDateGroups`/`snapshotControls`.
export interface ViewSnapshot {
  version: number;
  configId: string | null;
  dateGroups?: DateGroupsState;
  controls?: TabControls & {
    // v2 legacy per-tab fields (still read for migration):
    date?: { granularity?: Granularity; dateFrom?: string | null; dateTo?: string | null };
    group?: { granularity?: Granularity; ranges?: RangeState[] };
  };
  // v1 legacy fields (still read for migration):
  granularity?: Granularity;
  dateFrom?: string | null;
  dateTo?: string | null;
  ranges?: RangeState[];
  cohortSelection?: Record<string, string[]>;
  panels: Record<string, PanelState>;
  group: Record<string, GroupPanelState>;
  deepdive: Record<DeepdivePanel, DeepdivePanelState>;
  summaryTable?: SummaryTableState; // optional: pre-summary-table snapshots omit it
  // Global lifecycle groups (optional: pre-lifecycle snapshots omit them; an
  // early-format plain array is read as the day-unit definition).
  lifecycle?: LifecycleByUnit | LifecycleGroupState[];
  lifecycleAll?: boolean;
  rangeGroups?: RangeGroupDef[];
  // Global "Total bet" definitions per granularity (optional: older snapshots omit them).
  periodTotalGroups?: Partial<Record<Granularity, RangeGroupDef[]>>;
  // Active report-spec name when the view was saved; loading the view reloads it.
  reportSpec?: string | null;
}

function snapshotControls(snap: ViewSnapshot): TabControls {
  const pick = (t?: {
    cohortSelection?: Record<string, string[]>;
    rangeSelection?: string[];
    periodTotalSelection?: string[];
  }): TabCohortControls => ({
    cohortSelection: t?.cohortSelection ?? snap.cohortSelection ?? {},
    rangeSelection: t?.rangeSelection ?? [],
    periodTotalSelection: t?.periodTotalSelection ?? [],
  });
  const c = (snap.controls ?? {}) as Partial<Record<TabKey, { cohortSelection?: Record<string, string[]> }>>;
  return { date: pick(c.date), group: pick(c.group), viz: pick(c.viz), summaryTable: pick(c.summaryTable) };
}

function snapshotDateGroups(snap: ViewSnapshot): DateGroupsState {
  if (snap.dateGroups) {
    return { ...defaultDateGroups(), ...snap.dateGroups };
  }
  // v2: per-tab controls — take the Stats-by-Date tab's granularity; prefer the
  // comparison tabs' ranges, else build R1 from the date tab's window.
  const v2date = snap.controls?.date;
  const v2ranges = snap.controls?.group?.ranges ?? snap.ranges;
  const granularity = v2date?.granularity ?? snap.granularity ?? "day";
  if (v2ranges?.some((r) => r.start && r.end)) {
    return { granularity, ranges: v2ranges.slice(0, 3) };
  }
  const from = v2date?.dateFrom ?? snap.dateFrom ?? null;
  const to = v2date?.dateTo ?? snap.dateTo ?? null;
  const ranges = defaultRanges();
  ranges[0] = { start: from, end: to, show: true };
  return { granularity, ranges };
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
  // Global "Date groups": granularity + up to 3 date windows, defined once per
  // game (bar above the tab selector) and read by every tab's fetches.
  dateGroups: DateGroupsState;
  setDateGranularity: (g: Granularity) => void;
  setDateRange: (index: number, range: RangeState) => void;
  // Available cohort values per granularity (the data can differ by
  // granularity; keyed in case the user flips granularities back and forth).
  groupValuesByGran: Partial<Record<Granularity, Record<string, string[]>>>;
  // Range-scoped availability: cohort values absent from the current date
  // range render grayed/disabled. null = no range known (everything enabled).
  groupAvailableByGran: Partial<Record<Granularity, Record<string, string[]> | null>>;
  // "start|end" key the availability was fetched for, to skip refetches.
  groupAvailRangeByGran: Partial<Record<Granularity, string>>;

  // Global lifecycle groups (period ranges since first bet, one definition per
  // day/week/month unit): defined once per game, shared by every tab.
  // `lifecycleAll` overlays the full population.
  lifecycle: LifecycleByUnit;
  lifecycleAll: boolean;
  setLifecycleGroup: (unit: Granularity, index: number, group: LifecycleGroupState) => void;
  setLifecycleAll: (all: boolean) => void;

  // Global value-range group definitions (config.range_group_defaults seeds
  // them); selection is per tab in controls[tab].rangeSelection.
  rangeGroups: RangeGroupDef[];
  setRangeGroup: (index: number, group: RangeGroupDef) => void;
  setTabRangeSelection: (tab: TabKey, labels: string[]) => void;

  // Global "Total bet" (period-total) group definitions, one set per
  // granularity (edges can differ by day/week/month, like lifecycle units);
  // config.period_total_defaults seeds them. Selection is per tab in
  // controls[tab].periodTotalSelection.
  periodTotalGroups: Partial<Record<Granularity, RangeGroupDef[]>>;
  setPeriodTotalGroup: (unit: Granularity, index: number, group: RangeGroupDef) => void;
  setTabPeriodTotalSelection: (tab: TabKey, labels: string[]) => void;

  panels: Record<string, PanelState>; // Stats-by-Date
  group: Record<string, GroupPanelState>; // Stats-by-Group
  summaryTable: SummaryTableState; // Summary table
  deepdiveMetrics: { derived: string[]; user: string[] };
  deepdive: Record<DeepdivePanel, DeepdivePanelState>;

  views: string[]; // names of saved view snapshots
  currentView: string | null; // last saved/loaded view name (reports link to it)
  notifications: Toast[]; // transient status messages (toasts)
  loading: boolean;
  status: string | null; // what the dashboard is currently doing (header indicator)
  error: string | null;

  loadConfigs: () => Promise<void>;
  selectConfig: (id: string) => Promise<void>;
  setTabCohort: (tab: TabKey, col: string, values: string[]) => void;
  ensureGroupValues: (gran: Granularity) => Promise<void>;
  _fetchGroupValues: (
    gran: Granularity,
    configId: string,
    range: { start: string; end: string } | null,
    rangeKey: string,
  ) => Promise<void>;

  // Stats-by-Date
  setPanelMetrics: (panelId: string, side: "left" | "right", metrics: string[]) => void;
  setPanelLog: (panelId: string, log: boolean, threshold?: number) => void;
  loadDateBounds: () => Promise<void>;
  // Long-lived sessions: re-check the data edge (ETL lands while the tab is
  // open) and slide the default window's end forward if the user hasn't
  // moved it off the previous edge.
  dataMax: string | null;
  refreshDateBounds: () => Promise<void>;
  loadAllSeries: () => Promise<void>;

  // Stats-by-Group
  setGroupMetric: (panelId: string, metric: string | null) => void;
  setGroupMode: (panelId: string, mode: "box" | "bar") => void;
  setGroupClipFilter: (panelId: string, patch: { clip: ClipOpts; filter: FilterOpts; clipFilterRows: ClipFilterRows }) => void;
  loadGroupDistribution: () => Promise<void>;

  // Summary table
  setSummaryMetrics: (groupId: string, metrics: string[]) => void;
  setSummaryMetricOption: (metric: string, option: SummaryMetricOption) => void;
  setSummaryStats: (stats: SummaryStat[]) => void;
  setSummaryPValues: (show: boolean) => void;
  setSummaryReference: (key: string | null) => void;
  loadSummaryTable: () => Promise<void>;

  // Deep Dive
  loadDeepdiveMetrics: () => Promise<void>;
  patchDeepdive: (panel: DeepdivePanel, patch: Partial<DeepdivePanelState>) => void;
  loadDeepdive: () => Promise<void>;

  // Saved views (save/load the dashboard setting)
  captureView: () => ViewSnapshot;
  applyView: (snap: ViewSnapshot, opts?: { loadReport?: boolean }) => Promise<boolean>;
  loadViews: () => Promise<void>;
  saveView: (name: string) => Promise<void>;
  loadViewByName: (name: string, loadReport?: boolean) => Promise<void>;

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
// Per-panel signature of the request currently in flight (cleared on land),
// so duplicate loadAllSeries invocations don't abort-and-reissue identical
// fetches. Distinct from _panelCtx, which records the last APPLIED context.
const _panelInflight: Record<string, string> = {};
// In-flight group-values/availability fetches keyed (config|gran|range):
// concurrent callers (setDateRange, tab effects) share one request and can
// await its completion.
const _groupValuesInflight: Record<string, Promise<void>> = {};

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
  dateGroups: defaultDateGroups(),
  setDateGranularity: (g) => {
    set((s) => ({ dateGroups: { ...s.dateGroups, granularity: g } }));
    void get().ensureGroupValues(g);
  },
  setDateRange: (index, range) => {
    set((s) => ({
      dateGroups: { ...s.dateGroups, ranges: s.dateGroups.ranges.map((r, i) => (i === index ? range : r)) },
    }));
    void get().ensureGroupValues(get().dateGroups.granularity);
  },
  groupValuesByGran: {},
  groupAvailableByGran: {},
  groupAvailRangeByGran: {},
  lifecycle: defaultLifecycle(),
  lifecycleAll: true,
  setLifecycleGroup: (unit, index, group) =>
    set((s) => ({
      lifecycle: { ...s.lifecycle, [unit]: s.lifecycle[unit].map((g, i) => (i === index ? group : g)) },
    })),
  setLifecycleAll: (all) => set({ lifecycleAll: all }),
  rangeGroups: [],
  setRangeGroup: (index, group) =>
    set((s) => ({ rangeGroups: s.rangeGroups.map((g, i) => (i === index ? group : g)) })),
  setTabRangeSelection: (tab, labels) =>
    set((s) => ({ controls: { ...s.controls, [tab]: { ...s.controls[tab], rangeSelection: labels } } })),
  periodTotalGroups: {},
  setPeriodTotalGroup: (unit, index, group) =>
    set((s) => ({
      periodTotalGroups: {
        ...s.periodTotalGroups,
        [unit]: (s.periodTotalGroups[unit] ?? []).map((g, i) => (i === index ? group : g)),
      },
    })),
  setTabPeriodTotalSelection: (tab, labels) =>
    set((s) => ({ controls: { ...s.controls, [tab]: { ...s.controls[tab], periodTotalSelection: labels } } })),
  panels: {},
  group: {},
  summaryTable: emptySummaryTable(),
  deepdiveMetrics: { derived: [], user: [] },
  deepdive: { derived: emptyDeepdivePanel(), user: emptyDeepdivePanel() },
  views: [],
  currentView: null,
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
      groupAvailableByGran: {},
      groupAvailRangeByGran: {},
      summaryTable: emptySummaryTable(),
      deepdiveMetrics: { derived: [], user: [] },
      deepdive: { derived: emptyDeepdivePanel(), user: emptyDeepdivePanel() },
      // Keep each tab's granularity/windows; cohort values are config-specific.
      controls: {
        date: { ...s.controls.date, cohortSelection: {} },
        group: { ...s.controls.group, cohortSelection: {} },
        viz: { ...s.controls.viz, cohortSelection: {} },
        summaryTable: { ...s.controls.summaryTable, cohortSelection: {} },
      },
      // Lifecycle definitions are per-game — back to defaults on a switch.
      lifecycle: defaultLifecycle(),
      lifecycleAll: true,
    }));
    try {
      const config = await api.getConfig(id);
      // Range-group picker starts with an explicit 'all' (same rule as the
      // cohort pickers); configs without a range dimension keep it empty.
      const seededRange = config.range_group_col ? ["all"] : [];
      const seededPeriodTotal = config.period_total_col ? ["all"] : [];
      set((s) => ({
        config,
        controls: {
          date: { ...s.controls.date, rangeSelection: seededRange, periodTotalSelection: seededPeriodTotal },
          group: { ...s.controls.group, rangeSelection: seededRange, periodTotalSelection: seededPeriodTotal },
          viz: { ...s.controls.viz, rangeSelection: seededRange, periodTotalSelection: seededPeriodTotal },
          summaryTable: {
            ...s.controls.summaryTable,
            rangeSelection: seededRange,
            periodTotalSelection: seededPeriodTotal,
          },
        },
        rangeGroups: (config.range_group_defaults ?? []).map((g) => ({ label: g.label, min: g.min, max: g.max ?? null })),
        periodTotalGroups: periodTotalDefaults(config),
        panels: defaultPanels(config),
        group: defaultGroupPanels(config),
        summaryTable: { ...emptySummaryTable(), metrics: defaultSummaryMetrics(config) },
      }));
      await get().ensureGroupValues(get().dateGroups.granularity);
      await get().loadDeepdiveMetrics();
      await get().loadDateBounds();
      await get().loadAllSeries();
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  setTabCohort: (tab, col, values) =>
    set((s) => ({
      controls: {
        ...s.controls,
        [tab]: { ...s.controls[tab], cohortSelection: { ...s.controls[tab].cohortSelection, [col]: values } },
      },
    })),

  // Tabs AWAIT this before loading their data, so the pickers' grayed-out
  // (availability) state always updates BEFORE the plots for a new date
  // range. Concurrent callers of the same (config, granularity, range) share
  // one request, and the fetch shows in the header's loading status.
  ensureGroupValues: async (gran) => {
    const { configId, groupValuesByGran, groupAvailRangeByGran, dateGroups } = get();
    if (!configId) return;
    const range = overallDateRange(dateGroups);
    const rangeKey = range ? `${range.start}|${range.end}` : "";
    if (groupValuesByGran[gran] && groupAvailRangeByGran[gran] === rangeKey) return;
    const flightKey = `${configId}|${gran}|${rangeKey}`;
    const inflight = _groupValuesInflight[flightKey];
    if (inflight) return inflight;
    const request = get()._fetchGroupValues(gran, configId, range, rangeKey);
    _groupValuesInflight[flightKey] = request;
    _inflight += 1;
    set({ loading: true, status: "updating group labels…" });
    try {
      await request;
    } finally {
      delete _groupValuesInflight[flightKey];
      _inflight = Math.max(0, _inflight - 1);
      set(_inflight > 0 ? { loading: true } : { loading: false, status: null });
    }
  },

  _fetchGroupValues: async (gran, configId, range, rangeKey) => {
    try {
      const { values, available } = await api.groupValues(configId, gran, range?.start, range?.end);
      set((s) => {
        // A slow response from a config the user has already switched away
        // from must be dropped: applying it would render the OLD game's
        // cohort pickers (e.g. fishhunter's fish_value under ss03) until the
        // new config's values arrive and overwrite them.
        if (s.configId !== configId) return {};
        // Seed 'all' as the explicit selection for every picker that has no
        // selection yet, so the UI state matches what the request means and
        // an emptied picker (user unchecked everything) can mean "no data".
        const cols = Object.keys(visibleGroupValues(s.config, values));
        const controls = { ...s.controls };
        for (const tab of Object.keys(controls) as (keyof typeof controls)[]) {
          const sel = { ...controls[tab].cohortSelection };
          let changed = false;
          for (const c of cols) {
            if (!(c in sel)) {
              sel[c] = ["all"];
              changed = true;
            }
          }
          if (changed) controls[tab] = { ...controls[tab], cohortSelection: sel };
        }
        return {
          groupValuesByGran: { ...s.groupValuesByGran, [gran]: values },
          groupAvailableByGran: { ...s.groupAvailableByGran, [gran]: available ?? null },
          groupAvailRangeByGran: { ...s.groupAvailRangeByGran, [gran]: rangeKey },
          controls,
        };
      });
    } catch (e) {
      if (get().configId !== configId) return;
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
    const { configId, dateGroups } = get();
    if (!configId) return;
    try {
      const { max } = await api.dateBounds(configId, dateGroups.granularity);
      if (get().configId !== configId) return;
      if (max) {
        const { from, to } = lastThirtyDays(max);
        set((s) => ({
          dataMax: max,
          dateGroups: {
            ...s.dateGroups,
            ranges: s.dateGroups.ranges.map((r, i) => (i === 0 ? { ...r, start: from, end: to } : r)),
          },
        }));
      }
    } catch (e) {
      if (get().configId !== configId) return;
      set({ error: String(e) });
      get().notify("error", `Failed to load date bounds: ${e}`);
    }
  },

  dataMax: null,

  refreshDateBounds: async () => {
    const { configId, dateGroups, dataMax } = get();
    if (!configId || !dataMax) return;
    try {
      const { max } = await api.dateBounds(configId, dateGroups.granularity);
      if (get().configId !== configId) return;
      // FORWARD-ONLY: at week/month granularity the max is a period-START
      // label (e.g. Monday 07-27 while day data reaches 07-29), so a naive
      // comparison would drag the window backward. Only ever advance.
      if (!max || max <= dataMax) return;
      const untouched = get().dateGroups.ranges[0].end === dataMax;
      set((s) => ({
        dataMax: max,
        dateGroups: untouched
          ? {
              ...s.dateGroups,
              ranges: s.dateGroups.ranges.map((r, i) => (i === 0 ? { ...r, end: max } : r)),
            }
          : s.dateGroups,
      }));
      if (untouched) {
        get().notify("info", `New data through ${max} — date window updated`);
        await get().ensureGroupValues(get().dateGroups.granularity);
        void get().loadAllSeries();
      }
    } catch {
      // Periodic best-effort check; the next tick retries.
    }
  },

  // Fetch each Stats-by-Date panel independently AND incrementally: when only the
  // metric set changed (same context), fetch just the newly-added metrics and
  // append them; removed metrics are hidden by the option-builder and kept cached
  // (instant re-add). A context change (config/granularity/dates/cohorts) reloads
  // the whole panel. Each panel has its own cancellation key.
  loadAllSeries: async () => {
    const { configId, controls, dateGroups, panels } = get();
    if (!configId) return;
    const { granularity } = dateGroups;
    const { cohortSelection } = controls.date;
    if (nothingSelected(get(), "date")) {
      set((s) => ({
        panels: Object.fromEntries(Object.entries(s.panels).map(([id, p]) => [id, { ...p, series: [] }])),
      }));
      return;
    }
    const reqRanges = activeRanges(dateGroups.ranges);
    if (reqRanges.length === 0) return;
    const lifecycleGroups = activeLifecycleGroups(get(), granularity);
    const rangeGroups = activeRangeDimensions(get(), controls.date, granularity);
    const ctxSig = JSON.stringify({ configId, granularity, reqRanges, cohortSelection, lifecycleGroups, rangeGroups });
    await Promise.all(
      Object.entries(panels).map(([panelId, panel]) => {
        const key = `series:${panelId}`;
        const ctxChanged = _panelCtx[key] !== ctxSig;
        const desired = [...new Set([...panel.left, ...panel.right])];
        const loaded = ctxChanged ? new Set<string>() : new Set(panel.series.map((s) => s.metric));
        const toFetch = desired.filter((m) => !loaded.has(m));

        // Startup fires loadAllSeries from both selectConfig and the tab
        // effect; an identical request already in flight would be aborted and
        // re-issued (wasted round trip, and the abort churn is where empty
        // panels can linger). Let the in-flight one land instead.
        const flightSig = ctxSig + "|" + toFetch.join(",");
        if (_panelInflight[key] === flightSig) return;

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

        _panelInflight[key] = flightSig;
        return runExclusive(
          key,
          "loading metrics…",
          set,
          async (signal): Promise<Series[]> => {
            const resp = await api.series(
              {
                config: configId,
                granularity,
                metrics: toFetch,
                ranges: reqRanges,
                group_values: cohortSelection,
                lifecycle_groups: lifecycleGroups,
                range_groups: rangeGroups,
              },
              signal,
            );
            return resp.series;
          },
          (fetched, set) => {
            if (_panelInflight[key] === flightSig) delete _panelInflight[key];
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
  setGroupClipFilter: (panelId, patch) =>
    set((s) => (s.group[panelId] ? { group: { ...s.group, [panelId]: { ...s.group[panelId], ...patch } } } : {})),

  loadGroupDistribution: async () => {
    const { configId, controls, dateGroups, group } = get();
    if (!configId) return;
    const { granularity } = dateGroups;
    const { cohortSelection } = controls.group;
    if (nothingSelected(get(), "group")) {
      set((s) => ({
        group: Object.fromEntries(
          Object.entries(s.group).map(([id, p]) => [id, { ...p, stats: [], missing: false }]),
        ),
      }));
      return;
    }
    const reqRanges = activeRanges(dateGroups.ranges);
    if (reqRanges.length === 0) return;
    const lifecycleGroups = activeLifecycleGroups(get(), granularity);
    const rangeGroups = activeRangeDimensions(get(), controls.group, granularity);
    await Promise.all(
      Object.entries(group).map(([panelId, p]) => {
        const key = `group:${panelId}`;
        const sig = JSON.stringify({
          configId,
          granularity,
          reqRanges,
          cohortSelection,
          lifecycleGroups,
          rangeGroups,
          metric: p.metric,
          clip: p.clip,
          filter: p.filter,
        });
        if (_panelSig[key] === sig) return;
        return runExclusive(
          key,
          "CI bootstrapping…",
          set,
          async (signal) => {
            if (!p.metric) return { stats: [] as GroupStat[], missing: false };
            const resp = await api.groupDistribution(
              {
                config: configId,
                granularity,
                metric: p.metric,
                ranges: reqRanges,
                group_values: cohortSelection,
                lifecycle_groups: lifecycleGroups,
                range_groups: rangeGroups,
                clip: p.clip,
                filter: p.filter,
              },
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

  setSummaryMetrics: (groupId, metrics) =>
    set((s) => ({ summaryTable: { ...s.summaryTable, metrics: { ...s.summaryTable.metrics, [groupId]: metrics } } })),
  setSummaryMetricOption: (metric, option) =>
    set((s) => ({
      summaryTable: { ...s.summaryTable, metricOptions: { ...s.summaryTable.metricOptions, [metric]: option } },
    })),
  setSummaryStats: (stats) => set((s) => ({ summaryTable: { ...s.summaryTable, stats } })),
  setSummaryPValues: (show) => set((s) => ({ summaryTable: { ...s.summaryTable, showPValues: show } })),
  setSummaryReference: (key) => set((s) => ({ summaryTable: { ...s.summaryTable, referenceKey: key } })),

  // Fetch the whole summary grid in one request (all metrics × all columns).
  // Only config / granularity / ranges / cohorts / p-values toggle force a
  // refetch; stat-selection and the reference column are display-only.
  loadSummaryTable: async () => {
    const { configId, controls, dateGroups, summaryTable } = get();
    if (!configId) return;
    const { granularity } = dateGroups;
    const { cohortSelection } = controls.summaryTable;
    if (nothingSelected(get(), "summaryTable")) {
      set((s) => ({ summaryTable: { ...s.summaryTable, columns: [], rows: [], missing: false } }));
      return;
    }
    const reqRanges = activeRanges(dateGroups.ranges);
    if (reqRanges.length === 0) return;
    const lifecycleGroups = activeLifecycleGroups(get(), granularity);
    const rangeGroups = activeRangeDimensions(get(), controls.summaryTable, granularity);
    const key = "summary-table";
    const sig = JSON.stringify({
      configId,
      granularity,
      reqRanges,
      cohortSelection,
      lifecycleGroups,
      rangeGroups,
      metricOptions: summaryTable.metricOptions,
      pvalues: summaryTable.showPValues,
    });
    if (_panelSig[key] === sig) return;
    return runExclusive(
      key,
      summaryTable.showPValues ? "computing summary + p-values…" : "computing summary…",
      set,
      async (signal) =>
        api.summaryTable(
          {
            config: configId,
            granularity,
            ranges: reqRanges,
            group_values: cohortSelection,
            lifecycle_groups: lifecycleGroups,
            range_groups: rangeGroups,
            metric_options: summaryTable.metricOptions,
            pvalues: summaryTable.showPValues,
          },
          signal,
        ),
      (resp, set) => {
        _panelSig[key] = sig;
        set((s) => {
          // Keep the chosen reference if it's still a column; else pin to the first.
          const keys = resp.columns.map((c) => c.key);
          const referenceKey =
            s.summaryTable.referenceKey && keys.includes(s.summaryTable.referenceKey)
              ? s.summaryTable.referenceKey
              : (keys[0] ?? null);
          const missing = resp.rows.length > 0 && resp.rows.every((r) => r.missing);
          return { summaryTable: { ...s.summaryTable, columns: resp.columns, rows: resp.rows, referenceKey, missing } };
        });
      },
      (e, set) => {
        set({ error: String(e) });
        get().notify("error", `Failed to load summary table: ${e}`);
      },
    );
  },

  loadDeepdiveMetrics: async () => {
    const { configId, dateGroups } = get();
    if (!configId) return;
    try {
      const { derived, user } = await api.deepdiveMetrics(configId, dateGroups.granularity);
      if (get().configId !== configId) return;
      set({ deepdiveMetrics: { derived, user } });
    } catch (e) {
      if (get().configId !== configId) return;
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
      version: 3,
      configId: s.configId,
      dateGroups: s.dateGroups,
      controls: s.controls,
      lifecycle: s.lifecycle,
      lifecycleAll: s.lifecycleAll,
      rangeGroups: s.rangeGroups,
      periodTotalGroups: s.periodTotalGroups,
      panels: Object.fromEntries(Object.entries(s.panels).map(([k, v]) => [k, { ...v, series: [] }])),
      group: Object.fromEntries(Object.entries(s.group).map(([k, v]) => [k, { ...v, stats: [], missing: false }])),
      // Keep the display options (stats / p-values / reference); drop fetched data.
      summaryTable: { ...s.summaryTable, columns: [], rows: [], missing: false },
      deepdive: {
        derived: { ...s.deepdive.derived, histograms: [], heatmaps: [], scatters: [], missing: [] },
        user: { ...s.deepdive.user, histograms: [], heatmaps: [], scatters: [], missing: [] },
      },
      // Remember the active (saved) report spec so loading this view reloads it.
      reportSpec: useReportStore.getState().spec.id || null,
    };
  },

  // Restore a snapshot: load the config (for group/granularity metadata), merge
  // saved selections over fresh defaults (so renamed/added groups still work),
  // then let the tabs refetch. v1 snapshots are migrated by snapshotControls.
  // Cohort selections for columns the config doesn't define (or the lifecycle
  // column, which the global picker owns) are stripped rather than replayed as
  // invisible filters, so a stale view can't silently skew the data.
  // Re-saving the view under the same name persists the cleaned state.
  applyView: async (snap, opts) => {
    if (!snap?.configId) return false;
    set({ loading: true, error: null });
    try {
      const config = await api.getConfig(snap.configId);
      const validCols = new Set(config.user_group_cols.filter((c) => c !== config.lifecycle_col));
      // Same explicit-'all' rule as live pickers: seed every picker-visible
      // column, then let the snapshot's non-empty selections override. Saved
      // empty selections are treated as 'all' (pre-explicit-defaults views).
      const pickerCols = config.user_group_cols.filter(
        (c) => c !== config.lifecycle_col && c !== config.range_group_col && c !== config.period_total_col,
      );
      const sanitize = <
        T extends { cohortSelection: Record<string, string[]>; rangeSelection?: string[]; periodTotalSelection?: string[] },
      >(
        t: T,
      ): T => ({
        ...t,
        cohortSelection: {
          ...Object.fromEntries(pickerCols.map((c) => [c, ["all"]])),
          ...Object.fromEntries(
            Object.entries(t.cohortSelection ?? {}).filter(([col, v]) => validCols.has(col) && (v?.length ?? 0) > 0),
          ),
        },
        rangeSelection:
          config.range_group_col && !(t.rangeSelection ?? []).length ? ["all"] : (t.rangeSelection ?? []),
        periodTotalSelection:
          config.period_total_col && !(t.periodTotalSelection ?? []).length
            ? ["all"]
            : (t.periodTotalSelection ?? []),
      });
      const raw = snapshotControls(snap);
      const controls: TabControls = {
        date: sanitize(raw.date),
        group: sanitize(raw.group),
        viz: sanitize(raw.viz),
        summaryTable: sanitize(raw.summaryTable),
      };
      set({
        configId: snap.configId,
        config,
        controls,
        dateGroups: snapshotDateGroups(snap),
        // Early snapshots stored a plain array (day-unit only); merge over defaults.
        lifecycle: Array.isArray(snap.lifecycle)
          ? { ...defaultLifecycle(), day: snap.lifecycle }
          : { ...defaultLifecycle(), ...(snap.lifecycle ?? {}) },
        lifecycleAll: snap.lifecycleAll ?? true,
        rangeGroups:
          snap.rangeGroups ??
          (config.range_group_defaults ?? []).map((g) => ({ label: g.label, min: g.min, max: g.max ?? null })),
        periodTotalGroups: snap.periodTotalGroups ?? periodTotalDefaults(config),
        groupValuesByGran: {},
        groupAvailableByGran: {},
        groupAvailRangeByGran: {},
        panels: { ...defaultPanels(config), ...(snap.panels ?? {}) },
        // Merge each saved panel over a default one so snapshots from before a
        // field existed (e.g. pre-filter) still restore with complete state.
        group: {
          ...defaultGroupPanels(config),
          ...Object.fromEntries(
            Object.entries(snap.group ?? {}).map(([k, v]) => [
              k,
              {
                ...defaultGroupPanel(),
                ...v,
                // pre-percentile snapshots lack the flag; defaults keep the old meanings
                clip: { ...noClip(), ...(v.clip ?? {}) },
                filter: { ...noFilter(), ...(v.filter ?? {}) },
              },
            ]),
          ),
        },
        summaryTable: {
          ...emptySummaryTable(),
          ...(snap.summaryTable ?? {}),
          metrics: snap.summaryTable?.metrics ?? defaultSummaryMetrics(config),
          columns: [],
          rows: [],
          missing: false,
        },
        deepdive: {
          derived: normalizeDeepdivePanel(snap.deepdive?.derived),
          user: normalizeDeepdivePanel(snap.deepdive?.user),
        },
      });
      await get().ensureGroupValues(get().dateGroups.granularity);
      await get().loadDeepdiveMetrics();
      await get().loadAllSeries();
      // Reload the report spec this view was saved with (unless a spec load is
      // what triggered this view load — restoreView=false path avoids a cycle).
      const rep = useReportStore.getState();
      if (opts?.loadReport !== false && snap.reportSpec && snap.reportSpec !== rep.spec.id) {
        await rep.loadSpec(snap.reportSpec, false);
      }
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
      set({ views: r.views, currentView: r.name });
      get().notify("success", `Saved view “${r.name}” → ${r.path}`);
    } catch (e) {
      set({ error: String(e) });
      get().notify("error", `Failed to save view: ${e}`);
    }
  },

  loadViewByName: async (name, loadReport = true) => {
    let snap: ViewSnapshot;
    try {
      snap = (await api.loadView(name)) as ViewSnapshot;
    } catch (e) {
      set({ error: String(e) });
      get().notify("error", `Failed to load view “${name}”: ${e}`);
      return;
    }
    if (await get().applyView(snap, { loadReport })) {
      set({ currentView: name });
      get().notify("info", `Loaded view “${name}”`);
    }
  },

  notify: (kind, message) =>
    set((s) => ({
      notifications: [...s.notifications, { id: uid(), kind, message }].slice(-5),
    })),

  dismissNotification: (id) => set((s) => ({ notifications: s.notifications.filter((n) => n.id !== id) })),

  loadDeepdive: async () => {
    const { configId, controls, dateGroups, deepdive } = get();
    if (!configId) return;
    const { granularity } = dateGroups;
    const { cohortSelection } = controls.viz;
    if (nothingSelected(get(), "viz")) {
      set((s) => ({
        deepdive: {
          derived: { ...s.deepdive.derived, histograms: [], heatmaps: [], scatters: [], missing: [] },
          user: { ...s.deepdive.user, histograms: [], heatmaps: [], scatters: [], missing: [] },
        },
      }));
      return;
    }
    const reqRanges = activeRanges(dateGroups.ranges);
    if (reqRanges.length === 0) return;
    const lifecycleGroups = activeLifecycleGroups(get(), granularity);
    const rangeGroups = activeRangeGroups(get(), controls.viz.rangeSelection);
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
          lifecycleGroups,
          rangeGroups,
          mode: p.mode,
          metrics: p.metrics,
          nbins: p.nbins,
          normalize: p.normalize,
          clip: p.clip,
          filter: p.filter,
          outliersStd: p.outliersStd,
        });
        if (_panelSig[key] === sig) return;
        const label =
          p.mode === "histogram"
            ? "binning histograms…"
            : p.mode === "heatmap"
              ? "computing correlations…"
              : "sampling points…";
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
                lifecycle_groups: lifecycleGroups,
                range_groups: rangeGroups,
                clip: p.clip,
                filter: p.filter,
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
    // The report section comes BEFORE panels: the backend truncates the tail of
    // this JSON, and panels carry the bulk. description/summary are previews —
    // skills.md forbids the model from copying them back into patches.
    const s = get();
    const rep = useReportStore.getState();
    if (rep.specs.length === 0) void rep.loadSpecs(); // names ready by the next turn even if the Report tab was never opened
    const dashboard_state = {
      configs: s.configs.map((c) => c.id),
      configId: s.configId,
      activeTab: s.activeTab,
      granularities: s.config?.granularities ?? [],
      dateWindow: {
        ranges: activeRanges(s.dateGroups.ranges),
        granularity: s.dateGroups.granularity,
      },
      report: {
        specs: rep.specs,
        title: rep.spec.title,
        language: rep.spec.language,
        period: rep.spec.period,
        view: rep.spec.view,
        summary_preview: rep.spec.summary.slice(0, 200),
        figures: rep.spec.figures.map((f) => ({
          id: f.id,
          title: f.title,
          inherit_period: f.inherit_period,
          description_preview: f.description.slice(0, 150),
          render_error: rep.figureData[f.id]?.error ?? null,
          source: f.source,
        })),
        reference_urls: rep.spec.references.map((r) => r.url),
      },
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
