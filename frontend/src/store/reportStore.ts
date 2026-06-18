import { create } from "zustand";
import { uid } from "../lib/uid";
import type * as echarts from "echarts";
import { api } from "../api/client";
import { buildDataSummary } from "../charts/reportSummary";
import { useDashboardStore } from "./dashboardStore";
import type {
  CorrMatrix,
  DateRange,
  FigureSource,
  GroupStat,
  HistogramSeries,
  ReportFigure,
  ReportReference,
  ReportSpec,
  ScatterSeries,
  Series,
} from "../api/types";

// Report tab state (Phase 4). The working ReportSpec is the source of truth —
// figures are declarative recipes re-fetched from /api/data/* (NOT stored
// renders), which is what makes "regenerate with a new period" work. Fetched
// data and live chart instances are kept OUTSIDE the spec.

export interface FigureData {
  loading?: boolean;
  error?: string;
  series?: Series[];
  stats?: GroupStat[];
  histograms?: HistogramSeries[];
  heatmaps?: CorrMatrix[];
  scatters?: ScatterSeries[];
}

const emptySpec = (): ReportSpec => ({
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
});

const notify = (kind: "success" | "error" | "info", msg: string) => useDashboardStore.getState().notify(kind, msg);

// Live chart instances per figure id (registered by the cards via EChart
// onReady) — used to export PNGs. Not React state; never serialized.
const _charts: Record<string, echarts.ECharts | null> = {};

// Effective recipe for a render: figures with inherit_period follow the
// spec-level period (date window for the date kind; the single range for the
// ranged kinds).
function effectiveSource(fig: ReportFigure, period: DateRange | null | undefined): FigureSource {
  const src = fig.source;
  if (!fig.inherit_period || !period?.start || !period?.end) return src;
  if (src.kind === "stats-by-date") return { ...src, date_from: period.start, date_to: period.end };
  return { ...src, ranges: [{ start: period.start, end: period.end }] };
}

const hasPrompt = (text: string | undefined): boolean => /^\s*\/prompt\b/m.test(text ?? "");

interface ReportState {
  spec: ReportSpec;
  figureData: Record<string, FigureData>;
  specs: string[];
  busy: string | null; // human label of the in-flight LLM/export call

  patchSpec: (patch: Partial<ReportSpec>) => void;
  addFigure: (source: FigureSource, title: string) => void;
  removeFigure: (id: string) => void;
  patchFigure: (id: string, patch: Partial<ReportFigure>) => void;
  registerChart: (id: string, chart: echarts.ECharts | null) => void;

  renderFigure: (id: string) => Promise<void>;
  renderAll: () => Promise<void>;
  regenerate: () => Promise<void>;

  generateDescription: (id: string) => Promise<void>;
  generateSummary: () => Promise<void>;

  addReference: () => void;
  updateReference: (id: string, url: string) => Promise<void>;
  removeReference: (id: string) => void;

  loadSpecs: () => Promise<void>;
  saveSpec: (name: string) => Promise<void>;
  loadSpec: (name: string) => Promise<void>;

  // Apply a hand- or LLM-edited spec JSON: parse, normalize, re-render the
  // changed figures from it. Returns an error message, or null on success.
  applySpecJson: (json: string) => Promise<string | null>;
  // Same, from an already-parsed spec object (the agent's figure actions edit
  // the spec and route through here so normalize + render + the staleness
  // rule apply uniformly).
  applySpec: (parsed: unknown) => Promise<string | null>;

  exportReport: () => Promise<void>;
}

// Light normalization so hand/LLM-edited specs are forgiving: arrays exist,
// every figure has an id (so a new figure block can be added without inventing
// one), inherit_period defaults on. Field-level validation stays server-side
// (Pydantic on save) and per-card (render errors) — never breaks the tab.
function normalizeSpec(raw: unknown): ReportSpec {
  const r = (raw ?? {}) as Partial<ReportSpec>;
  return {
    version: r.version ?? 1,
    id: r.id ?? "",
    title: r.title ?? "",
    language: (["en", "zh-Hans", "zh-Hant"] as const).includes(r.language as never) ? (r.language as never) : "en",
    period: r.period ?? null,
    view: r.view ?? null,
    figures: (Array.isArray(r.figures) ? r.figures : []).map((f) => ({
      id: f?.id || uid(),
      title: f?.title ?? "",
      description: f?.description ?? "",
      inherit_period: f?.inherit_period ?? true,
      source: f?.source as ReportFigure["source"],
    })),
    references: (Array.isArray(r.references) ? r.references : []).map((ref) => ({
      ...ref,
      id: ref?.id || uid(),
      url: ref?.url ?? "",
    })),
    summary: r.summary ?? "",
    confluence_url: r.confluence_url ?? "",
  };
}

export const useReportStore = create<ReportState>((set, get) => ({
  spec: emptySpec(),
  figureData: {},
  specs: [],
  busy: null,

  patchSpec: (patch) => set((s) => ({ spec: { ...s.spec, ...patch } })),

  addFigure: (source, title) => {
    const fig: ReportFigure = {
      id: uid(),
      title,
      description: "",
      inherit_period: true,
      source,
    };
    set((s) => ({ spec: { ...s.spec, figures: [...s.spec.figures, fig] } }));
    void get().renderFigure(fig.id);
    notify("info", `Added “${title}” to the report (${get().spec.figures.length} figures)`);
  },

  removeFigure: (id) =>
    set((s) => {
      const { [id]: _, ...rest } = s.figureData;
      delete _charts[id];
      return { spec: { ...s.spec, figures: s.spec.figures.filter((f) => f.id !== id) }, figureData: rest };
    }),

  patchFigure: (id, patch) =>
    set((s) => ({
      spec: { ...s.spec, figures: s.spec.figures.map((f) => (f.id === id ? { ...f, ...patch } : f)) },
    })),

  registerChart: (id, chart) => {
    _charts[id] = chart;
  },

  // Re-fetch a figure's data from its recipe (independent of the live tabs).
  renderFigure: async (id) => {
    const { spec } = get();
    const fig = spec.figures.find((f) => f.id === id);
    if (!fig) return;
    const src = effectiveSource(fig, spec.period);
    const setData = (data: FigureData) => set((s) => ({ figureData: { ...s.figureData, [id]: data } }));
    setData({ loading: true });
    try {
      if (src.kind === "stats-by-date") {
        const metrics = [...new Set([...src.left, ...src.right])];
        if (metrics.length === 0) return setData({ series: [] });
        const resp = await api.series({
          config: src.config,
          granularity: src.granularity,
          metrics,
          date_from: src.date_from,
          date_to: src.date_to,
          group_values: src.cohort_selection,
        });
        setData({ series: resp.series });
      } else if (src.kind === "stats-by-group") {
        const resp = await api.groupDistribution({
          config: src.config,
          granularity: src.granularity,
          metric: src.metric,
          ranges: src.ranges,
          group_values: src.cohort_selection,
          clip: src.clip,
        });
        setData({ stats: resp.stats });
      } else {
        const resp = await api.deepdive({
          config: src.config,
          granularity: src.granularity,
          panel: src.panel,
          mode: src.mode,
          metrics: src.metrics,
          ranges: src.ranges,
          group_values: src.cohort_selection,
          clip: src.clip,
          nbins: src.nbins,
          normalize: src.normalize,
          outliers_std: src.outliers_std,
        });
        setData({ histograms: resp.histograms, heatmaps: resp.heatmaps, scatters: resp.scatters });
      }
    } catch (e) {
      setData({ error: String(e) });
    }
  },

  renderAll: async () => {
    await Promise.all(get().spec.figures.map((f) => get().renderFigure(f.id)));
  },

  // The "update the report with new data" lever: re-fetch every figure, then
  // refresh prose whose data changed — non-empty descriptions of
  // period-inheriting figures (their data follows the new period), plus any
  // /prompt-marked ones, plus a non-empty summary. Empty prose stays empty.
  regenerate: async () => {
    const { spec } = get();
    const periodSet = Boolean(spec.period?.start && spec.period?.end);
    await get().renderAll();
    for (const f of spec.figures) {
      const dataChanged = periodSet && f.inherit_period;
      if ((dataChanged && f.description.trim() !== "") || hasPrompt(f.description)) {
        await get().generateDescription(f.id);
      }
    }
    if (get().spec.summary.trim() !== "") await get().generateSummary();
    notify("success", "Report regenerated for the current period");
  },

  generateDescription: async (id) => {
    const { spec, figureData } = get();
    const fig = spec.figures.find((f) => f.id === id);
    const data = figureData[id];
    if (!fig || !data || data.loading || data.error) {
      notify("error", "Figure data not loaded yet — render it first");
      return;
    }
    set({ busy: `Generating description for “${fig.title}”…` });
    try {
      const resp = await api.generateDescription({
        data_summary: buildDataSummary(fig, data),
        existing_description: fig.description || null,
        references: spec.references.map((r) => ({ url: r.url, title: r.title, text: r.text, error: r.error })),
        language: spec.language,
      });
      get().patchFigure(id, { description: resp.text });
      notify("success", `Description ${resp.status} (${fig.title})`);
    } catch (e) {
      notify("error", `Description failed: ${e}`);
    } finally {
      set({ busy: null });
    }
  },

  generateSummary: async () => {
    const { spec, figureData } = get();
    if (spec.figures.length === 0) {
      notify("error", "Add figures before generating a summary");
      return;
    }
    set({ busy: "Generating summary…" });
    try {
      const resp = await api.generateSummary({
        figures: spec.figures.map((f) => ({
          data_summary: buildDataSummary(f, figureData[f.id] ?? {}),
          description: f.description,
        })),
        existing_summary: spec.summary || null,
        references: spec.references.map((r) => ({ url: r.url, title: r.title, text: r.text, error: r.error })),
        language: spec.language,
      });
      get().patchSpec({ summary: resp.text });
      notify("success", `Summary ${resp.status}`);
    } catch (e) {
      notify("error", `Summary failed: ${e}`);
    } finally {
      set({ busy: null });
    }
  },

  addReference: () =>
    set((s) => ({
      spec: { ...s.spec, references: [...s.spec.references, { id: uid(), url: "" }] },
    })),

  updateReference: async (id, url) => {
    const patchRef = (patch: Partial<ReportReference>) =>
      set((s) => ({
        spec: { ...s.spec, references: s.spec.references.map((r) => (r.id === id ? { ...r, ...patch } : r)) },
      }));
    patchRef({ url });
    if (!url.trim()) return patchRef({ status: null, title: null, text: null, error: null });
    try {
      const [ref] = await api.reportReferences([url.trim()]);
      patchRef({
        title: ref?.title ?? url,
        text: ref?.text ?? "",
        error: ref?.error ?? null,
        status: ref?.error ? "error" : url.includes("/wiki/") ? "ok" : "external",
      });
    } catch (e) {
      patchRef({ status: "error", error: String(e) });
    }
  },

  removeReference: (id) =>
    set((s) => ({ spec: { ...s.spec, references: s.spec.references.filter((r) => r.id !== id) } })),

  loadSpecs: async () => {
    try {
      set({ specs: await api.listReportSpecs() });
    } catch (e) {
      notify("error", `Failed to list report specs: ${e}`);
    }
  },

  saveSpec: async (name) => {
    try {
      const r = await api.saveReportSpec(name, { ...get().spec, id: name });
      set({ specs: r.specs });
      get().patchSpec({ id: r.name });
      notify("success", `Saved report spec “${r.name}” → ${r.path}`);
    } catch (e) {
      notify("error", `Failed to save report spec: ${e}`);
    }
  },

  loadSpec: async (name) => {
    try {
      const spec = normalizeSpec(await api.loadReportSpec(name));
      set({ spec, figureData: {} });
      // Restore the linked dashboard view (one view ↔ many reports) so the
      // whole authoring context comes back; figures render from their own
      // recipes regardless.
      if (spec.view) await useDashboardStore.getState().loadViewByName(spec.view);
      await get().renderAll();
      notify("info", `Loaded report spec “${name}”`);
    } catch (e) {
      notify("error", `Failed to load report spec “${name}”: ${e}`);
    }
  },

  applySpecJson: async (json) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch (e) {
      return `Invalid JSON: ${e instanceof Error ? e.message : e}`;
    }
    return get().applySpec(parsed);
  },

  applySpec: async (parsed) => {
    const bad = ((parsed as Partial<ReportSpec>)?.figures ?? []).find(
      (f) => !f?.source || !["stats-by-date", "stats-by-group", "stats-deepdive"].includes(f.source.kind),
    );
    if (bad) return `Figure "${bad?.title || bad?.id || "?"}" has a missing/unknown source.kind`;
    const prev = get().spec;
    const prevData = get().figureData;
    const spec = normalizeSpec(parsed);

    // Keep fetched data for figures whose effective recipe didn't change; only
    // changed/new/errored ones re-fetch (a one-figure patch costs one fetch).
    const kept: Record<string, FigureData> = {};
    const toRender: string[] = [];
    for (const f of spec.figures) {
      const old = prev.figures.find((o) => o.id === f.id);
      const unchanged =
        old && JSON.stringify(effectiveSource(f, spec.period)) === JSON.stringify(effectiveSource(old, prev.period));
      const data = prevData[f.id];
      if (unchanged && data && !data.error && !data.loading) kept[f.id] = data;
      else toRender.push(f.id);
    }
    set({ spec, figureData: kept });
    if (spec.view && spec.view !== prev.view) await useDashboardStore.getState().loadViewByName(spec.view);
    await Promise.all(toRender.map((id) => get().renderFigure(id)));

    // "Data changed, prose didn't → refresh the prose": auto-regenerate the
    // description of figures whose EFFECTIVE source changed in this edit,
    // unless the user also edited that description (their words win) or it's
    // empty (don't spontaneously write prose). Same rule for the summary.
    const staleFigs = spec.figures.filter((f) => {
      const old = prev.figures.find((o) => o.id === f.id);
      if (!old) return false; // new figure — author describes it when ready
      const srcChanged =
        JSON.stringify(effectiveSource(f, spec.period)) !== JSON.stringify(effectiveSource(old, prev.period));
      const descEdited = f.description !== old.description;
      return srcChanged && !descEdited && f.description.trim() !== "";
    });
    for (const f of staleFigs) await get().generateDescription(f.id);
    if (staleFigs.length > 0 && spec.summary === prev.summary && spec.summary.trim() !== "") {
      await get().generateSummary();
    }

    notify(
      "success",
      staleFigs.length > 0
        ? `Spec applied — re-rendered; refreshed ${staleFigs.length} stale description(s)`
        : "Spec applied — figures re-rendered",
    );
    return null;
  },

  exportReport: async () => {
    const { spec } = get();
    if (!spec.confluence_url.trim()) {
      notify("error", "Set the Confluence page URL first");
      return;
    }
    const figures = spec.figures
      .map((f) => {
        const chart = _charts[f.id];
        if (!chart) return null;
        const dataUrl = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fff" });
        return { title: f.title, description: f.description, png_base64: dataUrl.split(",")[1] ?? "" };
      })
      .filter((f): f is NonNullable<typeof f> => f !== null && f.png_base64.length > 0);
    if (figures.length === 0) {
      notify("error", "No rendered figures to export");
      return;
    }
    set({ busy: `Exporting ${figures.length} figures to Confluence…` });
    try {
      const r = await api.exportReport({
        confluence_url: spec.confluence_url,
        summary: spec.summary,
        references: spec.references,
        figures,
      });
      notify("success", `Exported ${r.figures_uploaded} figures → page v${r.page_version ?? "?"}`);
    } catch (e) {
      notify("error", `Export failed: ${e}`);
    } finally {
      set({ busy: null });
    }
  },
}));
