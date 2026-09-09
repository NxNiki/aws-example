import { useEffect, useMemo, useState } from "react";
import { EChart } from "../../charts/EChart";
import { buildStatsByDateOption } from "../../charts/statsByDateOption";
import { buildGroupDistributionOption, groupChartWidth } from "../../charts/groupDistributionOption";
import { buildHistogramOption } from "../../charts/histogramOption";
import { buildHeatmapOption, heatmapSize } from "../../charts/heatmapOption";
import { buildScatterOption } from "../../charts/scatterOption";
import { clipFilterInfo } from "../../charts/clipFilterInfo";
import { SummaryGrid } from "../summary-table/SummaryGrid";
import { useReportStore } from "../../store/reportStore";
import { useDashboardStore } from "../../store/dashboardStore";
import type { FigureData } from "../../store/reportStore";
import type { ReportFigure, ReportLanguage } from "../../api/types";

// Report tab (Phase 4): compose a report from figure RECIPES added on the data
// tabs, generate LLM descriptions/summary, save/load the spec (S3), regenerate
// for a new period, and export to Confluence. See docs/frontend_redesign.md §6.

const LANGUAGES: { value: ReportLanguage; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh-Hans", label: "中文（简体）" },
  { value: "zh-Hant", label: "中文（繁體）" },
];

function refPill(status?: string | null) {
  if (status === "ok") return <span className="text-green-600">✓</span>;
  if (status === "error") return <span className="text-red-600">✗</span>;
  if (status === "external") return <span className="text-gray-400">↗</span>;
  return null;
}

// Renders a figure card's chart from its recipe + fetched data. Only the first
// chart of multi-chart figures is registered for PNG export.
function FigureChart({ fig, data }: { fig: ReportFigure; data: FigureData }) {
  const registerChart = useReportStore((s) => s.registerChart);
  const src = fig.source;
  const onReady = (chart: Parameters<typeof registerChart>[1]) => registerChart(fig.id, chart);

  const option = useMemo(() => {
    if (src.kind === "stats-by-date" && data.series) {
      return buildStatsByDateOption(data.series, {
        leftMetrics: src.left,
        rightMetrics: src.right,
        log: src.log,
        threshold: src.threshold,
        granularity: src.granularity,
      });
    }
    if (src.kind === "stats-by-group" && data.stats) {
      return buildGroupDistributionOption(data.stats, src.mode, src.metric, clipFilterInfo(src.clip, src.filter));
    }
    return null;
  }, [src, data]);

  if (data.loading) return <div className="p-6 text-gray-400">Loading…</div>;
  if (data.error) return <div className="p-4 text-red-600">Failed to render: {data.error}</div>;
  if (option) {
    // Stats-by-Group bars carry the per-bar stat block (needs the source tab's
    // 400px) and a width sized to the bar count (wrapped to scroll).
    if (src.kind === "stats-by-group" && data.stats) {
      return (
        <div className="overflow-x-auto">
          <EChart option={option} height={400} width={groupChartWidth(data.stats.length)} onReady={onReady} />
        </div>
      );
    }
    return <EChart option={option} height={360} onReady={onReady} />;
  }

  if (src.kind === "summary-table" && data.columns && data.rows) {
    // Read-only grid (no reference radio); table figures aren't PNG-exported.
    const rows = data.rows.filter((r) => (src.metrics[r.group_id] ?? []).includes(r.metric));
    return (
      <SummaryGrid
        columns={data.columns}
        rows={rows}
        stats={src.stats}
        referenceKey={src.reference_key}
        showPValues={src.pvalues}
        metricOptions={src.metric_options}
      />
    );
  }

  if (src.kind === "stats-deepdive") {
    const info = clipFilterInfo(src.clip, src.filter);
    if (src.mode === "histogram" && data.histograms?.length) {
      const byMetric: Record<string, typeof data.histograms> = {};
      for (const h of data.histograms) (byMetric[h.metric] ??= []).push(h);
      return (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          {Object.entries(byMetric).map(([metric, series], i) => (
            <div key={metric}>
              <div className="text-sm text-gray-500">{metric}</div>
              <EChart
                option={buildHistogramOption(series, { logY: src.log_y, normalize: src.normalize, info })}
                height={300}
                onReady={i === 0 ? onReady : undefined}
              />
            </div>
          ))}
        </div>
      );
    }
    if (src.mode === "heatmap" && data.heatmaps?.length) {
      return (
        <div className="flex flex-col gap-3 overflow-x-auto">
          {data.heatmaps.map((m, i) => {
            const { width, height } = heatmapSize(m.metrics.length);
            return (
              <div key={i}>
                <div className="text-sm text-gray-500">
                  {m.cohort} · {m.range_label}
                </div>
                <EChart option={buildHeatmapOption(m, info)} width={width} height={height} onReady={i === 0 ? onReady : undefined} />
              </div>
            );
          })}
        </div>
      );
    }
    if (src.mode === "scatter" && data.scatters?.length) {
      const sm = data.scatters[0].metrics;
      if (sm.length >= 2) {
        return (
          <EChart
            option={buildScatterOption(data.scatters, 0, 1, {
              logX: src.scatter_log_x,
              logY: src.scatter_log_y,
              xLabel: sm[0],
              yLabel: sm[1],
              showLegend: true,
              info,
            })}
            width={520}
            height={520}
            onReady={onReady}
          />
        );
      }
    }
  }
  return <div className="p-4 text-gray-400">No data — check the recipe (config/metrics may have changed).</div>;
}

function FigureCard({ fig, index, total }: { fig: ReportFigure; index: number; total: number }) {
  const data = useReportStore((s) => s.figureData[fig.id] ?? {});
  const { patchFigure, removeFigure, generateDescription, renderFigure, moveFigure } = useReportStore();
  return (
    <div className="mb-6 rounded border bg-white p-4">
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <input
          className="min-w-0 flex-1 rounded border px-2 py-1 font-semibold"
          value={fig.title}
          onChange={(e) => patchFigure(fig.id, { title: e.target.value })}
        />
        <label className="flex items-center gap-1 text-sm text-gray-600" title="Follow the report period on Regenerate">
          <input
            type="checkbox"
            checked={fig.inherit_period}
            onChange={(e) => patchFigure(fig.id, { inherit_period: e.target.checked })}
          />
          inherit period
        </label>
        <button
          className="rounded border px-2 py-1 text-sm text-gray-600 disabled:opacity-30"
          title="Move figure up"
          disabled={index === 0}
          onClick={() => moveFigure(fig.id, -1)}
        >
          ↑
        </button>
        <button
          className="rounded border px-2 py-1 text-sm text-gray-600 disabled:opacity-30"
          title="Move figure down"
          disabled={index === total - 1}
          onClick={() => moveFigure(fig.id, 1)}
        >
          ↓
        </button>
        <button className="rounded border px-2 py-1 text-sm text-gray-600" onClick={() => void renderFigure(fig.id)}>
          ↻ re-render
        </button>
        <button className="rounded border px-2 py-1 text-sm text-red-600" onClick={() => removeFigure(fig.id)}>
          remove
        </button>
      </div>

      <FigureChart fig={fig} data={data} />

      <div className="mt-3">
        <div className="mb-1 flex items-center gap-3">
          <span className="text-sm text-gray-500">Description (use “/prompt …” lines to steer regeneration)</span>
          <button
            className="rounded border border-blue-300 bg-blue-50 px-2 py-0.5 text-sm text-blue-700 hover:bg-blue-100"
            onClick={() => void generateDescription(fig.id)}
          >
            Generate description
          </button>
        </div>
        <textarea
          className="min-h-24 w-full rounded border px-2 py-1 text-base"
          value={fig.description}
          onChange={(e) => patchFigure(fig.id, { description: e.target.value })}
        />
      </div>
    </div>
  );
}

// Collapsible JSON editor over the working spec: the spec IS the report (the
// data that recovers it), so editing it directly — by hand or via an LLM — and
// applying re-renders everything. Persisted as JSON in S3 (exact round-trip).
function SpecEditor() {
  const spec = useReportStore((s) => s.spec);
  const applySpecJson = useReportStore((s) => s.applySpecJson);
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    setDraft(JSON.stringify(spec, null, 2));
    setError(null);
  };

  return (
    <div className="mb-6 rounded border bg-white p-4">
      <div className="flex items-center gap-3">
        <h3 className="font-semibold text-gray-700">Spec (JSON)</h3>
        <button
          className="rounded border px-2 py-0.5 text-sm text-gray-600"
          onClick={() => {
            if (!open) refresh();
            setOpen(!open);
          }}
        >
          {open ? "hide" : "edit"}
        </button>
        {open && (
          <>
            <button className="rounded border px-2 py-0.5 text-sm text-gray-600" onClick={refresh} title="Reload from the current report state">
              ⟳ refresh
            </button>
            <button
              className="rounded border border-green-600 bg-green-50 px-2 py-0.5 text-sm text-green-700"
              onClick={() => void applySpecJson(draft).then(setError)}
            >
              Apply & render
            </button>
            <span className="text-sm text-gray-400">edit by hand or paste an LLM-edited spec</span>
          </>
        )}
      </div>
      {open && (
        <>
          {error && <div className="mt-2 rounded bg-red-50 px-2 py-1 text-sm text-red-700">{error}</div>}
          <textarea
            className="mt-2 h-96 w-full rounded border bg-gray-50 p-2 font-mono text-sm"
            spellCheck={false}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
        </>
      )}
    </div>
  );
}

export function ReportTab() {
  const r = useReportStore();
  const hasLifecycle = useDashboardStore((s) => Boolean(s.config?.lifecycle_col));
  const specName = r.specName;
  const setSpecName = r.setSpecName;

  useEffect(() => {
    void r.loadSpecs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="p-6 pt-0 w-full">
      {/* Pinned controls (below the sticky header + tab bar). */}
      <div
        className={`sticky ${hasLifecycle ? "top-[11.75rem]" : "top-[9rem]"} z-20 -mx-6 mb-6 flex flex-wrap items-end gap-4 border-b bg-gray-50 px-6 py-3`}
      >
        <div className="flex items-center gap-2">
          <select
            className="w-64 rounded border px-2 py-2"
            value=""
            onChange={(e) => {
              if (e.target.value) {
                setSpecName(e.target.value);
                void r.loadSpec(e.target.value);
              }
            }}
          >
            <option value="">Load spec…</option>
            {r.specs.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <input
            className="w-56 rounded border px-2 py-2"
            placeholder="spec name"
            value={specName}
            onChange={(e) => setSpecName(e.target.value)}
          />
          <button
            className="rounded border bg-blue-600 px-3 py-2 text-white disabled:opacity-40"
            disabled={!specName.trim()}
            onClick={() => void r.saveSpec(specName.trim())}
          >
            Save
          </button>
        </div>

        <input
          className="w-56 rounded border px-2 py-2"
          placeholder="Report title"
          value={r.spec.title}
          onChange={(e) => r.patchSpec({ title: e.target.value })}
        />

        <select
          className="rounded border px-2 py-2"
          value={r.spec.language}
          onChange={(e) => r.patchSpec({ language: e.target.value as ReportLanguage })}
        >
          {LANGUAGES.map((l) => (
            <option key={l.value} value={l.value}>
              {l.label}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-2" title="Figures with “inherit period” re-fetch this window on Regenerate">
          <span className="text-gray-600">Period</span>
          <input
            type="date"
            className="rounded border px-2 py-2"
            value={r.spec.period?.start ?? ""}
            onChange={(e) => r.patchSpec({ period: { start: e.target.value || null, end: r.spec.period?.end ?? null } })}
          />
          <span className="text-gray-400">→</span>
          <input
            type="date"
            className="rounded border px-2 py-2"
            value={r.spec.period?.end ?? ""}
            onChange={(e) => r.patchSpec({ period: { start: r.spec.period?.start ?? null, end: e.target.value || null } })}
          />
          <button
            className="rounded border bg-green-700 px-3 py-2 text-white"
            title="Re-fetch inheriting figures for the period; regenerate /prompt descriptions + summary"
            onClick={() => void r.regenerate()}
          >
            Regenerate
          </button>
        </div>

        {r.busy && <span className="animate-pulse text-blue-600">● {r.busy}</span>}
      </div>

      <SpecEditor />

      {/* Summary (kept above the figures) */}
      <div className="mb-6 rounded border bg-white p-4">
        <div className="mb-1 flex items-center gap-3">
          <h3 className="font-semibold text-gray-700">Summary</h3>
          <button
            className="rounded border border-blue-300 bg-blue-50 px-2 py-0.5 text-sm text-blue-700 hover:bg-blue-100"
            onClick={() => void r.generateSummary()}
          >
            Generate summary
          </button>
        </div>
        <textarea
          className="min-h-28 w-full rounded border px-2 py-1 text-base"
          value={r.spec.summary}
          onChange={(e) => r.patchSpec({ summary: e.target.value })}
        />
      </div>

      {r.spec.figures.length === 0 && (
        <div className="p-10 text-center text-gray-400">
          No figures yet — use <b>＋ Add to report</b> on any chart in the data tabs.
        </div>
      )}
      {r.spec.figures.map((f, i) => (
        <FigureCard key={f.id} fig={f} index={i} total={r.spec.figures.length} />
      ))}

      {/* References */}
      <div className="mb-6 rounded border bg-white p-4">
        <div className="mb-2 flex items-center gap-3">
          <h3 className="font-semibold text-gray-700">References</h3>
          <button className="rounded border px-2 py-0.5 text-sm text-gray-600" onClick={r.addReference}>
            + Add reference
          </button>
        </div>
        {r.spec.references.map((ref) => (
          <div key={ref.id} className="mb-2 flex items-center gap-2">
            <input
              className="flex-1 rounded border px-2 py-1"
              placeholder="https://…/wiki/… (Confluence) or external URL"
              value={ref.url}
              onChange={(e) => void r.updateReference(ref.id, e.target.value)}
            />
            <span title={ref.error ?? ref.title ?? ""}>{refPill(ref.status)}</span>
            <button className="rounded border px-2 py-1 text-sm text-red-600" onClick={() => r.removeReference(ref.id)}>
              remove
            </button>
          </div>
        ))}
      </div>

      {/* Export */}
      <div className="mb-10 rounded border bg-white p-4">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="font-semibold text-gray-700">Export</h3>
          <input
            className="min-w-0 flex-1 rounded border px-2 py-2"
            placeholder="Confluence page URL (report region is replaced; content above it is preserved)"
            value={r.spec.confluence_url}
            onChange={(e) => r.patchSpec({ confluence_url: e.target.value })}
          />
          <button
            className="rounded border bg-purple-700 px-4 py-2 text-white disabled:opacity-40"
            disabled={!r.spec.confluence_url.trim() || r.spec.figures.length === 0 || r.busy !== null}
            onClick={() => void r.exportReport()}
          >
            Export to Doc
          </button>
        </div>
      </div>
    </div>
  );
}
