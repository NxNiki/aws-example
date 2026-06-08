import { useEffect } from "react";
import { EChart } from "../../charts/EChart";
import { buildHeatmapOption, heatmapSize } from "../../charts/heatmapOption";
import { buildHistogramOption } from "../../charts/histogramOption";
import { CohortSelect } from "../../components/CohortSelect";
import { ClipControls } from "../../components/ClipControls";
import { DateRanges } from "../../components/DateRanges";
import { MetricCheckList } from "../../components/MetricCheckList";
import { useDashboardStore } from "../../store/dashboardStore";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import type { DeepdivePanel as PanelId, Granularity, HistogramSeries } from "../../api/types";
import type { DeepdivePanelState } from "../../store/dashboardStore";

// Deep Dive tab: distribution (histogram) and correlation (heatmap) views over
// selected metrics, for the derived (group-level) and user-level panels. Mirrors
// the legacy "Stats Deepdive" tab (scatter mode deferred); data from
// /api/data/deepdive. p-value stars / Yeo-Johnson are deferred (need scipy).
const NBINS = [50, 100, 200, 500];

function DeepdivePanelView(props: {
  title: string;
  options: string[];
  panel: DeepdivePanelState;
  onPatch: (patch: Partial<DeepdivePanelState>) => void;
}) {
  const { panel } = props;

  const histsByMetric: Record<string, HistogramSeries[]> = {};
  for (const h of panel.histograms) (histsByMetric[h.metric] ??= []).push(h);

  return (
    <div className="border rounded p-3 mb-6">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">{props.title}</h3>
      <div className="flex gap-4">
        <div className="flex flex-col gap-3 w-72 shrink-0">
          <div className="flex gap-3 text-xs text-gray-600">
            {(["histogram", "heatmap"] as const).map((m) => (
              <label key={m} className="flex items-center gap-1 cursor-pointer">
                <input type="radio" checked={panel.mode === m} onChange={() => props.onPatch({ mode: m })} />
                {m}
              </label>
            ))}
          </div>
          <MetricCheckList
            label="Metrics"
            options={props.options}
            selected={panel.metrics}
            onChange={(metrics) => props.onPatch({ metrics })}
            maxHeightClass="max-h-72"
          />
          {panel.mode === "histogram" && (
            <div className="flex flex-col gap-2 text-xs text-gray-600">
              <label className="flex items-center gap-2">
                bins
                <select
                  className="border rounded px-2 py-1"
                  value={panel.nbins}
                  onChange={(e) => props.onPatch({ nbins: Number(e.target.value) })}
                >
                  {NBINS.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={panel.logY} onChange={(e) => props.onPatch({ logY: e.target.checked })} />
                log y
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={panel.normalize}
                  onChange={(e) => props.onPatch({ normalize: e.target.checked })}
                />
                normalize (density)
              </label>
            </div>
          )}
          <ClipControls clip={panel.clip} onChange={(clip) => props.onPatch({ clip })} />
          {panel.missing.length > 0 && (
            <span className="text-xs text-amber-600">No data: {panel.missing.join(", ")}</span>
          )}
          {panel.mode === "heatmap" && (
            <span className="text-[11px] text-gray-400">Pearson r (significance/transform deferred)</span>
          )}
        </div>

        <div className="flex-1 min-w-0">
          {panel.metrics.length === 0 ? (
            <div className="text-sm text-gray-400 p-6">Select one or more metrics.</div>
          ) : panel.mode === "histogram" ? (
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(histsByMetric).map(([metric, series]) => (
                <div key={metric}>
                  <div className="text-xs text-gray-500 mb-1">{metric}</div>
                  <EChart option={buildHistogramOption(series, { logY: panel.logY, normalize: panel.normalize })} height={380} />
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col gap-6">
              {panel.heatmaps.map((m, i) => {
                const { width, height } = heatmapSize(m.metrics.length);
                return (
                  <div key={i} className="overflow-x-auto">
                    <div className="text-xs text-gray-500 mb-1">
                      {m.cohort} · {m.range_label}
                    </div>
                    <EChart option={buildHeatmapOption(m)} width={width} height={height} />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function DeepDive() {
  const s = useDashboardStore();

  // Refetch on config / granularity / range / cohort / per-panel
  // mode/metrics/nbins/normalize/clip changes (log-y is display-only).
  const fetchKey = JSON.stringify({
    cfg: s.configId,
    gran: s.granularity,
    ranges: s.ranges.map((r) => [r.start, r.end, r.show]),
    cohorts: s.cohortSelection,
    panels: (["derived", "user"] as PanelId[]).map((id) => {
      const p = s.deepdive[id];
      return [id, p.mode, p.metrics, p.nbins, p.normalize, p.clip];
    }),
  });
  const debouncedKey = useDebouncedValue(fetchKey);
  useEffect(() => {
    if (s.configId) void s.loadDeepdive();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  return (
    <div className="p-6 w-full">
      <div className="flex flex-wrap gap-6 items-start mb-6">
        <label className="flex flex-col text-sm">
          <span className="text-gray-600 mb-1">Granularity</span>
          <select
            className="border rounded px-3 py-2"
            value={s.granularity}
            onChange={(e) => s.setGranularity(e.target.value as Granularity)}
          >
            {(s.config?.granularities ?? ["day"]).map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </label>
        <DateRanges ranges={s.ranges} onChange={s.setRange} />
        <CohortSelect groupValues={s.groupValues} selection={s.cohortSelection} onSetCohort={s.setCohort} />
      </div>

      {s.error && <div className="mb-4 rounded bg-red-50 text-red-700 text-sm px-3 py-2">{s.error}</div>}
      {s.loading && <div className="mb-4 text-sm text-gray-500">Loading…</div>}

      <DeepdivePanelView
        title="Derived metrics (group-level)"
        options={s.deepdiveMetrics.derived}
        panel={s.deepdive.derived}
        onPatch={(patch) => s.patchDeepdive("derived", patch)}
      />
      <DeepdivePanelView
        title="User-level metrics"
        options={s.deepdiveMetrics.user}
        panel={s.deepdive.user}
        onPatch={(patch) => s.patchDeepdive("user", patch)}
      />
    </div>
  );
}
