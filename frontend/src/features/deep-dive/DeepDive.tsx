import { useEffect } from "react";
import { EChart } from "../../charts/EChart";
import { buildHeatmapOption, heatmapSize } from "../../charts/heatmapOption";
import { buildHistogramOption } from "../../charts/histogramOption";
import { buildScatterOption } from "../../charts/scatterOption";
import { clipFilterInfo } from "../../charts/clipFilterInfo";
import { CohortSelect } from "../../components/CohortSelect";
import { RangeGroupSelect } from "../../components/RangeGroupSelect";
import { ClipControls } from "../../components/ClipControls";
import { FilterControls } from "../../components/FilterControls";
import { MetricCheckList } from "../../components/MetricCheckList";
import { activeLifecycleGroups, activeRangeDimensions, rangeGroupValues, useDashboardStore, visibleGroupValues } from "../../store/dashboardStore";
import { AddToReportButton } from "../report/AddToReport";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import type { DeepdivePanel as PanelId, HistogramSeries } from "../../api/types";
import type { DeepdivePanelState } from "../../store/dashboardStore";

// Deep Dive tab: histogram, correlation heatmap, and scatter views over selected
// metrics, for the derived (group-level) and user-level panels. Mirrors the
// legacy "Stats Deepdive" tab; data from /api/data/deepdive. p-value stars,
// Yeo-Johnson, and symlog axes are deferred (need scipy/sklearn).
const NBINS = [50, 100, 200, 500];
const OUTLIER_STDS = [2, 2.5, 3, 3.5, 4];

function DeepdivePanelView(props: {
  title: string;
  panelId: PanelId;
  options: string[];
  panel: DeepdivePanelState;
  onPatch: (patch: Partial<DeepdivePanelState>) => void;
}) {
  const { panel } = props;
  const configId = useDashboardStore((s) => s.configId);
  const vizControls = useDashboardStore((s) => s.controls.viz);
  const dateGroups = useDashboardStore((s) => s.dateGroups);
  const info = clipFilterInfo(panel.clip, panel.filter);

  const histsByMetric: Record<string, HistogramSeries[]> = {};
  for (const h of panel.histograms) (histsByMetric[h.metric] ??= []).push(h);

  // Scatter: a single pair plot for 2 metrics, else an N×N matrix (off-diagonal
  // scatter, diagonal labels the metric). All cells reuse the same points.
  const renderScatter = () => {
    const sc = panel.scatters;
    if (sc.length === 0) return <div className="text-base text-gray-400 p-6">No scatter data.</div>;
    const sm = sc[0].metrics;
    if (sm.length < 2) return <div className="text-base text-gray-400 p-6">Select at least 2 metrics.</div>;
    const axes = { logX: panel.scatterLogX, logY: panel.scatterLogY };
    const PAIR = 600; // square px for the single pair plot
    const CELL = 260; // square px per matrix cell
    if (sm.length === 2) {
      return (
        <div className="overflow-x-auto">
          <EChart
            option={buildScatterOption(sc, 0, 1, { ...axes, xLabel: sm[0], yLabel: sm[1], showLegend: true, info })}
            width={PAIR}
            height={PAIR}
          />
        </div>
      );
    }
    const n = sm.length;
    return (
      <div className="overflow-x-auto">
        {info && <div className="text-base text-gray-500 mb-1">{info}</div>}
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${n}, ${CELL}px)` }}>
          {sm.flatMap((_, i) =>
            sm.map((__, j) =>
              i === j ? (
                <div
                  key={`${i}-${j}`}
                  className="flex items-center justify-center border rounded text-base font-medium text-gray-600 text-center p-1"
                  style={{ width: CELL, height: CELL }}
                >
                  {sm[i]}
                </div>
              ) : (
                <div key={`${i}-${j}`} className="border rounded" style={{ width: CELL, height: CELL }}>
                  <EChart
                    option={buildScatterOption(sc, j, i, {
                      ...axes,
                      xLabel: i === n - 1 ? sm[j] : "",
                      yLabel: j === 0 ? sm[i] : "",
                      compact: true,
                    })}
                    width={CELL}
                    height={CELL}
                  />
                </div>
              ),
            ),
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="border rounded p-3 mb-6">
      <h3 className="text-base font-semibold text-gray-700 mb-2">{props.title}</h3>
      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="flex w-full flex-col gap-3 lg:w-72 lg:shrink-0">
          <div className="flex gap-3 text-base text-gray-600">
            {(["histogram", "heatmap", "scatter"] as const).map((m) => (
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
            <div className="flex flex-col gap-2 text-base text-gray-600">
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
          {panel.mode === "scatter" && (
            <div className="flex flex-col gap-2 text-base text-gray-600">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={panel.outliersStd !== null}
                  onChange={(e) => props.onPatch({ outliersStd: e.target.checked ? 3 : null })}
                />
                remove outliers
                <select
                  className="border rounded px-2 py-1"
                  value={panel.outliersStd ?? 3}
                  disabled={panel.outliersStd === null}
                  onChange={(e) => props.onPatch({ outliersStd: Number(e.target.value) })}
                >
                  {OUTLIER_STDS.map((n) => (
                    <option key={n} value={n}>
                      {n}σ
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={panel.scatterLogX}
                  onChange={(e) => props.onPatch({ scatterLogX: e.target.checked })}
                />
                log x
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={panel.scatterLogY}
                  onChange={(e) => props.onPatch({ scatterLogY: e.target.checked })}
                />
                log y
              </label>
            </div>
          )}
          <ClipControls clip={panel.clip} onChange={(clip) => props.onPatch({ clip })} />
          <FilterControls filter={panel.filter} onChange={(filter) => props.onPatch({ filter })} />
          <AddToReportButton
            getFigure={() => ({
              title: `Deep Dive ${props.panelId} — ${panel.mode}: ${panel.metrics.join(", ") || "no metrics"}`,
              source: {
                kind: "stats-deepdive",
                config: configId ?? "",
                granularity: dateGroups.granularity,
                ranges: dateGroups.ranges
                  .filter((r) => r.show && r.start && r.end)
                  .map((r) => ({ start: r.start, end: r.end })),
                cohort_selection: vizControls.cohortSelection,
                lifecycle_groups: activeLifecycleGroups(useDashboardStore.getState(), dateGroups.granularity) ?? null,
                range_groups: activeRangeDimensions(useDashboardStore.getState(), vizControls, dateGroups.granularity) ?? null,
                panel: props.panelId,
                mode: panel.mode,
                metrics: panel.metrics,
                nbins: panel.nbins,
                normalize: panel.normalize,
                outliers_std: panel.outliersStd,
                scatter_log_x: panel.scatterLogX,
                scatter_log_y: panel.scatterLogY,
                log_y: panel.logY,
                clip: panel.clip,
                filter: panel.filter,
              },
            })}
          />
          {panel.missing.length > 0 && (
            <span className="text-base text-amber-600">No data: {panel.missing.join(", ")}</span>
          )}
          {panel.mode === "heatmap" && (
            <span className="text-base text-gray-400">Pearson r (significance/transform deferred)</span>
          )}
          {panel.mode === "scatter" && (
            <span className="text-base text-gray-400">≤5k sampled points/series (symlog deferred)</span>
          )}
        </div>

        <div className="flex-1 min-w-0">
          {panel.metrics.length === 0 ? (
            <div className="text-base text-gray-400 p-6">Select one or more metrics.</div>
          ) : panel.mode === "histogram" ? (
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(histsByMetric).map(([metric, series]) => (
                <div key={metric}>
                  <div className="text-base text-gray-500 mb-1">{metric}</div>
                  <EChart
                    option={buildHistogramOption(series, { logY: panel.logY, normalize: panel.normalize, info })}
                    height={380}
                  />
                </div>
              ))}
            </div>
          ) : panel.mode === "heatmap" ? (
            <div className="flex flex-col gap-6">
              {panel.heatmaps.map((m, i) => {
                const { width, height } = heatmapSize(m.metrics.length);
                return (
                  <div key={i} className="overflow-x-auto">
                    <div className="text-base text-gray-500 mb-1">
                      {m.cohort} · {m.range_label}
                    </div>
                    <EChart option={buildHeatmapOption(m, info)} width={width} height={height} />
                  </div>
                );
              })}
            </div>
          ) : (
            renderScatter()
          )}
        </div>
      </div>
    </div>
  );
}

export function DeepDive() {
  const s = useDashboardStore();
  const c = s.controls.viz; // this tab's own cohorts; dates/granularity are global
  const dg = s.dateGroups;

  // Refetch on config / date-group / cohort / per-panel
  // mode/metrics/nbins/normalize/clip/filter/outliers changes (log axes are display-only).
  const fetchKey = JSON.stringify({
    cfg: s.configId,
    gran: dg.granularity,
    ranges: dg.ranges.map((r) => [r.start, r.end, r.show]),
    cohorts: c.cohortSelection,
    lifecycle: activeLifecycleGroups(s, dg.granularity),
    rangeGroups: activeRangeDimensions(s, c, dg.granularity),
    panels: (["derived", "user"] as PanelId[]).map((id) => {
      const p = s.deepdive[id];
      return [id, p.mode, p.metrics, p.nbins, p.normalize, p.clip, p.filter, p.outliersStd];
    }),
  });
  const debouncedKey = useDebouncedValue(fetchKey);
  useEffect(() => {
    if (s.configId) {
      // Availability before data — see StatsByDate.
      void s.ensureGroupValues(dg.granularity).then(() => s.loadDeepdive());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  return (
    <div className="p-6 pt-0 w-full">
      {/* Pinned below the sticky header + global bars (see App.tsx height comment). */}
      <div
        className={`sticky ${s.config?.lifecycle_col ? "top-[11.75rem]" : "top-[9rem]"} z-20 -mx-6 mb-6 flex flex-wrap gap-6 items-start border-b bg-gray-50 px-6 py-3`}
      >
        <CohortSelect
          groupValues={visibleGroupValues(s.config, s.groupValuesByGran[dg.granularity] ?? {})}
          available={s.groupAvailableByGran[dg.granularity] ?? null}
          selection={c.cohortSelection}
          onSetCohort={(col, values) => s.setTabCohort("viz", col, values)}
        />
        {s.config?.range_group_col && (
          <RangeGroupSelect
            name={s.config.range_group_name ?? s.config.range_group_col}
            groups={s.rangeGroups}
            values={rangeGroupValues(s, dg.granularity)}
            selection={c.rangeSelection}
            onSetSelection={(labels) => s.setTabRangeSelection("viz", labels)}
            onSetGroup={s.setRangeGroup}
          />
        )}
        {s.config?.period_total_col && (
          <RangeGroupSelect
            name={`${s.config.period_total_name ?? s.config.period_total_col} / ${dg.granularity}`}
            groups={s.periodTotalGroups[dg.granularity] ?? []}
            values={[]}
            freeBounds
            selection={c.periodTotalSelection ?? []}
            onSetSelection={(labels) => s.setTabPeriodTotalSelection("viz", labels)}
            onSetGroup={(i, g) => s.setPeriodTotalGroup(dg.granularity, i, g)}
          />
        )}
      </div>

      {s.error && <div className="mb-4 rounded bg-red-50 text-red-700 text-base px-3 py-2">{s.error}</div>}

      <DeepdivePanelView
        title="Derived metrics (group-level)"
        panelId="derived"
        options={s.deepdiveMetrics.derived}
        panel={s.deepdive.derived}
        onPatch={(patch) => s.patchDeepdive("derived", patch)}
      />
      <DeepdivePanelView
        title="User-level metrics"
        panelId="user"
        options={s.deepdiveMetrics.user}
        panel={s.deepdive.user}
        onPatch={(patch) => s.patchDeepdive("user", patch)}
      />
    </div>
  );
}
