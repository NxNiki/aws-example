import { useEffect, useRef } from "react";
import { activeLifecycleGroups, activeRangeGroups, rangeGroupValues, useDashboardStore, visibleGroupValues } from "../../store/dashboardStore";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { CohortSelect } from "../../components/CohortSelect";
import { RangeGroupSelect } from "../../components/RangeGroupSelect";
import { Panel } from "./Panel";

// Stats-by-Date tab: per-game metric time-series across three dual-axis panels,
// with cohort filtering, hybrid-log scale, and weekend stripes. The date
// windows and granularity come from the global "Date groups" bar (App.tsx);
// when several windows are shown they render concatenated horizontally on each
// chart. Data comes from /api/data/series; this tab's own controls are just
// the cohort pickers (store.controls.date).
export function StatsByDate() {
  const s = useDashboardStore();
  const c = s.controls.date;
  const dg = s.dateGroups;

  // Metric / cohort / date-group / lifecycle changes → refetch series (log &
  // threshold are pure display transforms and intentionally excluded).
  const fetchKey = JSON.stringify({
    gran: dg.granularity,
    ranges: dg.ranges.map((r) => [r.start, r.end, r.show]),
    cohorts: c.cohortSelection,
    lifecycle: activeLifecycleGroups(s, dg.granularity),
    rangeGroups: activeRangeGroups(s, c.rangeSelection),
    metrics: Object.entries(s.panels).map(([id, p]) => [id, p.left, p.right]),
  });
  const debouncedKey = useDebouncedValue(fetchKey);
  const fetchReady = useRef(false);
  useEffect(() => {
    if (!fetchReady.current) {
      fetchReady.current = true;
      return;
    }
    if (s.configId) {
      void s.ensureGroupValues(dg.granularity);
      void s.loadAllSeries();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  const groups = s.config?.groups ?? [];

  return (
    <div className="p-6 pt-0 w-full">
      {/* Pinned below the sticky header + global bars while the charts scroll underneath. */}
      <div
        className={`sticky ${s.config?.lifecycle_col ? "top-[11.75rem]" : "top-[9rem]"} z-20 -mx-6 mb-6 flex flex-wrap gap-6 items-start border-b bg-gray-50 px-6 py-3`}
      >
        <CohortSelect
          groupValues={visibleGroupValues(s.config, s.groupValuesByGran[dg.granularity] ?? {})}
          available={s.groupAvailableByGran[dg.granularity] ?? null}
          selection={c.cohortSelection}
          onSetCohort={(col, values) => s.setTabCohort("date", col, values)}
        />
        {s.config?.range_group_col && (
          <RangeGroupSelect
            name={s.config.range_group_name ?? s.config.range_group_col}
            groups={s.rangeGroups}
            values={rangeGroupValues(s, dg.granularity)}
            selection={c.rangeSelection}
            onSetSelection={(labels) => s.setTabRangeSelection("date", labels)}
            onSetGroup={s.setRangeGroup}
          />
        )}
      </div>

      {s.error && <div className="mb-4 rounded bg-red-50 text-red-700 text-base px-3 py-2">{s.error}</div>}

      {groups.map((g) => {
        const panel = s.panels[g.id];
        if (!panel) return null;
        return (
          <Panel
            key={g.id}
            group={g}
            panel={panel}
            granularity={dg.granularity}
            onSetMetrics={(side, metrics) => s.setPanelMetrics(g.id, side, metrics)}
            onSetLog={(log, threshold) => s.setPanelLog(g.id, log, threshold)}
          />
        );
      })}
    </div>
  );
}
