import { useEffect, useRef } from "react";
import { useDashboardStore } from "../../store/dashboardStore";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { Controls } from "./Controls";
import { Panel } from "./Panel";

// Stats-by-Date tab: per-game metric time-series across three dual-axis panels,
// with cohort filtering, hybrid-log scale, and weekend stripes. Mirrors the
// legacy Dash "Stats by Date" tab; data comes from /api/data/series. Controls
// (granularity / date window / cohorts) are THIS tab's own (store.controls.date).
export function StatsByDate() {
  const s = useDashboardStore();
  const c = s.controls.date;

  // Config loading + the game-config picker live in App (dashboard-wide).
  // Granularity change → cohort values can differ, so ensure them then refetch.
  const granReady = useRef(false);
  useEffect(() => {
    if (!granReady.current) {
      granReady.current = true;
      return;
    }
    if (!s.configId) return;
    void s.ensureGroupValues(c.granularity).then(() => s.loadAllSeries());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [c.granularity]);

  // Metric / cohort / date-window changes → refetch series (log & threshold are
  // pure display transforms and intentionally excluded).
  const fetchKey = JSON.stringify({
    df: c.dateFrom,
    dt: c.dateTo,
    cohorts: c.cohortSelection,
    metrics: Object.entries(s.panels).map(([id, p]) => [id, p.left, p.right]),
  });
  const debouncedKey = useDebouncedValue(fetchKey);
  const fetchReady = useRef(false);
  useEffect(() => {
    if (!fetchReady.current) {
      fetchReady.current = true;
      return;
    }
    if (s.configId) void s.loadAllSeries();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  const groups = s.config?.groups ?? [];

  return (
    <div className="p-6 pt-0 w-full">
      {/* Pinned below the sticky header + tab bar while the charts scroll underneath. */}
      <div className="sticky top-[100px] z-20 -mx-6 mb-6 border-b bg-gray-50 px-6 py-3">
        <Controls
          granularity={c.granularity}
          granularities={s.config?.granularities ?? ["day"]}
          onSetGranularity={(g) => s.patchControls("date", { granularity: g })}
          dateFrom={c.dateFrom}
          dateTo={c.dateTo}
          onSetDateRange={(from, to) => s.patchControls("date", { dateFrom: from, dateTo: to })}
          groupValues={s.groupValuesByGran[c.granularity] ?? {}}
          cohortSelection={c.cohortSelection}
          onSetCohort={(col, values) => s.setTabCohort("date", col, values)}
        />
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
            granularity={c.granularity}
            onSetMetrics={(side, metrics) => s.setPanelMetrics(g.id, side, metrics)}
            onSetLog={(log, threshold) => s.setPanelLog(g.id, log, threshold)}
          />
        );
      })}
    </div>
  );
}
