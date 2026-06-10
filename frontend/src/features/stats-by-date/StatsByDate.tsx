import { useEffect, useRef } from "react";
import { useDashboardStore } from "../../store/dashboardStore";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { Controls } from "./Controls";
import { Panel } from "./Panel";

// Stats-by-Date tab: per-game metric time-series across three dual-axis panels,
// with cohort filtering, hybrid-log scale, and weekend stripes. Mirrors the
// legacy Dash "Stats by Date" tab; data comes from /api/data/series.
export function StatsByDate() {
  const s = useDashboardStore();

  // Config loading + the game-config picker live in App (dashboard-wide).
  // Granularity change → cohort values can differ, so refetch both.
  const granReady = useRef(false);
  useEffect(() => {
    if (!granReady.current) {
      granReady.current = true;
      return;
    }
    if (!s.configId) return;
    void s.loadGroupValues().then(() => s.loadAllSeries());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.granularity]);

  // Metric / cohort / date-window changes → refetch series (log & threshold are
  // pure display transforms and intentionally excluded).
  const fetchKey = JSON.stringify({
    df: s.dateFrom,
    dt: s.dateTo,
    cohorts: s.cohortSelection,
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
    <div className="p-6 w-full">
      <Controls
        granularity={s.granularity}
        granularities={s.config?.granularities ?? ["day"]}
        onSetGranularity={s.setGranularity}
        dateFrom={s.dateFrom}
        dateTo={s.dateTo}
        onSetDateRange={s.setDateRange}
        groupValues={s.groupValues}
        cohortSelection={s.cohortSelection}
        onSetCohort={s.setCohort}
      />

      {s.error && <div className="mb-4 rounded bg-red-50 text-red-700 text-sm px-3 py-2">{s.error}</div>}

      {groups.map((g) => {
        const panel = s.panels[g.id];
        if (!panel) return null;
        return (
          <Panel
            key={g.id}
            group={g}
            panel={panel}
            granularity={s.granularity}
            onSetMetrics={(side, metrics) => s.setPanelMetrics(g.id, side, metrics)}
            onSetLog={(log, threshold) => s.setPanelLog(g.id, log, threshold)}
          />
        );
      })}
    </div>
  );
}
