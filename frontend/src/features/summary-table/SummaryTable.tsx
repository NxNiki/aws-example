import { useEffect, useMemo } from "react";
import { CohortSelect } from "../../components/CohortSelect";
import { DateRanges } from "../../components/DateRanges";
import { activeLifecycleGroups, useDashboardStore, visibleGroupValues } from "../../store/dashboardStore";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { AddToReportButton } from "../report/AddToReport";
import { MetricSelector } from "./MetricSelector";
import { SummaryGrid } from "./SummaryGrid";
import type { Granularity, SummaryStat } from "../../api/types";

// Stat toggles, in display order (label is the control text; the in-cell tag
// comes from SummaryGrid's STAT_LABEL). Selecting one keeps the canonical order.
const STAT_OPTIONS: { key: SummaryStat; label: string }[] = [
  { key: "n", label: "Count" },
  { key: "mean", label: "Mean" },
  { key: "median", label: "Median" },
  { key: "q1", label: "25th pct" },
  { key: "q3", label: "75th pct" },
  { key: "min", label: "Min" },
  { key: "max", label: "Max" },
];
const STAT_ORDER = STAT_OPTIONS.map((o) => o.key);

// Summary-table tab: one grid of every selected metric (rows, grouped by metric
// group) across every selected cohort × date-range (columns). One column is the
// reference; other cells show ±% vs it per chosen stat. Each metric can be
// clipped / log-transformed independently, and an optional significance test
// (t-test for 2 columns, ANOVA for 3+) added. Data from /api/data/summary-table.
export function SummaryTable() {
  const s = useDashboardStore();
  const c = s.controls.summaryTable; // this tab's own granularity / ranges / cohorts
  const t = s.summaryTable;
  const groups = s.config?.groups ?? [];

  // Refetch on config / granularity / range / cohort / per-metric option /
  // p-values changes. Metric selection, stat selection and the reference column
  // are display-only (no refetch).
  const fetchKey = JSON.stringify({
    cfg: s.configId,
    gran: c.granularity,
    ranges: c.ranges.map((r) => [r.start, r.end, r.show]),
    cohorts: c.cohortSelection,
    lifecycle: activeLifecycleGroups(s, c.granularity),
    metricOptions: t.metricOptions,
    pvalues: t.showPValues,
  });
  const debouncedKey = useDebouncedValue(fetchKey);
  useEffect(() => {
    if (s.configId) {
      void s.ensureGroupValues(c.granularity);
      void s.loadSummaryTable();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  // Filter the fetched rows to the user's per-group metric selection (display only).
  const rows = useMemo(
    () => t.rows.filter((r) => (t.metrics[r.group_id] ?? []).includes(r.metric)),
    [t.rows, t.metrics],
  );

  return (
    <div className="p-6 pt-0 w-full">
      {/* Pinned below the sticky header + tab bar (see App.tsx height comment). */}
      <div
        className={`sticky ${s.config?.lifecycle_col ? "top-[9rem]" : "top-[6.25rem]"} z-20 -mx-6 mb-6 flex flex-wrap gap-6 items-start border-b bg-gray-50 px-6 py-3`}
      >
        <label className="flex flex-col text-base">
          <span className="text-gray-600 mb-1">Granularity</span>
          <select
            className="border rounded px-3 py-2"
            value={c.granularity}
            onChange={(e) => s.patchControls("summaryTable", { granularity: e.target.value as Granularity })}
          >
            {(s.config?.granularities ?? ["day"]).map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </label>
        <DateRanges ranges={c.ranges} onChange={(i, r) => s.setTabRange("summaryTable", i, r)} />
        <CohortSelect
          groupValues={visibleGroupValues(s.config, s.groupValuesByGran[c.granularity] ?? {})}
          selection={c.cohortSelection}
          onSetCohort={(col, values) => s.setTabCohort("summaryTable", col, values)}
        />
        <div className="flex flex-col gap-2 text-base">
          <span className="text-gray-600">Show stats</span>
          <div className="flex flex-wrap gap-x-3 gap-y-1 max-w-xs">
            {STAT_OPTIONS.map((opt) => (
              <label key={opt.key} className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={t.stats.includes(opt.key)}
                  onChange={(e) =>
                    s.setSummaryStats(
                      e.target.checked
                        ? STAT_ORDER.filter((k) => t.stats.includes(k) || k === opt.key)
                        : t.stats.filter((k) => k !== opt.key),
                    )
                  }
                />
                {opt.label}
              </label>
            ))}
          </div>
          <label className="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" checked={t.showPValues} onChange={(e) => s.setSummaryPValues(e.target.checked)} />
            Show p-values (t-test / ANOVA)
          </label>
          <AddToReportButton
            getFigure={() => ({
              title: "Summary table",
              source: {
                kind: "summary-table",
                config: s.configId ?? "",
                granularity: c.granularity,
                ranges: c.ranges.filter((r) => r.show && r.start && r.end).map((r) => ({ start: r.start, end: r.end })),
                cohort_selection: c.cohortSelection,
                lifecycle_groups: activeLifecycleGroups(s, c.granularity) ?? null,
                metrics: t.metrics,
                stats: t.stats,
                reference_key: t.referenceKey,
                metric_options: t.metricOptions,
                pvalues: t.showPValues,
              },
            })}
          />
        </div>
      </div>

      {s.error && <div className="mb-4 rounded bg-red-50 text-red-700 text-base px-3 py-2">{s.error}</div>}
      {t.missing && (
        <div className="mb-4 rounded bg-amber-50 text-amber-700 text-base px-3 py-2">
          No data for the selected ranges / cohorts.
        </div>
      )}

      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Left sidebar: per-group metric pickers with per-metric clip + log. */}
        <div className="flex w-full flex-col gap-4 lg:w-80 lg:shrink-0">
          {groups.map((g) => (
            <MetricSelector
              key={g.id}
              group={g}
              selected={t.metrics[g.id] ?? []}
              options={t.metricOptions}
              onToggle={(metric, checked) =>
                s.setSummaryMetrics(
                  g.id,
                  checked ? [...new Set([...(t.metrics[g.id] ?? []), metric])] : (t.metrics[g.id] ?? []).filter((x) => x !== metric),
                )
              }
              onOption={(metric, option) => s.setSummaryMetricOption(metric, option)}
            />
          ))}
        </div>
        <div className="flex-1 min-w-0">
          <SummaryGrid
            columns={t.columns}
            rows={rows}
            stats={t.stats}
            referenceKey={t.referenceKey}
            showPValues={t.showPValues}
            metricOptions={t.metricOptions}
            onSetReference={s.setSummaryReference}
          />
        </div>
      </div>
    </div>
  );
}
