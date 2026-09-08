import { useEffect, useMemo } from "react";
import { EChart } from "../../charts/EChart";
import { buildGroupDistributionOption, groupChartWidth } from "../../charts/groupDistributionOption";
import { clipFilterInfo } from "../../charts/clipFilterInfo";
import { CohortSelect } from "../../components/CohortSelect";
import { RangeGroupSelect } from "../../components/RangeGroupSelect";
import { ClipControls } from "../../components/ClipControls";
import { FilterControls } from "../../components/FilterControls";
import { activeLifecycleGroups, activeRangeDimensions, rangeGroupValues, useDashboardStore, visibleGroupValues } from "../../store/dashboardStore";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { AddToReportButton } from "../report/AddToReport";
import type { ClipOpts, FilterOpts, MetricGroup } from "../../api/types";
import type { GroupPanelState } from "../../store/dashboardStore";

// Stats-by-Group tab: per metric group, compare one metric's distribution across
// cohorts and up to 3 date ranges as a box plot or bar (mean ± 95% CI). Mirrors
// the legacy Dash "Stats by Group" tab; data from /api/data/group-distribution.
function GroupPanel(props: {
  group: MetricGroup;
  panel: GroupPanelState;
  onMetric: (m: string | null) => void;
  onMode: (mode: "box" | "bar") => void;
  onClip: (c: ClipOpts) => void;
  onFilter: (f: FilterOpts) => void;
}) {
  const { group, panel } = props;
  const configId = useDashboardStore((s) => s.configId);
  const groupControls = useDashboardStore((s) => s.controls.group);
  const dateGroups = useDashboardStore((s) => s.dateGroups);
  const option = useMemo(
    () =>
      buildGroupDistributionOption(panel.stats, panel.mode, panel.metric ?? "", clipFilterInfo(panel.clip, panel.filter)),
    [panel.stats, panel.mode, panel.metric, panel.clip, panel.filter],
  );
  return (
    <div className="border rounded p-3 mb-6">
      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="flex w-full flex-col gap-3 lg:w-72 lg:shrink-0">
          <h3 className="text-base font-semibold text-gray-700">{group.label}</h3>
          <label className="flex flex-col text-base">
            <span className="text-gray-500 mb-1">Metric</span>
            <select
              className="border rounded px-2 py-1"
              value={panel.metric ?? ""}
              onChange={(e) => props.onMetric(e.target.value || null)}
            >
              <option value="">{"<None>"}</option>
              {group.metrics.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <div className="flex gap-3 text-base text-gray-600">
            {(["bar", "box"] as const).map((m) => (
              <label key={m} className="flex items-center gap-1 cursor-pointer">
                <input type="radio" checked={panel.mode === m} onChange={() => props.onMode(m)} />
                {m}
              </label>
            ))}
          </div>
          <ClipControls clip={panel.clip} onChange={props.onClip} />
          <FilterControls filter={panel.filter} onChange={props.onFilter} />
          <AddToReportButton
            getFigure={() => ({
              title: `${group.label} — ${panel.metric ?? "?"} (${panel.mode})`,
              source: {
                kind: "stats-by-group",
                config: configId ?? "",
                granularity: dateGroups.granularity,
                ranges: dateGroups.ranges
                  .filter((r) => r.show && r.start && r.end)
                  .map((r) => ({ start: r.start, end: r.end })),
                cohort_selection: groupControls.cohortSelection,
                lifecycle_groups: activeLifecycleGroups(useDashboardStore.getState(), dateGroups.granularity) ?? null,
                range_groups: activeRangeDimensions(useDashboardStore.getState(), groupControls, dateGroups.granularity) ?? null,
                panel_id: group.id,
                metric: panel.metric ?? "",
                mode: panel.mode,
                clip: panel.clip,
                filter: panel.filter,
              },
            })}
          />
          {panel.missing && <span className="text-base text-amber-600">No data for this metric/range.</span>}
        </div>
        <div className="flex-1 min-w-0 overflow-x-auto">
          <EChart option={option} height={400} width={groupChartWidth(panel.stats.length)} />
        </div>
      </div>
    </div>
  );
}

export function StatsByGroup() {
  const s = useDashboardStore();
  const c = s.controls.group; // this tab's own cohorts; dates/granularity are global
  const dg = s.dateGroups;
  const groups = s.config?.groups ?? [];

  // Refetch on config / date-group / cohort / metric / clip / filter changes
  // (box-vs-bar is a pure display switch and intentionally excluded).
  const fetchKey = JSON.stringify({
    cfg: s.configId,
    gran: dg.granularity,
    ranges: dg.ranges.map((r) => [r.start, r.end, r.show]),
    cohorts: c.cohortSelection,
    lifecycle: activeLifecycleGroups(s, dg.granularity),
    rangeGroups: activeRangeDimensions(s, c, dg.granularity),
    panels: Object.entries(s.group).map(([id, p]) => [id, p.metric, p.clip, p.filter]),
  });
  const debouncedKey = useDebouncedValue(fetchKey);
  useEffect(() => {
    if (s.configId) {
      // Availability before data — see StatsByDate.
      void s.ensureGroupValues(dg.granularity).then(() => s.loadGroupDistribution());
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
          onSetCohort={(col, values) => s.setTabCohort("group", col, values)}
        />
        {s.config?.range_group_col && (
          <RangeGroupSelect
            name={s.config.range_group_name ?? s.config.range_group_col}
            groups={s.rangeGroups}
            values={rangeGroupValues(s, dg.granularity)}
            selection={c.rangeSelection}
            onSetSelection={(labels) => s.setTabRangeSelection("group", labels)}
            onSetGroup={s.setRangeGroup}
          />
        )}
        {s.config?.period_total_col && (
          <RangeGroupSelect
            name={`${s.config.period_total_name ?? s.config.period_total_col} / ${dg.granularity}`}
            groups={s.periodTotalGroups[dg.granularity] ?? []}
            values={[]}
            freeBounds
            rightExclusive
            boundsNote="([min, max) — right-exclusive)"
            selection={c.periodTotalSelection ?? []}
            onSetSelection={(labels) => s.setTabPeriodTotalSelection("group", labels)}
            onSetGroup={(i, g) => s.setPeriodTotalGroup(dg.granularity, i, g)}
          />
        )}
      </div>

      {s.error && <div className="mb-4 rounded bg-red-50 text-red-700 text-base px-3 py-2">{s.error}</div>}

      {groups.map((g) => {
        const panel = s.group[g.id];
        if (!panel) return null;
        return (
          <GroupPanel
            key={g.id}
            group={g}
            panel={panel}
            onMetric={(m) => s.setGroupMetric(g.id, m)}
            onMode={(mode) => s.setGroupMode(g.id, mode)}
            onClip={(c) => s.setGroupClip(g.id, c)}
            onFilter={(f) => s.setGroupFilter(g.id, f)}
          />
        );
      })}
    </div>
  );
}
