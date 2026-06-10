import { useEffect, useMemo } from "react";
import { EChart } from "../../charts/EChart";
import { buildGroupDistributionOption } from "../../charts/groupDistributionOption";
import { CohortSelect } from "../../components/CohortSelect";
import { ClipControls } from "../../components/ClipControls";
import { DateRanges } from "../../components/DateRanges";
import { useDashboardStore } from "../../store/dashboardStore";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import type { ClipOpts, Granularity, MetricGroup } from "../../api/types";
import type { GroupPanelState } from "../../store/dashboardStore";

// Stats-by-Group tab: per metric group, compare one metric's distribution across
// cohorts and up to 3 date ranges as a box plot or bar (mean ± 95% CI). Mirrors
// the legacy Dash "Stats by Group" tab; data from /api/data/group-distribution.
function GroupPanel(props: {
  group: MetricGroup;
  panel: GroupPanelState;
  onMetric: (m: string) => void;
  onMode: (mode: "box" | "bar") => void;
  onClip: (c: ClipOpts) => void;
}) {
  const { group, panel } = props;
  const option = useMemo(
    () => buildGroupDistributionOption(panel.stats, panel.mode, panel.metric ?? ""),
    [panel.stats, panel.mode, panel.metric],
  );
  return (
    <div className="border rounded p-3 mb-6">
      <div className="flex gap-4">
        <div className="flex flex-col gap-3 w-72 shrink-0">
          <h3 className="text-base font-semibold text-gray-700">{group.label}</h3>
          <label className="flex flex-col text-base">
            <span className="text-gray-500 mb-1">Metric</span>
            <select
              className="border rounded px-2 py-1"
              value={panel.metric ?? ""}
              onChange={(e) => props.onMetric(e.target.value)}
            >
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
          {panel.missing && <span className="text-base text-amber-600">No data for this metric/range.</span>}
        </div>
        <div className="flex-1 min-w-0">
          <EChart option={option} height={400} />
        </div>
      </div>
    </div>
  );
}

export function StatsByGroup() {
  const s = useDashboardStore();
  const c = s.controls.group; // this tab's own granularity / ranges / cohorts
  const groups = s.config?.groups ?? [];

  // Refetch on config / granularity / range / cohort / metric / clip changes
  // (box-vs-bar is a pure display switch and intentionally excluded).
  const fetchKey = JSON.stringify({
    cfg: s.configId,
    gran: c.granularity,
    ranges: c.ranges.map((r) => [r.start, r.end, r.show]),
    cohorts: c.cohortSelection,
    panels: Object.entries(s.group).map(([id, p]) => [id, p.metric, p.clip]),
  });
  const debouncedKey = useDebouncedValue(fetchKey);
  useEffect(() => {
    if (s.configId) {
      void s.ensureGroupValues(c.granularity);
      void s.loadGroupDistribution();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  return (
    <div className="p-6 pt-0 w-full">
      {/* Pinned below the sticky header + tab bar (see App.tsx height comment). */}
      <div className="sticky top-[100px] z-20 -mx-6 mb-6 flex flex-wrap gap-6 items-start border-b bg-gray-50 px-6 py-3">
        <label className="flex flex-col text-base">
          <span className="text-gray-600 mb-1">Granularity</span>
          <select
            className="border rounded px-3 py-2"
            value={c.granularity}
            onChange={(e) => s.patchControls("group", { granularity: e.target.value as Granularity })}
          >
            {(s.config?.granularities ?? ["day"]).map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </label>
        <DateRanges ranges={c.ranges} onChange={(i, r) => s.setTabRange("group", i, r)} />
        <CohortSelect
          groupValues={s.groupValuesByGran[c.granularity] ?? {}}
          selection={c.cohortSelection}
          onSetCohort={(col, values) => s.setTabCohort("group", col, values)}
        />
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
          />
        );
      })}
    </div>
  );
}
