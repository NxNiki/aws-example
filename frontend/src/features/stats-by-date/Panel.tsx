import { useMemo } from "react";
import { EChart } from "../../charts/EChart";
import { buildStatsByDateOption } from "../../charts/statsByDateOption";
import { AxisMetricPicker } from "../../components/AxisMetricPicker";
import { AddToReportButton } from "../report/AddToReport";
import { activeLifecycleGroups, useDashboardStore } from "../../store/dashboardStore";
import type { Granularity, MetricGroup } from "../../api/types";
import type { PanelState } from "../../store/dashboardStore";

// One Stats-by-Date chart panel: dual-axis metric pickers + hybrid-log toggle,
// rendered from the panel's fetched data series. Log/threshold are pure display
// transforms (no refetch); metric changes drive a server refetch upstream.
export function Panel(props: {
  group: MetricGroup;
  panel: PanelState;
  granularity: Granularity;
  onSetMetrics: (side: "left" | "right", metrics: string[]) => void;
  onSetLog: (log: boolean, threshold?: number) => void;
}) {
  const { group, panel, granularity } = props;
  const configId = useDashboardStore((s) => s.configId);
  const dateControls = useDashboardStore((s) => s.controls.date);

  const option = useMemo(
    () =>
      buildStatsByDateOption(panel.series, {
        leftMetrics: panel.left,
        rightMetrics: panel.right,
        log: panel.log,
        threshold: panel.threshold,
        granularity,
      }),
    [panel.series, panel.left, panel.right, panel.log, panel.threshold, granularity],
  );

  return (
    <div className="border rounded p-3 mb-6">
      <h3 className="text-base font-semibold text-gray-700 mb-2">{group.label}</h3>
      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Controls sidebar: combined metric/axis picker, then log scale. */}
        <div className="flex w-full flex-col gap-3 lg:w-80 lg:shrink-0">
          <AxisMetricPicker
            options={group.metrics}
            left={panel.left}
            right={panel.right}
            onChange={props.onSetMetrics}
            maxHeightClass="max-h-96"
          />

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-base text-gray-600">
              <input type="checkbox" checked={panel.log} onChange={(e) => props.onSetLog(e.target.checked)} />
              log scale
              <input
                type="number"
                className="border rounded px-2 py-1 w-16"
                value={panel.threshold}
                disabled={!panel.log}
                onChange={(e) => props.onSetLog(panel.log, Number(e.target.value))}
                title="hybrid-log threshold"
              />
            </label>
            <AddToReportButton
              getFigure={() => ({
                title: `${group.label} — ${[...panel.left, ...panel.right].join(", ") || "no metrics"}`,
                source: {
                  kind: "stats-by-date",
                  config: configId ?? "",
                  granularity: dateControls.granularity,
                  date_from: dateControls.dateFrom,
                  date_to: dateControls.dateTo,
                  cohort_selection: dateControls.cohortSelection,
                  lifecycle_groups: activeLifecycleGroups(useDashboardStore.getState()) ?? null,
                  panel_id: group.id,
                  left: panel.left,
                  right: panel.right,
                  log: panel.log,
                  threshold: panel.threshold,
                },
              })}
            />
          </div>
        </div>

        {/* Chart fills the remaining width (min-w-0 lets ECharts shrink/resize). */}
        <div className="flex-1 min-w-0">
          <EChart option={option} height={400} />
        </div>
      </div>
    </div>
  );
}
