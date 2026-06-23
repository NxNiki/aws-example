import { ClipControls } from "../../components/ClipControls";
import type { MetricGroup, SummaryMetricOption } from "../../api/types";

const NO_OPT: SummaryMetricOption = { log: false, clip: { enable: false, min: null, max: null } };

// One metric group's picker for the Summary-table tab: a checkbox per metric to
// include it, and — once selected — its own log toggle + clip (min/max) on the
// same row. Independent per metric. Lives in the tab's left sidebar (like the
// per-panel controls on Stats-by-Group).
export function MetricSelector(props: {
  group: MetricGroup;
  selected: string[];
  options: Record<string, SummaryMetricOption>;
  onToggle: (metric: string, checked: boolean) => void;
  onOption: (metric: string, option: SummaryMetricOption) => void;
}) {
  const { group, selected, options } = props;
  return (
    <div className="flex flex-col text-base">
      <span className="text-gray-600 mb-1 font-medium">{group.label}</span>
      <div className="border rounded px-2 py-1">
        {group.metrics.map((m) => {
          const isSel = selected.includes(m);
          const opt = options[m] ?? NO_OPT;
          return (
            <div key={m} className="flex flex-wrap items-center gap-x-2 gap-y-1 py-0.5">
              <label className="flex items-center gap-2 cursor-pointer whitespace-nowrap">
                <input type="checkbox" checked={isSel} onChange={(e) => props.onToggle(m, e.target.checked)} />
                <span>{m}</span>
              </label>
              {isSel && (
                <div className="flex flex-wrap items-center gap-2 pl-6">
                  <label className="flex items-center gap-1 text-gray-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={opt.log}
                      onChange={(e) => props.onOption(m, { ...opt, log: e.target.checked })}
                    />
                    log
                  </label>
                  <ClipControls clip={opt.clip} onChange={(clip) => props.onOption(m, { ...opt, clip })} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
