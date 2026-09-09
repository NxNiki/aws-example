import type { Granularity } from "../api/types";

// Global lifecycle-group picker: up to three user-defined cohorts by periods
// since the user's first bet, replacing the ETL's fixed new/beginner/old split.
// Defined once per game, above the tab bar, and shared by every tab. The unit
// follows the active tab's granularity — days on daily data, calendar weeks on
// weekly data, calendar months on monthly data (a week/month row aggregates
// the whole period, so day units there would slice users by start weekday
// rather than by age). Each row is a show toggle + editable name + a HALF-OPEN
// range [start, end): start inclusive, end exclusive, so adjacent groups
// sharing a boundary ([0,3), [3,7), [7,max)) tile with no gap and no
// double-count. "max" as the end means open-ended. Ranges may overlap.
export interface LifecycleGroupState {
  label: string;
  start: number;
  end: number | null; // exclusive; null = max (open-ended)
  show: boolean;
}

// Day boundaries mirror the retention metrics (day0/1/3/5/7/15/30). Weeks and
// months use plain period indices (0 = first week/month).
const UNIT_OPTIONS: Record<Granularity, number[]> = {
  day: [0, 1, 3, 5, 7, 15, 30],
  week: [0, 1, 2, 3, 4],
  month: [0, 1, 2, 3, 4],
};
const UNIT_LABEL: Record<Granularity, string> = {
  day: "days since first bet",
  week: "weeks since first bet",
  month: "months since first bet",
};

function PeriodSelect(props: {
  value: number | null;
  options: number[];
  allowMax: boolean;
  onChange: (v: number | null) => void;
}) {
  return (
    <select
      className="rounded border px-1 py-0.5"
      value={props.value === null ? "max" : String(props.value)}
      onChange={(e) => props.onChange(e.target.value === "max" ? null : Number(e.target.value))}
    >
      {props.options.map((d) => (
        <option key={d} value={d}>
          {d}
        </option>
      ))}
      {props.allowMax && <option value="max">max</option>}
    </select>
  );
}

export function LifecycleGroups(props: {
  unit: Granularity;
  groups: LifecycleGroupState[];
  all: boolean;
  onChange: (index: number, group: LifecycleGroupState) => void;
  onSetAll: (all: boolean) => void;
}) {
  const options = UNIT_OPTIONS[props.unit];
  return (
    <div className="flex items-center gap-x-4 whitespace-nowrap text-sm">
      <span
        className="text-gray-600"
        title="Users grouped by periods since their first bet; defined once per game, shared by all tabs. The unit follows the active tab's granularity."
      >
        Lifecycle groups <span className="text-gray-400">({UNIT_LABEL[props.unit]})</span>
      </span>
      <label className="flex items-center gap-1 cursor-pointer">
        <input type="checkbox" checked={props.all} onChange={(e) => props.onSetAll(e.target.checked)} />
        <span className="font-medium">all</span>
      </label>
      {props.groups.map((g, i) => (
        <div
          key={i}
          className="flex items-center gap-1 border-l pl-4"
          title="half-open range [start, end): start inclusive, end exclusive — groups sharing a boundary have no gap or overlap"
        >
          <input type="checkbox" checked={g.show} onChange={(e) => props.onChange(i, { ...g, show: e.target.checked })} />
          <input
            className="w-20 rounded border px-1 py-0.5"
            value={g.label}
            onChange={(e) => props.onChange(i, { ...g, label: e.target.value })}
            title="group name (chart legend label)"
          />
          <span className="text-gray-500">{props.unit}</span>
          <span className="text-gray-400">[</span>
          <PeriodSelect
            value={g.start}
            options={options}
            allowMax={false}
            onChange={(v) => props.onChange(i, { ...g, start: v ?? 0 })}
          />
          <span className="text-gray-400">,</span>
          <PeriodSelect value={g.end} options={options} allowMax onChange={(v) => props.onChange(i, { ...g, end: v })} />
          <span className="text-gray-400">)</span>
        </div>
      ))}
    </div>
  );
}
