// Global lifecycle-group picker: up to three user-defined cohorts by play-day
// range (days since the user's first bet), replacing the ETL's fixed
// new/beginner/old split. Defined once per game, above the tab bar, and shared
// by every tab (Stats-by-Date / by-Group / Summary / Deep Dive). Each row is a
// show toggle + editable name + day-from/day-to preset dropdowns; "max" as the
// end means open-ended. Ranges may overlap (e.g. day 0 vs day 0-3 vs rest).
export interface LifecycleGroupState {
  label: string;
  dayFrom: number;
  dayTo: number | null; // null = max (open-ended)
  show: boolean;
}

// Preset day offsets mirroring the retention metrics (day0/1/3/5/7/10/15/30),
// plus 4 and 8 so the default new(0-3)/beginner(4-7)/old(8-max) stay expressible.
const DAY_OPTIONS = [0, 1, 2, 3, 4, 5, 7, 8, 10, 15, 30];

function DaySelect(props: { value: number | null; allowMax: boolean; onChange: (v: number | null) => void }) {
  return (
    <select
      className="rounded border px-1 py-0.5"
      value={props.value === null ? "max" : String(props.value)}
      onChange={(e) => props.onChange(e.target.value === "max" ? null : Number(e.target.value))}
    >
      {DAY_OPTIONS.map((d) => (
        <option key={d} value={d}>
          {d}
        </option>
      ))}
      {props.allowMax && <option value="max">max</option>}
    </select>
  );
}

export function LifecycleGroups(props: {
  groups: LifecycleGroupState[];
  all: boolean;
  onChange: (index: number, group: LifecycleGroupState) => void;
  onSetAll: (all: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-x-4 whitespace-nowrap text-sm">
      <span className="text-gray-600" title="Users grouped by days since their first bet; defined once per game, shared by all tabs">
        Lifecycle groups <span className="text-gray-400">(days since first bet)</span>
      </span>
      <label className="flex items-center gap-1 cursor-pointer">
        <input type="checkbox" checked={props.all} onChange={(e) => props.onSetAll(e.target.checked)} />
        <span className="font-medium">all</span>
      </label>
      {props.groups.map((g, i) => (
        <div key={i} className="flex items-center gap-1 border-l pl-4">
          <input type="checkbox" checked={g.show} onChange={(e) => props.onChange(i, { ...g, show: e.target.checked })} />
          <input
            className="w-20 rounded border px-1 py-0.5"
            value={g.label}
            onChange={(e) => props.onChange(i, { ...g, label: e.target.value })}
            title="group name (chart legend label)"
          />
          <span className="text-gray-500">day</span>
          <DaySelect
            value={g.dayFrom}
            allowMax={false}
            onChange={(v) => props.onChange(i, { ...g, dayFrom: v ?? 0 })}
          />
          <span className="text-gray-400">→</span>
          <DaySelect value={g.dayTo} allowMax onChange={(v) => props.onChange(i, { ...g, dayTo: v })} />
        </div>
      ))}
    </div>
  );
}
