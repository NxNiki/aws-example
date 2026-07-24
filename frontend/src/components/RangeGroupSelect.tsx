// Value-range cohort picker (e.g. "Fish level" over fish_value): rendered in
// each tab's cohort row next to the other pickers. The checkbox SELECTION is
// per tab (like any cohort column), but the group DEFINITIONS — name and
// INCLUSIVE [min, max] — are global, so a range means the same thing on every
// tab. Bounds are picked from the values that exist in the data. "all" is the
// no-filter population; ranges may overlap.
export interface RangeGroupDef {
  label: string;
  min: number;
  max: number | null; // inclusive; null = open-ended
}

function BoundSelect(props: { values: number[]; value: number; onChange: (v: number) => void }) {
  // A bound from an older saved view may no longer exist in the data; keep it
  // selectable so the definition stays visible instead of silently changing.
  const options = props.values.includes(props.value)
    ? props.values
    : [...props.values, props.value].sort((a, b) => a - b);
  return (
    <select
      className="rounded border px-1 py-0.5 text-sm"
      value={String(props.value)}
      onChange={(e) => props.onChange(Number(e.target.value))}
    >
      {options.map((v) => (
        <option key={v} value={v}>
          {v}
        </option>
      ))}
    </select>
  );
}

export function RangeGroupSelect(props: {
  name: string; // display name from the config, e.g. "Fish level"
  groups: RangeGroupDef[];
  values: number[]; // distinct values of the range column present in the data
  selection: string[]; // per-tab: checked group labels (may include "all")
  onSetSelection: (labels: string[]) => void;
  onSetGroup: (index: number, group: RangeGroupDef) => void; // global defs
}) {
  const toggle = (label: string, checked: boolean) => {
    const next = checked ? [...new Set([...props.selection, label])] : props.selection.filter((v) => v !== label);
    props.onSetSelection(next);
  };

  return (
    <div className="flex flex-col text-base">
      <span className="text-gray-600 mb-1">
        {props.name} <span className="text-gray-400 text-sm">(inclusive [min, max])</span>
      </span>
      <div className="border rounded px-3 py-2">
        <div className="flex flex-col gap-y-1">
          <label className="flex items-center gap-2 cursor-pointer whitespace-nowrap">
            <input type="checkbox" checked={props.selection.includes("all")} onChange={(e) => toggle("all", e.target.checked)} />
            <span className="font-medium">all</span>
          </label>
          {props.groups.map((g, i) => (
            <div key={i} className="flex items-center gap-1 whitespace-nowrap">
              <input type="checkbox" checked={props.selection.includes(g.label)} onChange={(e) => toggle(g.label, e.target.checked)} />
              <input
                className="w-20 rounded border px-1 py-0.5 text-sm"
                value={g.label}
                onChange={(e) => props.onSetGroup(i, { ...g, label: e.target.value })}
                title="group name (chart legend label)"
              />
              <span className="text-gray-400">[</span>
              <BoundSelect values={props.values} value={g.min} onChange={(v) => props.onSetGroup(i, { ...g, min: v })} />
              <span className="text-gray-400">,</span>
              <BoundSelect
                values={props.values}
                value={g.max ?? props.values[props.values.length - 1] ?? g.min}
                onChange={(v) => props.onSetGroup(i, { ...g, max: v })}
              />
              <span className="text-gray-400">]</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
