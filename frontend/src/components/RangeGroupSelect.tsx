// Value-range cohort picker (e.g. "Fish level" over fish_value): rendered in
// each tab's cohort row next to the other pickers. The checkbox SELECTION is
// per tab (like any cohort column), but the group DEFINITIONS — name and
// INCLUSIVE [min, max] — are global, so a range means the same thing on every
// tab. "all" is the no-filter population; ranges may overlap.
export interface RangeGroupDef {
  label: string;
  min: number;
  max: number | null; // inclusive; null = open-ended
}

export function RangeGroupSelect(props: {
  name: string; // display name from the config, e.g. "Fish level"
  groups: RangeGroupDef[];
  selection: string[]; // per-tab: checked group labels (may include "all")
  onSetSelection: (labels: string[]) => void;
  onSetGroup: (index: number, group: RangeGroupDef) => void; // global defs
}) {
  const toggle = (label: string, checked: boolean) => {
    const next = checked ? [...new Set([...props.selection, label])] : props.selection.filter((v) => v !== label);
    props.onSetSelection(next);
  };
  const num = (v: string): number | null => (v === "" ? null : Number(v));

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
              <input
                type="number"
                className="w-16 rounded border px-1 py-0.5 text-sm"
                value={g.min}
                min={0}
                onChange={(e) => props.onSetGroup(i, { ...g, min: Number(e.target.value) })}
              />
              <span className="text-gray-400">,</span>
              <input
                type="number"
                className="w-16 rounded border px-1 py-0.5 text-sm"
                value={g.max ?? ""}
                placeholder="max"
                min={0}
                onChange={(e) => props.onSetGroup(i, { ...g, max: num(e.target.value) })}
              />
              <span className="text-gray-400">]</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
