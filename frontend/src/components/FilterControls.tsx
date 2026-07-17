import type { FilterOpts } from "../api/types";

// Percentile filter: DROP samples below the min / above the max percentile
// (0–100). Unlike clip (which pins values to the bound and keeps every
// sample), filtered samples are removed server-side before stats/binning.
// Per-panel control on the Stats-by-Group and Deep Dive tabs.
export function FilterControls(props: { filter: FilterOpts; onChange: (filter: FilterOpts) => void }) {
  const { filter } = props;
  const pct = (v: string) => (v === "" ? null : Math.max(0, Math.min(100, Number(v))));
  return (
    <div className="flex flex-col gap-1 text-base text-gray-600">
      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={filter.enable}
          onChange={(e) => props.onChange({ ...filter, enable: e.target.checked })}
        />
        filter
      </label>
      <div className="flex items-center gap-2">
        <input
          type="number"
          className="border rounded px-2 py-1 w-32"
          placeholder="min %"
          min={0}
          max={100}
          value={filter.min ?? ""}
          disabled={!filter.enable}
          onChange={(e) => props.onChange({ ...filter, min: pct(e.target.value) })}
        />
        <input
          type="number"
          className="border rounded px-2 py-1 w-32"
          placeholder="max %"
          min={0}
          max={100}
          value={filter.max ?? ""}
          disabled={!filter.enable}
          onChange={(e) => props.onChange({ ...filter, max: pct(e.target.value) })}
        />
      </div>
    </div>
  );
}
