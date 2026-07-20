import type { Granularity } from "../api/types";
import type { RangeState } from "./DateRanges";

// Global "Date groups" picker: the dashboard's granularity plus up to three
// comparable date windows, defined once per game in the bar above the tab
// selector (and above Lifecycle groups) — every tab reads the same windows.
// Stats-by-Date concatenates the shown ranges horizontally on its line charts;
// the other tabs compare them side by side as before.
export function DateGroups(props: {
  granularity: Granularity;
  granularities: Granularity[];
  ranges: RangeState[];
  onSetGranularity: (g: Granularity) => void;
  onSetRange: (index: number, range: RangeState) => void;
}) {
  return (
    <div className="flex items-center gap-x-4 whitespace-nowrap text-sm">
      <span className="text-gray-600" title="Granularity + up to 3 date windows; defined once per game, shared by all tabs">
        Date groups
      </span>
      <select
        className="rounded border px-1 py-0.5"
        value={props.granularity}
        onChange={(e) => props.onSetGranularity(e.target.value as Granularity)}
      >
        {props.granularities.map((g) => (
          <option key={g} value={g}>
            {g}
          </option>
        ))}
      </select>
      {props.ranges.map((r, i) => (
        <div key={i} className="flex items-center gap-1 border-l pl-4">
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={r.show}
              onChange={(e) => props.onSetRange(i, { ...r, show: e.target.checked })}
            />
            <span className="text-gray-500">R{i + 1}</span>
          </label>
          <input
            type="date"
            className="rounded border px-1 py-0.5"
            value={r.start ?? ""}
            onChange={(e) => props.onSetRange(i, { ...r, start: e.target.value || null })}
          />
          <span className="text-gray-400">→</span>
          <input
            type="date"
            className="rounded border px-1 py-0.5"
            value={r.end ?? ""}
            onChange={(e) => props.onSetRange(i, { ...r, end: e.target.value || null })}
          />
        </div>
      ))}
    </div>
  );
}
