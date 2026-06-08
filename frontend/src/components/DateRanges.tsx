// Up to three comparable date windows (Stats-by-Group / Deep Dive). Each row is
// a show toggle + from/to inputs; overlaid charts dim ranges 2 and 3. State is
// tab-level (shared by that tab's panels).
export interface RangeState {
  start: string | null;
  end: string | null;
  show: boolean;
}

export function DateRanges(props: { ranges: RangeState[]; onChange: (index: number, range: RangeState) => void }) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-gray-600 text-sm">Date ranges (compare up to 3)</span>
      {props.ranges.map((r, i) => (
        <div key={i} className="flex items-center gap-2 text-sm">
          <label className="flex items-center gap-1">
            <input type="checkbox" checked={r.show} onChange={(e) => props.onChange(i, { ...r, show: e.target.checked })} />
            <span className="text-gray-500">R{i + 1}</span>
          </label>
          <input
            type="date"
            className="border rounded px-2 py-1"
            value={r.start ?? ""}
            onChange={(e) => props.onChange(i, { ...r, start: e.target.value || null })}
          />
          <span className="text-gray-400">→</span>
          <input
            type="date"
            className="border rounded px-2 py-1"
            value={r.end ?? ""}
            onChange={(e) => props.onChange(i, { ...r, end: e.target.value || null })}
          />
        </div>
      ))}
    </div>
  );
}
