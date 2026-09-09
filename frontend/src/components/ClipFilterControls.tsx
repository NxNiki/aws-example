import type { ClipOpts, FilterOpts } from "../api/types";
import {
  CLIP_FILTER_KINDS,
  type ClipFilterBounds,
  type ClipFilterKind,
  type ClipFilterRows,
  activeKind,
  boundsFor,
  rememberActive,
  toWire,
} from "./clipFilter";

// The clip/filter box: one bordered group with a row per transform —
// clip (pin to bounds) and filter (drop outside bounds), each value-based or
// percentile-based. Checkboxes behave like a radio group that allows
// deselection: checking a row unchecks the others, unchecking leaves none
// active. Each row keeps its own min/max, so switching rows never
// reinterprets a value bound as a percentile (or vice versa).
const ROWS: Record<ClipFilterKind, { label: string; pct: boolean }> = {
  clip_value: { label: "clip", pct: false },
  clip_pct: { label: "clip %", pct: true },
  filter_value: { label: "filter", pct: false },
  filter_pct: { label: "filter %", pct: true },
};

export interface ClipFilterPatch {
  clip: ClipOpts;
  filter: FilterOpts;
  clipFilterRows: ClipFilterRows;
}

export function ClipFilterControls(props: {
  clip: ClipOpts;
  filter: FilterOpts;
  rows: ClipFilterRows;
  onChange: (patch: ClipFilterPatch) => void;
  kinds?: ClipFilterKind[]; // subset of rows to offer (e.g. clip-only tabs)
}) {
  const { clip, filter, rows } = props;
  const kinds = props.kinds ?? CLIP_FILTER_KINDS;
  const active = activeKind(clip, filter);

  const toggle = (kind: ClipFilterKind, checked: boolean) => {
    const remembered = rememberActive(clip, filter, rows);
    const next = checked ? kind : null;
    const bounds = next ? boundsFor(next, clip, filter, remembered) : { min: null, max: null };
    props.onChange({ ...toWire(next, bounds), clipFilterRows: remembered });
  };

  // Only the active row's inputs are enabled, so edits always go to the wire.
  const setBound = (kind: ClipFilterKind, which: "min" | "max", raw: string) => {
    let v = raw === "" ? null : Number(raw);
    if (v !== null && ROWS[kind].pct) v = Math.max(0, Math.min(100, v));
    const bounds: ClipFilterBounds = { ...boundsFor(kind, clip, filter, rows), [which]: v };
    props.onChange({ ...toWire(kind, bounds), clipFilterRows: rows });
  };

  return (
    <div className="flex flex-col gap-1 text-base text-gray-600 border rounded px-2 py-1.5">
      {kinds.map((kind) => {
        const row = ROWS[kind];
        const bounds = boundsFor(kind, clip, filter, rows);
        const enabled = kind === active;
        return (
          <div key={kind} className="flex items-center gap-2">
            <label className="flex items-center gap-2 w-20 shrink-0 whitespace-nowrap">
              <input type="checkbox" checked={enabled} onChange={(e) => toggle(kind, e.target.checked)} />
              {row.label}
            </label>
            <input
              type="number"
              className="border rounded px-2 py-1 w-[5.5rem] min-w-0"
              placeholder={row.pct ? "min %" : "min"}
              min={row.pct ? 0 : undefined}
              max={row.pct ? 100 : undefined}
              value={bounds.min ?? ""}
              disabled={!enabled}
              onChange={(e) => setBound(kind, "min", e.target.value)}
            />
            <input
              type="number"
              className="border rounded px-2 py-1 w-[5.5rem] min-w-0"
              placeholder={row.pct ? "max %" : "max"}
              min={row.pct ? 0 : undefined}
              max={row.pct ? 100 : undefined}
              value={bounds.max ?? ""}
              disabled={!enabled}
              onChange={(e) => setBound(kind, "max", e.target.value)}
            />
          </div>
        );
      })}
    </div>
  );
}
