import type { ClipOpts, FilterOpts } from "../api/types";

// The four rows of the clip/filter box. At most ONE is active at a time:
// clip pins values to the bounds, filter drops samples outside them; each
// comes in a value-based and a percentile-based (0–100) variant.
export type ClipFilterKind = "clip_value" | "clip_pct" | "filter_value" | "filter_pct";

export interface ClipFilterBounds {
  min: number | null;
  max: number | null;
}

// Per-row bound memory for the INACTIVE rows, so switching rows never
// reinterprets one row's numbers in the other row's unit (1000 ≠ 1000%).
// The active row's bounds live on the wire object (clip/filter) itself.
export type ClipFilterRows = Partial<Record<ClipFilterKind, ClipFilterBounds>>;

export const CLIP_FILTER_KINDS: ClipFilterKind[] = ["clip_value", "clip_pct", "filter_value", "filter_pct"];

export const noClip = (): ClipOpts => ({ enable: false, min: null, max: null, percentile: false });
export const noFilter = (): FilterOpts => ({ enable: false, min: null, max: null, percentile: true });

export function activeKind(clip: ClipOpts, filter: FilterOpts): ClipFilterKind | null {
  if (clip.enable) return clip.percentile ? "clip_pct" : "clip_value";
  if (filter.enable) return filter.percentile ? "filter_pct" : "filter_value";
  return null;
}

export function boundsFor(kind: ClipFilterKind, clip: ClipOpts, filter: FilterOpts, rows: ClipFilterRows): ClipFilterBounds {
  const active = activeKind(clip, filter);
  if (kind === active) {
    const src = kind.startsWith("clip") ? clip : filter;
    return { min: src.min ?? null, max: src.max ?? null };
  }
  return rows[kind] ?? { min: null, max: null };
}

// Wire objects for a given active row + bounds. Exactly one of clip/filter is
// enabled (or neither); the disabled one is reset to its inert default.
export function toWire(kind: ClipFilterKind | null, bounds: ClipFilterBounds): { clip: ClipOpts; filter: FilterOpts } {
  const clip = noClip();
  const filter = noFilter();
  if (kind === null) return { clip, filter };
  const pct = kind.endsWith("_pct");
  if (kind.startsWith("clip")) {
    return { clip: { enable: true, min: bounds.min, max: bounds.max, percentile: pct }, filter };
  }
  return { clip, filter: { enable: true, min: bounds.min, max: bounds.max, percentile: pct } };
}

// Stash the previously-active row's bounds back into the row memory.
export function rememberActive(clip: ClipOpts, filter: FilterOpts, rows: ClipFilterRows): ClipFilterRows {
  const active = activeKind(clip, filter);
  if (!active) return rows;
  const src = active.startsWith("clip") ? clip : filter;
  return { ...rows, [active]: { min: src.min ?? null, max: src.max ?? null } };
}
