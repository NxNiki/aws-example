// Number formatting for the Summary table (and any other tabular numerics).

const NUM = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 4 });

// Compact, readable value (4 significant digits, grouped thousands). Null → em-dash.
export function fmtNum(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return NUM.format(v);
}

// Raw percent difference of `val` vs the reference `ref` (e.g. -10 for -10%).
// Null when undefined (missing value, or a zero/absent reference).
export function pctValue(val: number | null | undefined, ref: number | null | undefined): number | null {
  if (val == null || ref == null || ref === 0 || Number.isNaN(val) || Number.isNaN(ref)) return null;
  return ((val - ref) / Math.abs(ref)) * 100;
}

// Signed percent difference as text, e.g. "+4.0%". Null when undefined.
export function fmtPct(val: number | null | undefined, ref: number | null | undefined): string | null {
  const pct = pctValue(val, ref);
  return pct == null ? null : `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

// Diverging colormap for a percent change, as an "rgb(r, g, b)" string for text.
// Interpolates neutral-gray → red (down) / green (up). `scale` is the
// normalization magnitude: |pct| == scale is fully saturated. Callers pass the
// per-metric (per-row) max |Δ%| so each metric is colored on its own scale.
// Shared by the on-screen grid and the exported HTML so they read identically.
const PCT_NEUTRAL: [number, number, number] = [110, 110, 110];
const PCT_DOWN: [number, number, number] = [200, 30, 30]; // red
const PCT_UP: [number, number, number] = [21, 128, 61]; // green

export function pctColor(pct: number | null | undefined, scale: number): string | null {
  if (pct == null || Number.isNaN(pct)) return null;
  const t = scale > 0 ? Math.min(1, Math.abs(pct) / scale) : 0;
  const end = pct < 0 ? PCT_DOWN : PCT_UP;
  const c = PCT_NEUTRAL.map((n, i) => Math.round(n + (end[i] - n) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// Per-row normalization magnitude = the largest |Δ%| among a metric's cells.
export function pctScale(pcts: Array<number | null | undefined>): number {
  let max = 0;
  for (const p of pcts) if (p != null && !Number.isNaN(p)) max = Math.max(max, Math.abs(p));
  return max;
}

// p-value with a small-value floor so tiny p's don't render as "0.000".
export function fmtPValue(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  if (p < 0.001) return "<0.001";
  return p.toFixed(3);
}
