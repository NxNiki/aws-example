// Number formatting for the Summary table (and any other tabular numerics).

const NUM = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 4 });

// Compact, readable value (4 significant digits, grouped thousands). Null → em-dash.
export function fmtNum(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return NUM.format(v);
}

// Signed percent difference of `val` vs the reference `ref`, e.g. "+4.0%".
// Returns null when undefined (missing value, or a zero/absent reference).
export function fmtPct(val: number | null | undefined, ref: number | null | undefined): string | null {
  if (val == null || ref == null || ref === 0 || Number.isNaN(val) || Number.isNaN(ref)) return null;
  const pct = ((val - ref) / Math.abs(ref)) * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

// p-value with a small-value floor so tiny p's don't render as "0.000".
export function fmtPValue(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  if (p < 0.001) return "<0.001";
  return p.toFixed(3);
}
