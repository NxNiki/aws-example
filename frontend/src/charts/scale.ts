// Hybrid linear/log y-scale, ported from the legacy Dash app
// (dashboards/game_stats_monitor.py: Styles.hybrid_transform / get_ticks).
// Linear below `thresh`, logarithmic above — keeps small values readable while
// compressing large spikes. The backend returns raw values; this transform is a
// pure display concern applied in the option-builder.

/** Transform a single finite value; linear below `thresh`, log-compressed above. */
export function hybridValue(v: number, thresh: number): number {
  if (!Number.isFinite(v)) return NaN;
  if (v <= thresh) return v;
  return thresh * (1 + Math.log(Math.max(v / thresh, 1)));
}

/** Inverse of {@link hybridValue}: map a compressed-axis position back to the
 * original value, so a hybrid axis can be labelled in original units. */
export function hybridInverse(t: number, thresh: number): number {
  if (!Number.isFinite(t)) return NaN;
  if (t <= thresh) return t;
  return thresh * Math.exp(t / thresh - 1);
}

/** Map an array of (nullable) values through {@link hybridValue}; null/NaN stay null. */
export function hybridTransform(values: (number | null)[], thresh: number): (number | null)[] {
  return values.map((v) => {
    if (v === null || !Number.isFinite(v)) return null;
    return hybridValue(v, thresh);
  });
}

function linspace(start: number, stop: number, n: number): number[] {
  if (n <= 1) return [start];
  const step = (stop - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + step * i);
}

function uniqueSorted(values: number[]): number[] {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

/**
 * Tick positions (in ORIGINAL value space) for a hybrid axis, mirroring
 * get_ticks: evenly spaced below `thresh`, log-spaced above. The caller maps
 * these through {@link hybridValue} for placement and labels them with the
 * original values.
 */
export function getTicks(allY: number[], thresh: number, nTicks = 8): number[] {
  const valid = allY.filter((v) => Number.isFinite(v));
  if (valid.length === 0) return [0, thresh];
  const lo = Math.min(...valid);
  const hi = Math.max(...valid);
  if (lo >= hi) return [lo];

  const half = Math.max(2, Math.floor(nTicks / 2));
  const below = linspace(lo, Math.min(hi, thresh), half);
  const above = valid.filter((v) => v > thresh);
  if (above.length > 0) {
    const logHi = Math.log(Math.max(...above) / thresh + 1e-12);
    if (logHi > 0) {
      const tAbove = linspace(0, logHi, half).map((t) => thresh * (1 + t));
      return uniqueSorted([...below, thresh, ...tAbove]);
    }
  }
  return uniqueSorted(below);
}
