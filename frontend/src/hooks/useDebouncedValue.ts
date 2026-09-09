import { useEffect, useState } from "react";

// How long to wait after the last control change before refetching. Keeps a
// burst of clicks (selecting several cohorts/metrics) collapsing into ONE
// request for the final state. Kept short because in-flight requests are now
// cancelled (see runExclusive) — a superseded fetch is aborted, so we don't need
// a long idle window to avoid pile-ups. Tune here.
export const REFETCH_DEBOUNCE_MS = 500;

// Returns `value` delayed by `delayMs`; rapid changes collapse to the last one.
// The initial value passes through immediately, so first load isn't delayed.
export function useDebouncedValue<T>(value: T, delayMs: number = REFETCH_DEBOUNCE_MS): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}
