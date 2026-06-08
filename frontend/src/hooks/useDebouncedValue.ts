import { useEffect, useState } from "react";

// How long to wait after the last control change before refetching. Selecting
// several cohorts/metrics in a row should fire ONE request for the final state,
// not one per click. Tune here if it feels laggy / too eager.
export const REFETCH_DEBOUNCE_MS = 3000;

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
