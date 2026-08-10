// Cohort (user-group) pickers: one checkbox group per user_group column, each
// with a bold "all" token that the backend treats as no-filter (so a subgroup
// can be overlaid against the full population). Values are the full (cached)
// vocabulary; ones absent from the current date range (per `available`) render
// grayed and unselectable rather than disappearing. Shared across tabs.
export function CohortSelect(props: {
  groupValues: Record<string, string[]>;
  selection: Record<string, string[]>;
  onSetCohort: (col: string, values: string[]) => void;
  available?: Record<string, string[]> | null;
}) {
  const toggle = (col: string, value: string, checked: boolean) => {
    const current = props.selection[col] ?? [];
    const next = checked ? [...new Set([...current, value])] : current.filter((v) => v !== value);
    props.onSetCohort(col, next);
  };

  return (
    <>
      {Object.entries(props.groupValues).map(([col, values]) => {
        const selected = props.selection[col] ?? [];
        return (
          <div key={col} className="flex flex-col text-base">
            <span className="text-gray-600 mb-1">{col} (cohorts)</span>
            {/* Up to 4 content-sized columns; extra values wrap into new rows
                and the box scrolls vertically past ~6 rows (same pattern as
                the metric pickers). "all" renders first so it stays visible. */}
            <div className="border rounded px-3 py-2 overflow-auto max-h-48">
              <div className="grid grid-cols-[repeat(4,max-content)] gap-x-6 gap-y-1 w-max">
                {["all", ...values].map((v) => {
                  const inRange = v === "all" || !props.available?.[col] || props.available[col].includes(v);
                  return (
                    <label
                      key={v}
                      className={`flex items-center gap-2 whitespace-nowrap ${inRange ? "cursor-pointer" : "cursor-not-allowed"}`}
                      title={inRange ? undefined : "no data in the selected date range"}
                    >
                      <input
                        type="checkbox"
                        disabled={!inRange}
                        checked={selected.includes(v)}
                        onChange={(e) => toggle(col, v, e.target.checked)}
                      />
                      <span className={v === "all" ? "font-medium" : inRange ? "" : "text-gray-400"}>{v}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        );
      })}
    </>
  );
}
