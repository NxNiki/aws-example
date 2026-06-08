// Cohort (user-group) pickers: one checkbox group per user_group column, each
// led by a bold "all" token that the backend treats as no-filter (so a subgroup
// can be overlaid against the full population). Shared across tabs.
export function CohortSelect(props: {
  groupValues: Record<string, string[]>;
  selection: Record<string, string[]>;
  onSetCohort: (col: string, values: string[]) => void;
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
          <div key={col} className="flex flex-col text-sm">
            <span className="text-gray-600 mb-1">{col} (cohorts)</span>
            {/* Fixed 4 rows; extra values flow into new columns (scroll if wide). */}
            <div className="border rounded px-3 py-2 overflow-x-auto">
              <div className="grid grid-flow-col grid-rows-4 gap-x-6 gap-y-1 w-max">
                {["all", ...values].map((v) => (
                  <label key={v} className="flex items-center gap-2 cursor-pointer whitespace-nowrap">
                    <input
                      type="checkbox"
                      checked={selected.includes(v)}
                      onChange={(e) => toggle(col, v, e.target.checked)}
                    />
                    <span className={v === "all" ? "font-medium" : ""}>{v}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </>
  );
}
