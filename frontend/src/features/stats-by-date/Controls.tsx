import type { Granularity } from "../../api/types";
import { CohortSelect } from "../../components/CohortSelect";

// Tab controls for Stats-by-Date: granularity, date window, and cohort
// (user-group) value pickers. These apply to all panels in this tab. The game
// config picker is dashboard-wide and lives in the top header (App.tsx).
export function Controls(props: {
  granularity: Granularity;
  granularities: Granularity[];
  onSetGranularity: (g: Granularity) => void;
  dateFrom: string | null;
  dateTo: string | null;
  onSetDateRange: (from: string | null, to: string | null) => void;
  groupValues: Record<string, string[]>;
  cohortSelection: Record<string, string[]>;
  onSetCohort: (col: string, values: string[]) => void;
}) {
  return (
    <div className="flex flex-wrap gap-4 items-end">
      <label className="flex flex-col text-base">
        <span className="text-gray-600 mb-1">Granularity</span>
        <select
          className="border rounded px-3 py-2"
          value={props.granularity}
          onChange={(e) => props.onSetGranularity(e.target.value as Granularity)}
        >
          {props.granularities.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col text-base">
        <span className="text-gray-600 mb-1">From</span>
        <input
          type="date"
          className="border rounded px-3 py-2"
          value={props.dateFrom ?? ""}
          onChange={(e) => props.onSetDateRange(e.target.value || null, props.dateTo)}
        />
      </label>
      <label className="flex flex-col text-base">
        <span className="text-gray-600 mb-1">To</span>
        <input
          type="date"
          className="border rounded px-3 py-2"
          value={props.dateTo ?? ""}
          onChange={(e) => props.onSetDateRange(props.dateFrom, e.target.value || null)}
        />
      </label>

      <CohortSelect groupValues={props.groupValues} selection={props.cohortSelection} onSetCohort={props.onSetCohort} />
    </div>
  );
}
