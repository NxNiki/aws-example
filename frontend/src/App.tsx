import { useEffect, useState } from "react";
import { useDashboardStore } from "./store/dashboardStore";
import { StatsByDate } from "./features/stats-by-date/StatsByDate";
import { StatsByGroup } from "./features/stats-by-group/StatsByGroup";
import { DeepDive } from "./features/deep-dive/DeepDive";

// Tab shell. Phase 1 ships the Stats-by-Date tab; the rest are stubbed and
// arrive in later phases (see docs/frontend_redesign.md §8). The game-config
// picker is dashboard-wide and lives here, above the tabs, so it drives every
// tab.
const TABS = [
  { id: "stats-by-date", label: "Stats by Date" },
  { id: "stats-by-group", label: "Stats by Group" },
  { id: "stats-deepdive", label: "Deep Dive" },
  { id: "weekly-report", label: "Weekly Report" },
] as const;

export default function App() {
  const { configs, configId, selectConfig, loadConfigs } = useDashboardStore();
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("stats-by-date");

  useEffect(() => {
    void loadConfigs();
  }, [loadConfigs]);

  return (
    <div className="min-h-full bg-gray-50 text-gray-900">
      <header className="flex items-center justify-between border-b bg-white px-4 py-3">
        <h1 className="text-lg font-semibold">Game Stats Dashboard</h1>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-gray-600">Game config</span>
          <select
            className="border rounded px-3 py-2 min-w-56"
            value={configId ?? ""}
            onChange={(e) => void selectConfig(e.target.value)}
          >
            {configs.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title} ({c.id})
              </option>
            ))}
          </select>
        </label>
      </header>

      <nav className="flex gap-1 border-b bg-white px-4">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={
              "px-4 py-3 text-sm border-b-2 " +
              (tab === t.id
                ? "border-blue-600 text-blue-600 font-semibold"
                : "border-transparent text-gray-500 hover:text-gray-700")
            }
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "stats-by-date" && <StatsByDate />}
      {tab === "stats-by-group" && <StatsByGroup />}
      {tab === "stats-deepdive" && <DeepDive />}
      {tab === "weekly-report" && (
        <div className="p-10 text-center text-gray-400">“Weekly Report” — coming in a later phase.</div>
      )}
    </div>
  );
}
