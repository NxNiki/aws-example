import { useEffect, useState } from "react";
import { useDashboardStore } from "./store/dashboardStore";
import { Notifications } from "./components/Notifications";
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
  const { configs, configId, selectConfig, loadConfigs, views, loadViews, saveView, loadViewByName, loading } =
    useDashboardStore();
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("stats-by-date");
  const [viewName, setViewName] = useState("");

  useEffect(() => {
    void loadConfigs();
    void loadViews();
  }, [loadConfigs, loadViews]);

  return (
    <div className="min-h-full bg-gray-50 text-gray-900">
      <header className="flex items-center justify-between border-b bg-white px-4 py-3">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold">Game Stats Dashboard</h1>
          {loading && <span className="text-xs text-blue-600 animate-pulse">● loading…</span>}
        </div>
        <div className="flex items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
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

          {/* Saved views: recover a dashboard setting (legacy save/load config). */}
          <div className="flex items-center gap-2 border-l pl-4">
            <select
              className="border rounded px-2 py-2 w-56"
              value=""
              onChange={(e) => {
                if (e.target.value) {
                  setViewName(e.target.value); // surface the loaded view's name
                  void loadViewByName(e.target.value);
                }
              }}
            >
              <option value="">Load view…</option>
              {views.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
            <input
              className="border rounded px-2 py-2 w-56"
              placeholder="view name"
              value={viewName}
              onChange={(e) => setViewName(e.target.value)}
            />
            <button
              className="border rounded px-5 py-2 bg-blue-600 text-white disabled:opacity-40 whitespace-nowrap"
              disabled={!viewName.trim()}
              onClick={() => {
                const name = viewName.trim();
                if (!name) return;
                if (views.includes(name) && !window.confirm(`Overwrite existing view “${name}”?`)) return;
                void saveView(name);
              }}
            >
              Save view
            </button>
          </div>
        </div>
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

      <Notifications />
    </div>
  );
}
