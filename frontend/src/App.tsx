import { useEffect, useState } from "react";
import { useDashboardStore } from "./store/dashboardStore";
import { DateGroups } from "./components/DateGroups";
import { LifecycleGroups } from "./components/LifecycleGroups";
import { Notifications } from "./components/Notifications";
import { ChatPanel } from "./features/agent/ChatPanel";
import { StatsByDate } from "./features/stats-by-date/StatsByDate";
import { StatsByGroup } from "./features/stats-by-group/StatsByGroup";
import { SummaryTable } from "./features/summary-table/SummaryTable";
import { DeepDive } from "./features/deep-dive/DeepDive";
import { ReportTab } from "./features/report/ReportTab";

// Tab shell. The game-config picker is dashboard-wide and lives here, above
// the tabs. The active tab lives in the store so the agent's navigate_tab
// action can drive it (Phase 3).
const TABS = [
  { id: "stats-by-date", label: "Stats by Date" },
  { id: "stats-by-group", label: "Stats by Group" },
  { id: "summary-table", label: "Summary Table" },
  { id: "stats-deepdive", label: "Deep Dive" },
  { id: "report", label: "Report" },
] as const;

export default function App() {
  const { configs, configId, selectConfig, loadConfigs, views, loadViews, saveView, loadViewByName, loading, status } =
    useDashboardStore();
  const tab = useDashboardStore((s) => s.activeTab);
  const setTab = useDashboardStore((s) => s.setActiveTab);
  const chatOpen = useDashboardStore((s) => s.chat.open);
  const chatWidth = useDashboardStore((s) => s.chat.width);
  const toggleChat = useDashboardStore((s) => s.toggleChat);
  const hasLifecycle = useDashboardStore((s) => Boolean(s.config?.lifecycle_col));
  const lifecycle = useDashboardStore((s) => s.lifecycle);
  const lifecycleAll = useDashboardStore((s) => s.lifecycleAll);
  const setLifecycleGroup = useDashboardStore((s) => s.setLifecycleGroup);
  const setLifecycleAll = useDashboardStore((s) => s.setLifecycleAll);
  const dateGroups = useDashboardStore((s) => s.dateGroups);
  const setDateGranularity = useDashboardStore((s) => s.setDateGranularity);
  const setDateRange = useDashboardStore((s) => s.setDateRange);
  const granularities = useDashboardStore((s) => s.config?.granularities ?? ["day" as const]);
  // The lifecycle picker edits the unit matching the global granularity.
  const lifecycleUnit = dateGroups.granularity;
  const [viewName, setViewName] = useState("");

  useEffect(() => {
    void loadConfigs();
    void loadViews();
  }, [loadConfigs, loadViews]);

  // Long-lived sessions: the daily ETL lands while the tab is open, so
  // re-check the data edge when the tab regains focus and every 10 minutes;
  // the store slides the default window forward if the user hasn't moved it.
  const refreshDateBounds = useDashboardStore((s) => s.refreshDateBounds);
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible") void refreshDateBounds();
    };
    document.addEventListener("visibilitychange", onVisible);
    // Hidden tabs don't poll: the interval is an /api/* request and counts as
    // user activity for scale-to-zero — a background tab would keep the
    // service awake all night. The visibilitychange handler re-checks the
    // data edge immediately on refocus, so nothing is missed.
    const timer = setInterval(() => {
      if (!document.hidden) void refreshDateBounds();
    }, 10 * 60 * 1000);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      clearInterval(timer);
    };
  }, [refreshDateBounds]);

  return (
    // Right padding reserves room for the docked chat panel (user-resizable);
    // ECharts re-sizes via its ResizeObserver when the panel opens/resizes.
    <div className="min-h-full bg-gray-50 text-gray-900" style={{ paddingRight: chatOpen ? chatWidth : 0 }}>
      {/* Sticky stack: header (top-0, h-14) → date-groups bar (top-14, h-11) →
          [lifecycle bar (top-[6.25rem], h-11), when the config has one] →
          tab bar (top-[9rem] or top-[6.25rem]) → per-tab controls
          (top-[11.75rem] or top-[9rem]). Keep the heights in sync. */}
      <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b bg-white px-4">
        <div className="flex min-w-0 items-center gap-3">
          {/* Full title on wide windows; compact glyph below lg. */}
          <h1 className="hidden truncate text-lg font-semibold lg:block">Game Stats Dashboard</h1>
          <h1 className="text-lg font-semibold lg:hidden">📊</h1>
          {loading && (
            <span className="hidden truncate text-sm text-blue-600 animate-pulse md:inline-block md:max-w-36 xl:max-w-none xl:text-base">
              ● {status ?? "loading…"}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-base xl:gap-4">
          <label className="flex items-center gap-2">
            <span className="hidden text-gray-600 xl:inline">Game config</span>
            <select
              className="w-36 rounded border px-2 py-2 xl:w-auto xl:min-w-56 xl:max-w-md xl:px-3"
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
          <div className="flex items-center gap-2 border-l pl-2 xl:pl-4">
            <select
              className="w-32 rounded border px-2 py-2 xl:w-56"
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
              className="w-28 rounded border px-2 py-2 xl:w-56"
              placeholder="view name"
              value={viewName}
              onChange={(e) => setViewName(e.target.value)}
            />
            <button
              className="whitespace-nowrap rounded border bg-blue-600 px-3 py-2 text-white disabled:opacity-40 xl:px-5"
              disabled={!viewName.trim()}
              onClick={() => {
                const name = viewName.trim();
                if (!name) return;
                if (views.includes(name) && !window.confirm(`Overwrite existing view “${name}”?`)) return;
                void saveView(name);
              }}
            >
              Save<span className="hidden xl:inline"> view</span>
            </button>
          </div>

          <button
            className={`border rounded px-4 py-2 whitespace-nowrap ${
              chatOpen ? "bg-orange-500 text-white" : "bg-orange-100 text-orange-700 hover:bg-orange-200"
            }`}
            onClick={toggleChat}
          >
            🤖<span className="hidden xl:inline"> AI Assistant</span>
          </button>
        </div>
      </header>

      {/* Global "Date groups" bar: granularity + up to 3 date windows, defined
          once per game and read by every tab. */}
      <div className="sticky top-14 z-30 flex h-11 items-center overflow-x-auto border-b bg-white px-4">
        <DateGroups
          granularity={dateGroups.granularity}
          granularities={granularities}
          ranges={dateGroups.ranges}
          onSetGranularity={setDateGranularity}
          onSetRange={setDateRange}
        />
      </div>

      {/* Global lifecycle-group picker: one definition per game, shared by every
          tab (so switching tabs never asks the user to redefine the day ranges).
          Only shown when the config has a lifecycle cohort dimension. */}
      {hasLifecycle && (
        <div className="sticky top-[6.25rem] z-30 flex h-11 items-center overflow-x-auto border-b bg-white px-4">
          <LifecycleGroups
            unit={lifecycleUnit}
            groups={lifecycle[lifecycleUnit]}
            all={lifecycleAll}
            onChange={(i, g) => setLifecycleGroup(lifecycleUnit, i, g)}
            onSetAll={setLifecycleAll}
          />
        </div>
      )}

      <nav
        className={`sticky ${hasLifecycle ? "top-[9rem]" : "top-[6.25rem]"} z-30 flex h-11 gap-1 border-b bg-white px-4`}
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={
              "flex items-center px-4 text-base border-b-2 " +
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
      {tab === "summary-table" && <SummaryTable />}
      {tab === "stats-deepdive" && <DeepDive />}
      {tab === "report" && <ReportTab />}
      <ChatPanel />
      <Notifications />
    </div>
  );
}
