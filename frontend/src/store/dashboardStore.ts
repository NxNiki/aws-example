import { create } from "zustand";
import { api } from "../api/client";
import type { ConfigDetail, ConfigSummary, SeriesResponse } from "../api/types";

// Canonical client-side dashboard state. Phase 3 extends this with the agent
// action dispatcher: agent-emitted DashboardActions become pure mutations on
// this store, so the agent drives the same state the UI does.
interface DashboardState {
  configs: ConfigSummary[];
  configId: string | null;
  config: ConfigDetail | null;
  series: SeriesResponse | null;
  loading: boolean;
  error: string | null;

  loadConfigs: () => Promise<void>;
  selectConfig: (id: string) => Promise<void>;
  loadSeries: (metrics: string[]) => Promise<void>;
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  configs: [],
  configId: null,
  config: null,
  series: null,
  loading: false,
  error: null,

  loadConfigs: async () => {
    set({ loading: true, error: null });
    try {
      const { configs } = await api.listConfigs();
      set({ configs });
      if (configs.length && !get().configId) {
        await get().selectConfig(configs[0].id);
      }
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  selectConfig: async (id: string) => {
    set({ loading: true, error: null, configId: id, series: null });
    try {
      const config = await api.getConfig(id);
      set({ config });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },

  loadSeries: async (metrics: string[]) => {
    const { configId, config } = get();
    if (!configId || !config || !metrics.length) return;
    set({ loading: true, error: null });
    try {
      const granularity = config.granularities[0] ?? "day";
      const series = await api.series({ config: configId, granularity, metrics });
      set({ series });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ loading: false });
    }
  },
}));
