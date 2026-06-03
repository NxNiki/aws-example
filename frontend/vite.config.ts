import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, proxy /api to the local dashboard_api (uvicorn on :8050) so the
// browser talks to one origin. In prod the SPA is served by dashboard_api
// itself, so same-origin /api works without a proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY ?? "http://localhost:8050",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
