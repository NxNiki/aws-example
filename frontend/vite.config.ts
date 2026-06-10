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
      // Order matters: /api/agent (SSE chat on the ai_agent service) must match
      // before the catch-all /api → dashboard_api rule. In prod the ALB does
      // the same path split (§9 of the redesign doc).
      "/api/agent": {
        target: process.env.VITE_AGENT_PROXY ?? "http://127.0.0.1:8051",
        changeOrigin: true,
      },
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
