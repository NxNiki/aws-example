import { defineConfig } from "vitest/config";

// Store/util tests run in a plain Node env (no DOM needed yet). When component
// tests land (React Testing Library, Phase 1+), switch `environment` to "jsdom"
// and add the jsdom devDependency.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
