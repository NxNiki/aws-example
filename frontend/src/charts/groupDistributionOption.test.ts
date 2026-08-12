import { describe, expect, it } from "vitest";
import {
  COMPACT_BARS_FROM,
  DENSE_BARS_FROM,
  buildGroupDistributionOption,
  groupChartWidth,
} from "./groupDistributionOption";
import type { GroupStat } from "../api/types";

const stat = (i: number): GroupStat =>
  ({
    cohort: `group${i}`,
    range_label: "2026-07-13 → 2026-08-11",
    range_index: 0,
    n: 10,
    mean: 1,
    median: 1,
    min: 0,
    max: 2,
    q1: 0.5,
    q3: 1.5,
    ci_lower: 0.8,
    ci_upper: 1.2,
  }) as GroupStat;

const stats = (n: number) => Array.from({ length: n }, (_, i) => stat(i));

describe("group distribution density tiers", () => {
  it("steps the per-bar width budget down at each tier", () => {
    expect(groupChartWidth(7)).toBe(120 + 7 * 180); // base tier
    expect(groupChartWidth(COMPACT_BARS_FROM)).toBe(120 + COMPACT_BARS_FROM * 150);
    expect(groupChartWidth(DENSE_BARS_FROM)).toBe(120 + DENSE_BARS_FROM * 120);
    // One more bar at a tier boundary yields a NARROWER chart — by design.
    expect(groupChartWidth(COMPACT_BARS_FROM)).toBeLessThan(groupChartWidth(COMPACT_BARS_FROM - 1));
    expect(groupChartWidth(DENSE_BARS_FROM)).toBeLessThan(groupChartWidth(DENSE_BARS_FROM - 1));
    expect(groupChartWidth(1)).toBe(500); // floor for tiny charts
  });

  it("always wraps the range's end date onto its own line", () => {
    const o = buildGroupDistributionOption(stats(2), "bar", "m");
    expect((o.xAxis as { data: string[] }).data[0]).toBe("group0\n2026-07-13 →\n2026-08-11");
  });
});
