import { describe, expect, it } from "vitest";
import { COMPACT_BARS_FROM, buildGroupDistributionOption, groupChartWidth } from "./groupDistributionOption";
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

describe("group distribution compact layout", () => {
  it("shrinks the per-bar width budget once many groups are selected", () => {
    const few = groupChartWidth(COMPACT_BARS_FROM - 1);
    const many = groupChartWidth(COMPACT_BARS_FROM);
    // One MORE bar yields a NARROWER chart at the threshold — that's the point.
    expect(many).toBeLessThan(few);
    expect(groupChartWidth(20)).toBe(120 + 20 * 160);
  });

  it("wraps the range's end date onto its own line only in compact mode", () => {
    const wide = buildGroupDistributionOption(stats(COMPACT_BARS_FROM - 1), "bar", "m");
    const compact = buildGroupDistributionOption(stats(COMPACT_BARS_FROM), "bar", "m");
    const cat = (o: ReturnType<typeof buildGroupDistributionOption>) =>
      (o.xAxis as { data: string[] }).data[0];
    expect(cat(wide)).toBe("group0\n2026-07-13 → 2026-08-11");
    expect(cat(compact)).toBe("group0\n2026-07-13 →\n2026-08-11");
    // Taller bottom gutter for the extra label line.
    expect((wide.grid as { bottom: number }).bottom).toBeLessThan((compact.grid as { bottom: number }).bottom);
  });
});
