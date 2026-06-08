import { describe, expect, it } from "vitest";
import { getTicks, hybridTransform, hybridValue } from "./scale";

describe("hybridValue", () => {
  it("is identity at or below the threshold", () => {
    expect(hybridValue(0, 10)).toBe(0);
    expect(hybridValue(5, 10)).toBe(5);
    expect(hybridValue(10, 10)).toBe(10);
  });

  it("log-compresses above the threshold", () => {
    // thresh * (1 + ln(v/thresh)); at v == thresh*e the bracket is 2.
    expect(hybridValue(10 * Math.E, 10)).toBeCloseTo(20, 6);
    // monotonic but compressed: 100 maps below its linear value.
    expect(hybridValue(100, 10)).toBeLessThan(100);
    expect(hybridValue(100, 10)).toBeGreaterThan(10);
  });

  it("returns NaN for non-finite input", () => {
    expect(Number.isNaN(hybridValue(NaN, 10))).toBe(true);
  });
});

describe("hybridTransform", () => {
  it("preserves nulls and transforms the rest", () => {
    expect(hybridTransform([null, 5, 10 * Math.E], 10)).toEqual([null, 5, hybridValue(10 * Math.E, 10)]);
  });
});

describe("getTicks", () => {
  it("falls back when there is no spread", () => {
    expect(getTicks([], 10)).toEqual([0, 10]);
    expect(getTicks([7, 7, 7], 10)).toEqual([7]);
  });

  it("returns sorted, unique, ascending ticks spanning the range", () => {
    const ticks = getTicks([1, 5, 50, 500], 10);
    expect(ticks.length).toBeGreaterThan(2);
    expect(ticks).toEqual([...ticks].sort((a, b) => a - b));
    expect(new Set(ticks).size).toBe(ticks.length);
    expect(Math.min(...ticks)).toBeCloseTo(1, 6);
    expect(ticks).toContain(10); // threshold is included once there are values above it
  });
});
