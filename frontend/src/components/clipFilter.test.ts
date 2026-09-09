import { describe, expect, it } from "vitest";

import { activeKind, boundsFor, noClip, noFilter, rememberActive, toWire } from "./clipFilter";

describe("clipFilter box logic", () => {
  it("maps each row to exactly one enabled wire object", () => {
    expect(toWire("clip_value", { min: 0, max: 100 })).toEqual({
      clip: { enable: true, min: 0, max: 100, percentile: false },
      filter: noFilter(),
    });
    expect(toWire("clip_pct", { min: 1, max: 99 })).toEqual({
      clip: { enable: true, min: 1, max: 99, percentile: true },
      filter: noFilter(),
    });
    expect(toWire("filter_value", { min: 5, max: null })).toEqual({
      clip: noClip(),
      filter: { enable: true, min: 5, max: null, percentile: false },
    });
    expect(toWire("filter_pct", { min: null, max: 95 })).toEqual({
      clip: noClip(),
      filter: { enable: true, min: null, max: 95, percentile: true },
    });
    // unchecking everything disables both
    expect(toWire(null, { min: 1, max: 2 })).toEqual({ clip: noClip(), filter: noFilter() });
  });

  it("derives the active row from the wire objects", () => {
    expect(activeKind(noClip(), noFilter())).toBeNull();
    expect(activeKind({ enable: true, min: 0, max: 1, percentile: false }, noFilter())).toBe("clip_value");
    expect(activeKind({ enable: true, min: 0, max: 1, percentile: true }, noFilter())).toBe("clip_pct");
    expect(activeKind(noClip(), { enable: true, min: 0, max: 1, percentile: false })).toBe("filter_value");
    expect(activeKind(noClip(), { enable: true, min: 0, max: 1, percentile: true })).toBe("filter_pct");
  });

  it("keeps per-row bounds when switching rows (never reinterprets units)", () => {
    // clip_value active with [0, 5000]; filter_pct remembered at [1, 99]
    const clip = { enable: true, min: 0, max: 5000, percentile: false };
    const filter = noFilter();
    const rows = { filter_pct: { min: 1, max: 99 } };
    // switching to filter_pct: stash clip_value's bounds, restore filter_pct's own
    const remembered = rememberActive(clip, filter, rows);
    expect(remembered.clip_value).toEqual({ min: 0, max: 5000 });
    expect(boundsFor("filter_pct", clip, filter, remembered)).toEqual({ min: 1, max: 99 });
    const wire = toWire("filter_pct", boundsFor("filter_pct", clip, filter, remembered));
    expect(wire.filter).toEqual({ enable: true, min: 1, max: 99, percentile: true });
    expect(wire.clip.enable).toBe(false);
    // and the active row's bounds read from the wire object, not the memory
    expect(boundsFor("clip_value", clip, filter, rows)).toEqual({ min: 0, max: 5000 });
  });
});
