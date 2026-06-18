import { afterEach, describe, expect, it, vi } from "vitest";
import { uid } from "./uid";

describe("uid", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("returns an id of the requested length", () => {
    expect(uid()).toHaveLength(8);
    expect(uid(12)).toHaveLength(12);
  });

  it("works in a non-secure context where crypto.randomUUID is undefined", () => {
    // Reproduces the prod crash: HTTP origin → crypto.randomUUID is not a
    // function. getRandomValues is still available and must be used.
    const getRandomValues = vi.fn((arr: Uint8Array) => {
      for (let i = 0; i < arr.length; i++) arr[i] = i;
      return arr;
    });
    vi.stubGlobal("crypto", { getRandomValues });
    const id = uid();
    expect(getRandomValues).toHaveBeenCalled();
    expect(id).toHaveLength(8);
    expect(id).toMatch(/^[0-9a-f]+$/);
  });

  it("falls back to Math.random when crypto is entirely absent", () => {
    vi.stubGlobal("crypto", undefined);
    expect(uid()).toHaveLength(8);
  });
});
