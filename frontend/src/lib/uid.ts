// Short unique id. crypto.randomUUID() exists only in a SECURE CONTEXT
// (HTTPS or localhost); the dashboard is served over plain HTTP on an internal
// ALB, where it is undefined ("crypto.randomUUID is not a function"). Fall back
// to crypto.getRandomValues (always available), then Math.random as a last
// resort. These ids are client-only keys (toasts, figure/reference slugs), so
// collision resistance only needs to be practical, not cryptographic.
export function uid(length = 8): string {
  const c = globalThis.crypto;
  if (c?.randomUUID) return c.randomUUID().slice(0, length);
  if (c?.getRandomValues) {
    const bytes = c.getRandomValues(new Uint8Array(Math.ceil(length / 2)));
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0"))
      .join("")
      .slice(0, length);
  }
  return Math.random()
    .toString(16)
    .slice(2, 2 + length)
    .padEnd(length, "0");
}
