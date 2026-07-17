import type { ClipOpts, FilterOpts } from "../api/types";

// "clip: [a, b], filter: [x%, y%]" tag describing the active display
// transforms. Stamped INSIDE the chart (not the surrounding DOM) so exported
// report PNGs carry the context of how the data was reshaped.
export function clipFilterInfo(clip?: ClipOpts | null, filter?: FilterOpts | null): string {
  const parts: string[] = [];
  if (clip?.enable && (clip.min != null || clip.max != null)) {
    parts.push(`clip: [${clip.min ?? "-inf"}, ${clip.max ?? "inf"}]`);
  }
  if (filter?.enable && (filter.min != null || filter.max != null)) {
    parts.push(`filter: [${filter.min ?? 0}%, ${filter.max ?? 100}%]`);
  }
  return parts.join(", ");
}

// Corner text element for the info tag; empty text renders nothing. `left`
// for charts whose top-right is occupied (e.g. the scatter legend).
export function infoGraphic(text: string, corner: "right" | "left" = "right") {
  if (!text) return undefined;
  return [
    {
      type: "text" as const,
      [corner]: 8,
      top: 4,
      silent: true,
      style: { text, fontSize: 13, fill: "#666" },
    },
  ];
}
