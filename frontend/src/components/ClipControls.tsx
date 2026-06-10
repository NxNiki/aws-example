import type { ClipOpts } from "../api/types";

// Display-only value clipping (pin to [min, max]). Per-panel control.
export function ClipControls(props: { clip: ClipOpts; onChange: (clip: ClipOpts) => void }) {
  const { clip } = props;
  return (
    <label className="flex items-center gap-2 text-base text-gray-600">
      <input type="checkbox" checked={clip.enable} onChange={(e) => props.onChange({ ...clip, enable: e.target.checked })} />
      clip
      <input
        type="number"
        className="border rounded px-2 py-1 w-16"
        placeholder="min"
        value={clip.min ?? ""}
        disabled={!clip.enable}
        onChange={(e) => props.onChange({ ...clip, min: e.target.value === "" ? null : Number(e.target.value) })}
      />
      <input
        type="number"
        className="border rounded px-2 py-1 w-16"
        placeholder="max"
        value={clip.max ?? ""}
        disabled={!clip.enable}
        onChange={(e) => props.onChange({ ...clip, max: e.target.value === "" ? null : Number(e.target.value) })}
      />
    </label>
  );
}
