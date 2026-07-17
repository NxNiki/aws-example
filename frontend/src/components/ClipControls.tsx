import type { ClipOpts } from "../api/types";

// Display-only value clipping (pin to [min, max]). Per-panel control.
export function ClipControls(props: { clip: ClipOpts; onChange: (clip: ClipOpts) => void }) {
  const { clip } = props;
  return (
    <div className="flex flex-col gap-1 text-base text-gray-600">
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={clip.enable} onChange={(e) => props.onChange({ ...clip, enable: e.target.checked })} />
        clip
      </label>
      <div className="flex items-center gap-2">
        <input
          type="number"
          className="border rounded px-2 py-1 w-32"
          placeholder="min"
          value={clip.min ?? ""}
          disabled={!clip.enable}
          onChange={(e) => props.onChange({ ...clip, min: e.target.value === "" ? null : Number(e.target.value) })}
        />
        <input
          type="number"
          className="border rounded px-2 py-1 w-32"
          placeholder="max"
          value={clip.max ?? ""}
          disabled={!clip.enable}
          onChange={(e) => props.onChange({ ...clip, max: e.target.value === "" ? null : Number(e.target.value) })}
        />
      </div>
    </div>
  );
}
