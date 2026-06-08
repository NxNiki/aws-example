// A scrollable checkbox list for picking metrics (single-click toggle, no Cmd).
// Shared by every tab's axis/metric pickers.
export function MetricCheckList(props: {
  label?: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  maxHeightClass?: string;
}) {
  const toggle = (m: string, checked: boolean) => {
    props.onChange(checked ? [...new Set([...props.selected, m])] : props.selected.filter((x) => x !== m));
  };
  return (
    <div className="flex flex-col text-xs">
      {props.label && <span className="text-gray-500 mb-1">{props.label}</span>}
      <div className={`border rounded px-2 py-1 overflow-auto ${props.maxHeightClass ?? "max-h-28"}`}>
        {props.options.map((m) => (
          <label key={m} className="flex items-center gap-2 py-0.5 cursor-pointer whitespace-nowrap">
            <input type="checkbox" checked={props.selected.includes(m)} onChange={(e) => toggle(m, e.target.checked)} />
            <span>{m}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
