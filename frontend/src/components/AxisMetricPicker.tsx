// Dual-axis metric picker: ONE row per metric with Left / Right checkboxes, so
// the metric list isn't duplicated across two selectors. The checkboxes are
// mutually exclusive per metric (a series renders on exactly one axis; checking
// one side unchecks the other). Long metric names truncate with an ellipsis —
// hover shows the full name via the title tooltip.
export function AxisMetricPicker(props: {
  options: string[];
  left: string[];
  right: string[];
  onChange: (side: "left" | "right", metrics: string[]) => void;
  className?: string;
  maxHeightClass?: string;
}) {
  const toggle = (side: "left" | "right", metric: string, checked: boolean) => {
    const mine = side === "left" ? props.left : props.right;
    const other = side === "left" ? props.right : props.left;
    const otherSide = side === "left" ? "right" : "left";
    props.onChange(side, checked ? [...new Set([...mine, metric])] : mine.filter((m) => m !== metric));
    if (checked && other.includes(metric)) {
      props.onChange(otherSide, other.filter((m) => m !== metric));
    }
  };

  // Fixed-width checkbox columns keep them vertically aligned; the name column
  // (minmax(0,1fr)) absorbs the rest and allows truncation.
  const row = "grid grid-cols-[minmax(0,1fr)_3rem_3rem] items-center gap-x-1";

  return (
    <div className={`flex flex-col text-base ${props.className ?? ""}`}>
      <div className={`${row} text-gray-500 mb-1 pr-2`}>
        <span>Metric</span>
        <span className="text-center">Left</span>
        <span className="text-center">Right</span>
      </div>
      <div className={`border rounded px-2 py-1 overflow-auto ${props.maxHeightClass ?? "max-h-96"}`}>
        {props.options.map((m) => (
          <div key={m} className={`${row} py-0.5`}>
            <span className="truncate" title={m}>
              {m}
            </span>
            <span className="flex justify-center">
              <input
                type="checkbox"
                checked={props.left.includes(m)}
                onChange={(e) => toggle("left", m, e.target.checked)}
              />
            </span>
            <span className="flex justify-center">
              <input
                type="checkbox"
                checked={props.right.includes(m)}
                onChange={(e) => toggle("right", m, e.target.checked)}
              />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
