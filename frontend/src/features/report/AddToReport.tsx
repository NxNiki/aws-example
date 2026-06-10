import { useReportStore } from "../../store/reportStore";
import type { FigureSource } from "../../api/types";

// "Add to report" button shown on every chart panel: captures the panel's
// current recipe (config/controls/metrics) into the working Report Spec. The
// Report tab re-renders the figure from this recipe — nothing is frozen.
export function AddToReportButton({ getFigure }: { getFigure: () => { source: FigureSource; title: string } }) {
  const addFigure = useReportStore((s) => s.addFigure);
  return (
    <button
      className="rounded border border-purple-300 bg-purple-50 px-2 py-1 text-sm text-purple-700 hover:bg-purple-100 whitespace-nowrap"
      title="Add this chart (as a re-renderable recipe) to the Report tab"
      onClick={() => {
        const { source, title } = getFigure();
        addFigure(source, title);
      }}
    >
      ＋ Add to report
    </button>
  );
}
