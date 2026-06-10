import * as echarts from "echarts";
import { useEffect, useRef } from "react";

// Thin ECharts wrapper: own the chart instance lifecycle and feed it a plain
// `option` object. The data→encoding split lives in the caller (option
// builders), which is what lets the agent treat a chart as data later.
export function EChart({
  option,
  height = 360,
  width = "100%",
  onReady,
}: {
  option: echarts.EChartsOption;
  height?: number | string;
  width?: number | string;
  // Hands the live chart instance to the caller (and null on dispose) — the
  // Report tab uses it to grab PNGs via chart.getDataURL() for Confluence export.
  onReady?: (chart: echarts.ECharts | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    onReady?.(chart);
    // Track the container's own size (flex/layout changes), not just the window,
    // so the chart always fills its column.
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      onReady?.(null);
      chart.dispose();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option, true);
  }, [option]);

  return <div ref={ref} style={{ width, height }} />;
}
