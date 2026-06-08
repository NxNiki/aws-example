import * as echarts from "echarts";
import { useEffect, useRef } from "react";

// Thin ECharts wrapper: own the chart instance lifecycle and feed it a plain
// `option` object. The data→encoding split lives in the caller (option
// builders), which is what lets the agent treat a chart as data later.
export function EChart({
  option,
  height = 360,
  width = "100%",
}: {
  option: echarts.EChartsOption;
  height?: number | string;
  width?: number | string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    // Track the container's own size (flex/layout changes), not just the window,
    // so the chart always fills its column.
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option, true);
  }, [option]);

  return <div ref={ref} style={{ width, height }} />;
}
