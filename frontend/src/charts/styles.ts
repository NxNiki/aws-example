// Visual parity with the legacy Dash app (Styles.COLORS / LINE_SHAPE in
// dashboards/game_stats_monitor.py). Color encodes the cohort, dash pattern
// encodes the metric — so the same (cohort, metric) reads the same across tabs.

// ColorBrewer Set1 — color per cohort.
export const COLORS = [
  "#E41A1C",
  "#377EB8",
  "#4DAF4A",
  "#FF7F00",
  "#984EA3",
  "#A65628",
  "#F781BF",
  "#999999",
];

// Plotly's LINE_SHAPE order, mapped to ECharts lineStyle.type — dash per metric.
// ECharts types `type` as "solid"|"dashed"|"dotted"|number (dash length), so the
// longer Plotly patterns become distinct dash lengths.
export type DashType = "solid" | "dashed" | "dotted" | number;

export const LINE_DASH: DashType[] = [
  "solid", // solid
  "dotted", // dot
  "dashed", // dash
  12, // longdash
  6, // dashdot
  16, // longdashdot
];

export const colorForIndex = (i: number): string => COLORS[i % COLORS.length];
export const dashForIndex = (i: number): DashType => LINE_DASH[i % LINE_DASH.length];

// Two-dimension cohort labels ("new[0, 3) | daily_x") get wide fast — break
// each dimension onto its own line in legends / category axis labels. Series
// identity keeps the full one-line name; this is display-only.
export const wrapCohort = (label: string): string => label.split(" | ").join("\n");
