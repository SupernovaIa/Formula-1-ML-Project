// Shared look for every chart — matches the site's "telemetry panel" theme
// (see .chart-panel in index.css), independent of the light/dark site theme.
export const CHART_TEXT = "#eceef2";
export const CHART_MUTED = "#9a9ca6";
export const CHART_GRID_LINE = "rgba(255,255,255,0.08)";
export const CHART_FONT = '"Titillium Web", system-ui, sans-serif';
export const CHART_MONO = '"JetBrains Mono", ui-monospace, monospace';

export const CHART_PALETTE = [
  "#9a7bff",
  "#ff5a52",
  "#ffc84a",
  "#2fe0a0",
  "#4dd2ff",
  "#ff8fd6",
  "#c2a5cf",
  "#7bdcb5",
  "#f0a35e",
  "#7aa2ff",
];

export const BASE_OPTION = {
  backgroundColor: "transparent",
  color: CHART_PALETTE,
  textStyle: { fontFamily: CHART_FONT, color: CHART_TEXT },
  tooltip: {
    backgroundColor: "#1d1f26",
    borderColor: "#2a2c33",
    textStyle: { color: CHART_TEXT, fontFamily: CHART_MONO, fontSize: 12 },
  },
  legend: {
    textStyle: { color: CHART_MUTED, fontFamily: CHART_FONT },
    top: 4,
  },
  title: {
    textStyle: { color: CHART_TEXT, fontFamily: CHART_FONT, fontWeight: 700, fontSize: 14 },
    left: 8,
    top: 8,
  },
};

export const axis = (overrides = {}) => ({
  axisLine: { lineStyle: { color: "rgba(255,255,255,0.2)" } },
  axisLabel: { color: CHART_MUTED, fontFamily: CHART_MONO, fontSize: 11 },
  splitLine: { lineStyle: { color: CHART_GRID_LINE } },
  ...overrides,
});
