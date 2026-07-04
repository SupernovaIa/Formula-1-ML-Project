import { lazy, Suspense } from "react";

// react-plotly.js pulls in the full plotly.js bundle (~1MB+ gzipped) — load it
// lazily so it doesn't bloat the initial app bundle for pages that don't
// render a chart yet.
const Plot = lazy(() => import("react-plotly.js"));

// Charts render as a permanently-dark "telemetry panel", like a broadcast
// overlay, regardless of the site's light/dark theme — consistent look no
// matter which plotly template the backend's figure happens to set.
export default function PlotlyChart({ figure }) {
  if (!figure) return null;

  return (
    <div className="chart-panel">
      <Suspense fallback={<div className="chart-panel-fallback" style={{ height: "520px" }} />}>
        <Plot
          data={figure.data}
          layout={{
            ...figure.layout,
            template: "plotly_dark",
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { family: '"Titillium Web", system-ui, sans-serif', color: "#eceef2" },
            margin: { t: 48, r: 24, b: 48, l: 56 },
          }}
          useResizeHandler
          style={{ width: "100%", height: "520px" }}
          config={{ responsive: true, displaylogo: false }}
        />
      </Suspense>
    </div>
  );
}
