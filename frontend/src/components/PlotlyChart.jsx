import Plot from "react-plotly.js";

export default function PlotlyChart({ figure }) {
  if (!figure) return null;

  return (
    <Plot
      data={figure.data}
      layout={{ ...figure.layout, autosize: true }}
      useResizeHandler
      style={{ width: "100%", height: "560px" }}
      config={{ responsive: true }}
    />
  );
}
