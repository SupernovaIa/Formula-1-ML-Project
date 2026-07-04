// Translates the Plotly figure JSON the backend already returns (fig.to_json())
// into ECharts `option` objects. The backend isn't touched — these adapters
// read whichever handful of Plotly trace shapes each endpoint actually
// produces (bar / scatter / scatterpolar / violin) and rebuild the same
// information as native ECharts series.
import { axis, CHART_MONO } from "./echartsTheme.js";

function titleOf(figure) {
  const t = figure?.layout?.title;
  if (!t) return undefined;
  return typeof t === "string" ? t : t.text;
}

// Single horizontal bar trace with one color per bar (e.g. fastest laps).
export function barOption(figure) {
  const trace = figure.data[0];
  const horizontal = trace.orientation === "h";
  const categories = horizontal ? trace.y : trace.x;
  const values = horizontal ? trace.x : trace.y;
  const colors = Array.isArray(trace.marker?.color) ? trace.marker.color : undefined;

  const valueAxis = axis();
  const categoryAxis = axis({ type: "category", data: categories });

  return {
    title: { text: titleOf(figure) },
    grid: { top: 56, right: 24, bottom: 40, left: horizontal ? 70 : 56, containLabel: true },
    legend: false,
    xAxis: horizontal ? { ...valueAxis, type: "value" } : categoryAxis,
    yAxis: horizontal ? categoryAxis : { ...valueAxis, type: "value" },
    series: [
      {
        type: "bar",
        data: values.map((v, i) => ({ value: v, itemStyle: colors ? { color: colors[i] } : undefined })),
      },
    ],
  };
}

// One trace per category, each carrying a single (x, y) point and its own
// color (e.g. mean value per cluster).
export function categoryBarOption(figure) {
  const data = figure.data.map((trace) => ({
    name: trace.name,
    value: Array.isArray(trace.y) ? trace.y[0] : trace.y,
    itemStyle: { color: trace.marker?.color },
  }));

  return {
    title: { text: titleOf(figure) },
    grid: { top: 56, right: 24, bottom: 40, left: 56, containLabel: true },
    legend: false,
    xAxis: axis({ type: "category", data: data.map((d) => d.name) }),
    yAxis: axis({ type: "value" }),
    series: [{ type: "bar", data }],
  };
}

// Multiple line traces sharing an x-axis (championship standings, position
// changes over laps).
export function multiLineOption(figure) {
  const series = figure.data.map((trace) => ({
    type: "line",
    name: trace.name,
    data: trace.x.map((x, i) => [x, trace.y[i]]),
    showSymbol: trace.mode?.includes("markers") ?? false,
    symbolSize: 6,
    lineStyle: {
      color: trace.line?.color,
      type: trace.line?.dash === "dash" ? "dashed" : "solid",
    },
    itemStyle: { color: trace.line?.color },
  }));

  return {
    title: { text: titleOf(figure) },
    grid: { top: 56, right: 24, bottom: 40, left: 56, containLabel: true },
    xAxis: axis({ type: figure.layout?.xaxis?.type === "category" ? "category" : "value" }),
    yAxis: axis({ type: "value" }),
    series,
  };
}

// One scatter trace per group (circuit clusters, PCA projection).
export function scatterGroupsOption(figure) {
  const series = figure.data.map((trace) => ({
    type: "scatter",
    name: trace.name,
    symbolSize: trace.marker?.size ?? 12,
    itemStyle: { color: trace.marker?.color, borderColor: "#00000066", borderWidth: 1 },
    data: trace.x.map((x, i) => [x, trace.y[i]]),
  }));

  return {
    title: { text: titleOf(figure) },
    grid: { top: 56, right: 24, bottom: 40, left: 56, containLabel: true },
    xAxis: axis({ type: "value", scale: true, name: figure.layout?.xaxis?.title?.text }),
    yAxis: axis({ type: "value", scale: true, name: figure.layout?.yaxis?.title?.text }),
    series,
  };
}

// scatterpolar traces -> ECharts radar.
export function radarOption(figure) {
  const categories = figure.data[0]?.theta ?? [];
  const max = Math.max(...figure.data.flatMap((t) => t.r)) * 1.1;

  return {
    title: { text: titleOf(figure) },
    legend: { bottom: 0 },
    radar: {
      indicator: categories.map((name) => ({ name, max })),
      axisName: { color: "#9a9ca6", fontFamily: CHART_MONO, fontSize: 11 },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
      splitArea: { show: false },
    },
    series: [
      {
        type: "radar",
        data: figure.data.map((trace) => ({
          name: trace.name,
          value: trace.r,
          areaStyle: { opacity: trace.opacity ?? 0.25 },
        })),
      },
    ],
  };
}

// draw_track(): a track outline + per-corner markers. The backend also sends
// a dotted reference line + text label per corner — we fold that into a
// single labelled scatter series instead of replaying each trace verbatim.
export function trackOption(figure) {
  const track = figure.data.find((t) => t.name === "Track");
  const finishLine = figure.data.find((t) => t.name === "Finish Line");
  const corners = figure.data.filter((t) => t.name?.startsWith("Corner") && t.mode === "markers");

  return {
    title: { text: titleOf(figure) },
    legend: false,
    grid: { top: 56, right: 24, bottom: 24, left: 24, containLabel: true },
    xAxis: axis({ type: "value", scale: true, show: false }),
    yAxis: axis({ type: "value", scale: true, show: false }),
    series: [
      {
        type: "line",
        data: track.x.map((x, i) => [x, track.y[i]]),
        showSymbol: false,
        lineStyle: { color: "#4dd2ff", width: 3 },
      },
      finishLine && {
        type: "line",
        data: finishLine.x.map((x, i) => [x, finishLine.y[i]]),
        showSymbol: false,
        lineStyle: { color: "#ff5a52", width: 3 },
      },
      {
        type: "scatter",
        symbolSize: 20,
        data: corners.map((c, i) => [c.x[0], c.y[0], i + 1]),
        itemStyle: { color: "rgba(77, 210, 255, 0.7)" },
        label: {
          show: true,
          formatter: (p) => p.data[2],
          color: "#0b0c10",
          fontFamily: CHART_MONO,
          fontWeight: 700,
        },
      },
    ].filter(Boolean),
  };
}

// plot_telemetry(): one line per driver, plus dotted vertical corner
// references — those become ECharts markLines instead of extra series.
export function telemetryOption(figure) {
  const driverLines = figure.data.filter((t) => t.mode === "lines" && t.name);
  const cornerRefs = figure.data.filter((t) => t.line?.dash === "dot" && Array.isArray(t.x));

  const markLineData = cornerRefs.map((ref, i) => ({ xAxis: ref.x[0], label: { formatter: `${i + 1}` } }));

  return {
    title: { text: titleOf(figure) },
    grid: { top: 56, right: 24, bottom: 40, left: 56, containLabel: true },
    xAxis: axis({ type: "value", scale: true, name: "Distance" }),
    yAxis: axis({ type: "value", scale: true }),
    series: driverLines.map((trace, i) => ({
      type: "line",
      name: trace.name,
      showSymbol: false,
      data: trace.x.map((x, j) => [x, trace.y[j]]),
      lineStyle: { color: trace.line?.color },
      itemStyle: { color: trace.line?.color },
      markLine:
        i === 0
          ? {
              silent: true,
              symbol: "none",
              label: { color: "#9a9ca6", fontFamily: CHART_MONO, fontSize: 10 },
              lineStyle: { color: "rgba(255,255,255,0.15)", type: "dotted" },
              data: markLineData,
            }
          : undefined,
    })),
  };
}

// plot_tyre_strat(): floating horizontal bars (Plotly encodes each stint's
// start as `base` and duration as `x`). ECharts has no floating-bar
// primitive, so a `custom` series draws each stint as its own rectangle.
export function tyreStrategyOption(figure) {
  const drivers = [...new Set(figure.data.map((t) => t.y[0]))];
  const stints = figure.data.map((t) => ({
    driver: t.y[0],
    start: t.base,
    duration: t.x[0],
    color: t.marker?.color,
    compound: t.name,
  }));

  return {
    title: { text: titleOf(figure) },
    legend: { bottom: 0, data: [...new Set(stints.map((s) => s.compound))] },
    grid: { top: 56, right: 24, bottom: 48, left: 70, containLabel: true },
    xAxis: axis({ type: "value", name: "Lap" }),
    yAxis: axis({ type: "category", data: drivers }),
    series: [
      {
        type: "custom",
        renderItem: (params, api) => {
          const driverIndex = api.value(0);
          const start = api.coord([api.value(1), driverIndex]);
          const end = api.coord([api.value(1) + api.value(2), driverIndex]);
          const height = api.size([0, 1])[1] * 0.6;
          return {
            type: "rect",
            shape: {
              x: start[0],
              y: start[1] - height / 2,
              width: Math.max(end[0] - start[0], 1),
              height,
            },
            style: api.style({ stroke: "#00000055" }),
          };
        },
        data: stints.map((s) => ({
          value: [drivers.indexOf(s.driver), s.start, s.duration],
          itemStyle: { color: s.color },
        })),
      },
    ],
  };
}

// px.violin(..., box=True): approximate as an ECharts boxplot (median/
// quartiles/whiskers) — the density silhouette itself is dropped since
// ECharts has no built-in violin series. Optionally overlay every raw lap
// time as a jittered point per category, which is what the violin's
// `points: 'all'` gave you for free.
export function paceBoxplotOption(figure, { showPoints = false } = {}) {
  function quartiles(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const q = (p) => {
      const idx = (sorted.length - 1) * p;
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    };
    return [sorted[0], q(0.25), q(0.5), q(0.75), sorted[sorted.length - 1]];
  }

  const drivers = figure.data.map((t) => t.name);
  const boxData = figure.data.map((t) => quartiles(t.y));
  const colors = figure.data.map((t) => t.marker?.color);

  const boxSeries = {
    type: "boxplot",
    data: boxData.map((d, i) => ({
      value: d,
      // Solid color for both fill and border made the box/whiskers/median
      // invisible against their own fill — translucent fill + solid
      // border of the same color keeps the internal structure visible.
      itemStyle: { color: `${colors[i]}33`, borderColor: colors[i], borderWidth: 2 },
    })),
  };

  const pointSeries = showPoints
    ? figure.data.map((trace, i) => ({
        type: "scatter",
        name: `${drivers[i]} laps`,
        symbolSize: 6,
        itemStyle: { color: colors[i], opacity: 0.6 },
        tooltip: { show: false },
        z: 3,
        // jitter around the category so overlapping laps don't stack in a line
        data: trace.y.map((v) => [i + (Math.random() - 0.5) * 0.5, v]),
      }))
    : [];

  return {
    title: { text: titleOf(figure) },
    legend: false,
    grid: { top: 56, right: 24, bottom: 40, left: 56, containLabel: true },
    xAxis: axis({ type: "category", data: drivers, boundaryGap: true }),
    yAxis: axis({ type: "value", scale: true, name: "Lap time (s)" }),
    series: [boxSeries, ...pointSeries],
  };
}
