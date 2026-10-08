// Translates the Plotly figure JSON the backend already returns (fig.to_json())
// into ECharts `option` objects. The backend isn't touched — these adapters
// read whichever handful of Plotly trace shapes each endpoint actually
// produces (bar / scatter / scatterpolar / violin) and rebuild the same
// information as native ECharts series.
import {
  axis,
  CHART_MONO,
  clusterColor,
  clusterSymbol,
  dataColor,
  rawColor,
} from "./echartsTheme.js";

// Colours the backend sends (team / driver / tyre) are data: keep their identity
// via dataColor(), but let the theme nudge them off the page background.
const data = (c) => (c ? dataColor(c) : undefined);

function titleOf(figure) {
  const t = figure?.layout?.title;
  if (!t) return undefined;
  const text = typeof t === "string" ? t : t.text;
  // Plotly titles are HTML; keep <br> (EChart.jsx splits it into title + subtitle), drop other tags.
  return text?.replace(/<(?!br\s*\/?>)[^>]+>/gi, "");
}

const axisTitle = (figure, which) => figure?.layout?.[which]?.title?.text;

// Single horizontal bar trace with one color per bar (e.g. fastest laps).
export function barOption(figure) {
  const trace = figure.data[0];
  const horizontal = trace.orientation === "h";
  const categories = horizontal ? trace.y : trace.x;
  const values = horizontal ? trace.x : trace.y;
  const colors = Array.isArray(trace.marker?.color) ? trace.marker.color : undefined;

  const valueAxis = axis({ name: axisTitle(figure, horizontal ? "xaxis" : "yaxis"), nameLocation: "middle", nameGap: 28 });
  // ECharts draws the first category at the bottom; the first row here is P1, which belongs on top.
  const categoryAxis = axis({ type: "category", data: categories, inverse: horizontal });

  return {
    title: { text: titleOf(figure) },
    grid: { top: 48, right: 24, bottom: 40, left: horizontal ? 70 : 56, containLabel: true },
    legend: false,
    xAxis: horizontal ? { ...valueAxis, type: "value" } : categoryAxis,
    yAxis: horizontal ? categoryAxis : { ...valueAxis, type: "value" },
    series: [
      {
        type: "bar",
        barMaxWidth: 18,
        itemStyle: { borderRadius: [0, 3, 3, 0] },
        label: {
          show: true,
          position: horizontal ? "right" : "top",
          color: "var(--ink-soft)",
          fontFamily: CHART_MONO,
          fontSize: 11,
          formatter: (p) => (Number.isInteger(p.value) ? p.value : p.value.toFixed(3)),
        },
        data: values.map((v, i) => ({ value: v, itemStyle: colors ? { color: data(colors[i]) } : undefined })),
      },
    ],
  };
}

// One trace per category, each carrying a single (x, y) point and its own
// color (e.g. mean value per cluster).
export function categoryBarOption(figure) {
  const bars = figure.data.map((trace, i) => ({
    name: trace.name,
    value: Array.isArray(trace.y) ? trace.y[0] : trace.y,
    itemStyle: { color: clusterColor(i) },
  }));

  return {
    title: { text: titleOf(figure) },
    grid: { top: 48, right: 24, bottom: 40, left: 56, containLabel: true },
    legend: false,
    xAxis: axis({ type: "category", data: bars.map((d) => d.name) }),
    yAxis: axis({ type: "value" }),
    series: [{ type: "bar", data: bars }],
  };
}

// Multiple line traces sharing an x-axis (championship standings, position
// changes over laps).
export function multiLineOption(figure) {
  // The backend's Plotly JSON doesn't always declare an explicit
  // layout.xaxis.type - infer it from the data instead (string x values,
  // e.g. race names for a championship, mean a category axis; numbers,
  // e.g. lap number, mean a value axis). A category axis needs its own
  // labels on xAxis.data, with series data as plain values indexed against
  // it - [x, y] pairs (which work for a value axis) silently fail to plot.
  const isCategory = typeof figure.data[0]?.x?.[0] === "string";
  const categories = isCategory ? figure.data[0].x : undefined;

  const series = figure.data.map((trace) => ({
    type: "line",
    name: trace.name,
    data: isCategory ? trace.y : trace.x.map((x, i) => [x, trace.y[i]]),
    showSymbol: trace.mode?.includes("markers") ?? false,
    symbolSize: 4,
    emphasis: { focus: "series" },
    labelLayout: { moveOverlap: "shiftY" }, // tied drivers end on the same point: fan their labels out
    endLabel: {
      show: figure.data.length > 2,
      formatter: "{a}",
      color: "var(--ink-soft)",
      fontFamily: CHART_MONO,
      fontSize: 11,
    },
    lineStyle: {
      color: data(trace.line?.color),
      type: trace.line?.dash === "dash" ? "dashed" : "solid",
    },
    itemStyle: { color: data(trace.line?.color) },
  }));

  return {
    title: { text: titleOf(figure) },
    grid: { top: 48, right: figure.data.length > 2 ? 64 : 24, bottom: 40, left: 56, containLabel: true },
    xAxis: axis({ type: isCategory ? "category" : "value", data: categories, boundaryGap: false }),
    yAxis: axis({ type: "value" }),
    series,
  };
}

// One scatter trace per group. With `clusters`, groups are circuit clusters:
// colour and symbol come from the cluster's position, not from the backend
// palette, so they match the badges on the Circuits page.
export function scatterGroupsOption(figure, { clusters = false } = {}) {
  const series = figure.data.map((trace, i) => ({
    type: "scatter",
    name: trace.name,
    symbol: clusters ? clusterSymbol(i) : "circle",
    symbolSize: trace.marker?.size ?? 12,
    itemStyle: {
      color: clusters ? clusterColor(i) : data(trace.marker?.color),
      borderColor: "var(--card)@0.7",
      borderWidth: 1,
    },
    data: trace.x.map((x, j) => [x, trace.y[j]]),
  }));

  return {
    title: { text: titleOf(figure) },
    grid: { top: 48, right: 24, bottom: 40, left: 56, containLabel: true },
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
      axisName: { color: "var(--ink-soft)", fontFamily: CHART_MONO, fontSize: 11 },
      axisLine: { lineStyle: { color: "var(--hairline)" } },
      splitLine: { lineStyle: { color: "var(--hairline)" } },
      splitArea: { show: false },
    },
    series: [
      {
        type: "radar",
        data: figure.data.map((trace, i) => ({
          name: trace.name,
          value: trace.r,
          symbol: clusterSymbol(i),
          lineStyle: { color: clusterColor(i), width: 2 },
          itemStyle: { color: clusterColor(i) },
          areaStyle: { color: clusterColor(i), opacity: trace.opacity ?? 0.2 },
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
    grid: { top: 48, right: 24, bottom: 24, left: 24, containLabel: true },
    xAxis: axis({ type: "value", scale: true, show: false }),
    yAxis: axis({ type: "value", scale: true, show: false }),
    series: [
      {
        type: "line",
        data: track.x.map((x, i) => [x, track.y[i]]),
        showSymbol: false,
        lineStyle: { color: "var(--cat-1)", width: 3 },
      },
      finishLine && {
        type: "line",
        data: finishLine.x.map((x, i) => [x, finishLine.y[i]]),
        showSymbol: false,
        lineStyle: { color: "var(--amber)", width: 3 },
      },
      {
        type: "scatter",
        symbolSize: 20,
        data: corners.map((c, i) => [c.x[0], c.y[0], i + 1]),
        itemStyle: { color: "var(--cat-1)@0.85" },
        label: {
          show: true,
          formatter: (p) => p.data[2],
          color: "var(--page)",
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
    grid: { top: 48, right: 24, bottom: 40, left: 56, containLabel: true },
    xAxis: axis({ type: "value", scale: true, name: "Distance (m)", nameLocation: "middle", nameGap: 28 }),
    yAxis: axis({ type: "value", scale: true, name: axisTitle(figure, "yaxis") }),
    series: driverLines.map((trace, i) => ({
      type: "line",
      name: trace.name,
      showSymbol: false,
      data: trace.x.map((x, j) => [x, trace.y[j]]),
      lineStyle: { color: data(trace.line?.color) },
      itemStyle: { color: data(trace.line?.color) },
      markLine:
        i === 0
          ? {
              silent: true,
              symbol: "none",
              label: { color: "var(--ink-faint)", fontFamily: CHART_MONO, fontSize: 10 },
              lineStyle: { color: "var(--hairline-strong)", type: "dotted" },
              data: markLineData,
            }
          : undefined,
    })),
  };
}

// plot_tyre_strat(): floating horizontal bars (Plotly encodes each stint's
// start as `base` and duration as `x`). ECharts has no floating-bar
// primitive, so a `custom` series draws each stint as its own rectangle - one
// series per compound, which also gives the legend real entries to toggle.
export function tyreStrategyOption(figure) {
  const drivers = [...new Set(figure.data.map((t) => t.y[0]))];
  const stints = figure.data.map((t) => ({
    driver: t.y[0],
    start: t.base,
    end: t.base + t.x[0],
    color: t.marker?.color,
    compound: t.name,
  }));
  const compounds = [...new Set(stints.map((s) => s.compound))];

  const renderItem = (params, api) => {
    const driverIndex = api.value(0);
    const start = api.coord([api.value(1), driverIndex]);
    const end = api.coord([api.value(2), driverIndex]);
    const height = api.size([0, 1])[1] * 0.62;
    return {
      type: "rect",
      shape: { x: start[0], y: start[1] - height / 2, width: Math.max(end[0] - start[0], 1), height, r: 3 },
      style: api.style(),
    };
  };

  return {
    title: { text: titleOf(figure) },
    legend: { bottom: 0, data: compounds },
    tooltip: {
      trigger: "item",
      formatter: (p) => `${drivers[p.value[0]]} · ${p.seriesName}<br/>Laps ${p.value[1] + 1}–${p.value[2]}`,
    },
    grid: { top: 24, right: 24, bottom: 56, left: 70, containLabel: true },
    xAxis: axis({ type: "value", name: "Lap", nameLocation: "middle", nameGap: 28 }),
    yAxis: axis({ type: "category", data: drivers, inverse: true }),
    series: compounds.map((compound) => ({
      type: "custom",
      name: compound,
      // legend swatch = the tyre's own colour, not the palette slot
      color: rawColor(stints.find((s) => s.compound === compound)?.color, 1),
      renderItem,
      // dims: [driver index, first lap, last lap] -> x spans both laps, y is the driver row
      encode: { x: [1, 2], y: 0 },
      data: stints
        .filter((s) => s.compound === compound)
        .map((s) => ({
          value: [drivers.indexOf(s.driver), s.start, s.end],
          // Tyre colours are the meaning (soft = red, hard = white...), so no legibility
          // nudge; the outline keeps a white stint visible on a light page.
          itemStyle: { color: rawColor(s.color, 1), borderColor: "var(--ink-faint)", borderWidth: 1 },
        })),
    })),
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
    z: 2,
    data: boxData.map((d, i) => ({
      value: d,
      // Solid color for both fill and border made the box/whiskers/median
      // invisible against their own fill — translucent fill + solid
      // border of the same color keeps the internal structure visible.
      itemStyle: { color: rawColor(colors[i], 0.2), borderColor: data(colors[i]), borderWidth: 2 },
    })),
  };

  const pointSeries = showPoints
    ? figure.data.map((trace, i) => ({
        type: "scatter",
        name: `${drivers[i]} laps`,
        symbolSize: 7,
        itemStyle: {
          color: data(colors[i]),
          opacity: 0.85,
          borderColor: "var(--card)@0.8",
          borderWidth: 1,
        },
        emphasis: { itemStyle: { opacity: 1, borderColor: "var(--ink)", borderWidth: 1.5 } },
        tooltip: { show: false },
        z: 4,
        // jitter around the category so overlapping laps don't stack in a line
        data: trace.y.map((v) => [i + (Math.random() - 0.5) * 0.3, v]),
      }))
    : [];

  return {
    title: { text: titleOf(figure) },
    legend: false,
    grid: { top: 48, right: 24, bottom: 40, left: 56, containLabel: true },
    xAxis: axis({ type: "category", data: drivers, boundaryGap: true }),
    yAxis: axis({ type: "value", scale: true, name: "Lap time (s)" }),
    series: [boxSeries, ...pointSeries],
  };
}
