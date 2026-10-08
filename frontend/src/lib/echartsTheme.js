// Chart look, driven by the Atlas design tokens.
//
// ECharts draws on a canvas and can't read CSS variables, so adapters write
// colours as *token strings* and EChart.jsx resolves them to real colours at
// render time, per theme (see `resolveOption`). That keeps adapters free of
// theme state and makes light/dark switching a re-render of one component.
//
//   "var(--teal)"            a design token
//   "var(--ink-faint)@0.3"   a token with alpha
//   dataColor("#ff8700")     a colour that comes from the data (team, driver,
//                            tyre...). Identity is kept, but pushed away from
//                            the page background when it would vanish into it
//                            (a white team on a light page, a black one on dark).
//   rawColor("#ffd12e", .2)  same, but never adjusted (fills with an outline)

export const CHART_FONT = '"Geist", system-ui, sans-serif';
export const CHART_MONO = '"Geist Mono", ui-monospace, Menlo, monospace';

export const CLUSTER_COUNT = 12;
const SYMBOLS = ["circle", "rect", "triangle", "diamond"];

// Cluster i gets the same colour everywhere (badge, scatter, bar, radar):
// `--cat-N` by absolute position, never by size or by what's currently shown.
export const clusterColor = (i) => `var(--cat-${(i % CLUSTER_COUNT) + 1})`;
// Past ~7 colours alone can't separate groups (see design/f1.css), so charts
// add a shape as a second channel.
export const clusterSymbol = (i) => SYMBOLS[i % SYMBOLS.length];

export const dataColor = (color) => `data:${color}`;
export const rawColor = (color, alpha = 1) => `raw:${color}@${alpha}`;

export const CHART_PALETTE = Array.from({ length: CLUSTER_COUNT }, (_, i) => `var(--cat-${i + 1})`);

export const baseOption = () => ({
  backgroundColor: "transparent",
  color: CHART_PALETTE,
  textStyle: { fontFamily: CHART_FONT, color: "var(--ink)" },
  tooltip: {
    backgroundColor: "var(--card)",
    borderColor: "var(--hairline-strong)",
    textStyle: { color: "var(--ink)", fontFamily: CHART_MONO, fontSize: 12 },
    extraCssText: "box-shadow: var(--shadow); border-radius: 8px;",
  },
  legend: {
    textStyle: { color: "var(--ink-soft)", fontFamily: CHART_FONT, fontSize: 12 },
    inactiveColor: "var(--ink-faint)",
    itemWidth: 14,
    itemHeight: 8,
    top: 4,
  },
});

export const axis = (overrides = {}) => ({
  axisLine: { lineStyle: { color: "var(--hairline-strong)" } },
  axisTick: { lineStyle: { color: "var(--hairline-strong)" } },
  axisLabel: { color: "var(--ink-faint)", fontFamily: CHART_MONO, fontSize: 11 },
  nameTextStyle: { color: "var(--ink-faint)", fontFamily: CHART_MONO, fontSize: 11 },
  splitLine: { lineStyle: { color: "var(--hairline)" } },
  ...overrides,
});

// ── token resolution ────────────────────────────────────────────────────────

let canvasCtx;
// Normalises any CSS colour (oklch included) to "#rrggbb" / "rgba(...)" via the
// canvas, which serialises fillStyle in sRGB.
function toRgb(css) {
  canvasCtx ??= document.createElement("canvas").getContext("2d");
  canvasCtx.fillStyle = "#000000";
  canvasCtx.fillStyle = css;
  return canvasCtx.fillStyle;
}

function parseRgb(color) {
  const c = toRgb(color);
  if (c.startsWith("#")) {
    return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)).concat(1);
  }
  const [r, g, b, a = 1] = c.match(/[\d.]+/g).map(Number);
  return [r, g, b, a];
}

const rgba = ([r, g, b], a) => `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${a})`;

function luminance([r, g, b]) {
  const lin = (v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

const mix = (rgb, target, t) => rgb.slice(0, 3).map((v, i) => v + (target[i] - v) * t);

function legible(color, mode) {
  const rgb = parseRgb(color);
  const l = luminance(rgb);
  if (mode === "light" && l > 0.5) return rgba(mix(rgb, [20, 24, 32], 0.55), 1);
  // Dark pages need real lift: Red Bull's #0600EF (l≈0.06) is unreadable as a thin line on them.
  if (mode === "dark" && l < 0.2) return rgba(mix(rgb, [240, 244, 248], Math.min(0.7, (0.2 - l) * 3.6)), 1);
  return rgba(rgb, 1);
}

function resolveString(value, mode, rootStyle) {
  const token = value.match(/^var\((--[\w-]+)\)(?:@([\d.]+))?$/);
  if (token) {
    const css = rootStyle.getPropertyValue(token[1]).trim();
    if (!css) return value;
    return token[2] ? rgba(parseRgb(css), Number(token[2])) : toRgb(css);
  }
  if (value.startsWith("data:")) return legible(value.slice(5), mode);
  if (value.startsWith("raw:")) {
    const [color, alpha] = value.slice(4).split("@");
    return rgba(parseRgb(color), Number(alpha));
  }
  return value;
}

// Deep-copies `option`, resolving every token string. Functions (formatters,
// renderItem) and everything else are passed through untouched.
export function resolveOption(option, mode) {
  const rootStyle = getComputedStyle(document.documentElement);
  const walk = (node) => {
    if (typeof node === "string") return resolveString(node, mode, rootStyle);
    if (Array.isArray(node)) return node.map(walk);
    if (node && typeof node === "object" && Object.getPrototypeOf(node) === Object.prototype) {
      return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, walk(v)]));
    }
    return node;
  };
  return walk(option);
}
