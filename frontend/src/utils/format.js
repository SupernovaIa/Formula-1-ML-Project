// "GridPosition" -> "Grid Position", "Q1" -> "Q1" (kept as-is)
export function humanizeHeader(key) {
  if (/^[A-Z0-9]+$/.test(key)) return key;
  return key.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
}

// "albert_park" -> "Albert Park"
export function humanizeSlug(slug) {
  if (!slug) return slug;
  return slug
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

// Table cells from the backend: NaT/NaN/null -> an em dash, lap times keep
// their milliseconds ("01:30" -> "1:30.000", "01:31.295" -> "1:31.295").
export function formatCell(value) {
  if (value == null || value === "NaT" || value === "NaN" || value === "") return "—";
  if (typeof value === "string") {
    const t = value.match(/^0?(\d{1,2}:\d{2})(?:\.(\d{1,3}))?$/);
    if (t) return `${t[1]}.${(t[2] ?? "").padEnd(3, "0")}`;
  }
  return String(value);
}
