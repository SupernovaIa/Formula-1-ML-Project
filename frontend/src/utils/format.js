// "GridPosition" -> "Grid Position", "Q1" -> "Q1" (kept as-is)
export function humanizeHeader(key) {
  if (/^[A-Z0-9]+$/.test(key)) return key;
  return key.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
}
