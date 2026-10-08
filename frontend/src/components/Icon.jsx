// Lucide icons from the Atlas design system (src/design/icons). Only the files
// copied there are available - add one by copying its SVG from Atlas.
const FILES = import.meta.glob("../design/icons/*.svg", { query: "?raw", import: "default", eager: true });

const INNER = Object.fromEntries(
  Object.entries(FILES).map(([path, svg]) => [
    path.split("/").pop().replace(".svg", ""),
    svg.replace(/^<svg[^>]*>/, "").replace(/<\/svg>\s*$/, ""),
  ])
);

export default function Icon({ name, className = "" }) {
  const inner = INNER[name];
  if (!inner) throw new Error(`Unknown icon "${name}" - copy it to src/design/icons/`);
  return (
    <svg
      className={`a-icon ${className}`.trim()}
      viewBox="0 0 24 24"
      aria-hidden="true"
      dangerouslySetInnerHTML={{ __html: inner }}
    />
  );
}
