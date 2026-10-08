import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { baseOption, resolveOption } from "../lib/echartsTheme";
import { useThemeMode } from "../hooks/useThemeMode";

// echarts is a large dependency - load it lazily so pages without a chart
// (like Home) don't pay for it in their initial bundle.
const ReactECharts = lazy(() => import("echarts-for-react"));

export default function EChart({ option, height = 440 }) {
  const mode = useThemeMode();
  const fontsReady = useFontsReady();

  const resolved = useMemo(() => {
    if (!option) return null;
    const base = baseOption();
    return resolveOption(
      {
        ...base,
        ...option,
        // The title lives in the panel header (Atlas label), not in the canvas.
        title: undefined,
        textStyle: { ...base.textStyle, ...option.textStyle },
        tooltip: { ...base.tooltip, ...option.tooltip },
        legend: option.legend === false ? undefined : { ...base.legend, ...option.legend },
      },
      mode
    );
    // fontsReady: redraw once Geist is available so canvas text doesn't stay in the fallback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [option, mode, fontsReady]);

  if (!resolved) return null;
  const [title, ...subtitle] = (option.title?.text ?? "").split(/<br\s*\/?>/i);

  return (
    <figure className="chart a-panel">
      {title && (
        <figcaption className="chart-caption">
          <span className="a-label">{title}</span>
          {subtitle.length > 0 && <span className="chart-sub">{subtitle.join(" · ")}</span>}
        </figcaption>
      )}
      <Suspense fallback={<div className="chart-fallback" style={{ height }} />}>
        <ReactECharts option={resolved} style={{ width: "100%", height }} notMerge lazyUpdate />
      </Suspense>
    </figure>
  );
}

function useFontsReady() {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    document.fonts?.ready.then(() => setReady(true));
  }, []);
  return ready;
}
