import { lazy, Suspense } from "react";
import { BASE_OPTION } from "../lib/echartsTheme";

// echarts is a large dependency — load it lazily so pages without a chart
// (like Home) don't pay for it in their initial bundle.
const ReactECharts = lazy(() => import("echarts-for-react"));

export default function EChart({ option, height = 480 }) {
  if (!option) return null;

  const merged = {
    ...BASE_OPTION,
    ...option,
    textStyle: { ...BASE_OPTION.textStyle, ...option.textStyle },
    tooltip: { ...BASE_OPTION.tooltip, ...option.tooltip },
    legend: option.legend === false ? undefined : { ...BASE_OPTION.legend, ...option.legend },
  };

  return (
    <div className="chart-panel">
      <Suspense fallback={<div className="chart-panel-fallback" style={{ height }} />}>
        <ReactECharts
          option={merged}
          style={{ width: "100%", height }}
          notMerge
          lazyUpdate
        />
      </Suspense>
    </div>
  );
}
