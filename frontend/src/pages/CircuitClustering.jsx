import { useMemo, useState } from "react";
import AsyncSection from "../components/AsyncSection";
import EChart from "../components/EChart";
import { useAsync } from "../hooks/useAsync";
import { axis } from "../lib/echartsTheme";
import { categoryBarOption, radarOption, scatterGroupsOption } from "../lib/plotlyAdapters";
import {
  getClusterAssignments,
  getClusterMeanPlot,
  getClusterPcaPlot,
  getClusterRadarPlot,
  getClusterScatterPlot,
  getClusteringColumns,
  getSilhouetteScores,
} from "../api/client";
import { humanizeSlug } from "../utils/format";

const COMPARE_OPTIONS = ["Average by variable", "Scatter plot", "Radar profile", "PCA projection"];

// A simple, transparent heuristic for turning the 3 raw clustering features
// into a plain-language type - not a scientifically tuned label, just enough
// for a fan to recognize what a cluster is about at a glance.
function labelCluster(stats, overall) {
  const speedDelta = stats.avg_speed - overall.avg_speed;
  const slowDelta = stats.slow_corners_prop - overall.slow_corners_prop;
  const straightDelta = stats.straight_prop - overall.straight_prop;

  if (slowDelta > 0.03) return "Technical & Twisty";
  if (speedDelta > 10 && slowDelta < 0) return "Fast & Flowing";
  if (straightDelta > 0.05) return "Power Circuit";
  return "Balanced";
}

function average(rows, key) {
  return rows.reduce((sum, r) => sum + r[key], 0) / rows.length;
}

export default function CircuitClustering() {
  const [nClusters, setNClusters] = useState(7);
  const [compareView, setCompareView] = useState(COMPARE_OPTIONS[0]);
  const [meanColumn, setMeanColumn] = useState("avg_speed");
  const [col1, setCol1] = useState("avg_speed");
  const [col2, setCol2] = useState("straight_prop");
  const [markerSize, setMarkerSize] = useState(15);

  const columns = useAsync(() => getClusteringColumns(), []);
  const silhouette = useAsync(() => getSilhouetteScores(2, 12), []);
  const assignments = useAsync(() => getClusterAssignments(nClusters), [nClusters]);

  const top3 = useMemo(() => {
    if (!silhouette.data) return [];
    return [...silhouette.data].sort((a, b) => b.silhouette_score - a.silhouette_score).slice(0, 3);
  }, [silhouette.data]);

  const silhouetteOption = useMemo(() => {
    if (!silhouette.data) return null;
    const topKs = new Set(top3.map((d) => d.k));
    return {
      title: { text: "Selection of the Top 3 k by Silhouette Score" },
      legend: false,
      grid: { top: 56, right: 24, bottom: 40, left: 56, containLabel: true },
      xAxis: axis({ type: "category", data: silhouette.data.map((d) => d.k), name: "Number of clusters (k)" }),
      yAxis: axis({ type: "value", name: "Silhouette Score" }),
      series: [
        {
          type: "line",
          data: silhouette.data.map((d) => ({
            value: d.silhouette_score,
            itemStyle: topKs.has(d.k) ? { color: "#ff5a52" } : undefined,
            symbolSize: topKs.has(d.k) ? 12 : 6,
          })),
        },
      ],
    };
  }, [silhouette.data, top3]);

  const clusterLabels = useMemo(() => {
    if (!assignments.data) return {};
    const overall = {
      avg_speed: average(assignments.data, "avg_speed"),
      slow_corners_prop: average(assignments.data, "slow_corners_prop"),
      straight_prop: average(assignments.data, "straight_prop"),
    };
    const byCluster = {};
    for (const row of assignments.data) {
      (byCluster[row.cluster] ??= []).push(row);
    }
    return Object.fromEntries(
      Object.entries(byCluster).map(([cluster, rows]) => [
        cluster,
        labelCluster(
          {
            avg_speed: average(rows, "avg_speed"),
            slow_corners_prop: average(rows, "slow_corners_prop"),
            straight_prop: average(rows, "straight_prop"),
          },
          overall
        ),
      ])
    );
  }, [assignments.data]);

  const meanPlot = useAsync(
    () => getClusterMeanPlot(meanColumn, nClusters),
    [meanColumn, nClusters],
    compareView === "Average by variable"
  );
  const scatterPlot = useAsync(
    () => getClusterScatterPlot(col1, col2, nClusters, markerSize),
    [col1, col2, nClusters, markerSize],
    compareView === "Scatter plot"
  );
  const radarPlot = useAsync(
    () => getClusterRadarPlot(nClusters, 0.3),
    [nClusters],
    compareView === "Radar profile"
  );
  const pcaPlot = useAsync(
    () => getClusterPcaPlot(nClusters, markerSize),
    [nClusters, markerSize],
    compareView === "PCA projection"
  );

  return (
    <div className="page">
      <h1>🧭 Circuits</h1>
      <p className="page-intro">
        Every track drives differently. We grouped them by how they actually
        feel to drive — from real lap telemetry, not opinion.
      </p>

      <div className="controls-row">
        <label>
          Group circuits into
          <input
            type="range"
            min={2}
            max={12}
            value={nClusters}
            onChange={(e) => setNClusters(Number(e.target.value))}
          />
          {nClusters} types
        </label>
      </div>

      <AsyncSection loading={assignments.loading} error={assignments.error}>
        <div className="circuit-grid">
          {assignments.data
            ?.slice()
            .sort((a, b) => humanizeSlug(a.index).localeCompare(humanizeSlug(b.index)))
            .map((row) => (
              <div className="circuit-card" key={row.index}>
                <div className="circuit-card-header">
                  <span className="circuit-card-name">{humanizeSlug(row.index)}</span>
                  <span className={`badge cluster-${row.cluster % 10}`}>
                    {clusterLabels[row.cluster] ?? `Type ${row.cluster}`}
                  </span>
                </div>
                <div className="circuit-card-stats">
                  <span>Avg. speed: {row.avg_speed.toFixed(0)} km/h</span>
                  <span>Straight: {(row.straight_prop * 100).toFixed(0)}%</span>
                  <span>Slow corners: {(row.slow_corners_prop * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
        </div>
      </AsyncSection>

      <details className="disclosure">
        <summary>How are these groups formed?</summary>
        <p className="hint">
          A K-Means model groups circuits by average speed, proportion of
          straights, and proportion of slow corners. The chart below shows
          how well-separated the groups are (silhouette score) for different
          numbers of types — higher is better.
        </p>
        <AsyncSection loading={silhouette.loading} error={silhouette.error}>
          <EChart option={silhouetteOption} />
          <div className="top-k-strip">
            {top3.map((t, i) => (
              <div className="top-k-card" key={t.k}>
                <span className="top-k-rank">#{i + 1}</span>
                <span className="top-k-value">{t.k} types</span>
                <span className="top-k-score">{t.silhouette_score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </AsyncSection>
      </details>

      <details className="disclosure">
        <summary>Compare circuits visually</summary>
        <div className="controls-row">
          <label>
            View
            <select value={compareView} onChange={(e) => setCompareView(e.target.value)}>
              {COMPARE_OPTIONS.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>

          {compareView === "Average by variable" && columns.data && (
            <label>
              Variable
              <select value={meanColumn} onChange={(e) => setMeanColumn(e.target.value)}>
                {columns.data.all.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
          )}

          {compareView === "Scatter plot" && columns.data && (
            <>
              <label>
                X
                <select value={col1} onChange={(e) => setCol1(e.target.value)}>
                  {columns.data.clustering_features.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label>
                Y
                <select value={col2} onChange={(e) => setCol2(e.target.value)}>
                  {columns.data.clustering_features.filter((c) => c !== col1).map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </label>
            </>
          )}

          {(compareView === "Scatter plot" || compareView === "PCA projection") && (
            <label>
              Marker size
              <input
                type="range"
                min={1}
                max={30}
                value={markerSize}
                onChange={(e) => setMarkerSize(Number(e.target.value))}
              />
              {markerSize}
            </label>
          )}
        </div>

        {compareView === "Average by variable" && (
          <AsyncSection loading={meanPlot.loading} error={meanPlot.error}>
            <EChart option={meanPlot.data && categoryBarOption(meanPlot.data)} />
          </AsyncSection>
        )}
        {compareView === "Scatter plot" && (
          <AsyncSection loading={scatterPlot.loading} error={scatterPlot.error}>
            <EChart option={scatterPlot.data && scatterGroupsOption(scatterPlot.data)} />
          </AsyncSection>
        )}
        {compareView === "Radar profile" && (
          <AsyncSection loading={radarPlot.loading} error={radarPlot.error}>
            <EChart option={radarPlot.data && radarOption(radarPlot.data)} />
          </AsyncSection>
        )}
        {compareView === "PCA projection" && (
          <AsyncSection loading={pcaPlot.loading} error={pcaPlot.error}>
            <EChart option={pcaPlot.data && scatterGroupsOption(pcaPlot.data)} />
          </AsyncSection>
        )}
      </details>
    </div>
  );
}
