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

const VIZ_OPTIONS = ["Clusters", "Scatter", "Radar", "PCA"];

export default function CircuitClustering() {
  const [started, setStarted] = useState(false);
  const [nClusters, setNClusters] = useState(7);
  const [vizType, setVizType] = useState(VIZ_OPTIONS[0]);
  const [meanColumn, setMeanColumn] = useState("avg_speed");
  const [col1, setCol1] = useState("avg_speed");
  const [col2, setCol2] = useState("straight_prop");
  const [markerSize, setMarkerSize] = useState(15);

  const columns = useAsync(() => getClusteringColumns(), [], started);
  const silhouette = useAsync(() => getSilhouetteScores(2, 12), [], started);
  const assignments = useAsync(() => getClusterAssignments(nClusters), [nClusters], started);

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

  const meanPlot = useAsync(
    () => getClusterMeanPlot(meanColumn, nClusters),
    [meanColumn, nClusters],
    started && vizType === "Clusters"
  );
  const scatterPlot = useAsync(
    () => getClusterScatterPlot(col1, col2, nClusters, markerSize),
    [col1, col2, nClusters, markerSize],
    started && vizType === "Scatter"
  );
  const radarPlot = useAsync(
    () => getClusterRadarPlot(nClusters, 0.3),
    [nClusters],
    started && vizType === "Radar"
  );
  const pcaPlot = useAsync(
    () => getClusterPcaPlot(nClusters, markerSize),
    [nClusters, markerSize],
    started && vizType === "PCA"
  );

  return (
    <div className="page">
      <h1>Magic circuit cluster 🪄</h1>

      {!started && (
        <div className="controls-row">
          <button onClick={() => setStarted(true)}>Load Data</button>
        </div>
      )}

      {started && (
        <>
          <AsyncSection loading={silhouette.loading} error={silhouette.error}>
            <EChart option={silhouetteOption} />
            <div className="top-k-strip">
              {top3.map((t, i) => (
                <div className="top-k-card" key={t.k}>
                  <span className="top-k-rank">#{i + 1}</span>
                  <span className="top-k-value">k = {t.k}</span>
                  <span className="top-k-score">{t.silhouette_score.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </AsyncSection>

          <div className="controls-row">
            <label>
              Number of clusters
              <input
                type="range"
                min={2}
                max={12}
                value={nClusters}
                onChange={(e) => setNClusters(Number(e.target.value))}
              />
              {nClusters}
            </label>
          </div>

          <AsyncSection loading={assignments.loading} error={assignments.error}>
            <div className="cluster-badges">
              {assignments.data
                ?.slice()
                .sort((a, b) => a.cluster - b.cluster)
                .map((row) => (
                  <span key={row.index} className={`badge cluster-${row.cluster % 10}`}>
                    {humanizeSlug(row.index)} (Cluster {row.cluster})
                  </span>
                ))}
            </div>
          </AsyncSection>

          <div className="controls-row">
            <label>
              Visualization
              <select value={vizType} onChange={(e) => setVizType(e.target.value)}>
                {VIZ_OPTIONS.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </label>

            {vizType === "Clusters" && columns.data && (
              <label>
                Variable
                <select value={meanColumn} onChange={(e) => setMeanColumn(e.target.value)}>
                  {columns.data.all.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </label>
            )}

            {vizType === "Scatter" && columns.data && (
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

            {(vizType === "Scatter" || vizType === "PCA") && (
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

          {vizType === "Clusters" && (
            <AsyncSection loading={meanPlot.loading} error={meanPlot.error}>
              <EChart option={meanPlot.data && categoryBarOption(meanPlot.data)} />
            </AsyncSection>
          )}
          {vizType === "Scatter" && (
            <AsyncSection loading={scatterPlot.loading} error={scatterPlot.error}>
              <EChart option={scatterPlot.data && scatterGroupsOption(scatterPlot.data)} />
            </AsyncSection>
          )}
          {vizType === "Radar" && (
            <AsyncSection loading={radarPlot.loading} error={radarPlot.error}>
              <EChart option={radarPlot.data && radarOption(radarPlot.data)} />
            </AsyncSection>
          )}
          {vizType === "PCA" && (
            <AsyncSection loading={pcaPlot.loading} error={pcaPlot.error}>
              <EChart option={pcaPlot.data && scatterGroupsOption(pcaPlot.data)} />
            </AsyncSection>
          )}
        </>
      )}
    </div>
  );
}
