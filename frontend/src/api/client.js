const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request(path, options) {
  const response = await fetch(`${BASE_URL}${path}`, options);
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Non-JSON error body (e.g. network-level failure) — fall back to statusText.
    }
    throw new Error(message);
  }
  return response.json();
}

function query(params) {
  const usable = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")
  );
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(usable)) {
    if (Array.isArray(value)) {
      value.forEach((v) => search.append(key, v));
    } else {
      search.append(key, value);
    }
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

// Races
export const getRaceResults = (year, round, sessionType) =>
  request(`/races/${year}/${round}/${sessionType}/results`);

export const getSessionDrivers = (year, round, sessionType) =>
  request(`/races/${year}/${round}/${sessionType}/drivers`);

export const getFastestLaps = (year, round, sessionType) =>
  request(`/races/${year}/${round}/${sessionType}/fastest-laps`);

export const getTrack = (year, round, sessionType) =>
  request(`/races/${year}/${round}/${sessionType}/track`);

export const getTelemetry = (year, round, sessionType, mode, drivers) =>
  request(`/races/${year}/${round}/${sessionType}/telemetry${query({ mode, drivers })}`);

export const getStandingsChart = (year, round, sessionType) =>
  request(`/races/${year}/${round}/${sessionType}/standings`);

export const getPositionChanges = (year, round, sessionType) =>
  request(`/races/${year}/${round}/${sessionType}/position-changes`);

export const getDriverPace = (year, round, sessionType, driver, threshold) =>
  request(`/races/${year}/${round}/${sessionType}/driver-pace${query({ driver, threshold })}`);

export const getPace = (year, round, sessionType, kind, threshold, box) =>
  request(`/races/${year}/${round}/${sessionType}/pace${query({ kind, threshold, box })}`);

export const getTyreStrategy = (year, round, sessionType) =>
  request(`/races/${year}/${round}/${sessionType}/tyre-strategy`);

// Seasons
export const getDriversChampionship = (year, top) =>
  request(`/seasons/${year}/drivers-championship${query({ top })}`);

export const getConstructorsChampionship = (year, top) =>
  request(`/seasons/${year}/constructors-championship${query({ top })}`);

// Clustering
export const getClusteringColumns = () => request("/clustering/columns");

export const getSilhouetteScores = (kMin, kMax) =>
  request(`/clustering/silhouette-scores${query({ k_min: kMin, k_max: kMax })}`);

export const getClusterAssignments = (nClusters) =>
  request(`/clustering/assignments${query({ n_clusters: nClusters })}`);

export const getClusterMeanPlot = (column, nClusters) =>
  request(`/clustering/plot/mean-by-cluster${query({ column, n_clusters: nClusters })}`);

export const getClusterScatterPlot = (col1, col2, nClusters, markerSize) =>
  request(`/clustering/plot/scatter${query({ col1, col2, n_clusters: nClusters, marker_size: markerSize })}`);

export const getClusterRadarPlot = (nClusters, opacity) =>
  request(`/clustering/plot/radar${query({ n_clusters: nClusters, opacity })}`);

export const getClusterPcaPlot = (nClusters, markerSize) =>
  request(`/clustering/plot/pca${query({ n_clusters: nClusters, marker_size: markerSize })}`);

// Predictions
export const predictWinner = (payload) =>
  request("/predictions/winner", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

// Reference data
export const getSeasonRounds = (year) => request(`/reference/seasons/${year}/rounds`);

export const getRoundEntrants = (year, round) =>
  request(`/reference/seasons/${year}/rounds/${round}/entrants`);
